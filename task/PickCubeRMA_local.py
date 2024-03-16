from typing import Dict, Union
from collections import OrderedDict

import torch
import numpy as np
import sapien.core as sapien
from mani_skill2.envs.pick_and_place.pick_cube import PickCubeEnv
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.common import flatten_state_dict
from mani_skill2.utils.sapien_utils import (vectorize_pose, 
                                            get_pairwise_contact_impulse)
from algo.misc import get_task_id, get_object_id, linear_schedule

@register_env("PickCube-v1", max_episode_steps=200, override=True)
class PickCubeRMA(PickCubeEnv):
    '''
    PickCubeEnv with observation statedict that has keys: ['observation',
    'privaleged_info', 'goal_info'] which can be more easily used in RMA 
    training.
    '''
    def __init__(self, *args, obj_init_rot_z=True, 
                randomized_training=False, 
                noisy_observation=False,
                 **kwargs):
        
        self.obj_init_rot_z = obj_init_rot_z
        self.randomized_env = randomized_training 
        self._elapsed_steps = 0
        # --- domain randomization
        if self.randomized_env:
            init_step = 1.5e7
            end_step = 2.5e7
            self.scale_h_scdl = linear_schedule(1.0, 1.2, init_step, end_step)
            self.scale_l_scdl = linear_schedule(1.0, 0.7, init_step, end_step)
            self.mass_h_scdl = linear_schedule(0.064, 0.5, init_step, end_step)
            self.mass_l_scdl = linear_schedule(0.064, 0.01, init_step, end_step)
            self.fric_h_scdl = linear_schedule(1., 1.1, init_step, end_step)
            self.fric_l_scdl = linear_schedule(1., 0.5, init_step, end_step)
            scale_h = self.scale_h_scdl(elapsed_steps=self.elapsed_steps)
            scale_l = self.scale_l_scdl(elapsed_steps=self.elapsed_steps)
            self.cube_scale = np.random.uniform(scale_l, scale_h)
        else:
            self.cube_scale = 1.0
        # --- noisy observation
        if self.noisy_observation:
            # todo: calibrate the ranges
            init_step = 1.5e7
            end_step = 2.5e7
            self.obj_pose_h_scdl = linear_schedule(0., 0.1, init_step, end_step)
            self.obj_pose_l_scdl = linear_schedule(0., -0.1, init_step, end_step)
            # self.agent_pose_h_scdl = linear_schedule(0., 0.1, init_step, end_step)
            # self.agent_pose_l_scdl = linear_schedule(0., -0.1, init_step, end_step)
            obj_pose_h = self.obj_pose_h_scdl(elapsed_steps=self._elapsed_steps)
            obj_pose_l = self.obj_pose_l_scdl(elapsed_steps=self._elapsed_steps)
            # agent_pose_h = self.agent_pose_h_scdl(elapsed_steps=self._elapsed_steps)
            # agent_pose_l = self.agent_pose_l_scdl(elapsed_steps=self._elapsed_steps)
            self.obj_pose_noise = np.random.uniform(obj_pose_l, obj_pose_h, 7)
            # self.agent_pose_noise = np.random.uniform(agent_pose_l, agent_pose_h, 7)
        else:
            self.obj_pose_noise = np.zeros(7)
            self.agent_pose_noise = np.zeros(7)



        self.cube_half_size = np.array([0.02]*3, np.float32) * self.cube_scale
        self.bbox_size = self.cube_half_size[0] * 2 * np.sqrt(3)
        super(PickCubeEnv, self).__init__(*args, **kwargs)

    def _get_obs_state_dict(self):
        """Get (GT) state-based observations."""
        contacts = self._scene.get_contacts()
        limpulse = np.linalg.norm(get_pairwise_contact_impulse(contacts, #3dim
                                self.agent.finger1_link, self.obj), ord=2)
        rimpulse = np.linalg.norm(get_pairwise_contact_impulse(contacts, 
                                self.agent.finger2_link, self.obj), ord=2)
        agent_state = flatten_state_dict(OrderedDict(
                    proprioception=self.agent.get_proprioception(),
                    base_pose=vectorize_pose(self.agent.robot.pose),
                    tcp_pose=vectorize_pose(self.tcp.pose),
                    ))
        obj_pose = vectorize_pose(self.obj.pose) + self.obj_pose_noise
        
        return OrderedDict(
            # task_id=get_task_id("PickCube"),
            agent_state=agent_state,
            # object1_id=get_object_id(task_name="PickCube"),
            object1_state=flatten_state_dict(OrderedDict(
                        # add center of mass
                        bbox_size=self.bbox_size,
                        obj_pose=obj_pose,
                        tcp_to_obj_pos=self.obj.pose.p - self.tcp.pose.p,
                    )),
            obj1_priv_info=flatten_state_dict(OrderedDict(
                        obj_density=self.obj_density,
                        obj_friction=self.obj_friction,
                        limpulse = limpulse,
                        rimpulse = rimpulse,
                        agent_pose_noise = self.agent_pose_noise,
                        obj_pose_noise = self.obj_pose_noise,
                    )),
            goal_info=flatten_state_dict(OrderedDict(
                        target_pos=self.goal_pos,
                        tcp_to_goal_pos=self.goal_pos - self.tcp.pose.p,
                        obj_to_goal_pos=self.goal_pos - self.obj.pose.p
                    )).astype("float32"),
        )

    def _build_cube(
        self,
        half_size,
        color=(1, 0, 0),
        name="cube",
        static=False,
        render_material: sapien.RenderMaterial = None,
    ):
        '''build cube with varying mass
        '''
        if render_material is None:
            render_material = self._renderer.create_material()
            render_material.set_base_color(np.hstack([color, 1.0]))

        builder = self._scene.create_actor_builder()

        # --- randomize mass ---
        # default mass: 0.064 kg
        if self.randomized_env:
            mass_h = self.mass_h_scdl(elapsed_steps=self.elapsed_steps)
            mass_l = self.mass_l_scdl(elapsed_steps=self.elapsed_steps)
            mass = self._episode_rng.uniform(mass_l, mass_h)
        else:
            mass = np.array(0.064)
        self.obj_density = mass / ((self.cube_half_size[0]*2)**3)

        inertia_pose = sapien.Pose(np.zeros(3))
        # principle moments of inertia (a 3D vector)
        # https://sapien.ucsd.edu/docs/latest/apidoc/sapien.core.html#sapien.core.pysapien.ActorBuilder.set_mass_and_inertia
        pmi = np.zeros(3)
        builder.set_mass_and_inertia(mass, inertia_pose, pmi)

        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, material=render_material)
        if static:
            return builder.build_static(name)
        else:
            return builder.build(name)

    def _initialize_actors(self):
        '''called in `initialize_episode()`
        things doesn't require reconfiguring the simulation scene
        '''
        super()._initialize_actors()

        # --- randomize friction ---
        if self.randomized_env:
            fric_h = self.fric_h_scdl(elapsed_steps=self.elapsed_steps)
            fric_l = self.fric_l_scdl(elapsed_steps=self.elapsed_steps)
            self.obj_friction = self._episode_rng.uniform(fric_l, fric_h)
        else:
            self.obj_friction = np.array(1.)
        phys_mtl = self._scene.create_physical_material(
                        static_friction=self.obj_friction, 
                        dynamic_friction=self.obj_friction, restitution=0.1
        )
        # physical material only have friction related properties
        # https://github.com/haosulab/SAPIEN/blob/ab1d9a9fa1428484a918e61185ae9df2beb7cb30/python/py_package/core/pysapien/__init__.pyi#L1088
        for cs in self.obj.get_collision_shapes():
            cs.set_physical_material(phys_mtl)

        # # randomize damping
        # # from ActorDynamicBase class
        # # https://github.com/haosulab/SAPIEN/blob/ab1d9a9fa1428484a918e61185ae9df2beb7cb30/python/py_package/core/pysapien/__init__.pyi#L161
        # linear_damping = self._episode_rng.uniform(0, 1.0)
        # angular_damping = self._episode_rng.uniform(0, 1.0)
        # self.obj.set_damping(linear_damping, angular_damping)

@register_env("DR-PickCube-v0", max_episode_steps=100, override=True)
class PickCube(PickCubeEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_actors(self):
        '''called in `reset()` and `reconfigure()`
        things require reconfiguring the simulation scene.
        '''
        # --- randomize object size ---
        # self.cube_half_size += self._episode_rng.uniform(-0.005, 0.01, size=3)
        super()._load_actors()

    def _initialize_actors(self):
        '''called in `initialize_episode()`
        things doesn't require reconfiguring the simulation scene
        '''
        super()._initialize_actors()

        # # --- randomize friction ---
        # self.obj_friction = self._episode_rng.uniform(0.5, 1.0)
        # phys_mtl = self._scene.create_physical_material(
        #     static_friction=self.obj_friction, 
        #     dynamic_friction=self.obj_friction, restitution=0.1
        # )
        # # physical material only have friction related properties
        # # https://github.com/haosulab/SAPIEN/blob/ab1d9a9fa1428484a918e61185ae9df2beb7cb30/python/py_package/core/pysapien/__init__.pyi#L1088
        # for cs in self.obj.get_collision_shapes():
        #     cs.set_physical_material(phys_mtl)

        # # randomize damping
        # # from ActorDynamicBase class
        # # https://github.com/haosulab/SAPIEN/blob/ab1d9a9fa1428484a918e61185ae9df2beb7cb30/python/py_package/core/pysapien/__init__.pyi#L161
        # linear_damping = self._episode_rng.uniform(0, 1.0)
        # angular_damping = self._episode_rng.uniform(0, 1.0)
        # self.obj.set_damping(linear_damping, angular_damping)

    def _initialize_agent(self):
        if self.robot_uid == "panda":
            # fmt: off
            # EE at [0.615, 0, 0.17]
            # --- randomize joint position ---
            # joint angle - original
            #   the first 7dim: the 7dof of the arm, 
            #   the last  2dim: the end effector position
            qpos = np.array(
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, 
                 np.pi / 4, 0.04, 0.04])

            # fmt: on
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)

            # base position - original
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

        elif self.robot_uid == "xmate3_robotiq":
            qpos = np.array(
                [0, np.pi / 6, 0, np.pi / 3, 0, np.pi / 2, -np.pi / 2, 0, 0]
            )
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.562, 0, 0]))
        else:
            raise NotImplementedError(self.robot_uid)
        
        # --- randomize joint stiffness and damping ---
        # for j in self.agent.robot.get_active_joints():
        #     # access with: j.stiffness, j.damping
        #     # reference: stiffness=1000.0, damping=100.0
        #     sti, dam = j.stiffness, j.damping
        #     sti_scl = self._episode_rng.uniform(0.9, 1.1)
        #     dam_scl = self._episode_rng.uniform(0.99, 1.01)
        #     # sti_scl = self._episode_rng.uniform(1, 1)
        #     # dam_scl = self._episode_rng.uniform(10, 10)
        #     j.set_drive_property(stiffness=sti*sti_scl, damping=dam*dam_scl)


    def _build_cube(
        self,
        half_size,
        color=(1, 0, 0),
        name="cube",
        static=False,
        render_material: sapien.RenderMaterial = None,
    ):
        '''build cube with varying mass
        '''
        if render_material is None:
            render_material = self._renderer.create_material()
            render_material.set_base_color(np.hstack([color, 1.0]))

        builder = self._scene.create_actor_builder()

        # --- randomize mass ---
        # self.randomize_mass_inertia:
        # mass = self._episode_rng.uniform(0.01, 1)
        # inertia_pose = sapien.Pose(self._episode_rng.uniform(-0.005, 0.005, 
        #                                                      size=3))
        # # inertia_pose = sapien.Pose(self._episode_rng.uniform(0,0, size=3))
        # # principle moments of inertia (a 3D vector)
        # # https://sapien.ucsd.edu/docs/latest/apidoc/sapien.core.html#sapien.core.pysapien.ActorBuilder.set_mass_and_inertia
        # pmi = np.zeros(3)
        # builder.set_mass_and_inertia(mass, inertia_pose, pmi)

        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, material=render_material)
        if static:
            return builder.build_static(name)
        else:
            return builder.build(name)

    # def _get_obs_state_dict(self):
    #     """Get (GT) state-based observations."""
    #     return OrderedDict(
    #         observation=np.concatenate([
    #             flatten_state_dict(self._get_obs_agent()),
    #             flatten_state_dict(OrderedDict(
    #                                         obj_pos=self.obj.pose.p,
    #                                         obj_ori=self.obj.pose.q))],axis=-1),
    #         privaleged_info=flatten_state_dict(OrderedDict(
    #                                         obj_mass=self.obj.mass,
    #                                         obj_scale=self.cube_half_size,
    #                                         obj_com=self.obj.cmass_local_pose.p,
    #                                         obj_fri=self.obj_friction)),
    #         goal_info=flatten_state_dict(OrderedDict(target_pos=self.goal_pos)
    #                                                             ).astype("float32"),
    #     )
    def _get_obs_state_dict(self):
        """Get (GT) state-based observations."""
        # --- randomize external disturbance, by adding noise to obs ---
        return OrderedDict(
            observation=flatten_state_dict(OrderedDict(
                    proprio=self.agent.get_proprioception(),
                    tcp_pose=vectorize_pose(self.tcp.pose),
                    tcp_to_goal_pos=self.goal_pos - self.tcp.pose.p,
                    obj_pose=vectorize_pose(self.obj.pose),
                    tcp_to_obj_pos=self.obj.pose.p - self.tcp.pose.p,
                    obj_to_goal_pos=self.goal_pos - self.obj.pose.p)
                                                            ).astype("float32"),
            privaleged_info=np.zeros(1).astype("float32"),
            # privaleged_info=flatten_state_dict(OrderedDict(
            #                             obj_mass=self.obj.mass,
            #                             obj_scale=self.cube_half_size,
            #                             obj_com=self.obj.cmass_local_pose.p,
            #                             obj_fri=self.obj_friction)),
            goal_info=flatten_state_dict(OrderedDict(target_pos=self.goal_pos)
                                                            ).astype("float32"),
            )