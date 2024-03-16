from collections import OrderedDict
import random
from pathlib import Path
import os

from transforms3d.quaternions import axangle2quat, qmult
import numpy as np
from mani_skill2 import format_path
from numpy.linalg import norm
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.common import flatten_state_dict, random_choice
from mani_skill2.envs.misc.turn_faucet import TurnFaucetEnv
# from mani_skill2.utils.sapien_utils import (vectorize_pose, 
#                                             get_pairwise_contact_impulse)
from mani_skill2.utils.sapien_utils import (
    get_entity_by_name,
    hex2rgba,
    look_at,
    set_articulation_render_material,
    vectorize_pose,
    get_pairwise_contact_impulse
)
from mani_skill2.sensors.camera import parse_camera_cfgs

from algo.misc import get_task_id, get_object_id, linear_schedule

@register_env("TurnFaucet-v1", max_episode_steps=200)
class TurnFaucetRMA(TurnFaucetEnv):
    def set_randomization(self):
        if self.test_eval:
            self.l_scl, self.h_scl = 0.8, 1.2
        else:
            self.l_scl, self.h_scl = 1.0, 1.0

        if self.ext_disturbance:
            self.max_force = -np.inf
            self.force_scale_h, self.force_decay = 2 * self.h_scl, 0.8
            self.disturb_force = np.zeros(3)

        if self.randomized_env:
            self.scl_scdl_h = 1.2 * self.h_scl
            self.scl_scdl_l = 0.7 * self.l_scl
            self.dens_scdl_h = 5 * self.h_scl
            self.dens_scdl_l = 0.5 * self.l_scl
            self.fric_scdl_h = 1.1 * self.h_scl
            self.fric_scdl_l = 0.5 * self.l_scl

        if self.obs_noise:
            self.prop_scdl_h = 0.005 * self.h_scl
            self.prop_scdl_l = -0.005 * self.h_scl
            self.pos_scdl_h = 0.005 * self.h_scl
            self.pos_scdl_l = -0.005 * self.h_scl
            self.rot_scdl_h = np.pi * (10/180) * self.h_scl
            self.rot_scdl_l = -np.pi * (10/180) * self.h_scl

        if self.auto_dr:
            self.dr_params = OrderedDict(obj_scale=[1.,1.],
                                        obj_density=[1.,1.], 
                                        obj_friction=[1.,1.],
                                        force_scale=[0.,0.],
                                        obj_position=[0.,0.],
                                        obj_rotation=[0.,0.],
                                        prop_position=[0.,0.],
                                        )
        else:
            init_step, end_step  = 1e6, 2e6 # 15M
            if self.ext_disturbance:
                self.force_scale_scdl = linear_schedule(0., self.force_scale_h, 
                                                        init_step, end_step)
            if self.randomized_env:
                self.scale_h_scdl = linear_schedule(1., self.scl_scdl_h, init_step, end_step)
                self.scale_l_scdl = linear_schedule(1., self.scl_scdl_l, init_step, end_step)
                self.dens_h_scdl = linear_schedule(1., self.dens_scdl_h, init_step, end_step)
                self.dens_l_scdl = linear_schedule(1., self.dens_scdl_l, init_step, end_step)
                self.fric_h_scdl = linear_schedule(1., self.fric_scdl_h, init_step, end_step)
                self.fric_l_scdl = linear_schedule(1., self.fric_scdl_l, init_step, end_step)

            if self.obs_noise:
                # noise in agent joint position and object pose
                self.proprio_h_scdl = linear_schedule(0, self.prop_scdl_h, init_step, end_step)
                self.proprio_l_scdl = linear_schedule(0, self.prop_scdl_l, init_step, end_step)
                self.pos_h_scdl = linear_schedule(0, self.pos_scdl_h, init_step, end_step)
                self.pos_l_scdl = linear_schedule(0, self.pos_scdl_l, init_step, end_step)
                self.rot_h_scdl = linear_schedule(0, self.rot_scdl_h, init_step, 
                                                                end_step)
                self.rot_l_scdl = linear_schedule(0, self.rot_scdl_l, init_step, 
                                                                    end_step)

    def set_step_counter(self, n):
        self.step_counter = n

    def __init__(self, *args,
                auto_dr=False,
                randomized_training=False, 
                obs_noise=False, 
                ext_disturbance=False,
                test_eval=False,
                inc_obs_noise_in_priv=False,
                **kwargs):
        self.step_counter = 0

        self.test_eval = test_eval
        self.randomized_env = randomized_training
        self.obs_noise = obs_noise
        self.ext_disturbance = ext_disturbance
        self.inc_obs_noise_in_priv = inc_obs_noise_in_priv

        self.auto_dr = auto_dr
        self.eval_env = False
        self.randomized_param = None
        self.set_randomization()

        # get object list for computing ids
        parent_folder = "data/partnet_mobility/dataset"
        self.object_list = [f for f in os.listdir(parent_folder) 
                        if os.path.isdir(os.path.join(parent_folder, f))]
        super().__init__(*args, **kwargs)

    def reset(self, seed=None, options=None):
        self.set_episode_rng(seed)
        if self.auto_dr:
            if self.eval_env and self.randomized_param == 'obj_position':
                self.pos_noise = self._episode_rng.choice(self.dr_params['obj_position'])
            else:
                self.pos_noise = nself._episode_rng.uniform(*self.dr_params['obj_position'], 3)

            if self.eval_env and self.randomized_param == 'obj_rotation':
                self.rot_ang = self._episode_rng.choice(self.dr_params['obj_rotation'])
            else:
                self.rot_ang = self._episode_rng.uniform(*self.dr_params['obj_rotation'])
            rot_axis = self._episode_rng.uniform(0, 1, 3)
            self.rot_noise = axangle2quat(rot_axis, self.rot_ang)

            if self.eval_env and self.randomized_param == 'prop_position':
                self.proprio_noise = self._episode_rng.choice(self.dr_params['prop_position'])
            else:
                self.proprio_noise = self._episode_rng.uniform(*self.dr_params['prop_position'], 9)
        elif self.obs_noise:
            # noise to proprioception
            self.proprio_h = self.proprio_h_scdl(elapsed_steps=self.step_counter)
            self.proprio_l = self.proprio_l_scdl(elapsed_steps=self.step_counter)
            self.proprio_noise = self._episode_rng.uniform(self.proprio_l, self.proprio_h, 9)
            self.pos_h = self.pos_h_scdl(elapsed_steps=self.step_counter)
            self.pos_l = self.pos_l_scdl(elapsed_steps=self.step_counter)
            self.pos_noise = self._episode_rng.uniform(self.pos_h, self.pos_l, 3)
            self.rot_h = self.rot_h_scdl(elapsed_steps=self.step_counter) 
            self.rot_l = self.rot_l_scdl(elapsed_steps=self.step_counter)
            rot_axis = self._episode_rng.uniform(0, 1, 3)
            self.rot_ang = self._episode_rng.uniform(self.rot_h, self.rot_l)
            self.rot_noise = axangle2quat(rot_axis, self.rot_ang)
        
        return super().reset(seed, options)

    def _set_model(self, model_id, model_scale):
        """rewrite to save object id and type id
        Set the model id and scale. If not provided, choose one randomly."""
        reconfigure = False

        # Model ID
        if model_id is None:
            model_id = random_choice(self.model_ids, self._episode_rng)
        if model_id != self.model_id:
            reconfigure = True
        self.model_id = model_id
        self.model_info = self.model_db[self.model_id]

        self.obj1_num_id = get_object_id(task_name="TurnFaucet", 
                                        model_id=self.model_id,
                                        object_list=self.object_list)
        self.obj1_type_num_id = get_object_id(task_name="TurnFaucet",
                                            model_id=self.model_id)

        # Scale
        if model_scale is None:
            model_scale = self.model_info.get("scale")
        if model_scale is None:
            bbox = self.model_info["bbox"]
            bbox_size = np.float32(bbox["max"]) - np.float32(bbox["min"])
            model_scale = 0.3 / max(bbox_size)  # hardcode
        if model_scale != self.model_scale:
            reconfigure = True
        self.model_scale = model_scale

        # added to set scale
        if self.auto_dr:
            if self.eval_env and self.randomized_param == 'obj_scale':
                self.model_scale_mult = self._episode_rng.choice(self.dr_params['obj_scale'])
            else:
                self.model_scale_mult = self._episode_rng.uniform(*self.dr_params['obj_scale'])
        elif self.randomized_env:
            self.scale_h = self.scale_h_scdl(elapsed_steps=self.step_counter)
            self.scale_l = self.scale_l_scdl(elapsed_steps=self.step_counter)
            self.model_scale_mult = self._episode_rng.uniform(self.scale_l, self.scale_h)
        else:
            self.model_scale_mult = 1.0
        self.model_scale *= self.model_scale_mult

        if "offset" in self.model_info:
            self.model_offset = np.float32(self.model_info["offset"])
        else:
            self.model_offset = -np.float32(bbox["min"]) * self.model_scale
        # Add a small clearance
        self.model_offset[2] += 0.01

        return reconfigure

 
    def _load_faucet(self):
        loader = self._scene.create_urdf_loader()
        loader.scale = self.model_scale
        loader.fix_root_link = True

        model_dir = self.asset_root / str(self.model_id)
        urdf_path = model_dir / "mobility_cvx.urdf"
        loader.load_multiple_collisions_from_file = True

        density = self.model_info.get("density", 8e3)

        # randomize density
        if self.auto_dr:
            if self.eval_env and self.randomized_param == 'obj_density':
                self.dens_mult = self._episode_rng.choice(self.dr_params['obj_density'])
            else:
                self.dens_mult = self._episode_rng.uniform(*self.dr_params['obj_density'])
        elif self.randomized_env:
            self.dens_h = self.dens_h_scdl(elapsed_steps=self.step_counter)
            self.dens_l = self.dens_l_scdl(elapsed_steps=self.step_counter)
            self.dens_mult = self._episode_rng.uniform(self.dens_l, self.dens_h)
        else:
            self.dens_mult = np.array(1)
        density *= self.dens_mult
        # normalize density
        self.obj_density = density / 8e3

        articulation = loader.load(str(urdf_path), config={"density": density})
        articulation.set_name("faucet")

        set_articulation_render_material(
            articulation, color=hex2rgba("#AAAAAA"), metallic=1, roughness=0.4
        )

        return articulation

    def _initialize_actors(self):
        '''add randomization in friction
        '''
        super()._initialize_actors()
        # --- randomize friction ---
        if self.auto_dr:
            if self.eval_env and self.randomized_param == 'obj_friction':
                self.obj_friction = self._episode_rng.choice(self.dr_params['obj_friction'])
            else:
                self.obj_friction = self._episode_rng.uniform(
                                                *self.dr_params['obj_friction'])
        elif self.randomized_env:
            self.fric_h = self.fric_h_scdl(elapsed_steps=self.step_counter)
            self.fric_l = self.fric_l_scdl(elapsed_steps=self.step_counter)
            self.obj_friction = self._episode_rng.uniform(self.fric_l, 
                                                          self.fric_h)
        else:
            self.obj_friction = np.array(1.)
        phys_mtl = self._scene.create_physical_material(
            static_friction=self.obj_friction, 
            dynamic_friction=self.obj_friction, restitution=0.1
        )
        # physical material only have friction related properties
        # https://github.com/haosulab/SAPIEN/blob/ab1d9a9fa1428484a918e61185ae9df2beb7cb30/python/py_package/core/pysapien/__init__.pyi#L1088
        for l in self.switch_links:
            for cs in l.get_collision_shapes():
                cs.set_physical_material(phys_mtl)

    def step(self, action):
        self.step_counter += 1
        return super().step(action)

    def _get_obs_images(self) -> OrderedDict:
        if self._renderer_type == "client":
            # NOTE: not compatible with StereoDepthCamera
            cameras = [x.camera for x in self._cameras.values()]
            self._scene._update_render_and_take_pictures(cameras)
        else:
            self.update_render()
            self.take_picture()
        # update the obs from _get_obs_state_dict() with camera info
        obs_dict = self._get_obs_state_dict()
        
        camera_param_dict = {}
        for k, v in self.get_camera_params()['hand_camera'].items():
            camera_param_dict[k] = v.flatten()

        obs_dict.update({
                'camera_param': flatten_state_dict(camera_param_dict),
                'image': self.get_images(),
                }) 
        return obs_dict
    
    def _configure_cameras(self):
        '''modified to only include agent camera'''
        self._camera_cfgs = OrderedDict()
        # self._camera_cfgs.update(parse_camera_cfgs(self._register_cameras()))

        self._agent_camera_cfgs = OrderedDict()
        if self._agent_cfg is not None:
            self._agent_camera_cfgs = parse_camera_cfgs(self._agent_cfg.cameras)
            self._camera_cfgs.update(self._agent_camera_cfgs)
            
    def _get_obs_state_dict(self):
        
        self.step_counter += 1
        cmass_pose = self.target_link.pose * self.target_link.cmass_local_pose

        # add external disturbance force
        grasped = self.agent.check_grasp(self.target_link)
        if self.ext_disturbance:
            # dist_force *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)
            # decay the prev force
            self.disturb_force *= self.force_decay
            # sample whether to apply new force with probablity 0.1
            if self._episode_rng.uniform() < 0.1:
                # sample 3D force for guassian distribution
                self.disturb_force = self._episode_rng.normal(0, 0.1, 3)
                self.disturb_force /= np.linalg.norm(self.disturb_force, ord=2)
                # sample force scale
                if self.auto_dr:
                    if self.eval_env and self.randomized_param == 'force_scale':
                        self.force_scale = self.dr_params['force_scale'][1]
                    else:
                        self.force_scale = self._episode_rng.uniform(
                                                *self.dr_params['force_scale'])
                else:
                    self.fs_h = self.force_scale_scdl(elapsed_steps=self.step_counter)
                    self.force_scale = self._episode_rng.uniform(0, self.fs_h)
                if self.force_scale > self.max_force:
                    self.max_force = self.force_scale
                # scale by object mass
                self.disturb_force *= self.target_link.mass * self.force_scale
                # apply the force to object
            # only apply if the object is grasped
            if grasped:
                self.target_link.add_force_at_point(self.disturb_force, 
                                                    cmass_pose.p)

        contacts = self._scene.get_contacts()
        limpulse = np.linalg.norm(get_pairwise_contact_impulse(contacts,
                            self.agent.finger1_link, self.target_link), ord=2)
        rimpulse = np.linalg.norm(get_pairwise_contact_impulse(contacts, 
                            self.agent.finger2_link, self.target_link), ord=2)


        if self.obs_noise:
            # noise to proprioception
            qpos = self.agent.robot.get_qpos() + self.proprio_noise
            proprio = np.concatenate([qpos, self.agent.robot.get_qvel()])
            # noise to obj position
            obj_pos = cmass_pose.p
            obj_pos += self.pos_noise
            # noise to obj rotation
            obj_ang = cmass_pose.q
            obj_ang = qmult(obj_ang, self.rot_noise)
            # obj_pose = np.concatenate([obj_pos, obj_ang])
        else:
            proprio = self.agent.get_proprioception()
            # obj_pose = vectorize_pose(cmass_pose)
            obj_pos = cmass_pose.p
            obj_ang = cmass_pose.q

        priv_info_dict = OrderedDict(
                            obj_ang=obj_ang, # 4dim
                            angle_dist=np.array(self.target_angle - self.current_angle),
                            target_joint_axis=self.target_joint_axis, # 3dim
                            obj_density=self.obj_density, # 1dim
                            obj_friction=self.obj_friction, # 1dim
                            limpulse=limpulse, # 1dim
                            rimpulse=rimpulse) # 1dim

        if self.inc_obs_noise_in_priv:
            if grasped:
                f = self.disturb_force
            else:
                f = np.zeros_like(self.disturb_force)
            # 9 + 7 + 3
            priv_info_dict.update(
                        proprio_noise=proprio_noise, 
                        pos_noise=pos_noise,
                        rot_noise=rot_noise,
                        disturb_force=f)

        return OrderedDict(
            # the same as the others
            agent_state=flatten_state_dict(OrderedDict(
                    proprioception=proprio,
                    base_pose=vectorize_pose(self.agent.robot.pose),
                    tcp_pose=vectorize_pose(self.tcp.pose),
                    )),
            # object 1
            # 1, 7, 3 -> 14 (1, 7, 3, 3)
            object1_state=flatten_state_dict(OrderedDict(
                        # bbox_size=np.array([0]),
                        obj_pos=obj_pos,
                        tcp_to_obj_pos=obj_pos - self.tcp.pose.p,
                    )),
            object1_type_id=self.obj1_type_num_id,
            object1_id=self.obj1_num_id,
            obj1_priv_info=flatten_state_dict(priv_info_dict),
            # 7, 3, 3, 1 -> 1, 1
            goal_info=flatten_state_dict(OrderedDict(
                        target_angle_diff=np.array(self.target_angle_diff),
                        # target_pose=vectorize_pose(self.box_hole_pose),
                        # tcp_to_goal_pos=self.box_hole_pose.p - self.tcp.pose.p,
                        # obj_to_goal_pos=self.box_hole_pose.p - self.peg.pose.p,
                        # box_hole_radius=self.box_hole_radius,
                    )).astype("float32"),
        )
