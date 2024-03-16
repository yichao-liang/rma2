
from functools import partial

from algo.misc import ManiSkillRGBDWrapper
import numpy as np
import torch as th
# import gym
import mani_skill2.envs
from mani_skill2.utils.registration import register_env
from mani_skill2.vector import VecEnv
from mani_skill2.vector import make as make_vec_env
from mani_skill2.utils.common import flatten_state_dict
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from mani_skill2.envs.pick_and_place.pick_cube import PickCubeEnv
from mani_skill2.utils.wrappers import RecordEpisode
from mani_skill2.utils.wrappers.sb3 import ContinuousTaskWrapper, \
                                           SuccessInfoWrapper
import gymnasium as gym
from task import gym_task_map
from algo.misc import make_env

# test 8: new robot
def main():
    env_id = "PickSingleYCB-v1"
    # env_id = "PickSingleEGAD-v2"
    control_mode = "pd_ee_delta_pose"

    obs_mode = "state_dict"
    env_kwargs = dict(
                        robot='xarm7',
                        randomized_training=True,
                        obs_noise=True,
                        ext_disturbance=True,
                        inc_obs_noise_in_priv=False,
                        test_eval=False,
                        sim_freq=120,
                )
    env = gym.make(env_id, obs_mode=obs_mode, control_mode=control_mode,
                   **env_kwargs)
    obs = env.reset()

if __name__ == "__main__":
    main()

# test 7: new init for PickSingle (kwargs)
# def main():
#     # env_id = "PickSingleYCB-v2"
#     env_id = "PickSingleEGAD-v2"
#     control_mode = "pd_ee_delta_pose"

#     obs_mode = "state_dict"
#     env_kwargs = dict(
#                         randomized_training=True,
#                         obs_noise=True,
#                         ext_disturbance=True,
#                         inc_obs_noise_in_priv=True,
#                         test_eval=False,
#                         sim_freq=120,
#                 )
#     env = gym.make(env_id, obs_mode=obs_mode, control_mode=control_mode,
#                    **env_kwargs)
#     obs = env.reset()

# if __name__ == "__main__":
#     main()

# test 1
# def main():
#     env_id = "PickSingleYCB-v1"
#     num_envs = 20
#     control_mode = "pd_ee_delta_pose"
#     max_episode_steps = 100

#     # obs_mode = "rgbd"
#     # env: VecEnv = make_vec_env(
#     #                 env_id,
#     #                 num_envs=num_envs,
#     #                 obs_mode=obs_mode,
#     #                 control_mode=control_mode,
#     #                 wrappers=[partial(ContinuousTaskWrapper)],
#     #                 max_episode_steps=max_episode_steps,
#     #             )

#     obs_mode = "state_dict"
#     env = SubprocVecEnv(
#                     [make_env(env_id, max_episode_steps=max_episode_steps,
#                                 obs_mode=obs_mode, control_mode=control_mode, 
#                                 )
#                         for _ in range(num_envs)
#                     ],
#                 )
#     obs = env.reset()
#     print(env.get_attr("seedd"))
#     print(obs['object1_id'])
#     breakpoint()

# if __name__ == "__main__":
#     main()

# test 2: observation noise
# env = gym.make("PickSingleYCB-v1",obs_mode="state_dict")
# obs = env.reset()

# test 3: check gpu avilablity in sbatch
# print("cuda is available:", th.cuda.is_available())

# test 4: PegInsertionRMA
# env = gym.make("TurnFaucet-v1", obs_mode="state_dict", 
#                 inc_obs_noise_in_priv=True, obs_noise=True, ext_disturbance=True)
# # env = gym.make("PegInsertionSide-v1", obs_mode="state_dict")
# env.reset()
# action_space = env.action_space
# for _ in range(100):
#     print(flatten_state_dict(env.step(action_space.sample())[0]).shape)

# test 5: measure torque
# import mani_skill2.envs, gymnasium as gym
# import numpy as np

# env=gym.make('PickCube-v0')
# obs, _ = env.reset()

# action = env.action_space.sample()
# env.agent.controller.controllers['arm'].set_action(action[:-1])
# env.agent.controller.controllers['gripper'].set_action(action[-1:])

# joints = env.agent.robot.get_joints()
# valid_joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 
#                      'panda_joint4', 'panda_joint5', 'panda_joint6', 
#                      'panda_joint7', 'panda_finger_joint1', 
#                      'panda_finger_joint2']
# joints = [x for x in joints if x.name in valid_joint_names]

# stiffness = np.array([x.stiffness for x in joints])
# damping = np.array([x.damping for x in joints])
# force_limit = np.array([x.force_limit for x in joints])

# drive_target = env.agent.robot.get_drive_target()
# drive_velocity_target = env.agent.robot.get_drive_velocity_target()
# cur_qpos = env.agent.robot.get_qpos()
# cur_qvel = env.agent.robot.get_qvel()

# torque = stiffness * (drive_target - cur_qpos) + \
#          damping * (drive_velocity_target - cur_qvel)
# torque = np.clip(torque, -force_limit, force_limit)
# torque = torque + env.agent.robot.compute_passive_force(True, True, True)
# print("torque", torque)

# env.step(action)
# print(env.agent.robot.get_qpos() - cur_qpos)
# print("energy consumption for this control step", np.sum(torque * 
#         (env.agent.robot.get_qpos() - cur_qpos)))
# control frequency: env.agent.controller.controllers['arm']._control_freq, env.agent.controller.controllers['gripper']._control_freq, which are 20hz, so power is approximately the above energy consumption * 20

# # test 6: update step counter
# @register_env("PickCube-v2", max_episode_steps=200)
# class PickCubeEnvMod(PickCubeEnv):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.step_counter = 0

#     def set_step_counter(self, n):
#         self.step_counter = n

#     def step(self, action):
#         self.step_counter += 1
#         return super().step(action)

# def main():
#     def make_env(
#         env_id: str,
#         max_episode_steps: int = None,
#         record_dir: str = None,
#     ):
#         def _init() -> gym.Env:
#             # NOTE: Import envs here so that they are registered with gym in subprocesses
#             import mani_skill2.envs

#             env = gym.make(
#                 env_id,
#                 obs_mode="state",
#                 reward_mode="normalized_dense",
#                 control_mode="pd_ee_delta_pose",
#                 render_mode="cameras",
#                 max_episode_steps=max_episode_steps,
#             )
#             return env
#         return _init

#     env = SubprocVecEnv([make_env('PickCube-v2', max_episode_steps=100)
#                         for _ in range(2)])
#     env.reset()

#     for _ in range(10):
#         action = env.action_space.sample()
#         action = np.repeat(action[None, :], 2, axis=0)
#         env.step(action)

#     # check the step counter, we should expect 10
#     # we do get 10
#     print(env.get_attr('step_counter'))

#     # manually set the counter
#     env.env_method('set_step_counter', 100)
#     for _ in range(10):
#         action = env.action_space.sample()
#         action = np.repeat(action[None, :], 2, axis=0)
#         env.step(action)

#     # check the step counter again, we should expect 110
#     # but we still get 100
#     print(env.get_attr('step_counter'))

# if __name__ == "__main__":
#     main()