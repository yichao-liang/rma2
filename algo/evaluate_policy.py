import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch as th

from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.utils import (obs_as_tensor, 
                                            get_device)
def compute_energy_consumption(env):
    joints = env.agent.robot.get_joints()
    valid_joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 
                        'panda_joint4', 'panda_joint5', 'panda_joint6', 
                        'panda_joint7', 'panda_finger_joint1', 
                        'panda_finger_joint2']
    joints = [x for x in joints if x.name in valid_joint_names]

    stiffness = np.array([x.stiffness for x in joints])
    damping = np.array([x.damping for x in joints])
    force_limit = np.array([x.force_limit for x in joints])

    drive_target = env.agent.robot.get_drive_target()
    drive_velocity_target = env.agent.robot.get_drive_velocity_target()
    cur_qpos = env.agent.robot.get_qpos()
    cur_qvel = env.agent.robot.get_qvel()

    torque = stiffness * (drive_target - cur_qpos) + \
            damping * (drive_velocity_target - cur_qvel)
    torque = np.clip(torque, -force_limit, force_limit)
    torque = torque + env.agent.robot.compute_passive_force(True, True, True)
    # print("torque", torque)

    # print(env.agent.robot.get_qpos() - cur_qpos)
    # print("energy consumption for this control step", np.sum(torque * 
    #         (env.agent.robot.get_qpos() - cur_qpos)))
    return np.sum(torque * (env.agent.robot.get_qpos() - cur_qpos))

def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
    test_mode: bool = False,
    expert_adapt: bool = False,
    only_dr: bool = False,
    without_adapt_module: bool = False,
    # compute_e_consump: bool = False
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    # model.policy.set_training_mode(False)
    if test_mode:
        model.test_eval(expert_adapt=expert_adapt, only_dr=only_dr, 
                    without_adapt_module=without_adapt_module)
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    # energy_consumption = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    # current_energy_consumption = np.zeros(n_envs)
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    model.policy.reset_buffer(1)
    model.policy.reset_prev_action()
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)

        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        # energy_consumption.append(current_energy_consumption[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    model.policy.reset_buffer(1)
                    model.policy.reset_prev_action()
                    # current_energy_consumption[i] = 0

        observations = new_observations

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    model.policy.prev_actions = th.zeros(model.env.num_envs, 
                                         model.action_space.shape[0])
    model.policy.reset_buffer()
    # model.policy.proprio_hist_buffer = th.zeros(model.env.num_envs, 
    #                                             model.policy.hist_buffer_size, 
    #                                             model.policy.proprio_dim)
    # if compute_energy_consumption:  
    #     return episode_rewards, episode_lengths, energy_consumption
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
