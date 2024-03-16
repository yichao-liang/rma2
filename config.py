# Import required packages
import argparse
import os
import os.path as osp
from functools import partial

from stable_baselines3.common.utils import set_random_seed, get_latest_run_id
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from mani_skill2.vector.wrappers.sb3 import SB3VecEnvWrapper
from mani_skill2.vector import VecEnv
from mani_skill2.vector import make as make_vec_env
from mani_skill2.utils.wrappers.sb3 import ContinuousTaskWrapper,\
                                           SuccessInfoWrapper

from algo.misc import make_env, ManiSkillRGBDVecEnvWrapper

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple script demonstrating how to use Stable Baselines 3 with ManiSkill2 and RGBD Observations"
    )
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1")
    parser.add_argument("--robot", type=str, default="panda", choices=["panda", 
                                                            "xarm7",
                                                            "xmate3_robotiq"])
    # parser.add_argument("-e", "--env-id", type=str, default="PickSingleYCB-v1")
    parser.add_argument("--obs_mode", type=str, 
                        default="state_dict")
    parser.add_argument(
        "-n", "--n-envs", type=int,
        default=50,
        help="number of parallel envs to run.",
    )
    parser.add_argument(
        "-bs", "--batch_size", type=int,
        default=5000,
        help="batch size for training",
    )
    parser.add_argument(
        "-rs", "--rollout_steps", type=int,
        default=2000,
        help="rollout steps per env",
    )
    parser.add_argument(
        "-kl", "--target_kl", type=float,
        default=.05,
        help="upper bound for the KL divergence",
    )
    parser.add_argument(
        "-cr", "--clip_range", type=float,
        default=.2,
        help="clip range for PPO",
    )
    parser.add_argument(
        "-nep", "--n_epochs", type=int,
        default=10,
        help="number of parallel envs to run. Note that increasing this does not increase rollout size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed to initialize training with",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=50,
        help="Max steps per episode before truncating them",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=100_000_000_000_000,
        help="Total timesteps for training",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="/om/user/ycliang/logs",
        help="path for where logs, checkpoints, and videos are saved",
    )
    parser.add_argument(
        "--log_name", type=str, 
        default="PPO",
        help="model name, e.g., PPO, PPO-pc0-bs400_1, ..."
        # specify log_name in continue training to resume the logging
    )
    parser.add_argument(
        "--ckpt_name", type=str, help="path to sb3 model for evaluation"
        # specify log_name to continue training from checkpoint
        # e.g., model_320000_steps.zip, latest.zip
    )
    parser.add_argument(
        "--eval", action="store_true", help="whether to only evaluate policy"
    )
    parser.add_argument(
        "-ct", "--continue_training", action="store_true", 
        help="continue training from checkpoint"
        # for continue training, specify:
        # - log_name for logging to the correct dir, 
        # - ckpt_name for loading the correct model
    )
    parser.add_argument(
        # "--policy_arch", type=list, default=[256, 256],
        "--policy_arch", type=list, default=[512, 256, 128], # config in hora
        help="policy network architecture"
    )
    parser.add_argument(
        "--randomized_training",  action="store_true",
        help="whether to randomize the training environment"
    )
    parser.add_argument(
        "-on", "--obs_noise",  action="store_true",
        help="whether to add noise to the observations."
    )
    parser.add_argument(
        "--lr_schedule",  default=0, type=int,
        help="whether to use learning rate schedule, if not specified"
    )
    parser.add_argument(
        "--clip_range_schedule", default=1, type=int,
        help="whether to use learning rate schedule, if not specified"
    )
    parser.add_argument(
        "-ae", "--anneal_end_step", type=float, default=1e7,
        help="end step for annealing learning rate and clip range",
    )
    parser.add_argument(
        "--adaptation_training", action="store_true",
        help="perform stage 2, adaptation training when the tag is specified"+\
        "when using this, `log_dir`, `log_name`, `ckpt_name` must be specified"
    )
    parser.add_argument(
        "--transfer_learning", action="store_true",
        help="perform transfer learning on another env specified by env-id."+\
        "When used, specify `log_dir`, `log_name`, `ckpt_name` to choose the"+\
        "base model."
    )
    parser.add_argument(
        "--use_depth_adaptation", action="store_true",
        help="use depth information in the observation. This entails using rgbd"+\
             "observation and have a CNN feature extractor."
    )
    parser.add_argument(
        "--use_depth_base", action="store_true",
        help="doesn't use object position and privileged information."
    )
    parser.add_argument(
        "--use_prop_history_base", action="store_true",
        help="doesn't use object position and privileged information."
    )
    parser.add_argument(
        "--ext_disturbance", action="store_true",
        help="whether to add external disturbance force to the environment."
    )
    parser.add_argument(
        "--inc_obs_noise_in_priv", action="store_true",
        help="add obs noise as part of the privileged observation."
    )
    parser.add_argument(
        "--expert_adapt", action="store_true",
    )
    parser.add_argument(
        "--without_adapt_module", action="store_true",
    )
    parser.add_argument(
        "--only_DR", action="store_true",
    )
    parser.add_argument(
        "--sys_iden", action="store_true",
    )
    parser.add_argument(
        "--auto_dr", action="store_true",
    )
    parser.add_argument(
        "--no_obj_embedding", action="store_true",
    )
    parser.add_argument(
        "--obj_emb_dim", default=32, type=int,
        help=""
    )
    parser.add_argument(
        "--eval_model_id", default="002_master_chef_can", 
        help="The model to eval the model on"
    )
    parser.add_argument(
        "--compute_adaptation_loss", action="store_true",
        help="perform stage 2, adaptation training when the tag is specified"+\
        "when using this, `log_dir`, `log_name`, `ckpt_name` must be specified"
    )
    

    args = parser.parse_args()
    return args

env_name_to_abbrev = {
    'PickCube-v0': 'pc0',
    'PickCube-v1': 'pc',
    'StackCube-v1': 'sc',
    'PickSingleYCB-v1': 'ps',
    'PegInsertionSide-v1': 'pi',
    'TurnFaucet-v1': 'tf',
}

def config_log_path(args):
    # ---- config save, load path
    log_dir = args.log_dir
    ckpt_path = None
    if args.continue_training:
        log_name = f"{args.log_name}"
        ckpt_path = osp.join(log_dir, log_name, 'ckpt', args.ckpt_name)
    elif args.adaptation_training or args.transfer_learning or args.eval:
        log_name = f"{args.log_name}"
        ckpt_path = osp.join(log_dir, log_name, 'ckpt', args.ckpt_name)
        if args.adaptation_training:
            log_name = f"{args.log_name}_stage2"
            if args.use_depth_adaptation:
                log_name += "_dep"
        elif args.transfer_learning:
            log_name = f"{args.log_name}_to-{env_name_to_abbrev[args.env_id]}"
        latest_run_id = get_latest_run_id(args.log_dir, log_name)
        log_name = f"{log_name}_{latest_run_id + 1}"
    else:
        log_name = f"{args.log_name}-{env_name_to_abbrev[args.env_id]}"+\
                    f"-bs{args.batch_size}-rs{args.rollout_steps}"+\
                    f"-kl{args.target_kl}-neps{args.n_epochs}"+\
                    f"-cr{args.clip_range}-lr_scdl{args.lr_schedule}"+\
                    f"-cr_scdl{args.clip_range_schedule}"+\
                    f"-ms{args.max_episode_steps}"+\
                    f"-incObsN{int(args.inc_obs_noise_in_priv)}"
        if args.use_depth_base: log_name += "-dep"
        if args.use_prop_history_base: log_name += "-prop"
        if args.only_DR: log_name += "-onlyDR"
        if args.auto_dr: log_name += "-ADR"
        if args.sys_iden: log_name += "-SyId"
        if args.no_obj_embedding: log_name += "-noObjEmb"
        latest_run_id = get_latest_run_id(args.log_dir, log_name)
        log_name = f"{log_name}_{latest_run_id + 1}"
    print(f"## Saving to {log_name}")

    # record dir is: <log_dir>/<log_name>/video
    record_dir = osp.join(log_dir, log_name, "video")
    ckpt_dir = osp.join(log_dir, log_name, "ckpt")
    tb_path_root = osp.join(log_dir, log_name)
    if not osp.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    return record_dir, ckpt_dir, ckpt_path, tb_path_root

def config_envs(args, record_dir):
    env_id = args.env_id
    num_envs = args.n_envs
    max_episode_steps = args.max_episode_steps
    if args.use_depth_adaptation or args.use_depth_base:
        obs_mode = "rgbd"
    else:
        obs_mode = "state_dict"

    control_mode = "pd_ee_delta_pose"
    reward_mode = "normalized_dense"
    if args.seed is not None:
        set_random_seed(args.seed)

    # create eval environment
    if args.eval:
        record_dir = osp.join(record_dir, "eval")

    model_ids = args.eval_model_id
    eval_env_kwargs = dict(randomized_training=args.randomized_training,
                        # auto_dr=args.auto_dr,
                        robot=args.robot,
                        obs_noise=args.obs_noise,
                        ext_disturbance=args.ext_disturbance,
                        inc_obs_noise_in_priv=args.inc_obs_noise_in_priv,
                        test_eval=args.eval,
                        sim_freq=120,
                        # seed=args.seed,
                        # model_ids=model_ids if model_ids else [],
                )
    # begin
    if args.use_depth_adaptation or args.use_depth_base:
    # Create vectorized environments for training
        # env :: mani_skill2.vector.vec_env.VecEnv
        # RGBDVecEnv(<ContinuousTaskWrapper<TimeLimit<PickSingleYCBRMA<PickSingleYCB-v1>>>>)
        eval_env = make_vec_env(env_id, 
                            num_envs=1,
                            # record_dir=record_dir, 
                            obs_mode=obs_mode,
                            control_mode=control_mode, 
                            reward_mode=reward_mode,
                            wrappers=[partial(SuccessInfoWrapper)],
                            **eval_env_kwargs
                            )
        # ManiSkillRGBDVecEnvWrapper(<ContinuousTaskWrapper<TimeLimit<PickSingleYCBRMA<PickSingleYCB-v1>>>>)
        eval_env = ManiSkillRGBDVecEnvWrapper(eval_env)
        # <mani_skill2.vector.wrappers.sb3.SB3VecEnvWrapper object at 0x150c7a9c1750>
        eval_env = SB3VecEnvWrapper(eval_env)

    else:
        # <stable_baselines3.common.vec_env.subproc_vec_env.SubprocVecEnv object at 0x154358125490>
        eval_env = SubprocVecEnv(
                    [make_env(env_id, record_dir=record_dir, obs_mode=obs_mode,
                    control_mode=control_mode, reward_mode=reward_mode,
                    **eval_env_kwargs
                ) for _ in range(1)])
    # end new
    # old
    # eval_env = SubprocVecEnv(
    #                 [make_env(env_id, record_dir=record_dir, obs_mode=obs_mode,
    #                 control_mode=control_mode, reward_mode=reward_mode,
    #                 **eval_env_kwargs
    #             ) for _ in range(1)])
    # end old
    eval_env = VecMonitor(eval_env)  # attach this so SB3 can log reward metrics
    eval_env.seed(args.seed)
    eval_env.reset()

    env_kwargs = dict(randomized_training=args.randomized_training,
                        robot=args.robot,
                        auto_dr=args.auto_dr,
                        obs_noise=args.obs_noise,
                        ext_disturbance=args.ext_disturbance,
                        inc_obs_noise_in_priv=args.inc_obs_noise_in_priv,
                        test_eval=args.eval,
                        sim_freq=120,
                )
    if args.eval:
        env = eval_env
    else:
        if args.use_depth_adaptation or args.use_depth_base:
        # Create vectorized environments for training
            # env :: mani_skill2.vector.vec_env.VecEnv
            # RGBDVecEnv(<ContinuousTaskWrapper<TimeLimit<PickSingleYCBRMA<PickSingleYCB-v1>>>>)
            env: VecEnv = make_vec_env(
                            env_id, num_envs=num_envs, obs_mode=obs_mode,
                            control_mode=control_mode,
                            wrappers=[partial(ContinuousTaskWrapper), 
                                      partial(SuccessInfoWrapper)],
                            max_episode_steps=max_episode_steps,
                            **env_kwargs
                            )
            # ManiSkillRGBDVecEnvWrapper(<ContinuousTaskWrapper<TimeLimit<PickSingleYCBRMA<PickSingleYCB-v1>>>>)
            env = ManiSkillRGBDVecEnvWrapper(env)
            # <mani_skill2.vector.wrappers.sb3.SB3VecEnvWrapper object at 0x150c7a9c1750>
            env = SB3VecEnvWrapper(env)

        else:
            # <stable_baselines3.common.vec_env.subproc_vec_env.SubprocVecEnv object at 0x154358125490>
            env = SubprocVecEnv(
                [make_env(env_id, max_episode_steps=max_episode_steps,
                            obs_mode=obs_mode, control_mode=control_mode, 
                            reward_mode=reward_mode, 
                            **env_kwargs
                            )
                    for _ in range(num_envs)
                ],
            )

        env = VecMonitor(env)
        env.seed(args.seed)
        env.reset()
    return env, eval_env
