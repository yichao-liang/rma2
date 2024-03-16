# Import required packages
import json
import os.path as osp

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from config import parse_args, config_log_path, config_envs
from algo.models import FeaturesExtractorRMA
from algo.ppo_rma import PPORMA
from algo.adaptation import ProprioAdapt
from algo.policy_rma import ActorCriticPolicyRMA
from task import gym_task_map
from algo.evaluate_policy import evaluate_policy
from algo.callbacks_rma import EvalCallbackRMA, CheckpointCallbackRMA
from algo.misc import linear_schedule

def main():
    args = parse_args()
    print("args:", args)
    num_envs = args.n_envs
    log_dir = args.log_dir
    rollout_steps = args.rollout_steps 
    record_dir, ckpt_dir, ckpt_path, tb_path_root = config_log_path(args)
    
    # ---- Save the dictionary to a JSON file
    args_dict = vars(args)
    with open(ckpt_dir+'/args.json', 'w') as f:
        json.dump(args_dict, f, indent=4)

    # ---- Config the environments
    env, eval_env = config_envs(args, record_dir)

    # ---- policy configuration and algorithm configuration
    env_name, env_version = args.env_id.split("-")
    policy_kwargs = dict(
                        net_arch={"pi": args.policy_arch,
                                   "vf": args.policy_arch},
                        n_envs=num_envs,
                        use_obj_emb=not args.no_obj_embedding,
                        sys_iden=args.sys_iden,
                        inc_obs_noise_in_priv=args.inc_obs_noise_in_priv,
                        object_emb_dim=args.obj_emb_dim,
                        use_depth_adaptation=args.use_depth_adaptation,
                        env_name=env_name,
                        features_extractor_kwargs=dict(
                            object_emb_dim=args.obj_emb_dim,
                            env_name=env_name,
                            # use_depth_adaptation=args.use_depth_adaptation,
                            use_depth_base=args.use_depth_base,
                            use_prop_history_base=args.use_prop_history_base,
                            no_obj_emb=args.no_obj_embedding,
                            only_dr=args.only_DR or args.auto_dr,
                            sys_iden=args.sys_iden,
                            without_adapt_module=args.without_adapt_module,
                            inc_obs_noise_in_priv=args.inc_obs_noise_in_priv,
                    ))

    if env_version in ['v1', 'v2']:
        policy_kwargs.update(features_extractor_class=FeaturesExtractorRMA)

    end_step = args.anneal_end_step
    if args.lr_schedule:
        lr = linear_schedule(3e-4, 1e-5, total_steps=args.total_timesteps, 
                             end_step=end_step)
    else:
        lr = 3e-4
    if args.clip_range_schedule:
        clip_range = linear_schedule(args.clip_range, 0.05, 
                        total_steps=args.total_timesteps, end_step=end_step)
    else:
        clip_range = args.clip_range

    model = PPORMA(
        ActorCriticPolicyRMA,
        env,
        auto_dr=args.auto_dr,
        use_prop_history_base=args.use_prop_history_base,
        policy_kwargs=policy_kwargs,
        learning_rate=lr,
        clip_range=clip_range,
        verbose=2,
        n_steps=rollout_steps,
        batch_size=args.batch_size,
        gamma=0.85,
        n_epochs=args.n_epochs,
        tensorboard_log=tb_path_root,
        target_kl=args.target_kl,
        eval=args.eval,
    )
    exclude_adaptor_net=(
            (args.adaptation_training and not args.continue_training) or # just starting adaptor training
            (not args.adaptation_training) and (not args.eval) or # base training
            args.only_DR or args.auto_dr or args.use_depth_base or 
            args.expert_adapt
        ) # DR methods
    if args.eval:
        # if args.ckpt_name is None:
        #     ckpt_path = osp.join(log_dir, "best_model")
        model = model.load(ckpt_path, env=env, tensorboard_log=tb_path_root,
                           policy_kwargs=policy_kwargs,
                           exclude_adaptor_net=exclude_adaptor_net,
                           )
        model.policy.use_depth_base = args.use_depth_base
        model.policy.use_prop_history_base = args.use_prop_history_base
        model.policy.use_depth_adaptation = args.use_depth_adaptation
        model.policy.adapt_tconv.use_depth = args.use_depth_adaptation
    else:
        if (args.continue_training or args.adaptation_training or 
            args.transfer_learning):
            # if the file specified by ckpt_path doesn't exist
            if not osp.exists(ckpt_path):
                print(f"###  Warning: ckpt_path {ckpt_path} doesn't exist -- for debugging only")
            else:
                model = model.load(ckpt_path, env=env, 
                                   tensorboard_log=tb_path_root,
                                   policy_kwargs=policy_kwargs,
                                   exclude_adaptor_net=exclude_adaptor_net,)

                assert model.policy.adapt_tconv.use_depth == args.use_depth_adaptation
                model.observation_space = env.observation_space
                model.policy.observation_space = env.observation_space
                reset_num_timesteps = False
        else:
            reset_num_timesteps = True

        env.env_method('set_step_counter', model.num_timesteps//num_envs)
        eval_env.env_method('set_step_counter', model.num_timesteps//num_envs)

        # define callbacks to periodically save our model and evaluate it to 
        # help monitor training the below freq values will save every 10 
        # rollouts.
        # the callback is called every rollout_steps * n_envs steps
        eval_freq = args.rollout_steps
        save_freq = args.rollout_steps * 10
        eval_callback = EvalCallbackRMA(
            eval_env,
            best_model_save_path=ckpt_dir,
            log_path=ckpt_dir,
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            num_envs=num_envs,
        )
        checkpoint_callback = CheckpointCallbackRMA(
            save_path=ckpt_dir,
            save_freq=save_freq,
            name_prefix="model",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )

        # Train an agent with PPO for args.total_timesteps interactions
        if args.adaptation_training:
            # adaptation training assumes the policy has been trained
            # create a tensorboard write
            writer = SummaryWriter(log_dir=tb_path_root) 
            algo = ProprioAdapt(
                model=model,
                env=env,
                writer=writer,
                save_dir=ckpt_dir,
            )
            if args.compute_adaptation_loss:
                print("Computing adaptation loss")
                adaptation_loss = algo.compute_mean_adaptor_loss()
            else:
                algo.learn()
        else:
            model.learn(
                args.total_timesteps,
                callback=[checkpoint_callback, eval_callback],
                reset_num_timesteps=False,
                tb_log_name='',
            )
        # Save the final model
        model.save(osp.join(log_dir, "latest_model"))

    env.env_method('set_step_counter', 1e8)
    eval_env.env_method('set_step_counter', 1e8)

    if not args.compute_adaptation_loss:
        # n_eval_eps = 1
        n_eval_eps = 100
        # observations = env.reset()
        # from stable_baselines3.common.utils import obs_as_tensor
        # model.policy(obs_as_tensor(observations,"cuda"), adapt_trn=True)
        returns, ep_lens = evaluate_policy(
            model,
            # observations,
            eval_env,
            deterministic=True,
            render=False,
            return_episode_rewards=True,
            n_eval_episodes=n_eval_eps,
            test_mode=args.eval,
            expert_adapt=args.expert_adapt,
            only_dr=args.only_DR or args.auto_dr,
            without_adapt_module=args.without_adapt_module,
        )
        print("Returns", returns)
        print("Episode Lengths", ep_lens)
        mean_ep_lens = np.mean(np.array(ep_lens))
        success = np.array(ep_lens) < 200
        success_rate = success.mean()
        print("Mean Episode Lengths", mean_ep_lens)
        print("Success Rate:", success_rate)

    # save the results in the logs/eval_results.csv
    # create the file if it doesn't exist
    # eval_results_path = osp.join(log_dir, "eval_results.csv")
    eval_results_path = osp.join(log_dir, "eval_results_finegrained.csv")
    if not osp.exists(eval_results_path):
        with open(eval_results_path, "w") as f:
            # create the header if it's the first time creating the file
            f.write("log_name,expert_adapt,inc_obs_noise_in_priv,only_DR,"+\
                "without_adapt_module,sys_iden,no_obj_embedding,success_rate,"+\
                "mean_ep_len,n_eval_eps,env_name,model_id,adapt_loss\n")
    # save the log_name, args.expert_adapt, args.inc_obs_noise_in_priv
    # args.only_DR, args.without_adapt_module, args.sys_iden
    # args.no_obj_embedding, success_rate to the file
    log_model_id = args.eval_model_id if args.eval_model_id else "All"
    with open(eval_results_path, "a") as f:
        if args.compute_adaptation_loss:
            f.write(f"{args.log_name},{args.expert_adapt},"+\
                f"{args.inc_obs_noise_in_priv},{args.only_DR},"+\
                f"{args.without_adapt_module},{args.sys_iden},"+\
                f"{args.no_obj_embedding},,,"+\
                f",{env_name},{log_model_id},{adaptation_loss}\n")
        else:
            f.write(f"{args.log_name},{args.expert_adapt},"+\
                f"{args.inc_obs_noise_in_priv},{args.only_DR},"+\
                f"{args.without_adapt_module},{args.sys_iden},"+\
                f"{args.no_obj_embedding},{success_rate:.3f},{mean_ep_lens},"+\
                f"{n_eval_eps},{env_name},{log_model_id}\n")


if __name__ == "__main__":
    main()
