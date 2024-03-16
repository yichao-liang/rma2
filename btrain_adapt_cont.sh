# -- Contiual adaptation training --
# PickSingle
sbatch -J Apsd execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise --use_depth_adaptation\
    -e PickSingleYCB-v1 --adaptation_training\
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_1_stage2_dep_1 \
    --ckpt_name latest_model.zip 

sbatch -J Apsd execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise --use_depth_adaptation\
    -e PickSingleYCB-v1 --adaptation_training\
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_2_stage2_dep_1 \
    --ckpt_name latest_model.zip

sbatch -J Atfd execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise --use_depth_adaptation\
    -e TurnFaucet-v1 --adaptation_training\
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_1_stage2_dep_2 \
    --ckpt_name latest_model.zip 

sbatch -J Atfd execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise --use_depth_adaptation\
    -e TurnFaucet-v1 --adaptation_training\
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_2_stage2_dep_2 \
    --ckpt_name latest_model.zip