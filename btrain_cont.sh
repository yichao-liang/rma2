# -- Contiual training --
# PickSingle
sbatch -J ps execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise \
    -e PickSingleYCB-v1 \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_1 \
    --ckpt_name model_latest.zip 

sbatch -J ps execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise \
    -e PickSingleYCB-v1 \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_2 \
    --ckpt_name model_latest.zip

sbatch -J ps execute.sh python main.py -n 50 -bs 50000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise \
    -e PickSingleYCB-v1 \
    --log_name PPO-ps-bs50000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_1 \
    --ckpt_name model_latest.zip 

sbatch -J ps execute.sh python main.py -n 50 -bs 50000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise \
    -e PickSingleYCB-v1 \
    --log_name PPO-ps-bs50000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_2 \
    --ckpt_name model_latest.zip

# sbatch -J psD execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
#     --randomized_training --ext_disturbance --obs_noise --use_depth_base\
#     -e PickSingleYCB-v1 \
#     \
#     --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-dep_1 \
#     --ckpt_name model_latest.zip 

# sbatch -J psD execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
#     --randomized_training --ext_disturbance --obs_noise --use_depth_base\
#     -e PickSingleYCB-v1 \
#     \
#     --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-dep_2 \
#     --ckpt_name model_latest.zip

# sbatch -J psi execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
#     --randomized_training --ext_disturbance --obs_noise --inc_obs_noise_in_priv \
#     -e PickSingleYCB-v1 \
#     \
#     --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN1_1 \
#     --ckpt_name model_latest.zip 

# sbatch -J psSI execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
#     --randomized_training --ext_disturbance --obs_noise --sys_iden \
#     -e PickSingleYCB-v1 \
#     \
#     --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-SyId_1 \
#     --ckpt_name model_latest.zip

# sbatch -J psSI execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
#     --randomized_training --ext_disturbance --obs_noise --sys_iden \
#     -e PickSingleYCB-v1 \
#     \
#     --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-SyId_2 \
#     --ckpt_name model_latest.zip

sbatch -J psNoOE execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct\
    --randomized_training --ext_disturbance --obs_noise --no_obj_embedding \
    -e PickSingleYCB-v1 \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-noObjEmb_1 \
    --ckpt_name model_latest.zip

sbatch -J psNoOE execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct\
    --randomized_training --ext_disturbance --obs_noise --no_obj_embedding \
    -e PickSingleYCB-v1 \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-noObjEmb_2 \
    --ckpt_name model_latest.zip

sbatch -J psADR execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise --auto_dr\
    -e PickSingleYCB-v1 \
    \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-ADR_1 \
    --ckpt_name model_latest.zip

sbatch -J psADR execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise --auto_dr\
    -e PickSingleYCB-v1 \
    \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-ADR_2 \
    --ckpt_name model_latest.zip

sbatch -J psDR execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise --only_DR\
    -e PickSingleYCB-v1 \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-onlyDR_1 \
    --ckpt_name model_latest.zip

sbatch -J psDR execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise --only_DR\
    -e PickSingleYCB-v1 \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-onlyDR_2 \
    --ckpt_name model_latest.zip

# with both depth and proprioception-action history
sbatch -J psdp execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
            --use_depth_base --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-dep-prop_1\
            --randomized_training --ext_disturbance --obs_noise \
            -e PickSingleYCB-v1 --use_prop_history_base --ckpt_name model_latest.zip
sbatch -J psdp execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
            --use_depth_base --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-dep-prop_2\
            --randomized_training --ext_disturbance --obs_noise \
            -e PickSingleYCB-v1 --use_prop_history_base --ckpt_name model_latest.zip

# with proprioception-action history
sbatch -J psp execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
            --use_prop_history_base --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-prop_1\
            --randomized_training --ext_disturbance --obs_noise \
            -e PickSingleYCB-v1 --ckpt_name model_latest.zip
sbatch -J psp execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
            --use_prop_history_base --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-prop_2\
            --randomized_training --ext_disturbance --obs_noise \
            -e PickSingleYCB-v1 --ckpt_name model_latest.zip

# PegInsertionSide
sbatch -J pi2 execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise \
    -e PegInsertionSide-v1 \
    --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_7 \
    --ckpt_name model_latest.zip 

sbatch -J pi2 execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise \
    -e PegInsertionSide-v1 \
    --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_8 \
    --ckpt_name model_latest.zip

sbatch -J pi2 execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise \
    -e PegInsertionSide-v1 \
    --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_9 \
    --ckpt_name model_latest.zip 

sbatch -J pi2 execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise \
    -e PegInsertionSide-v1 \
    --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_10 \
    --ckpt_name model_latest.zip


sbatch -J piD execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise --use_depth_base\
    -e PegInsertionSide-v1 \
    --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-dep_1 \
    --ckpt_name model_latest.zip 

sbatch -J piD execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise --use_depth_base\
    -e PegInsertionSide-v1 \
    --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-dep_2 \
    --ckpt_name model_latest.zip

# sbatch -J pii execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
#     --randomized_training --ext_disturbance --obs_noise --inc_obs_noise_in_priv \
#     -e PegInsertionSide-v1 \
#     \
#     --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN1_1 \
#     --ckpt_name model_latest.zip 

# sbatch -J piSI execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
#     --randomized_training --ext_disturbance --obs_noise --sys_iden \
#     -e PegInsertionSide-v1 \
#     \
#     --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-SyId_1 \
#     --ckpt_name model_latest.zip

# sbatch -J piSI execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
#     --randomized_training --ext_disturbance --obs_noise --sys_iden \
#     -e PegInsertionSide-v1 \
#     \
#     --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-SyId_2 \
#     --ckpt_name model_latest.zip

sbatch -J piADR execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise --auto_dr\
    -e PegInsertionSide-v1 \
    --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-ADR_1 \
    --ckpt_name model_latest.zip

sbatch -J piADR execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise --auto_dr\
    -e PegInsertionSide-v1 \
    --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-ADR_2 \
    --ckpt_name model_latest.zip

sbatch -J piDR execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise --only_DR\
    -e PegInsertionSide-v1 \
    --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-onlyDR_1 \
    --ckpt_name model_latest.zip

sbatch -J piDR execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise --only_DR\
    -e PegInsertionSide-v1 \
    --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-onlyDR_2 \
    --ckpt_name model_latest.zip

# TrunFaucet
# sbatch -J tf execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
#     --randomized_training --ext_disturbance --obs_noise \
#     -e TurnFaucet-v1 \
#     \
#     --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_1 \
#     --ckpt_name model_latest.zip 

# sbatch -J tf execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
#     --randomized_training --ext_disturbance --obs_noise \
#     -e TurnFaucet-v1 \
#     \
#     --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_2 \
#     --ckpt_name model_latest.zip

sbatch -J tfD execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise --use_depth_base\
    -e TurnFaucet-v1 \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-dep_1 \
    --ckpt_name model_latest.zip 

sbatch -J tfD execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise --use_depth_base\
    -e TurnFaucet-v1 \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-dep_2 \
    --ckpt_name model_latest.zip

sbatch -J tfADR execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise --auto_dr\
    -e TurnFaucet-v1 \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-ADR_1 \
    --ckpt_name model_latest.zip

sbatch -J tfADR execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise --auto_dr\
    -e TurnFaucet-v1 \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-ADR_2\
    --ckpt_name model_latest.zip

sbatch -J tfDR execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise --only_DR\
    -e TurnFaucet-v1 \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-onlyDR_1 \
    --ckpt_name model_latest.zip

sbatch -J tfDR execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
    --randomized_training --ext_disturbance --obs_noise --only_DR\
    -e TurnFaucet-v1 \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-onlyDR_2\
    --ckpt_name model_latest.zip

# sbatch -J tfi execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
#     --randomized_training --ext_disturbance --obs_noise --inc_obs_noise_in_priv \
#     -e TurnFaucet-v1 \
#     \
#     --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN1_1 \
#     --ckpt_name model_latest.zip 

# sbatch -J tfi execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
#     --randomized_training --ext_disturbance --obs_noise --inc_obs_noise_in_priv \
#     -e TurnFaucet-v1 \
#     \
#     --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN1_2 \
#     --ckpt_name model_latest.zip

# sbatch -J tfSI execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
#     --randomized_training --ext_disturbance --obs_noise --sys_iden \
#     -e TurnFaucet-v1 \
#     \
#     --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-SyId_1 \
#     --ckpt_name model_latest.zip

# sbatch -J tfSI execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct \
#     --randomized_training --ext_disturbance --obs_noise --sys_iden \
#     -e TurnFaucet-v1 \
#     \
#     --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-SyId_2 \
#     --ckpt_name model_latest.zip

# sbatch -J tfNoOE execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct\
#     --randomized_training --ext_disturbance --obs_noise --no_obj_embedding \
#     -e TurnFaucet-v1 \
#     \
#     --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-noObjEmb_1 \
#     --ckpt_name model_latest.zip

# sbatch -J tfNoOE execute.sh python main.py -n 50 -bs 5000 -rs 2000 -ct\
#     --randomized_training --ext_disturbance --obs_noise --no_obj_embedding \
#     -e TurnFaucet-v1 \
#     \
#     --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-noObjEmb_2 \
#     --ckpt_name model_latest.zip
