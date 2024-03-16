# -- Adaptation Training --
# --- TurnFaucet ---
sbatch -J Atfd execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 \
    --randomized_training --ext_disturbance --obs_noise --use_depth_adaptation\
    -e TurnFaucet-v1 --adaptation_training \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_1 \
    --ckpt_name best_model.zip

sbatch -J Atfd execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 \
    --randomized_training --ext_disturbance --obs_noise --use_depth_adaptation\
    -e TurnFaucet-v1 --adaptation_training \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_2 \
    --ckpt_name best_model.zip

sbatch -J AtfNoVA execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 \
    --randomized_training --ext_disturbance --obs_noise \
    -e TurnFaucet-v1 --adaptation_training \
    \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_1 \
    --ckpt_name best_model.zip

sbatch -J AtfNoVA execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 \
    --randomized_training --ext_disturbance --obs_noise \
    -e TurnFaucet-v1 --adaptation_training \
    \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_2 \
    --ckpt_name best_model.zip

# --- Ablation models --- 
# ablations NoOE
sbatch -J tfNoOEAd execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 \
    --randomized_training --ext_disturbance --obs_noise --no_obj_embedding --use_depth_adaptation\
    -e TurnFaucet-v1 --adaptation_training \
    \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-noObjEmb_1 \
    --ckpt_name best_model.zip

sbatch -J AtfNoOEd execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 \
    --randomized_training --ext_disturbance --obs_noise --no_obj_embedding --use_depth_adaptation\
    -e TurnFaucet-v1 --adaptation_training \
    \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-noObjEmb_2 \
    --ckpt_name best_model.zip

# # ablations SysIden
# sbatch -J AtfSI execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 \
#     --randomized_training --ext_disturbance --obs_noise --sys_iden\
#     -e TurnFaucet-v1 --adaptation_training \
#     \
#     --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-SyId_3 \
#     --ckpt_name best_model.zip

# sbatch -J AtfSI execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 \
#     --randomized_training --ext_disturbance --obs_noise --sys_iden\
#     -e TurnFaucet-v1 --adaptation_training \
#     \
#     --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-SyId_4 \
#     --ckpt_name best_model.zip

# --- PickSingle ---
# use depth
sbatch -J Apsd execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 \
    --randomized_training --ext_disturbance --obs_noise --use_depth_adaptation\
    -e PickSingleYCB-v1 --adaptation_training \
    \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_1 \
    --ckpt_name best_model.zip

sbatch -J Apsd execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 \
    --randomized_training --ext_disturbance --obs_noise --use_depth_adaptation\
    -e PickSingleYCB-v1 --adaptation_training \
    \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_2 \
    --ckpt_name best_model.zip

# doesn't use depth
sbatch -J ApsNoVA execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 \
    --randomized_training --ext_disturbance --obs_noise\
    -e PickSingleYCB-v1 --adaptation_training\
    \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_1 \
    --ckpt_name best_model.zip

sbatch -J ApsNoVA execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 \
    --randomized_training --ext_disturbance --obs_noise\
    -e PickSingleYCB-v1 --adaptation_training\
    \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_2 \
    --ckpt_name best_model.zip

# --- Ablation models --- 
# ablations NoOE
sbatch -J ApsNoOE execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 \
    --randomized_training --ext_disturbance --obs_noise --no_obj_embedding\
    -e PickSingleYCB-v1 --adaptation_training --use_depth_adaptation\
    \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-noObjEmb_1 \
    --ckpt_name best_model.zip

sbatch -J ApsNoOE execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 \
    --randomized_training --ext_disturbance --obs_noise --no_obj_embedding\
    -e PickSingleYCB-v1 --adaptation_training --use_depth_adaptation\
    \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-noObjEmb_2 \
    --ckpt_name best_model.zip

# # ablations SysIden
# sbatch -J psSIA execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 \
#     --randomized_training --ext_disturbance --obs_noise --sys_iden\
#     -e PickSingleYCB-v1 --adaptation_training \
#     \
#     --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-SyId_1 \
#     --ckpt_name best_model.zip

# sbatch -J psSIA execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 \
#     --randomized_training --ext_disturbance --obs_noise --sys_iden\
#     -e PickSingleYCB-v1 --adaptation_training \
#     \
#     --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-SyId_2 \
#     --ckpt_name best_model.zip

# --- PegInset ---
# sbatch -J piA execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 \
#     --randomized_training --ext_disturbance --obs_noise \
#     -e PegInsertionSide-v1 --adaptation_training \
#     \
#     --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_1 \
#     --ckpt_name best_model.zip

# sbatch -J piA execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 \
#     --randomized_training --ext_disturbance --obs_noise \
#     -e PegInsertionSide-v1 --adaptation_training \
#     \
#     --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_2 \
#     --ckpt_name best_model.zip

# # with noise in input
# sbatch -J piiA execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 \
#     --randomized_training --ext_disturbance --obs_noise --inc_obs_noise_in_priv\
#     -e PegInsertionSide-v1 --adaptation_training \
#     \
#     --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN1_1 \
#     --ckpt_name best_model.zip

# sbatch -J piiA execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 \
#     --randomized_training --ext_disturbance --obs_noise --inc_obs_noise_in_priv\
#     -e PegInsertionSide-v1 --adaptation_training \
#     \
#     --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN1_2 \
#     --ckpt_name best_model.zip

# --- Ablation models --- 
# ablations NoOE
# sbatch -J piNoOEA execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 \
#     --randomized_training --ext_disturbance --obs_noise --no_obj_embedding\
#     -e PegInsertionSide-v1 --adaptation_training \
#     \
#     --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-noObjEmb_1 \
#     --ckpt_name best_model.zip

# sbatch -J piNoOEA execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 \
#     --randomized_training --ext_disturbance --obs_noise --no_obj_embedding\
#     -e PegInsertionSide-v1 --adaptation_training \
#     \
#     --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-noObjEmb_2 \
#     --ckpt_name best_model.zip

# # ablations SysIden
# sbatch -J piSIA execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 \
#     --randomized_training --ext_disturbance --obs_noise --sys_iden\
#     -e PegInsertionSide-v1 --adaptation_training \
#     \
#     --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-SyId_1 \
#     --ckpt_name best_model.zip

# sbatch -J piSIA execute.sh python train_sb.py -n 50 -bs 5000 -rs 2000 \
#     --randomized_training --ext_disturbance --obs_noise --sys_iden\
#     -e PegInsertionSide-v1 --adaptation_training \
#     \
#     --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-SyId_2 \
#     --ckpt_name best_model.zip
