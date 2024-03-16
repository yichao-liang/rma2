# -- Evuation -- 
# TurnFaucet
# --- GMS ---
# --- RMA^2 ---
sbatch -J EvTf execute.sh python train_sb.py --eval --seed 6\
    --randomized_training --ext_disturbance --obs_noise --use_depth_adaptation\
    -e TurnFaucet-v1 --ckpt_name best_model.zip \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_1_stage2_dep_1

sbatch -J EvTf execute.sh python train_sb.py --eval\
    --randomized_training --ext_disturbance --obs_noise --use_depth_adaptation\
    -e TurnFaucet-v1 --ckpt_name best_model.zip \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_2_stage2_dep_1

# --- expert adaptation model ---
sbatch -J EvTfEA execute.sh python train_sb.py --eval --expert_adapt  --seed 6\
    --randomized_training --ext_disturbance --obs_noise \
    -e TurnFaucet-v1 --ckpt_name best_model.zip \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_1

sbatch -J EvTfEA execute.sh python train_sb.py --eval --expert_adapt \
    --randomized_training --ext_disturbance --obs_noise\
    -e TurnFaucet-v1 --ckpt_name best_model.zip \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_2

# domain randomization model (not using priv. info.)
sbatch -J EvTfDR execute.sh python train_sb.py --eval --seed 6 \
    --randomized_training --ext_disturbance --obs_noise --only_DR\
    -e TurnFaucet-v1 --ckpt_name best_model.zip \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-onlyDR_1

sbatch -J EvTfDR execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise --only_DR\
    -e TurnFaucet-v1 --ckpt_name best_model.zip \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-onlyDR_2

# domain randomization model, vision-based (not using priv. info.)
sbatch -J EvTfDRV execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise --use_depth_base\
    -e TurnFaucet-v1 --ckpt_name best_model.zip \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-dep_1

sbatch -J EvTfDRV execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise --use_depth_base\
    -e TurnFaucet-v1 --ckpt_name best_model.zip \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-dep_2

# --- ADR ---
sbatch -J EvTfADR execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise --auto_dr\
    -e TurnFaucet-v1 --ckpt_name best_model.zip \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-ADR_1

sbatch -J EvTfADR execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise --auto_dr\
    -e TurnFaucet-v1 --ckpt_name best_model.zip \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-ADR_2

# without obj embeddings
sbatch -J EvTfNoOE execute.sh python train_sb.py --eval --use_depth_adaptation\
    --randomized_training --ext_disturbance --obs_noise --no_obj_embedding\
    -e TurnFaucet-v1 --ckpt_name best_model.zip \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-noObjEmb_1_stage2_dep_1

sbatch -J EvTfNoOE execute.sh python train_sb.py --eval --use_depth_adaptation\
    --randomized_training --ext_disturbance --obs_noise --no_obj_embedding\
    -e TurnFaucet-v1 --ckpt_name best_model.zip \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-noObjEmb_2_stage2_dep_1

# --- NoVA model ---
sbatch -J EvTfNoVA execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise \
    -e TurnFaucet-v1 --ckpt_name best_model.zip \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_1_stage2_1

sbatch -J EvTfNoVA execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise \
    -e TurnFaucet-v1 --ckpt_name best_model.zip \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_2_stage2_1

# PickSingle
# --- RMA^2 --- 
sbatch -J Evps execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise --use_depth_adaptation\
    -e PickSingleYCB-v1 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_1_stage2_dep_1

sbatch -J Evps execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise --use_depth_adaptation\
    -e PickSingleYCB-v1 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_2_stage2_dep_1

sbatch -J EvpsEGAD execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise --use_depth_adaptation\
    -e PickSingleEGAD-v2 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_1_stage2_dep_1

sbatch -J EvpsEGAD execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise --use_depth_adaptation\
    -e PickSingleEGAD-v2 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_2_stage2_dep_1

# --- expert adaptation model ---
sbatch -J EvpsEA execute.sh python train_sb.py --eval --expert_adapt --seed 6\
    --randomized_training --ext_disturbance --obs_noise \
    -e PickSingleYCB-v1 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_1

sbatch -J EvpsEA execute.sh python train_sb.py --eval --expert_adapt \
    --randomized_training --ext_disturbance --obs_noise\
    -e PickSingleYCB-v1 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_2

# domain randomization model (not using priv. info.)
sbatch -J EvpsDr execute.sh python train_sb.py --eval  --seed 6\
    --randomized_training --ext_disturbance --obs_noise --only_DR\
    -e PickSingleYCB-v1 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-onlyDR_1

sbatch -J EvpsDr execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise --only_DR\
    -e PickSingleYCB-v1 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-onlyDR_2

sbatch -J EvpsDrEGDA execute.sh python train_sb.py --eval --seed 6\
    --randomized_training --ext_disturbance --obs_noise --only_DR\
    -e PickSingleEGAD-v2 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-onlyDR_1

sbatch -J EvpsDrEGDA execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise --only_DR\
    -e PickSingleEGAD-v2 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-onlyDR_2

# domain randomization model, vision-based (not using priv. info.)
sbatch -J EvpsDRV execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise --use_depth_base\
    -e PickSingleYCB-v1 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-dep_1

sbatch -J EvpsDRV execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise --use_depth_base\
    -e PickSingleYCB-v1 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-dep_2

sbatch -J EvpsEgdaDRV execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise --use_depth_base\
    -e PickSingleEGAD-v2 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-dep_1

sbatch -J EvpsEgdaDRV execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise --use_depth_base\
    -e PickSingleEGAD-v2 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-dep_2

# --- ADR: automatic domain randomization model (not using priv. info.) ---
sbatch -J EvpsADr execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise --auto_dr\
    -e PickSingleYCB-v1 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-ADR_1

sbatch -J EvpsADr execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise --auto_dr\
    -e PickSingleYCB-v1 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-ADR_2

sbatch -J EvpsEgdaADr execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise --auto_dr\
    -e PickSingleEGAD-v2 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-ADR_1

sbatch -J EvpsEgdaADr execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise --auto_dr\
    -e PickSingleEGAD-v2 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-ADR_2

# no_obj_embedding
sbatch -J EvpsNoOE execute.sh python train_sb.py --eval --use_depth_adaptation \
    --randomized_training --ext_disturbance --obs_noise --no_obj_embedding\
    -e PickSingleYCB-v1 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-noObjEmb_1_stage2_dep_1

sbatch -J EvpsNoOE execute.sh python train_sb.py --eval --use_depth_adaptation \
    --randomized_training --ext_disturbance --obs_noise --no_obj_embedding\
    -e PickSingleYCB-v1 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-noObjEmb_2_stage2_dep_1

sbatch -J EvpsEgdaNoOE execute.sh python train_sb.py --eval --use_depth_adaptation \
    --randomized_training --ext_disturbance --obs_noise --no_obj_embedding\
    -e PickSingleEGAD-v2 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-noObjEmb_1_stage2_dep_1

sbatch -J EvpsEgdaNoOE execute.sh python train_sb.py --eval --use_depth_adaptation \
    --randomized_training --ext_disturbance --obs_noise --no_obj_embedding\
    -e PickSingleEGAD-v2 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-noObjEmb_2_stage2_dep_1

# --- no_VA: no vision in adaptation
sbatch -J EvpsNoVA execute.sh python train_sb.py --eval --seed 6\
    --randomized_training --ext_disturbance --obs_noise \
    -e PickSingleYCB-v1 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_1_stage2_1

sbatch -J EvpsNoVA execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise \
    -e PickSingleYCB-v1 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_2_stage2_1

sbatch -J EvpsEgdaNoVA execute.sh python train_sb.py --eval --seed 6\
    --randomized_training --ext_disturbance --obs_noise \
    -e PickSingleEGAD-v2 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_1_stage2_1

sbatch -J EvpsEgdaNoVA execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise \
    -e PickSingleEGAD-v2 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_2_stage2_1

# PegInsert
# --- RMA^2 ---
sbatch -J EvPi execute.sh python train_sb.py --eval\
    --randomized_training --ext_disturbance --obs_noise --use_depth_adaptation\
    -e PegInsertionSide-v1 --ckpt_name best_model.zip \
    --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_1_stage2_dep_1

sbatch -J EvPi execute.sh python train_sb.py --eval\
    --randomized_training --ext_disturbance --obs_noise --use_depth_adaptation\
    -e PegInsertionSide-v1 --ckpt_name best_model.zip \
    --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_2_stage2_dep_1

# --- expert adaptation model ---
sbatch -J EvPiEA execute.sh python train_sb.py --eval --expert_adapt \
    --randomized_training --ext_disturbance --obs_noise \
    -e PegInsertionSide-v1 --ckpt_name best_model.zip \
    --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_1

sbatch -J EvPiEA execute.sh python train_sb.py --eval --expert_adapt \
    --randomized_training --ext_disturbance --obs_noise\
    -e PegInsertionSide-v1 --ckpt_name best_model.zip \
    --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_2

# domain randomization model (not using priv. info.)
sbatch -J EvPiDR execute.sh python train_sb.py --eval --seed 6\
    --randomized_training --ext_disturbance --obs_noise --only_DR\
    -e PegInsertionSide-v1 --ckpt_name best_model.zip \
    --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-onlyDR_1

sbatch -J EvPiDR execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise --only_DR\
    -e PegInsertionSide-v1 --ckpt_name best_model.zip \
    --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-onlyDR_2

# domain randomization model, vision-based (not using priv. info.)
sbatch -J EvPiDRV execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise --use_depth_base\
    -e PegInsertionSide-v1 --ckpt_name best_model.zip \
    --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-dep_1

sbatch -J EvPiDRV execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise --use_depth_base\
    -e PegInsertionSide-v1 --ckpt_name best_model.zip \
    --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-dep_2

# --- ADR ---
sbatch -J EvPiADR execute.sh python train_sb.py --eval --seed 6\
    --randomized_training --ext_disturbance --obs_noise --auto_dr\
    -e PegInsertionSide-v1 --ckpt_name best_model.zip \
    --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-ADR_1

sbatch -J EvPiADR execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise --auto_dr\
    -e PegInsertionSide-v1 --ckpt_name best_model.zip \
    --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-ADR_2

# --- NoVA model ---
sbatch -J EvPiNoVA execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise \
    -e PegInsertionSide-v1 --ckpt_name best_model.zip \
    --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_1_stage2_1

sbatch -J EvPiNoVA execute.sh python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise \
    -e PegInsertionSide-v1 --ckpt_name best_model.zip \
    --log_name PPO-pi-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_2_stage2_1
