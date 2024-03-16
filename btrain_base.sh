# # --- PickCube ---
# sbatch -J pc execute.sh python main.py -n 52 -bs 5200 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise \
#             -e PickCube-v1
# sleep 15
# sbatch -J pc execute.sh python main.py -n 52 -bs 5200 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise \
#             -e PickCube-v1

# --- PickSingle ---
# sbatch -J ps execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise \
#             -e PickSingleYCB-v1
# sleep 15
# sbatch -J ps execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise \
#             -e PickSingleYCB-v1

sbatch -J ps execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
            --randomized_training --ext_disturbance --obs_noise \
            -e PickSingleYCB-v1 --robot xarm7
sleep 15
sbatch -J ps execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
            --randomized_training --ext_disturbance --obs_noise \
            -e PickSingleYCB-v1 --robot xarm7

# sbatch -J psD execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --use_depth_base\
#             -e PickSingleYCB-v1
# sleep 15
# sbatch -J psD execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --use_depth_base\
#             -e PickSingleYCB-v1

# sbatch -J psi execute.sh python main.py -n 50 -bs 5000 -rs 2000 --inc_obs_noise_in_priv\
#             --randomized_training --ext_disturbance --obs_noise \
#             -e PickSingleYCB-v1
# sleep 15
# sbatch -J psi execute.sh python main.py -n 50 -bs 5000 -rs 2000 --inc_obs_noise_in_priv\
#             --randomized_training --ext_disturbance --obs_noise \
#             -e PickSingleYCB-v1

# # ablations NoOE
# sbatch -J psNoOE execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --no_obj_embedding\
#             -e PickSingleYCB-v1
# sleep 15
# sbatch -J psNoOE execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --no_obj_embedding\
#             -e PickSingleYCB-v1

# # baseline Automatic DR
# sbatch -J psADR execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --auto_dr\
#             -e PickSingleYCB-v1
# sleep 15
# sbatch -J psADR execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --auto_dr\
#             -e PickSingleYCB-v1

# # # ablations DR
# sbatch -J psDR execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --only_DR\
#             -e PickSingleYCB-v1
# sleep 15
# sbatch -J psDR execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --only_DR\
#             -e PickSingleYCB-v1

# # # ablations SysIden
# sbatch -J psSI execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --sys_iden\
#             -e PickSingleYCB-v1
# sleep 15
# sbatch -J psSI execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --sys_iden\
#             -e PickSingleYCB-v1

# depth
sbatch -J psd execute.sh python main.py -n 50 -bs 5000 -rs 2000 --use_depth_base\
            --randomized_training --ext_disturbance --obs_noise \
            -e PickSingleYCB-v1
sleep 15
sbatch -J psd execute.sh python main.py -n 50 -bs 5000 -rs 2000 --use_depth_base\
            --randomized_training --ext_disturbance --obs_noise \
            -e PickSingleYCB-v1

# with both depth and proprioception-action history
sbatch -J psdp execute.sh python main.py -n 50 -bs 5000 -rs 2000 --use_depth_base\
            --randomized_training --ext_disturbance --obs_noise \
            -e PickSingleYCB-v1 --use_prop_history_base
sleep 15
sbatch -J psdp execute.sh python main.py -n 50 -bs 5000 -rs 2000 --use_depth_base\
            --randomized_training --ext_disturbance --obs_noise \
            -e PickSingleYCB-v1 --use_prop_history_base

# with proprioception-action history
sbatch -J psp execute.sh python main.py -n 50 -bs 5000 -rs 2000 --use_prop_history_base\
            --randomized_training --ext_disturbance --obs_noise \
            -e PickSingleYCB-v1
sleep 15
sbatch -J psp execute.sh python main.py -n 50 -bs 5000 -rs 2000 --use_prop_history_base\
            --randomized_training --ext_disturbance --obs_noise \
            -e PickSingleYCB-v1


# --- PegInsert ---
# sbatch -J pi execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise\
#             -e PegInsertionSide-v1
# sleep 15
# sbatch -J pi execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise\
#             -e PegInsertionSide-v1

# sbatch -J piD execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --use_depth_base\
#             -e PegInsertionSide-v1
# sleep 15
# sbatch -J piD execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --use_depth_base\
#             -e PegInsertionSide-v1

# sbatch -J pii execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --inc_obs_noise_in_priv\
#             -e PegInsertionSide-v1
# sleep 15
# sbatch -J pii execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --inc_obs_noise_in_priv\
#             -e PegInsertionSide-v1

# # ablations NoOE
# sbatch -J piNoOE execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --no_obj_embedding\
#             -e PegInsertionSide-v1
# sleep 15
# sbatch -J piNoOE execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --no_obj_embedding\
#             -e PegInsertionSide-v1

# sbatch -J piADR execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --auto_dr\
#             -e PegInsertionSide-v1
# sleep 15
# sbatch -J piADR execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --auto_dr\
#             -e PegInsertionSide-v1

# # ablations DR
# sbatch -J piDR execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --only_DR\
#             -e PegInsertionSide-v1
# sleep 15
# sbatch -J piDR execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --only_DR\
#             -e PegInsertionSide-v1

# # ablations SysIden
# sbatch -J piSI execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --sys_iden\
#             -e PegInsertionSide-v1
# sleep 15
# sbatch -J piSI execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --sys_iden\
#             -e PegInsertionSide-v1

# # --- TurnFaucet ---
# sbatch -J tf execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise \
#             -e TurnFaucet-v1
# sleep 15
# sbatch -J tf execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise \
#             -e TurnFaucet-v1

# sbatch -J tfD execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --use_depth_base\
#             -e TurnFaucet-v1
# sleep 15
# sbatch -J tfD execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --use_depth_base\
#             -e TurnFaucet-v1

# # # --- Ablation models --- 
# # # ablations NoOE
# sbatch -J tfNoOE execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --no_obj_embedding\
#             -e TurnFaucet-v1
# sleep 15
# sbatch -J tfNoOE execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --no_obj_embedding\
#             -e TurnFaucet-v1

# # # ablations DR
# sbatch -J tfADR execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --auto_dr\
#             -e TurnFaucet-v1
# sleep 15
# sbatch -J tfADR execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --auto_dr\
#             -e TurnFaucet-v1

# sbatch -J tfDR execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --only_DR\
#             -e TurnFaucet-v1
# sleep 15
# sbatch -J tfDR execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --only_DR\
#             -e TurnFaucet-v1

# # # ablations SysIden
# sbatch -J tfSI execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --sys_iden\
#             -e TurnFaucet-v1
# sleep 15
# sbatch -J tfSI execute.sh python main.py -n 50 -bs 5000 -rs 2000 \
#             --randomized_training --ext_disturbance --obs_noise --sys_iden\
#             -e TurnFaucet-v1