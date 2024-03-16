#!/bin/bash

# Directory containing the .obj files
directory="data/mani_skill2_egad/egad_train_set"
checked_models="checked_eg_model_ids_rma.txt"
# checked_models="checked_eg_model_ids_dr.txt"

# counter for the numbe of iterations
counter=0
# Loop through each .obj file in the directory
for filepath in "$directory"/*_0.obj; do
  # Extract the filename without the path and the .obj extension
  filename=$(basename -- "$filepath")
  model_id="${filename%.*}"

  # Check if model_id exists in the file
  if grep -Fxq "$model_id" "$checked_models"; then
    continue
  fi

  echo $model_id

  # Pass the model_id to the python command as an argument
  # sbatch -J EgDR execute.sh python train_sb.py --eval \
  #   --randomized_training --ext_disturbance --obs_noise --use_depth_base\
  #   -e PickSingleEGAD-v2 --ckpt_name best_model.zip \
  #   --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-dep_1\
  #   --eval_model_id "$model_id"

  sbatch -J egRMA execute.sh python train_sb.py --eval \
      --randomized_training --ext_disturbance --obs_noise --use_depth_adaptation\
      -e PickSingleEGAD-v2 --ckpt_name best_model.zip \
      --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_2_stage2_dep_1\
      --eval_model_id "$model_id"
  sleep 5
  # # Increment the counter
  # ((counter++))

  # # Break the loop after the second iteration
  # if [ $counter -ge 2 ]; then
  # break
  # fi
done
