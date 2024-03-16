#!/bin/bash

# Directory containing the .obj files
directory="data/partnet_mobility/dataset"
checked_models="checked_tf_model_ids_sr.txt"

# counter for the numbe of iterations
counter=0
# Loop through each .obj file in the directory
for filepath in "$directory"/*; do
  # Extract the filename without the path and the .obj extension
  filename=$(basename -- "$filepath")
  model_id="${filename%.*}"
  
  # Check if model_id exists in the file
  if grep -Fxq "$model_id" "$checked_models"; then
    continue
  fi
  
  echo $model_id

  # # Pass the model_id to the python command as an argument

  # sbatch -J tfDR execute.sh python train_sb.py --eval \
  #   --randomized_training --ext_disturbance --obs_noise --only_DR\
  #   -e TurnFaucet-v1 --ckpt_name best_model.zip \
  #   --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0-onlyDR_1\
  #   --eval_model_id "$model_id"

  sbatch -J tfRMA execute.sh python train_sb.py --eval --expert_adapt \
    --randomized_training --ext_disturbance --obs_noise \
    -e TurnFaucet-v1 --ckpt_name best_model.zip \
    --log_name PPO-tf-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_1\
    --eval_model_id "$model_id"
  sleep 30
  # # Increment the counter
  # ((counter++))

  # # Break the loop after the second iteration
  # if [ $counter -ge 2 ]; then
  # break
  # fi
done
