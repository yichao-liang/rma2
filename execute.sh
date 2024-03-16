#!/bin/bash

#SBATCH --job-name=exp
# SBATCH --partition=tenenbaum
# SBATCH --qos=tenenbaum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
# SBATCH --gres=gpu:RTXA6000:1
# SBATCH --constraint=high-capacity
#SBATCH --constraint=12GB
# SBATCH --constraint=3GB
# SBATCH --time=2:00:00
#SBATCH --time=2-00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=ycliang6@gmail.edu
#SBATCH --output=output/%x.%j.out
#SBATCH --error=output/%x.%j.err

source activate mani5
cd /om2/user/ycliang/gms

$@