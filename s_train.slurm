#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-sxm4:1
#SBATCH --job-name=huesandcues
#SBATCH --output=log_huesandcues.log
 
module purge
eval "$(conda shell.bash hook)"
conda activate imagenet21k

python -u loop_openclip.py
