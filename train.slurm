#!/bin/sh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2 
#SBATCH --mem-per-cpu=16G


#SBATCH --gres=gpu:1 

#SBATCH --time=2-23:00:00            
#SBATCH --job-name=mobile_net 

#SBATCH --output=out_files/mobile_net_simple_classification.out
nvidia-smi
conda activate dhenv
python train.py
