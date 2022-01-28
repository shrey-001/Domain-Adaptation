#!/bin/sh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2 
#SBATCH --mem-per-cpu=16G


#SBATCH --gres=gpu:1 

#SBATCH --time=2-23:00:00            
#SBATCH --job-name=all_m 

#SBATCH --output=out_files/all_m.out
source activate da
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
python train.py
