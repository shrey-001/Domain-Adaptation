#!/bin/sh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2 
#SBATCH --mem-per-cpu=16G


#SBATCH --gres=gpu:1 

#SBATCH --time=2-23:00:00            
#SBATCH --job-name=test 

#SBATCH --output=out_files/test.out
source activate da
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
python train.py
