#!/bin/sh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2 
#SBATCH --mem-per-cpu=16G


#SBATCH --gres=gpu:1 

#SBATCH --time=2-23:00:00            
#SBATCH --job-name=mobile_net 

#SBATCH --output=out_files/mobile_net_simple_classification_pretrained_augmentation_scheduler_new.out
source activate dhenv
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
python train.py
