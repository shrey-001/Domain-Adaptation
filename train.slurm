#!/bin/sh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2 
#SBATCH --mem-per-cpu=16G


#SBATCH --gres=gpu:1 

#SBATCH --time=2-23:00:00            
#SBATCH --job-name=simple-dmy

#SBATCH --output=out_files/simple-dmy.out

source activate da
python train.py
