#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2 
#SBATCH --mem-per-cpu=16G


#SBATCH --gres=gpu:1 

#SBATCH --time=8:00:00            
#SBATCH --job-name=da

#SBATCH --output=out_files/domain_adaptation.out

python train.py