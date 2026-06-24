#!/bin/bash
#SBATCH -A   berzelius-2026-117 
#SBATCH --gpus 1
#SBATCH -t 240

module load Mambaforge/23.3.1-1-hpc1-bdist; 
conda activate /proj/beyondfold/apps/.conda/envs/pytorch_2.1.0
python DistoMove/train.py "$@" 
