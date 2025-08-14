#!/bin/bash
#SBATCH -p evlab
#SBATCH --mem=23G
#SBATCH --gres=shard:a100:1
#SBATCH -c 1
#SBATCH -t 10:00:00
#SBATCH --output=perturb_output_%j.out

source /cm/shared/openmind/anaconda/3-2021.05/etc/profile.d/conda.sh
conda activate reasoning-models
cd multilingual-models-brain/additional_analyses/pereira

python perturbation_encoding.py