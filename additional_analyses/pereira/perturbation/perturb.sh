#!/bin/bash
#SBATCH -p evlab
#SBATCH --mem=23G
#SBATCH --gres=shard:a100:1
#SBATCH -c 1
#SBATCH -t 10:00:00
#SBATCH --output=perturb_output_%j.out
#SBATCH --chdir=/om2/user/devar_ag/multilingual-models-brain/additional_analyses/pereira/perturbation

source /cm/shared/openmind/anaconda/3-2021.05/etc/profile.d/conda.sh
conda activate reasoning-models
# cd multilingual-models-brain/additional_analyses/pereira

# python perturbation_encoding.py
python embeddings_sim_perturbation_pereira.py