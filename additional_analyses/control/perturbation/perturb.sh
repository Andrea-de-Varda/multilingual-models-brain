#!/bin/bash
#SBATCH -p evlab
#SBATCH --qos=evlab
#SBATCH --mem=23G
#SBATCH --gres=shard:a100:1
#SBATCH -c 1
#SBATCH -t 100:00:00
#SBATCH --output=perturb_output_%j.out
#SBATCH --error=perturb_error_%j.err
#SBATCH --chdir=/om2/user/devar_ag/multilingual-models-brain/additional_analyses/control

# Load conda
source /cm/shared/openmind/anaconda/3-2021.05/etc/profile.d/conda.sh
conda activate reasoning-models

echo "Running on $(hostname)"
echo "PWD: $(pwd)"
which python
python --version

# python perturbation/perturbation_encoding.py
python perturbation/embeddings_sim_perturbation.py
