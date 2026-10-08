#!/bin/bash
# Example SLURM submission script for the Study III perturbation analyses on the Tuckute2024 dataset.
# Adapt the partition, resources, conda environment and paths to your cluster.
#SBATCH -p <partition>
#SBATCH --mem=23G
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH -t 100:00:00
#SBATCH --output=perturb_output_%j.out
#SBATCH --error=perturb_error_%j.err
#SBATCH --chdir=<path/to/repository>/additional_analyses/control

source <path/to/conda>/etc/profile.d/conda.sh
conda activate <env>

# 1. train encoding models on perturbed stimuli (writes perturbation/registered_models/)
python perturbation/perturbation_encoding.py
# 2. embedding similarity between intact and perturbed stimuli (writes perturbation/similarity_outputs/)
python perturbation/embeddings_sim_perturbation.py
