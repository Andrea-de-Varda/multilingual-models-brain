#!/bin/bash
# Example SLURM submission script for the Study III perturbation analyses on the Pereira2018 dataset.
# Adapt the partition, resources, conda environment and paths to your cluster.
#SBATCH -p <partition>
#SBATCH --mem=23G
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH -t 10:00:00
#SBATCH --output=perturb_output_%j.out
#SBATCH --chdir=<path/to/repository>/additional_analyses/pereira/perturbation

source <path/to/conda>/etc/profile.d/conda.sh
conda activate <env>

# 1. train encoding models on perturbed stimuli (writes perturbation/registered_models/)
python perturbation_encoding.py
# 2. embedding similarity between intact and perturbed stimuli (writes similarity_outputs_pereira/)
python embeddings_sim_perturbation_pereira.py
