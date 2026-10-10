#!/bin/bash
# Example SLURM submission script for the Study III perturbation encoding models on the Tuckute2024 dataset.
# Adapt the partition, resources, conda environment and paths to your cluster.
#SBATCH -p <partition>
#SBATCH --mem=23G
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH -t 24:00:00
#SBATCH --output=perturb_output_%j.out
#SBATCH --error=perturb_error_%j.err

source <path/to/conda>/etc/profile.d/conda.sh
conda activate <env>

cd <path/to/repository>
# trains the encoding models on the perturbed stimuli (writes additional_analyses/control/perturbation/registered_models/)
python additional_analyses/control/perturbation/perturbation_encoding.py
