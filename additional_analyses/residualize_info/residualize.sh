#!/bin/bash
# Example SLURM submission script for the Study IV INLP ablation analysis (residualize_encoding.py; GPU recommended).
# Adapt the partition, resources, conda environment and paths to your cluster.
#SBATCH -p <partition>
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -c 1
#SBATCH --mem=23G
#SBATCH --job-name=residualize
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source <path/to/conda>/etc/profile.d/conda.sh
conda activate <env>

export HF_HOME=<path/to/huggingface/cache>
export TOKENIZERS_PARALLELISM=false

cd <path/to/repository>/additional_analyses/residualize_info
mkdir -p logs

python residualize_encoding.py
