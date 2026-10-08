#!/bin/bash
# Example SLURM submission script for the Study I encoding analyses (fit_encoding.py).
# Adapt the partition, resources, conda environment and paths to your cluster.
#SBATCH -p <partition>
#SBATCH --gres=gpu:1
#SBATCH --mem=6G
#SBATCH -c 1
#SBATCH -t 100:00:00
#SBATCH --output=logs/encoding_%j.out

source <path/to/conda>/etc/profile.d/conda.sh
conda activate <env>

cd <path/to/repository>
mkdir -p logs

# One --mode per job (see fit_encoding.py for the full list):
#   within, across, across-random, RH, MD, native-within, native-across, within-split, across-split
python fit_encoding.py --mode within
