#!/bin/bash
#SBATCH -p evlab
#SBATCH --mem=10G
#SBATCH -c 1
#SBATCH -t 30:00:00
#SBATCH --output=output_%j.out

source /cm/shared/openmind/anaconda/3-2021.05/etc/profile.d/conda.sh
conda activate reasoning-models

# python fit_encoding.py --mode within
# python fit_encoding.py --mode across
# python fit_encoding.py --mode RH
# python fit_encoding.py --mode MD