#!/bin/bash
#SBATCH -p pi_evelina9
#SBATCH --gres=gpu:a100:1
#SBATCH -t 24:00:00
#SBATCH -c 1
#SBATCH --mem=40G
#SBATCH --job-name=syntsem_enc_control
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /etc/profile.d/modules.sh

module purge
module load deprecated-modules
module load anaconda3/2022.05-x86_64
source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda activate modularity-conda
conda deactivate
conda activate modularity-conda

export HF_HOME=/orcd/data/evelina9/001/USERS/devar_ag/.hf_cache_new
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

export HF_OFFLOAD_DIR=/orcd/data/evelina9/001/USERS/devar_ag/offload
mkdir -p "$HF_OFFLOAD_DIR"

PROJECT_DIR=/orcd/data/evelina9/001/USERS/devar_ag/multilingual-models-brain/additional_analyses/syntax_semantics
cd "$PROJECT_DIR"

echo "Running on $(hostname)"
echo "PWD: $(pwd)"
which python
python --version

python encoding.py --dataset control
