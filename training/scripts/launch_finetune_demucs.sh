#!/bin/bash
#SBATCH --job-name=demucs-ft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=work
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/finetune_demucs_%j.out
#SBATCH --error=logs/finetune_demucs_%j.err

# Exact submission used for the Demucs "parity" dereverberation fine-tune
# (DGX, 1 x A100 40 GB): lr 1e-4, 50 epochs, the MUSE-recipe settings.
# Best validation PESQ 2.237 at epoch 45 (the checkpoint probed as "parity").
# Only the repo / env / data paths were made generic.

set -o errexit -o pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
CONDA_ENV=${CONDA_ENV:-meta-interface-py310}
RIR_DIR=${RIR_DIR:-data/rirs_clipped}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export PYTHONPATH="${PYTHONPATH}:${REPO}"
export PYTHONUNBUFFERED=1

cd "$REPO"
mkdir -p logs

python train_demucs.py \
  --config config_finetune_demucs.json \
  --checkpoint_path checkpoints/dereverb_demucs \
  --training_epochs 50 \
  --lr 1e-4 \
  --val_rir_ratio 0.05 \
  --val_subset 30 \
  --input_clean_wavs_dir data/VB_DEMAND_16K/clean_train \
  --input_training_file VoiceBank+DEMAND/training.txt \
  --input_validation_file VoiceBank+DEMAND/test.txt \
  --val_clean_wavs_dir data/VB_DEMAND_16K/clean_test \
  --rir_dir "$RIR_DIR" \
  --rir_metadata data/rir_metadata.csv \
  --rir_split data/rir_split.json
