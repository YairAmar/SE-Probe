#!/bin/bash
#SBATCH --job-name=demucs-ft-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=work
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/finetune_demucs_v2_%j.out
#SBATCH --error=logs/finetune_demucs_v2_%j.err

# Exact submission used for the Demucs "v2" dereverberation fine-tune
# (DGX, 1 x A100 40 GB). Best validation PESQ 2.335 at epoch 57; epoch 57 is the
# Demucs checkpoint probed under reverberation in the paper.
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

# v2 vs parity (checkpoints/dereverb_demucs, config_finetune_demucs.json):
#   - lr 1e-4 -> 3e-4 (denoiser/Demucs upstream default; the parity 1e-4 was
#     chosen for MUSE-recipe parity, not Demucs's own optimum)
#   - training_epochs 50 -> 60 (a bit longer to exploit the higher lr)
#   - optimizer kept AdamW(betas=[0.8,0.99]), scheduler kept ExponentialLR(0.99),
#     loss kept L1 + MultiResolutionSTFTLoss (same weights) -- unchanged from
#     parity so the lr/epoch bump is isolated as the only real delta.
#   - output dir checkpoints/dereverb_demucs_v2 (parity dir untouched)
python train_demucs.py \
  --config config_finetune_demucs_v2.json \
  --checkpoint_path checkpoints/dereverb_demucs_v2 \
  --training_epochs 60 \
  --lr 3e-4 \
  --val_rir_ratio 0.05 \
  --val_subset 30 \
  --input_clean_wavs_dir data/VB_DEMAND_16K/clean_train \
  --input_training_file VoiceBank+DEMAND/training.txt \
  --input_validation_file VoiceBank+DEMAND/test.txt \
  --val_clean_wavs_dir data/VB_DEMAND_16K/clean_test \
  --rir_dir "$RIR_DIR" \
  --rir_metadata data/rir_metadata.csv \
  --rir_split data/rir_split.json
