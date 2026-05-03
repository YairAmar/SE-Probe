#!/bin/bash
#SBATCH --job-name=muse-dereverb-ft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=work
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err

set -o errexit -o pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate meta-interface-py310
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

mkdir -p logs

python train.py \
  --config config_finetune.json \
  --pretrained_checkpoint paper_result/g_best \
  --lr 0.0001 \
  --training_epochs 50 \
  --validation_interval 850 \
  --checkpoint_path checkpoints/dereverb \
  --input_clean_wavs_dir data/VB_DEMAND_16K/clean_train \
  --input_training_file VoiceBank+DEMAND/training.txt \
  --input_validation_file VoiceBank+DEMAND/test.txt \
  --val_clean_wavs_dir data/VB_DEMAND_16K/clean_test \
  --rir_dir data/rirs_clipped \
  --rir_metadata data/rir_metadata.csv \
  --rir_split data/rir_split.json \
  --val_rir_ratio 0.05
