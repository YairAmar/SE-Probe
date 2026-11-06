#!/bin/bash
#SBATCH --job-name=mpse-ft-ga
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=work
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/finetune_mpsenet_ga_%j.out
#SBATCH --error=logs/finetune_mpsenet_ga_%j.err

# This is the exact submission used for the paper's MP-SENet dereverberation
# fine-tune (DGX cluster, 2 x A100 40 GB, three chained auto-resume jobs). The
# only edits relative to the original are the generic repo / env / data paths.
#
# Memory-safe, NaN-guarded dereverb FT on 40 GB A100s. Global batch 28 preserved via
# gradient accumulation: config batch_size 4 (-> 2/GPU on 2 GPUs) * grad_accum 7 = 28.
# NaN-guard skips optimizer steps on non-finite grad norm (prevents the atan2(0,0)
# NaN-poisoning that killed the prior run). Validation + checkpoint once per epoch
# (5800 micro-steps; epoch = 23250 chunks / (2 GPU * 2 micro) = 5812). Base = DNS
# remapped weights (scripts/convert_dns_to_trainer.py); STFT config matches DNS so
# the output reloads into MPSENet.from_pretrained('JacobLinCool/MP-SENet-DNS').

set -o errexit -o pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
CONDA_ENV=${CONDA_ENV:-meta-interface-py310}
RIR_DIR=${RIR_DIR:-data/rirs_clipped}
CKPT_DIR=${CKPT_DIR:-checkpoints_mpsenet/all_v2}
BASE_CKPT=${BASE_CKPT:-checkpoints_mpsenet/dns_base_converted.pt}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export PYTHONPATH="${PYTHONPATH}:${REPO}"
export PYTHONUNBUFFERED=1
# torch 1.12 does not understand expandable_segments; use max_split_size_mb instead.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export MASTER_PORT=$((29500 + ${SLURM_JOB_ID:-0} % 10000))

cd "$REPO"
mkdir -p logs

# On re-submit the trainer auto-resumes from its own latest g_/do_ checkpoint
# in $CKPT_DIR (saved once per epoch via --checkpoint_interval).
python train_mpsenet.py \
  --config config_mpsenet_finetune_ga.json \
  --mode reverb \
  --pretrained_checkpoint "$BASE_CKPT" \
  --unfreeze all \
  --lr 1e-4 \
  --grad_accum 7 \
  --training_epochs 50 \
  --stdout_interval 20 \
  --checkpoint_interval 5800 \
  --validation_interval 5800 \
  --checkpoint_path "$CKPT_DIR" \
  --save_every_epoch \
  --input_clean_wavs_dir data/VB_DEMAND_16K/clean_train \
  --input_noisy_wavs_dir data/VB_DEMAND_16K/noisy_train \
  --rir_dir "$RIR_DIR" \
  --rir_metadata data/rir_metadata.csv \
  --rir_split data/rir_split.json \
  --val_rir_ratio 0.05 \
  --input_training_file VoiceBank+DEMAND/training.txt \
  --input_validation_file VoiceBank+DEMAND/test.txt \
  --val_clean_wavs_dir data/VB_DEMAND_16K/clean_test \
  --val_noisy_wavs_dir data/VB_DEMAND_16K/noisy_test
