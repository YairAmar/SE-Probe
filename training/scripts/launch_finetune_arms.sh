#!/bin/bash
#SBATCH --job-name=muse-ft-arms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=work
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/ft_arms_%j.out
#SBATCH --error=logs/ft_arms_%j.err

# Profile-guided fine-tuning arms (the "Job A" experiment): MUSE fine-tuned for
# dereverberation under three freezing regimes x three seeds, 50 epochs each,
# lr 1e-4, batch 28, on-the-fly RIR-Mega reverb.
#   full_ft         every generator parameter trainable (the baseline)
#   freeze_encoder  dense_encoder + TCFTransformer.encoder_level* frozen
#   freeze_decoder  TCFTransformer.decoder_level* + mask/phase decoders frozen
# Result (mean +/- sd over seeds, best-epoch PESQ on the held-out RIR-Mega val
# subset): full_ft 2.783 +/- 0.011, freeze_encoder 2.697 +/- 0.004,
# freeze_decoder 2.700 +/- 0.003 (STOI 0.9333 / 0.9300 / 0.9293).
#
# One (arm, seed) per submission: set ARM and SEED in the environment, e.g.
#   for arm in full_ft freeze_encoder freeze_decoder; do
#     for seed in 1234 2345 3456; do ARM=$arm SEED=$seed sbatch scripts/launch_finetune_arms.sh; done
#   done
# All paths are generic; override via environment variables.

set -o errexit -o pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
CONDA_ENV=${CONDA_ENV:-meta-interface-py310}
ARM=${ARM:-full_ft}
SEED=${SEED:-1234}
EPOCHS=${EPOCHS:-50}
RIR_DIR=${RIR_DIR:-data/rirs_clipped}
OUT_ROOT=${OUT_ROOT:-checkpoints/arms}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH}:${REPO}"

cd "$REPO"
mkdir -p logs

RUN_ID="${ARM}__seed${SEED}"
python train.py \
  --config config_finetune.json \
  --run_name "$RUN_ID" \
  --freeze_arm "$ARM" \
  --seed "$SEED" \
  --checkpoint_path "$OUT_ROOT/$RUN_ID" \
  --pretrained_checkpoint paper_result/g_best \
  --lr 0.0001 \
  --training_epochs "$EPOCHS" \
  --checkpoint_interval 5000 \
  --validation_interval 413 \
  --input_clean_wavs_dir data/VB_DEMAND_16K/clean_train \
  --input_training_file VoiceBank+DEMAND/training.txt \
  --input_validation_file VoiceBank+DEMAND/test.txt \
  --val_clean_wavs_dir data/VB_DEMAND_16K/clean_test \
  --rir_dir "$RIR_DIR" \
  --rir_metadata data/rir_metadata.csv \
  --rir_split data/rir_split.json \
  --val_rir_ratio 0.05
