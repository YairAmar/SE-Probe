# MUSE — Fine-Tuning for Dereverberation

End-to-end fine-tuning of [MUSE](https://arxiv.org/pdf/2406.04589) (Lin et al.,
Interspeech 2024) for **dereverberation**. The base MUSE was trained on
VoiceBank+DEMAND for denoising; this repo adapts it to reverberant speech by
fine-tuning every parameter of the generator on clean utterances convolved
on-the-fly with real RIRs.

---

This folder now also carries the **MP-SENet** and **Demucs** dereverberation
fine-tunes, the held-out test evaluation of all three architectures, and the
selective-freezing / profile-guided arms used in the TASLP paper. See the
sections after "Evaluation" below.

## Provenance

This directory is a vendored, self-contained copy of the research trainer
(`Muse-Reverb-FN`, branches `mpsenet-ft`, `demucs-reverb-ft` and
`reverb-ft-jul2026`; the MUSE part was first carved out as `muse-dereverb-ft`).
Those research repositories are private and lived on the Technion clusters; every
cluster-specific path has been replaced here by a repo-relative one, and every
launch script is the exact SLURM submission that produced the published
checkpoints with only its paths made generic.

The fine-tuned checkpoint `checkpoints/g_00051852` is the same file as
`Muse-Reverb-FN/checkpoints/all/epoch_48/g_00051852` (the "all-block-unfrozen"
winner of an earlier selective-FT experiment, now repackaged as plain full FT).
It is also published as `yairamr/SE-Probe-models/muse_reverb_e48.pt` on
HuggingFace, next to the nine profile-guided arm checkpoints
`cluster-2026-07/<arm>__seed<S>/g_00040000`.

---

## Quick start: load + dereverberate one wav

```python
import json, torch, librosa, soundfile as sf
from env import AttrDict
from datasets.dataset import mag_pha_stft, mag_pha_istft
from models.generator import MUSE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
h = AttrDict(json.load(open('checkpoints/config.json')))      # hyperparams used at training time

model = MUSE(h).to(device).eval()
ckpt = torch.load('checkpoints/g_00051852', map_location=device)
model.load_state_dict(ckpt['generator'])

# Audio MUST be mono, 16 kHz
wav, _ = librosa.load('reverb.wav', sr=h.sampling_rate)
x = torch.from_numpy(wav).float().to(device)

# Energy-normalize, then chunk to segment_size=30700 samples (~1.92 s) and run
norm = torch.sqrt(len(x) / torch.sum(x**2))
x = (x * norm).unsqueeze(0)            # [1, T]
mag, pha, _ = mag_pha_stft(x, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
with torch.no_grad():
    mag_g, pha_g, _ = model(mag, pha)
y = mag_pha_istft(mag_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
y = (y / norm).squeeze().cpu().numpy()

sf.write('dereverbed.wav', y, h.sampling_rate, 'PCM_16')
```

**For full-length audio** (≥ `segment_size`), chunk first and concatenate. See
`process_audio()` in `train.py` or `process_audio_segment()` in `inference.py`
for the reference implementation. Or just shell out:

```bash
python inference.py \
  --checkpoint_file checkpoints/g_00051852 \
  --input_noisy_wavs_dir /path/to/reverb/wavs \
  --output_dir generated_files
```

`inference.py` reads its config from a sibling `config.json` in the
checkpoint's directory, which is already in place at `checkpoints/config.json`.

---

## Model I/O contract

| | |
|---|---|
| **Sample rate** | 16,000 Hz (mandatory — model was trained at 16 kHz only) |
| **Channels** | mono |
| **Normalization** | per-utterance: `x ← x · √(N / Σx²)` (RMS-style); divide by the same factor on the way back out |
| **STFT** | `n_fft=510`, `hop_size=100`, `win_size=510`, Hann window |
| **Magnitude compression** | `mag^β` with `β = compress_factor = 0.3` (model predicts a compressed-mag mask + phase) |
| **Segment size** | 30,700 samples (~1.92 s). For longer audio: split on `segment_size`, append a `[-segment_size:]` tail-segment if there's a remainder, run, drop overlap, concatenate, trim to original length |
| **Output** | `(mag_g, pha_g, com_g)` — feed `mag_g` and `pha_g` to `mag_pha_istft` to get the time-domain waveform |
| **Generator parameters** | 513,015 (~0.51 M) |

`mag_pha_stft` / `mag_pha_istft` live in `datasets/dataset.py`.

---

## Achieved metrics (vs. pretrained baseline)

Both checkpoints in this repo, evaluated on identical test sets:

| Test set | Checkpoint | PESQ | STOI |
|----------|-----------|------|------|
| Reverb (4,120 utts = 824 × 5 RIRs) | `paper_result/g_best` (baseline) | 2.172 | 0.834 |
| Reverb | `checkpoints/g_00051852` (fine-tuned) | **3.024** | **0.944** |
| Noise (824 VB+DEMAND test) | `paper_result/g_best` (baseline) | 3.349 | 0.950 |
| Noise (forgetting check) | `checkpoints/g_00051852` (fine-tuned) | 2.030 | 0.910 |

The fine-tuning shifts the model from denoising to dereverberation: +0.85 PESQ
on reverb at the cost of −1.32 PESQ on noise (catastrophic forgetting; expected).

---

## Environment

All published runs used one conda environment on the Technion clusters
(Python 3.10, torch 1.12.1+cu113; the launch scripts read its name from
`$CONDA_ENV`, default `meta-interface-py310`). CUDA-12 stacks did not work on
the DGX nodes used for training (driver 470 caps at CUDA 11.4); the held-out
"before" rows were later re-measured under torch 2.11 and reproduce the
published fine-tuned row to 8.6e-5 PESQ (see "Held-out test evaluation").

`requirements.txt` lists the original MUSE pins plus the packages the added
trainers need. Extra packages used by `train.py` / `evaluate.py`:
`wandb`, `pesq`, `pystoi`, `speechmos`, `matplotlib`, `onnx2torch` (for
GPU-accelerated DNSMOS in `evaluate.py`); `train_demucs.py` and
`evaluate_demucs_testset.py` additionally need `denoiser`, and
`scripts/convert_dns_to_trainer.py` needs `safetensors` + `huggingface_hub`.

---

## Repo layout

```
.
├── train.py                       # MUSE fine-tuning loop (full FT by default; --unfreeze / --freeze_arm for selective freezing)
├── train_mpsenet.py               # MP-SENet dereverberation fine-tuning (grad accumulation + NaN guard)
├── train_demucs.py                # Demucs (dns64) dereverberation fine-tuning (L1 + multi-resolution STFT loss)
├── inference.py                   # Wav-folder inference (reads config from ckpt dir)
├── evaluate.py                    # PESQ/STOI/SI-SDR/DNSMOS, supports reverb + noise sets
├── evaluate_mpsenet_testset.py    # Held-out 4,120-mixture test-set evaluation, MP-SENet
├── evaluate_demucs_testset.py     # Same harness for Demucs (identical metrics and schema)
├── config_finetune.json           # Hyperparameters used for the MUSE FT run
├── config_mpsenet_finetune.json   # MP-SENet FT at batch 28 (single large GPU)
├── config_mpsenet_finetune_ga.json# MP-SENet FT, batch 4 x grad_accum 7 = 28 (the run used)
├── config_finetune_demucs.json    # Demucs "parity" FT (lr 1e-4, 50 epochs)
├── config_finetune_demucs_v2.json # Demucs "v2" FT (lr 3e-4, 60 epochs; epoch 57 probed)
├── env.py                         # AttrDict + build_env helpers
├── utils.py                       # load/save_checkpoint, scan_checkpoint
├── models/
│   ├── generator.py               # MUSE = U-Net wrapping TCFTransformer (mask + phase head)
│   ├── discriminator.py           # MetricDiscriminator (PESQ-aligned GAN loss)
│   ├── MUSE_net.py                # Multi-path Enhanced Taylor Transformer
│   ├── mpsenet_generator.py       # MP-SENet generator (MPNet) used by train_mpsenet.py
│   └── mpsenet_discriminator.py   # MP-SENet MetricDiscriminator + batch_pesq
├── datasets/
│   ├── dataset.py                 # VB+DEMAND denoising dataset, mag_pha_stft/istft, file lists
│   ├── reverb_dataset.py          # ReverbDataset + ReverbValDataset (MUSE) and MPSENetReverb{,Val}Dataset
│   └── reverb_waveform_dataset.py # DemucsReverb{,Val}Dataset: raw (clean, reverb) waveform pairs
├── results/                       # Held-out test-set evaluations (see below)
├── tests/test_freeze_arms.py      # Unit test of the selective-freezing rules
├── scripts/
│   ├── prepare_vb_demand.py       # Download VB-DEMAND 16 kHz from HuggingFace
│   ├── prepare_rirs.py            # Compute RT60/DRR/C50/C80 metadata + onset-clip RIRs
│   ├── split_rirs.py              # 80/20 stratified-by-RT60 train/test split
│   ├── generate_test_sets.py      # Build 824×5-RIR reverb test set + noise test set symlinks
│   ├── convert_dns_to_trainer.py  # JacobLinCool/MP-SENet-DNS weights -> trainer key layout (+ round-trip proof)
│   ├── consolidate_test_eval.py   # Per-epoch MP-SENet eval JSONs -> summary + leakage audit
│   ├── consolidate_demucs_test_eval.py  # Same for Demucs, plus test-set identity proof
│   ├── launch_finetune.sh         # SLURM launcher, MUSE (1× A100, ~24 h for 50 epochs)
│   ├── launch_finetune_arms.sh    # SLURM template for the three freeze arms x three seeds
│   ├── launch_finetune_mpsenet_ga.sh  # SLURM launcher, MP-SENet (2× A100 40 GB, grad_accum 7)
│   ├── launch_finetune_demucs.sh      # SLURM launcher, Demucs parity run (1× A100)
│   └── launch_finetune_demucs_v2.sh   # SLURM launcher, Demucs v2 run (1× A100)
├── paper_result/
│   ├── config.json                # Pretrained-baseline config
│   └── g_best                     # Pretrained MUSE generator (denoising baseline, FT starting point)
├── checkpoints/
│   ├── config.json                # Config used for the FT run (== config_finetune.json)
│   └── g_00051852                 # Best fine-tuned dereverb checkpoint (epoch 48, ~2.3 MB)
└── VoiceBank+DEMAND/
    ├── training.txt               # 11,572 utterance IDs (e.g. p226_001|<path>)
    └── test.txt                   # 824 utterance IDs
```

---

## Reproducing the fine-tuning run

### Data layout the launcher expects

```
data/
├── VB_DEMAND_16K/
│   ├── clean_train/   *.wav  (11,572 files, 16 kHz mono)
│   ├── noisy_train/   *.wav  (unused for FT; kept for compat)
│   ├── clean_test/    *.wav  (824)
│   └── noisy_test/    *.wav  (824)
├── rirs_clipped/      *.wav  (1,000 RIRs from RIR-Mega; onset-clipped is fine)
├── rir_metadata.csv          (filename, wav_path, family, rt60, drr, c50, c80)
└── rir_split.json            ({filename: "train"|"test"}, ~798/202)
```

To build these from scratch (RIR-Mega is downloaded by `scripts/download_rirmega.sh`
in the research repo; any local copy of the 1000-RIR `rir_output_small` subset works):

```bash
python scripts/prepare_vb_demand.py
python scripts/prepare_rirs.py --src-dir data/rirmega
python scripts/split_rirs.py
python scripts/generate_test_sets.py --rir-dir data/rirmega
```

`prepare_vb_demand.py` pulls from HuggingFace (`JacobLinCool/VoiceBank-DEMAND-16k`).
There's a known shadowing trick in that script — the project's local
`datasets/` package shadows the HuggingFace `datasets` package, so the script
manipulates `sys.path` to import the right one. Don't "clean it up."

### Launch

```bash
sbatch scripts/launch_finetune.sh
```

Edit the script first if your data paths differ. The launcher requests 1× A100
40 GB and runs `train.py` directly (no DDP). One epoch ≈ 28 min, 50 epochs ≈
23 h. Checkpoints land in `checkpoints/dereverb/{,epoch_N/}{g_,do_}XXXXXXXX`.

### Hyperparameters (`config_finetune.json` + `launch_finetune.sh`)

| Setting | Value | Source |
|---------|-------|--------|
| Starting weights | `paper_result/g_best` | `--pretrained_checkpoint` |
| Trainable params | All 513,015 generator + discriminator params | full FT |
| Optimizer | AdamW(β₁=0.8, β₂=0.99) | config |
| Learning rate | 1e-4, exp decay γ=0.99/epoch | `--lr` overrides config |
| Batch size | 28 | config |
| Epochs | 50 | `--training_epochs` |
| Validation | every 850 steps (PESQ/STOI/DNSMOS on 10-sample subset) | `--validation_interval` |
| Loss | `0.05·L_metric + 0.9·L_mag + 0.3·L_phase + 0.1·L_complex` | `train.py` |
| RIR validation holdout | 5 % of train RIRs (~40) | `--val_rir_ratio` |
| W&B project | `muse-dereverb-ft` | `train.py` |

---

## Evaluation

```bash
python evaluate.py \
  --checkpoint checkpoints/g_00051852 \
  --test_set both \
  --config checkpoints/config.json \
  --output results/dereverb.csv
```

`--test_set` ∈ `{reverb, noise, both}`. Defaults look for
`test_sets/reverb/{wavs,clean_refs}` and `test_sets/noise/{wavs,clean_refs}`
(generated by `scripts/generate_test_sets.py`). DNSMOS runs on GPU via
`onnx2torch` for ~130× speedup over `speechmos`.

---

## MP-SENet dereverberation fine-tuning

`train_mpsenet.py` mirrors `train.py` for MP-SENet, with the model's own loss
(magnitude, anti-wrapping phase, complex, ISTFT→STFT consistency, time-domain
L1 and the metric-discriminator term), and matches the MUSE protocol: the same
11,572 clean VoiceBank utterances convolved on-the-fly with the RIR-Mega
training RIRs, all parameters unfrozen, AdamW(β₁=0.8, β₂=0.99), lr 1e-4,
per-epoch exponential decay γ=0.99, effective batch 28, 50 epochs. The
**epoch-50 checkpoint** is the one probed under reverberation in the paper.

**Base weights.** The starting point is the MP-SENet authors' DNS checkpoint,
`JacobLinCool/MP-SENet-DNS` on HuggingFace (the same weights `se_probe` probes
under additive noise). Its `state_dict` keys differ from this trainer's `MPNet`
only in the container names, so `scripts/convert_dns_to_trainer.py` remaps
them, loads `strict=True`, proves the round-trip back to the HF layout is
lossless, and writes `checkpoints_mpsenet/dns_base_converted.pt`:

| HF `MPSENet` prefix | trainer `MPNet` prefix |
|---|---|
| `dense_encoder.` | `encoder.` |
| `TSTransformer.` | `enhancer.` |
| `mask_decoder.` | `decoder.mask_decoder.` |
| `phase_decoder.` | `decoder.phase_decoder.` |

The inverse remap lives in `se_probe/mpsenet/model.py` (`remap_trainer_keys`),
so a fine-tuned checkpoint reloads into `MPSENet.from_pretrained(...)` for probing.

**STFT config must match DNS**: `n_fft=400, hop_size=100, win_size=400,
beta=2.0, compress_factor=0.3, dense_channel=64, num_tsblocks=4`
(`config_mpsenet_finetune*.json`). Any other `n_fft` shape-mismatches the
per-bin `mask_decoder.lsigmoid.slope` parameter.

**Memory and NaN guard.** On 40 GB A100s MP-SENet fits ~2 segments/GPU under
DDP, so the effective batch of 28 is reached by gradient accumulation:
`config_mpsenet_finetune_ga.json` (batch 4 → 2 per GPU on 2 GPUs) ×
`--grad_accum 7`. The generator accumulates; the discriminator still steps
per micro-batch. An unguarded first run diverged to NaN in epoch 3 because
`atan2(0, 0)` in the phase decoder produces a NaN gradient that
`clip_grad_norm_` then spreads to every weight. Both optimizers therefore skip a
step whose clipped gradient norm is non-finite (`[nan-guard]` log lines); in
the published run this fired for 23 generator steps (0.05–0.09 %) and never
for the discriminator.

```bash
python scripts/convert_dns_to_trainer.py              # writes checkpoints_mpsenet/dns_base_converted.pt
sbatch scripts/launch_finetune_mpsenet_ga.sh           # 2x A100 40 GB; auto-resumes on re-submit
python evaluate_mpsenet_testset.py \
  --checkpoint checkpoints_mpsenet/all_v2/epoch_50/g_00295909 \
  --config checkpoints_mpsenet/all_v2/config.json \
  --output results/mpsenet_ft_test_eval_epoch_50
```

Validation on a 10-utterance subset plateaued around epoch 42 at PESQ
3.17–3.19 (reverberant input 1.52); the test-set numbers below are higher
because the held-out set is slightly easier.

---

## Demucs dereverberation fine-tuning

`train_demucs.py` fine-tunes the pretrained `denoiser` DNS64 Demucs in the
time domain on the same convolutive mixtures, with Demucs's own loss
`L1(enhanced, clean) + MultiResolutionSTFTLoss` (spectral convergence +
log-magnitude over FFT sizes 512/1024/2048, lifted from `denoiser.stft_loss`),
AdamW(β₁=0.8, β₂=0.99) with per-epoch `ExponentialLR(0.99)`, batch 28, all
parameters trainable. `datasets/reverb_waveform_dataset.py` reproduces the MUSE
reverb pair generation (onset-clipped RIRs, `fftconvolve`, energy
normalisation by the reverberant signal) but returns raw waveforms. Checkpoints
are `{'generator': state_dict}` and reload into a fresh `dns64()`.

Two runs were trained; the learning rate and epoch count are the only delta:

| run | config / launcher | lr | epochs | best val PESQ | checkpoint probed |
|---|---|---|---|---|---|
| parity (MUSE-recipe lr) | `config_finetune_demucs.json`, `launch_finetune_demucs.sh` | 1e-4 | 50 | 2.237 @ ep 45 | `epoch_45/g_00037350` |
| v2 (Demucs upstream lr) | `config_finetune_demucs_v2.json`, `launch_finetune_demucs_v2.sh` | 3e-4 | 60 | 2.335 @ ep 57 | `epoch_57/g_00047310` |

The paper reports both; the v2 epoch-57 checkpoint is the Demucs arm of the
reverberation probe.

```bash
sbatch scripts/launch_finetune_demucs_v2.sh
python evaluate_demucs_testset.py \
  --checkpoint checkpoints/dereverb_demucs_v2/epoch_57/g_00047310 \
  --reverb_dir test_sets/reverb --output results/demucs_ft_test_eval_v2_epoch_57
```

---

## Held-out test evaluation

`evaluate_mpsenet_testset.py` and `evaluate_demucs_testset.py` score PESQ
(wide-band), STOI, ESTOI and SI-SDR for both the enhanced output and the
reverberant input on `test_sets/reverb`: **4,120 mixtures = the 824 VoiceBank
test utterances × 5 draws from the 202 held-out RIR-Mega impulse responses**
(`data/rir_split.json`, disjoint from the 798 training RIRs; the leakage audit
is in `results/*_test_eval.json`). The two harnesses share the metric code and
the Demucs consolidation script proves the two evaluations consumed the same
4,120 files (identical input-metric rows). DNSMOS is not reported. The pretrained
"before" rows were measured with the same harness, unmodified, in
`PRETRAINED_REVERB_EVAL` (athena job 128960); a rerun of the fine-tuned MP-SENet
row under that newer software stack reproduces the published value to 8.6e-5
PESQ, so the rows are comparable.

| system | PESQ | STOI | ESTOI | SI-SDR (dB) |
|---|---|---|---|---|
| reverberant input | 1.658 | 0.816 | 0.620 | −10.13 |
| MUSE pretrained (`paper_result/g_best`) | 1.817 | 0.787 | 0.596 | −10.03 |
| MP-SENet pretrained (DNS) | 1.693 | 0.802 | 0.595 | −9.69 |
| Demucs pretrained (DNS64) | 1.729 | 0.800 | 0.624 | −9.23 |
| MP-SENet fine-tuned, epoch 50 | **3.361** | **0.966** | **0.904** | **+9.44** |
| Demucs fine-tuned v2, epoch 57 | 2.441 | 0.918 | 0.799 | +3.77 |
| Demucs fine-tuned parity, epoch 45 | 2.339 | 0.914 | 0.790 | +3.34 |

MP-SENet epochs 48/49/50 lie within 0.014 PESQ of each other
(`results/mpsenet_ft_test_eval_summary.csv`). None of the pretrained
denoising checkpoints dereverberates: they gain at most +0.16 PESQ over the
input, lose STOI, and stay within 1 dB of the input's SI-SDR. The MUSE
fine-tune (`checkpoints/g_00051852`) was evaluated on a different held-out set
of the same size (824 utterances × 5 RIRs, mean C50 12.0 dB; PESQ 3.02 / STOI
0.944 vs 2.17 / 0.834 pretrained, see "Achieved metrics" above), so absolute
values are not comparable across the two evaluations.

Files under `results/`:

| file | contents |
|---|---|
| `mpsenet_ft_test_eval_epoch_{48,49,50}.{csv,json}` | per-mixture metrics and summary, MP-SENet |
| `mpsenet_ft_test_eval{.json,_summary.csv}` | consolidated MP-SENet results + leakage audit |
| `demucs_ft_test_eval_{parity_epoch_45,v2_epoch_57}.{csv,json}` | per-mixture metrics and summary, Demucs |
| `demucs_ft_test_eval{.json,_summary.csv}` | consolidated Demucs results + test-set identity proof |
| `pretrained_reverb_eval.csv` | the pretrained "before" rows and the control rerun, at full precision |

Checkpoint paths inside the JSONs are written as `<Muse-Reverb-FN>/...`, i.e.
relative to the research repository they were produced in.

---

## Selective freezing and the profile-guided arms

`train.py` keeps its default behaviour (every generator parameter trainable)
but exposes the freezing options used in two experiments:

* `--unfreeze {all|encoder_l1|encoder_l2|latent|decoder_l2|decoder_l1|refinement}`
  trains one MUSE stage (its transformer block plus its patch-embedding,
  down/up-sampling and output heads, see `BLOCK_TO_PREFIXES`), and
  `--unfreeze <block>.<layer>` (layer 0–3) a single transformer layer.
  `--unfreeze_boundary` additionally trains `dense_encoder`, `mask_decoder` and
  `phase_decoder`; `--freeze_discriminator` freezes the MetricDiscriminator.
  This is the single-block freeze study whose `all` arm is
  `checkpoints/g_00051852`.
* `--freeze_arm {full_ft,freeze_encoder,freeze_decoder}` reproduces the
  profile-guided arms: `freeze_encoder` freezes every parameter whose name
  contains `dense_encoder` or `TCFTransformer.encoder_level` (76 % of the
  generator stays trainable); `freeze_decoder` freezes
  `TCFTransformer.decoder_level`, `mask_decoder` and `phase_decoder` (71 %
  trainable; `mag_refinement` and `latent` are frozen by neither arm).
  `--seed` overrides the config seed; the arms were run with seeds 1234, 2345
  and 3456 (`scripts/launch_finetune_arms.sh`).

Only `requires_grad` parameters reach the optimizers. The freezing rules are
pinned by `tests/test_freeze_arms.py`.

Result of the three-arm experiment (50 epochs, lr 1e-4, batch 28, RIR-Mega
798-RIR training split, best-epoch PESQ/STOI on the held-out RIR-Mega validation
subset, mean ± sd over the three seeds):

| arm | PESQ | STOI |
|---|---|---|
| `full_ft` | 2.783 ± 0.011 | 0.9333 ± 0.0012 |
| `freeze_encoder` | 2.697 ± 0.004 | 0.9300 ± 0.0010 |
| `freeze_decoder` | 2.700 ± 0.003 | 0.9293 ± 0.0006 |

Freezing either the profile-robust or the profile-sensitive stages costs the
same 0.08–0.09 PESQ relative to full fine-tuning (freeze_encoder minus
freeze_decoder: −0.002 PESQ, 95 % CI [−0.007, +0.002]); the profile does not
localise where adaptation happens. Re-probing all ten checkpoints (pre-FT and
3 arms × 3 seeds) under reverberation shows the frozen stages holding their
pre-FT normalised AUC while the other stages move. The nine arm checkpoints are
on HuggingFace (`yairamr/SE-Probe-models/cluster-2026-07/<arm>__seed<S>/g_00040000`).

---

## Probing the fine-tuned checkpoints

The checkpoints written here load directly into the `se_probe` activation
extractors used for the reverberation axis of the paper:

```python
from se_probe.activation_extraction import (
    load_muse_activation_extractor_reverb,
    load_mpsenet_activation_extractor_reverb,
    load_demucs_activation_extractor_reverb,
)
muse = load_muse_activation_extractor_reverb(checkpoint_path="training/checkpoints/g_00051852")
mpsenet = load_mpsenet_activation_extractor_reverb(
    checkpoint_path="checkpoints_mpsenet/all_v2/epoch_50/g_00295909")   # trainer keys are remapped on load
demucs = load_demucs_activation_extractor_reverb(
    checkpoint_path="checkpoints/dereverb_demucs_v2/epoch_57/g_00047310")
```

Pretrained arms (no `checkpoint_path`) give the "before fine-tuning" probes.

---

## Known caveats / gotchas

1. **Inherited-bug note.** The original MUSE / `Muse-Reverb-FN` `train.py`
   contained `noisy_audio = clean_audio.to(...)` (clean tensor reassigned to
   the `noisy_audio` variable). It was harmless because the generator is fed
   `noisy_mag/noisy_pha` directly from the dataset, and `noisy_audio` is
   unused downstream. **This repo's `train.py` was rewritten and no longer
   contains that line** — but if you ever sync changes back from
   `Muse-Reverb-FN`, watch out for it.

2. **Sample rate is hard-coded at 16 kHz** in `config_*.json`. The model has
   no resampling layer; feeding 8 / 22.05 / 48 kHz audio will silently
   produce garbage.

3. **`segment_size = 30700` is not a multiple of `hop_size = 100`**, but the
   STFT layer pads internally so it works. If you change `n_fft`, you'll need
   to retrain — the architecture's freq-axis dimensions are baked in at
   construction.

4. **`datasets/` package shadows HuggingFace `datasets`.** Any script that
   needs both has to do the `sys.path` dance shown in
   `scripts/prepare_vb_demand.py`.

5. **The base MUSE was trained on noise, not reverb.** If the input wav has
   strong stationary noise *and* reverb, this checkpoint will dereverberate
   well but won't denoise — it's been adapted away from that. Run the noise
   model first, then this one, if you need both.

6. **The discriminator (`do_*` files) is not in this repo.** Only the generator
   weights (`g_00051852`) ship here, which is all you need for inference. To
   resume training you'd need the matching `do_00051852` from
   `Muse-Reverb-FN/checkpoints/all/epoch_48/`.

---

## Citation

```
@inproceedings{lin2024muse,
  title={MUSE: Flexible Voiceprint Receptive Fields and Multi-Path Fusion Enhanced Taylor Transformer for U-Net-based Speech Enhancement},
  author={Lin, Zizhen and Chen, Xiaoting and Wang, Junyu},
  booktitle={Interspeech 2024},
  year={2024}
}
```

Builds on [MP-SENet](https://github.com/yxlu-0102/MP-SENet) and
[MB-TaylorFormer](https://github.com/FVL2020/ICCV-2023-MB-TaylorFormer).
