# MUSE — Fine-Tuning for Dereverberation

End-to-end fine-tuning of [MUSE](https://arxiv.org/pdf/2406.04589) (Lin et al.,
Interspeech 2024) for **dereverberation**. The base MUSE was trained on
VoiceBank+DEMAND for denoising; this repo adapts it to reverberant speech by
fine-tuning every parameter of the generator on clean utterances convolved
on-the-fly with real RIRs.

---

## Where this lives

| | |
|---|---|
| GitHub | https://github.com/YairAmar/muse-dereverb-ft (private) |
| Local clone (Athena/DGX) | `/rg/iscohen_prj/yairamr/code/muse-dereverb-ft` |
| Source repo it was carved out of | `/rg/iscohen_prj/yairamr/code/Muse-Reverb-FN` (https://github.com/YairAmar/Muse-Reverb-FN, branch `mpsenet-ft`) |

The fine-tuned checkpoint `checkpoints/g_00051852` is the same file as
`Muse-Reverb-FN/checkpoints/all/epoch_48/g_00051852` (the "all-block-unfrozen"
winner of an earlier selective-FT experiment, now repackaged as plain full FT).

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

This repo runs in the project's standard conda env on Athena and DGX:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate meta-interface-py310           # Python 3.10, torch 1.12.1+cu113
```

The env is identical on both clusters, so code runs unchanged. CUDA-12 stacks
will not work on the DGX (driver 470 caps at CUDA 11.4). See
`~/.claude/CLAUDE.md` for the cluster comparison.

`requirements.txt` lists the original MUSE pins, but the project env is what's
actually used. Extra packages used by `train.py` / `evaluate.py`:
`wandb`, `pesq`, `pystoi`, `speechmos`, `matplotlib`, `onnx2torch` (for
GPU-accelerated DNSMOS in `evaluate.py`).

---

## Repo layout

```
.
├── train.py                       # Fine-tuning loop (full FT, no selective freezing)
├── inference.py                   # Wav-folder inference (reads config from ckpt dir)
├── evaluate.py                    # PESQ/STOI/SI-SDR/DNSMOS, supports reverb + noise sets
├── config_finetune.json           # Hyperparameters used for the FT run
├── env.py                         # AttrDict + build_env helpers
├── utils.py                       # load/save_checkpoint, scan_checkpoint
├── models/
│   ├── generator.py               # MUSE = U-Net wrapping TCFTransformer (mask + phase head)
│   ├── discriminator.py           # MetricDiscriminator (PESQ-aligned GAN loss)
│   └── MUSE_net.py                # Multi-path Enhanced Taylor Transformer
├── datasets/
│   ├── dataset.py                 # VB+DEMAND denoising dataset, mag_pha_stft/istft, file lists
│   └── reverb_dataset.py          # ReverbDataset + ReverbValDataset (on-the-fly RIR convolution)
├── scripts/
│   ├── prepare_vb_demand.py       # Download VB-DEMAND 16 kHz from HuggingFace
│   ├── prepare_rirs.py            # Compute RT60/DRR/C50/C80 metadata + onset-clip RIRs
│   ├── split_rirs.py              # 80/20 stratified-by-RT60 train/test split
│   ├── generate_test_sets.py      # Build 824×5-RIR reverb test set + noise test set symlinks
│   └── launch_finetune.sh         # SLURM launcher (1× A100, ~24 h for 50 epochs)
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

These files already exist in `/rg/iscohen_prj/yairamr/code/Muse-Reverb-FN/data/`
and `/home/yairamr/work/data/rirs/rirmega/`. To rebuild from scratch:

```bash
python scripts/prepare_vb_demand.py
python scripts/prepare_rirs.py --src-dir /home/yairamr/work/data/rirs/rirmega
python scripts/split_rirs.py
python scripts/generate_test_sets.py --rir-dir /home/yairamr/work/data/rirs/rirmega
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
| Batch size | 8 (~20 GB peak on A100 40 GB) | config |
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
