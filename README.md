# SE-Probe

> Probing internal representations of deep speech enhancement models under controlled noise and reverberation.

SE-Probe is the public companion code for *"Where Does Speech Enhancement Adapt? Probing Study Under Controlled Degradation"* (Amar, Ivry, Cohen, 2026). It computes layer-wise CKA, diffusion maps, and PESQ/STOI correlations across MUSE, MP-SENet, and Demucs, and ships six notebooks that reproduce the poster figures from a 50 MB demo subset.

## TL;DR

We treat SE networks as black boxes and ask **which layers actually adapt to a given degradation**. The pipeline:

1. Run clean and noisy/reverberant utterances through a frozen SE model.
2. Extract activations layer-by-layer.
3. Compare clean vs degraded representations with linear CKA.
4. Regress CKA against degradation severity (SNR / C50) to find the layers most sensitive to each condition.
5. Cross-check with diffusion-map distances and downstream PESQ/STOI.

The shipped demo runs the analysis end-to-end on precomputed CKA tables (`results_demo/`); two notebooks (01, 06) optionally re-run model inference if hardware is available.

## Quick start

```bash
git clone https://github.com/YairAmar/SE-Probe.git && cd SE-Probe
pip install -e .
python scripts/setup.py
```

`scripts/setup.py` fetches the MUSE pretrained weights from the upstream repo, downloads the reverb fine-tuned checkpoint placeholder from HuggingFace, and probes the local device. Add `--full-data` to also pull the full 3.1 GB precomputed CKA tables.

Then:

```bash
jupyter lab notebooks/
```

## Hardware

The notebooks autodetect the best available device through `se_probe.device.get_device()` (CUDA → MPS → CPU). The same notebook code runs unchanged on all three.

| Backend | Status | Notes |
|---|---|---|
| **NVIDIA CUDA** | Recommended | Original research used A100s. Required for fast inference in notebooks 01 and 06. |
| **Apple Silicon (MPS)** | Supported | Demo target — MacBook Pro M-series. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically by `se_probe.device` so unsupported ops fall back to CPU. |
| **CPU only** | Works | All analysis notebooks (02–05) run quickly. Inference cells in 01/06 are slow but functional. |

Notebooks that read precomputed parquets (02, 03, 04, 05, and the figure cells of 06) are device-agnostic — any laptop runs them in seconds.

## Data setup

Demo data ships in `results_demo/` (≤50 MB; subset of 5 SNRs, 1 noise type, 10 utterances per cell). Notebooks fall back to it automatically when `results_df/` is absent.

For the full reproduction, run `python scripts/setup.py --full-data` to populate `results_df/` from HuggingFace (`HF_REPO_DATA`, default `YairAmar/SE-Probe-data`).

To recompute from raw audio you also need the source corpora; point the following env vars at local copies:

| Variable | Dataset |
|---|---|
| `SEPROBE_VCTK_DIR` | VCTK-Corpus (clean speech) |
| `SEPROBE_DEMAND_DIR` | DEMAND noise database |
| `SEPROBE_AIR_RIR_DIR` | AIR-10 room impulse responses |

See `data/README.md` for download URLs and expected layouts.

## Notebooks

Run them in order; each starts with a markdown cell stating what's shown, what to look for, and a runtime estimate.

| # | Notebook | What it shows | Runtime (CUDA / MPS / CPU) |
|---|---|---|---|
| 01 | `01_pipeline_overview.ipynb` | One utterance through the full pipeline: clean vs noisy CKA on a single layer. | ~20 s / ~60 s / ~3 min |
| 02 | `02_cka_per_layer.ipynb` | Per-layer CKA heatmap and SNR-sensitivity profile across MUSE / MP-SENet / Demucs. | <30 s on any device |
| 03 | `03_cross_architectures.ipynb` | Slope-vs-intercept scatter (poster Fig. 5) comparing the three architectures. | <30 s on any device |
| 04 | `04_cka_to_pesq.ipynb` | Within-SNR-group CKA-PESQ correlation (residualised). | <30 s on any device |
| 05 | `05_diffusion_maps.ipynb` | Per-layer diffusion-distance + Spearman SNR ordering. | ~1 min on any device |
| 06 | `06_reverb_probing.ipynb` | C50-sensitivity figure for the reverb fine-tuned MUSE; optional inference cells gated by `SE_PROBE_RUN_INFERENCE=1`. | Figure: <30 s. Inference: ~1 min CUDA / ~5 min MPS. |

## Reproducing the full results

```bash
python scripts/setup.py --full-data         # pulls results_df/ from HuggingFace
SE_PROBE_RUN_INFERENCE=1 jupyter lab notebooks/
```

To regenerate everything from raw audio, set the dataset env vars listed above and run the scripts under `scripts/` in the order documented in `docs/architecture.md`.

## Package layout

`se_probe/` contains the analysis library: `cka.py` (linear CKA on centered kernels), `activation_extraction.py` (forward hooks on each model), `diffusion_maps.py` and `diffusion_analysis.py` (manifold distances), `metrics.py` (PESQ/STOI/SI-SDR), `data_generation.py` (noise/reverb mixing), `consts.py` (probe-layer indices and dataset paths), `plotting.py` (paper rcParams + colour palette), and `device.py` (CUDA/MPS/CPU autodetect). Model adapters live under `se_probe/muse/`, `se_probe/mpsenet/`, and `se_probe/demucs/`. See `docs/architecture.md` for a per-module tour.

## Citation

```bibtex
@article{amar2026seprobe,
  title   = {Where Does Speech Enhancement Adapt? Probing Study Under Controlled Degradation},
  author  = {Amar, Yair and Ivry, Amir and Cohen, Israel},
  year    = {2026},
  journal = {arXiv preprint}
}
```

## License

MIT — see `LICENSE`.

## Contact

Yair Amar · open an issue at https://github.com/YairAmar/SE-Probe/issues
