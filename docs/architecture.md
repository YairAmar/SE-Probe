# SE-Probe architecture

SE-Probe is a small library plus three frozen-model adapters. The library is plain PyTorch + NumPy + pandas; nothing is locked to a specific GPU or to the cluster the original research ran on. The pipeline is: **load model → register hooks → push (clean, degraded) pairs through it → compute CKA / diffusion distances on the activations → regress against degradation severity → plot**.

## Top-level modules — `se_probe/`

### `cka.py`
Linear CKA between two batches of activations. The implementation centers features (columns), forms the per-batch cross-covariance `XᵀY` and self-covariances `XᵀX`, `YᵀY`, and combines them into `‖XᵀY‖_F² / (‖XᵀX‖_F · ‖YᵀY‖_F)` averaged across the batch. Inputs may be raw `(B, T, F, H)` or already time-averaged `(B, F, H)`; the function handles both. This is the only similarity metric used in notebooks 02–04.

### `activation_extraction.py`
Forward-hook layer used to harvest intermediate activations from each frozen SE model. Exports `ActivationsExtractor`, `get_activations()`, `extract_activations_on_audios()`, plus model-specific loaders: `load_muse_activation_extractor`, `load_mpsenet_activation_extractor`, `load_demucs_activation_extractor`, and reverb-tuned variants. Loaders return a model wrapped with hooks already registered on every `target_layer`. All loaders accept a `device` argument; pass `get_device()` to stay device-agnostic.

### `consts.py`
Probe-layer constants per architecture, default SNR ladders, sample rate (`16000`), reverb constants (target C50 levels, AIR room splits, RIR counts per utterance), and the dataset path resolvers backed by `SEPROBE_VCTK_DIR`, `SEPROBE_DEMAND_DIR`, `SEPROBE_AIR_RIR_DIR`. Calling `set_paths(...)` programmatically overrides the env vars — tests use this against the smoke fixture.

### `data_generation.py`
Audio degradation utilities. `load_demand_noise()` lazily reads and caches DEMAND noise files. `add_noise_at_snr()` mixes a clean utterance with a noise track at a target SNR. `convolve_audio()` applies a room impulse response and `compute_ratio_for_target_c50()` rescales early-vs-late RIR energy to hit a desired C50 — together they generate the controlled-reverb conditions used in notebook 06.

### `diffusion_maps.py`
PyTorch implementation of diffusion maps. `diffusion_map_torch()` builds a kernel matrix, applies the alpha-normalisation, and returns the leading eigenvectors / eigenvalues either via a full eigendecomposition or LOBPCG for large `N`. Supports `cutoff` (cumulative-energy stopping criterion) or fixed `k`. Honours `get_device()` for CUDA/MPS acceleration on dense kernels.

### `diffusion_analysis.py`
Post-processing for diffusion-map embeddings: per-layer distances between SNR conditions, Spearman correlations between diffusion distance and SNR ordering, and the layer-grouping constants (`REPRESENTATIVE_LAYERS`, `BLOCK_NAMES`, `BLOCK_ORDER`) that align the per-layer plots with the encoder/latent/decoder/refinement structure of MUSE.

### `metrics.py`
Audio quality and acoustic metrics. CPU helpers (`c50`, `drr`, `sisdr`, `compute_audio_metrics`) wrap PESQ, STOI, SI-SDR, and RIR-derived quantities. GPU-accelerated evaluators (`gpu_sisdr`, `GPUSTOIEvaluator`, `GPUDNSMOSEvaluator`, `GPUMetricsEvaluator`) batch many utterances at once for `results_df/` regeneration. Notebook 04 uses `compute_audio_metrics` to align CKA with PESQ/STOI.

### `io.py`
Disk helpers for the source corpora: `load_clean_wavs()` walks `$SEPROBE_VCTK_DIR` and downsamples test-speaker utterances to 16 kHz; `load_air_rirs()` / `load_air_test_rirs()` read AIR `.mat` files, align them from the first peak, and resample 48 → 16 kHz. Only used by scripts that recompute from raw audio; the demo notebooks bypass it.

### `plotting.py`
`apply_paper_rcparams()` sets matplotlib to the poster style (serif fonts, CM mathtext, 10 / 12 / 14-pt size hierarchy, 300 dpi savefig, type-42 PDF fonts so figures embed in LaTeX). `MODEL_COLORS` and `MODEL_LABELS` give a single source of truth for the MUSE / MP-SENet / Demucs colour palette across all six notebooks.

### `device.py`
`get_device(prefer=None)` autodetects CUDA → MPS → CPU and sets `PYTORCH_ENABLE_MPS_FALLBACK=1` automatically when MPS is selected, so unsupported ops fall back to CPU silently. `device_info(device)` returns a one-line human-readable summary used in notebook bootstraps. This module is the only place CUDA/MPS strings should appear; everything downstream takes a `torch.device` argument.

## Model adapters — `se_probe/{muse,mpsenet,demucs}/`

Each subpackage vendors the upstream model definition and a thin loader compatible with `activation_extraction.py`. Weights are not redistributed — `scripts/setup.py` clones the upstream MUSE repo to retrieve `g_best`, and the reverb fine-tuned MUSE checkpoint is fetched from the SE-Probe HuggingFace model repo. Adapters expose the layer name list that the probe-layer constants in `consts.py` reference.

## Data flow

```
raw audio  ─┐
            ├─►  data_generation.add_noise_at_snr  ─►  (clean_wav, degraded_wav)
RIR  ───────┘                                                │
                                                             ▼
                                          activation_extraction.get_activations
                                                             │
                                                  per-layer activations
                                                             │
                                                       cka.cka(...)
                                                             │
                                                       results_df/*.parquet
                                                             │
                                       ┌─────────────────────┼─────────────────────┐
                                       ▼                     ▼                     ▼
                              regress vs SNR/C50    diffusion_analysis    correlate with PESQ
                                  (notebook 02)        (notebook 05)        (notebook 04)
```

The expensive step is activation extraction. Once `results_df/` (or `results_demo/`) is on disk, every analysis notebook is a few-second pandas / matplotlib operation.

## Building the book locally

The notebooks are also published as a Jupyter Book at <https://yairamar.github.io/SE-Probe/>. To rebuild it locally:

```bash
pip install -e .[docs]
jupyter-book build .
```

Output lands in `_build/html/index.html`. The first build executes every notebook end-to-end against `results_demo/` and caches the results under `_build/.jupyter_cache/`; subsequent builds reuse the cache and finish in well under a minute. The CI deploy workflow at `.github/workflows/deploy-book.yml` runs the same command on every push to `main` and publishes `_build/html` to the `gh-pages` branch via `peaceiris/actions-gh-pages`. `SE_PROBE_RUN_INFERENCE` is intentionally left unset during the build, so notebooks 01 and 06 render their figure-from-parquet branches and skip the model-inference cells.
