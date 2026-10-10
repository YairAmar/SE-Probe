# SE-Probe architecture

SE-Probe is a small library plus three frozen-model adapters. The library is plain PyTorch + NumPy + pandas; nothing is locked to a specific GPU or to the cluster the original research ran on. The pipeline is: **load model → register hooks → push (clean, degraded) pairs through it → compute CKA / diffusion distances on the activations → regress against degradation severity → plot**.

## Top-level modules, `se_probe/`

### `cka.py`
Linear CKA between two batches of activations. Also `linear_cka_unbiased()` (the unbiased HSIC estimator of Song et al.), `representations_from_activation()` (pooled `[C, F]`, frames `[T, C*F]`, Demucs-style `[T, C]`) and `cka_variants()` for the estimator / sample-convention ablation. The implementation centers features (columns), forms the per-batch cross-covariance `XᵀY` and self-covariances `XᵀX`, `YᵀY`, and combines them into `‖XᵀY‖_F² / (‖XᵀX‖_F · ‖YᵀY‖_F)` averaged across the batch. Inputs may be raw `(B, T, F, H)` or already time-averaged `(B, F, H)`; the function handles both. It is the similarity metric behind every table the analysis notebooks read.

### `activation_extraction.py`
Forward-hook layer used to harvest intermediate activations from each frozen SE model. Exports `ActivationsExtractor`, `get_activations()`, `extract_activations_on_audios()`, plus model-specific loaders: `load_muse_activation_extractor`, `load_mpsenet_activation_extractor`, `load_demucs_activation_extractor`, and reverb-tuned variants. Loaders return a model wrapped with hooks already registered on every `target_layer`. All loaders accept a `device` argument; pass `get_device()` to stay device-agnostic.

### `consts.py`
Probe-layer constants per architecture, default SNR ladders, sample rate (`16000`), reverb constants (target C50 levels, AIR room splits, RIR counts per utterance), and the dataset path resolvers backed by `SEPROBE_VCTK_DIR`, `SEPROBE_DEMAND_DIR`, `SEPROBE_AIR_RIR_DIR`. Calling `set_paths(...)` programmatically overrides the env vars, tests use this against the smoke fixture.

### `data_generation.py`
Audio degradation utilities. `load_demand_noise()` lazily reads and caches DEMAND noise files. `add_noise_at_snr()` mixes a clean utterance with a noise track at a target SNR. `convolve_audio()` applies a room impulse response and `compute_ratio_for_target_c50()` rescales early-vs-late RIR energy to hit a desired C50, together they generate the controlled-reverb conditions behind the reverberation chapter and `scripts/run_reverb_grid.py`.

### `diffusion_maps.py`
PyTorch implementation of diffusion maps. `diffusion_map_torch()` builds a kernel matrix, applies the alpha-normalisation, and returns the leading eigenvectors / eigenvalues either via a full eigendecomposition or LOBPCG for large `N`. Supports `cutoff` (cumulative-energy stopping criterion) or fixed `k`. Honours `get_device()` for CUDA/MPS acceleration on dense kernels.

### `diffusion_analysis.py`
Post-processing for diffusion-map embeddings, including the 824-utterance analyses of the paper (`embed_block()` joint per-block embedding at `t = 0.5`, `block_trajectory_from_psi()`, `block_statistics()`, `arc_length_ratio()`, `architecture_distance_matrices()` at `t = 5` and `group_distances()`): per-layer distances between SNR conditions, Spearman correlations between diffusion distance and SNR ordering, and the layer-grouping constants (`REPRESENTATIVE_LAYERS`, `BLOCK_NAMES`, `BLOCK_ORDER`) that align the per-layer plots with the encoder/latent/decoder/refinement structure of MUSE.

### `metrics.py`
Audio quality and acoustic metrics. CPU helpers (`c50`, `drr`, `sisdr`, `compute_audio_metrics`) wrap PESQ, STOI, SI-SDR, and RIR-derived quantities. GPU-accelerated evaluators (`gpu_sisdr`, `GPUSTOIEvaluator`, `GPUDNSMOSEvaluator`, `GPUMetricsEvaluator`) batch many utterances at once for `results_df/` regeneration. `compute_audio_metrics` produced the PESQ/STOI columns that the perceptual-quality chapter aligns with CKA.

### `io.py`
Disk helpers for the source corpora: `load_clean_wavs()` walks `$SEPROBE_VCTK_DIR` and downsamples test-speaker utterances to 16 kHz; `load_air_rirs()` / `load_air_test_rirs()` read AIR `.mat` files, align them from the first peak, and resample 48 → 16 kHz. Only used by scripts that recompute from raw audio; the demo notebooks bypass it.

### `layers.py`
The probed-layer sets of the paper (24 MUSE `.norm1` hooks, 8 MP-SENet `.norm1` hooks, 11 Demucs block outputs) and their **architectural depth order**: `probed_layers(model, all_layers)`, `depth_order(model, layers)`, `short_label(s)`, `layout(model)` for depth-axis figures. Sorting layer names alphabetically is not depth order (it scrambles Demucs and MUSE), which is why this lives in one place. Also the SNR and C50 grids, the DEMAND split lists (VoiceBank-DEMAND test five, training eight, neither five), the 88-RIR AIR pool histogram and the speaker index ranges of the two utterance sets.

### `profiles.py`
The paper's per-layer analysis: `mean_level_curves()` (utterance-first averaging over noises/RIRs), `fit_curves()` / `fit_profile()` (closed-form OLS `alpha + beta * s`, `R^2`, empirical endpoints `c_low` / `c_high`, normalised trapezoidal `auc`), `build_cube()` + `utterance_bootstrap()` (percentile intervals over whole utterances) and `two_way_cluster_bootstrap()` (utterance x noise-type / RIR, drawing as the published cluster script did; with `stream_layers` = every hooked layer it replays the published stream at `seed=0`). `tradeoff_summary()` returns a Table I row: `r(alpha, beta)`, Spearman, the saturation spread `f = sd(c_high)/sd(alpha)`, `r_c`, the identity-predicted `r`, the floor `sqrt(1 - f^2)`, PC1/PC2 of the layer x level matrix and the depth trends. Plus `profile_agreement()` / `layer_resample_ci()` (cross-task), `noise_set_independence()`, `random_init_summary()`, `emergence_convergence()`, `finetune_shift()` and `bottleneck_gap()`. The tests reproduce Table I from `results_tables/` to 1e-9.

### `bootstrap.py`
NumPy-only hierarchical (cluster) bootstrap. `hierarchical_bootstrap(statistic, levels, unit=...)` resamples the named sampling unit, carries nested levels whole, holds coarser levels fixed (warning with `PseudoReplicationWarning`), supports strata and crossed margins, and returns a `BootstrapResult` that states its estimand. `t_interval()` is the small-sample speaker-level estimator. The unit has no default on purpose.

### `perceptual.py`
Quality association: `add_improvements()`, `center_within()`, `per_layer_correlations()` (Pearson/Spearman per layer, optional utterance-cluster CI via `cluster_bootstrap_pearson_ci()` on sufficient sums), `speaker_level_analysis()` (utterance-level, per-speaker and speaker-level `t` tables), `sisdr_sign_convention()` and `ceiling_free_gain()`.

### `centroids.py`
Loading, decoding and averaging the per-(layer, noise, SNR) activation centroids that feed the diffusion maps, in both on-disk schemas.

### `plotting.py`
`apply_paper_rcparams()` sets matplotlib to the poster style (serif fonts, CM mathtext, 10 / 12 / 14-pt size hierarchy, 300 dpi savefig, type-42 PDF fonts so figures embed in LaTeX). `MODEL_COLORS` and `MODEL_LABELS` give a single source of truth for the MUSE / MP-SENet / Demucs colour palette across the notebooks.

### `device.py`
`get_device(prefer=None)` autodetects CUDA → MPS → CPU and sets `PYTORCH_ENABLE_MPS_FALLBACK=1` automatically when MPS is selected, so unsupported ops fall back to CPU silently. `device_info(device)` returns a one-line human-readable summary used in notebook bootstraps. This module is the only place CUDA/MPS strings should appear; everything downstream takes a `torch.device` argument.

## Model adapters, `se_probe/{muse,mpsenet,demucs}/`

Each subpackage vendors the upstream model definition and a thin loader compatible with `activation_extraction.py`. All three loaders accept a `checkpoint_path` for the dereverberation fine-tunes produced by `training/` (MP-SENet trainer key names are remapped by `remap_trainer_keys`), each has a `*_reverb` extractor variant, and `se_probe.muse.model.load_random_init_muse_model(seed)` builds the untrained control from the same configuration. Weights are not redistributed, `scripts/setup.py` clones the upstream MUSE repo to retrieve `g_best`, and the reverb fine-tuned MUSE checkpoint is fetched from the SE-Probe HuggingFace model repo. Adapters expose the layer name list that the probe-layer constants in `consts.py` reference.

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
                                  (chapter 02)         (chapter 07)         (chapter 06)
```

The expensive step is activation extraction. Once `results_df/` (or `results_demo/`) is on disk, every analysis notebook is a few-second pandas / matplotlib operation. The drivers that produced the 824-utterance sweeps of the TASLP paper live under `scripts/` (`run_*_grid.py`) and their reductions under `scripts/analyze_*.py`; the reduced tables are shipped in `results_tables/`, so the chapters run without the raw parquets.

## Building the book locally

The notebooks are also published as a Jupyter Book at <https://yairamar.github.io/SE-Probe/>. To rebuild it locally:

```bash
pip install -e .[docs]
jupyter-book build .
```

Output lands in `_build/html/index.html`. The first build executes every notebook end-to-end against `results_demo/` and caches the results under `_build/.jupyter_cache/`; subsequent builds reuse the cache and finish in well under a minute. The CI deploy workflow at `.github/workflows/deploy-book.yml` runs the same command on every push to `main` and publishes `_build/html` to the `gh-pages` branch via `peaceiris/actions-gh-pages`. `SE_PROBE_RUN_INFERENCE` is intentionally left unset during the build, so chapters 01, 04, 05 and 08 render their figure-from-table branches and skip the model-inference cells.
