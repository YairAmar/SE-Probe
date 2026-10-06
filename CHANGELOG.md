# Changelog

All notable changes to SE-Probe are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to semantic versioning.

## v0.2.0, TASLP revision (2026-10-06)

Everything the closed research repositories implemented for the journal version of the paper, *Probing Layer-Wise Robustness and Sensitivity of Speech Enhancement Models Under Noise and Reverberation* (IEEE/ACM TASLP), ported into the public package. Nothing from v0.1.x was removed.

### Added
- `se_probe.layers`: the probed-layer sets (24 / 8 / 11), the architectural depth order (never alphabetical), short labels, figure layouts, the SNR and C50 grids, the DEMAND split lists (VoiceBank-DEMAND test five with SCAFE, training eight, neither five) and the 88-RIR AIR pool description.
- `se_probe.profiles`: mean degradation-level curves, closed-form per-layer OLS with endpoint and AUC statistics, utterance-cluster and two-way utterance x noise/RIR bootstraps (the latter replays the stream of the cluster script behind the published CIs), the saturation spread `f`, the identity `r(alpha,beta) = (r_c f - 1)/sqrt(f^2 - 2 r_c f + 1)` and its floor, PC1/PC2 shares, Table I summaries, profile agreement with layer-resample CIs, noise-set independence, random-init and emergence summaries, before/after fine-tuning shifts and the Demucs bottleneck gap.
- `se_probe.bootstrap`: a NumPy-only hierarchical (cluster) bootstrap with an explicit, mandatory sampling unit, strata, crossed margins and `PseudoReplicationWarning`; reproduces the paper's 0.041 vs 0.505 interval-width comparison.
- `se_probe.perceptual`: within-group centring, per-layer correlations for PESQ / STOI / SI-SDR / DNSMOS, utterance-cluster bootstrap on Pearson r via sufficient sums, per-speaker and speaker-level (Student-t) inference, SI-SDR sign-convention detection and the ceiling-free PESQ gain.
- `se_probe.cka`: the unbiased HSIC estimator and the frames-as-samples conventions of the estimator ablation.
- `se_probe.centroids` and the 824-scale diffusion helpers in `se_probe.diffusion_analysis` (joint per-block embedding, arc lengths, between/within group distances).
- Model adapters: `load_mpsenet_model` / `load_demucs_model` accept fine-tuned checkpoints (trainer key names are remapped for MP-SENet), `load_demucs_activation_extractor_reverb`, and `load_random_init_muse_model` / `load_random_init_muse_activation_extractor` for the control.
- `se_probe.plotting`: column-width rcParams, Okabe-Ito palette, the two-panel profile figure, the tradeoff scatter and depth-axis helpers.
- `results_tables/`: 86 small tables with provenance and SHA-256 manifest, including the six-arm C50 mean curves that reproduce the C50 rows of Table I.
- Ten new book chapters (07–16) that recompute every manuscript number from those tables; `_toc.yml` now has two parts.
- `scripts/`: resumable, cluster-agnostic drivers for the noise grid (824 x 41 x 18), the reverberation grid (824 x 13 x 5-of-88 RIRs, six arms), the random-init and emergence grids, the perceptual-metric grid and join, diffusion centroids and psi tables, the HSIC ablation, the analysis CLIs behind every table, and a dry-run HuggingFace upload manifest.
- `training/`: MP-SENet and Demucs dereverberation fine-tuning (trainers, datasets, models, configs, launch scripts, DNS key converter), the held-out evaluation harnesses with their results (4,120 mixtures, 202 held-out RIR-Mega RIRs) and the pretrained "before" rows, and selective freezing / the three profile-guided arms in `train.py` (`--unfreeze`, `--freeze_arm`, `--seed`).
- Tests for all of the above (`tests/test_{layers,profiles,perceptual,cka_variants,diffusion_824,model_loaders,metrics_sign,bootstrap}.py`, `training/tests/test_freeze_arms.py`); CI also runs the training tests.

### Fixed
- `se_probe.metrics.sisdr` returned the negated SI-SDR; the sign is corrected and pinned by a test. The shipped demo and result tables already carried the correct sign.
- `torchvision` (required by the vendored MUSE deformable convolution) is now a declared dependency; `MPSENet` / `denoiser` and the DNSMOS stack are declared optional extras (`[models]`, `[metrics]`).

### Changed
- README, book intro, `docs/architecture.md` and `data/README.md` describe the TASLP scope, the 824-utterance sweeps and where their raw artifacts live; `CITATION.cff` carries the journal citation alongside the preprint.

## v0.1.2, Reverb FT pipeline (2026-05-03)

### Added
- Real epoch-48 reverb fine-tuned MUSE checkpoint on HuggingFace (`yairamr/SE-Probe-models/muse_reverb_e48.pt`), replacing the noise-only `g_best` placeholder shipped in v0.1.0/v0.1.1. Notebook 06 inference cells now reproduce paper numbers.
- Training pipeline that produced the checkpoint, vendored directly under `training/`. The repo is now self-contained: a plain `git clone` gives both probing and training in one tree.
- Notebook 06 placeholder warning markdown removed; replaced with a one-line pointer to the training repo.

### Changed
- `_workspace/cluster_artifacts_needed.md` item 1 (reverb FT checkpoint retrieval) marked resolved.

## v0.1.1, Hosted book (2026-05-03)

### Added
- Jupyter Book hosted at <https://yairamar.github.io/SE-Probe/>, deployed automatically from `main` via GitHub Actions (`.github/workflows/deploy-book.yml`). The book renders the six notebooks in chapter order from the in-tree `results_demo/` parquets; inference cells stay gated by `SE_PROBE_RUN_INFERENCE` and are skipped on CI.
- `[project.optional-dependencies].docs` extra (`jupyter-book>=1.0,<2`, `sphinx-copybutton`, `sphinx-design`) so `pip install -e .[docs]` is enough to rebuild the book locally.

## v0.1.0, Initial public release (2026-05-02)

First public release accompanying the paper *"Where Does Speech Enhancement Adapt? Probing Study Under Controlled Degradation"*.

### Added
- `se_probe` Python package: linear CKA, activation extraction (MUSE / MP-SENet / Demucs), diffusion maps, audio-quality metrics, paper-style plotting, and CUDA / MPS / CPU device autodetection.
- Six numbered Jupyter notebooks under `notebooks/` reproducing the poster figures qualitatively from a 50 MB demo subset:
  - `01_pipeline_overview`, end-to-end CKA on a single utterance.
  - `02_cka_per_layer`, per-layer heatmap and SNR sensitivity.
  - `03_cross_architectures`, slope-vs-intercept scatter across the three SE models.
  - `04_cka_to_pesq`, within-group CKA-PESQ correlation.
  - `05_diffusion_maps`, per-layer diffusion-distance and Spearman SNR ordering.
  - `06_reverb_probing`, C50 sensitivity for reverb fine-tuned MUSE; inference cells gated by `SE_PROBE_RUN_INFERENCE=1`.
- `scripts/setup.py`, one-shot installer that fetches the upstream MUSE pretrained, the reverb fine-tuned checkpoint placeholder, and (with `--full-data`) the full 3.1 GB precomputed CKA tables from HuggingFace.
- `scripts/build_demo_subset.py`, regenerates `results_demo/` from a full `results_df/` checkout.
- Smoke-test fixture under `tests/fixtures/` and a pytest suite that runs without any external download.
- Documentation: top-level `README.md` (with hardware table), `data/README.md` (external dataset URLs and env vars), `docs/architecture.md` (per-module tour), and 13 demo audio samples under `docs/audio_samples/`.
- GitHub Actions CI: `ruff` lint, `nbstripout` notebook-output check, and `pytest` against the smoke fixture.
- Pre-commit config (`nbstripout` + `ruff`).

### Known limitations
- The reverb fine-tuned MUSE checkpoint shipped via `scripts/setup.py` is a **placeholder** (= upstream noise-only `g_best`). The published reverb figure in notebook 06 is correct because it is read from a precomputed parquet, but inference cells produce non-paper numbers until the real checkpoint lands on HuggingFace. Tracked in `_workspace/cluster_artifacts_needed.md`.
- No training pipeline is shipped. v0.1.0 is checkpoint-only per design.
- Apple Silicon (MPS) runs use a CPU fallback for a handful of unsupported PyTorch ops; numerics are spot-checked against CUDA but not exhaustively certified.

### Notes on locked decisions
- D14 (Git LFS for `results_demo/*.parquet`) was a contingency for the case where the demo-data subset exceeded a comfortable in-tree size. The actual subset totals ~200 KB across five parquets (three per-model SNR tables, one reverb table, one diffusion-maps table), so LFS was not configured for v0.1.0. If a future release expands the demo set, ship `.gitattributes` with `*.parquet filter=lfs diff=lfs merge=lfs -text`.

[Unreleased]: https://github.com/YairAmar/SE-Probe/compare/v0.2.0...HEAD
