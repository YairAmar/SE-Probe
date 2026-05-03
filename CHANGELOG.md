# Changelog

All notable changes to SE-Probe are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to semantic versioning.

## v0.1.2, Reverb FT pipeline (2026-05-03)

### Added
- Real epoch-48 reverb fine-tuned MUSE checkpoint on HuggingFace (`yairamr/SE-Probe-models/muse_reverb_e48.pt`), replacing the noise-only `g_best` placeholder shipped in v0.1.0/v0.1.1. Notebook 06 inference cells now reproduce paper numbers.
- Training pipeline that produced the checkpoint, vendored directly under `training/` (originally <https://github.com/YairAmar/muse-dereverb-ft>). The repo is now self-contained: a plain `git clone` gives both probing and training in one tree.
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

[Unreleased]: https://github.com/YairAmar/SE-Probe/compare/v0.1.2...HEAD
