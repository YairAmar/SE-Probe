# scripts/: regenerating the paper's artifacts

Every driver is cluster-agnostic (paths come from flags or `SEPROBE_*` environment
variables), writes one parquet per grid cell and skips the cells that already exist,
so a job cut by a walltime limit resumes where it stopped. The statistics live in the
`se_probe` package (`layers`, `profiles`, `perceptual`, `cka`, `diffusion_analysis`,
`centroids`, `bootstrap`); the analysis CLIs here are thin wrappers over it. The
reduced tables every published number is read from are shipped under
`results_tables/`, so none of this is needed to run the notebooks.

Clean utterances come from the VoiceBank-DEMAND 16 kHz test split: pass
`--clean <clean_test dir>` or `--clean hf` (HuggingFace `JacobLinCool/VoiceBank-DEMAND-16k`),
or set `SEPROBE_CLEAN_TEST_DIR`. DEMAND noises and AIR RIRs are found through
`SEPROBE_DEMAND_DIR` and `SEPROBE_AIR_RIR_DIR` (see `data/README.md`).

| script | paper section / grid | inputs | outputs | cost |
|---|---|---|---|---|
| `setup.py` | fetches the MUSE checkpoints and, with `--full-data`, the HuggingFace tables | network | `pretrained_models/`, `results_df/` | minutes |
| `run_snr_grid.py` | II-C noise sweep: 824 utts x 41 SNRs x 18 DEMAND noises, one model (`--checkpoint` for a fine-tune) | clean dir, DEMAND dir | `<model>__<noise>__snr<NN>.parquet` x 738 per model | ~10-40 GPU-h per model (A100) |
| `run_reverb_grid.py` | II-C reverberation sweep: 824 utts x 13 target C50 x 5-of-88 AIR RIRs, one of six arms | clean dir, AIR dir, FT checkpoint | `shard_<arm>_<start>_<stop>.parquet` | ~5-15 GPU-h per arm |
| `run_random_init_grid.py` | III-E random-initialisation control, one seed (0, 1, 2) | clean dir, DEMAND dir | `randominit_seed<S>__<noise>__snr<NN>.parquet` x 205 | ~3 GPU-h per seed |
| `run_emergence_grid.py` | III-E emergence: the intermediate dereverb-FT checkpoints on the noise axis | clean dir, `epoch_N/g_*` checkpoints | `e<N>/snr/e<N>__<noise>__snr<NN>.parquet` | ~3 GPU-h per epoch |
| `run_perceptual_grid.py` + `join_perceptual.py` | II-D / III-B per-mixture PESQ, STOI, SI-SDR, DNSMOS, three models | clean dir, DEMAND dir | `perc_<model>__<noise>__snr<NN>.parquet`, `cka_snr_metrics.parquet`, `coverage.md` | ~4 GPU-h per model (DNSMOS dominates) |
| `run_diffusion_centroids.py` + `build_diffusion_psi.py` | II-D / III-G centroids of the 24 MUSE layers and their diffusion-map embeddings (t = 0.5 per block, t = 5 across layers) | clean dir, DEMAND dir | `centroids__<noise>__snr<NN>.parquet` x 205; `diffusion_maps_{per_layer_t0.5,architecture_t5}.parquet` | ~2 GPU-h, then CPU minutes |
| `run_hsic_ablation.py` | II-D unbiased-HSIC and sample-convention ablation (150 utts x 9 SNRs x 2 noises, non-pooled activations) | clean dir, DEMAND dir | `ablation__<noise>__snr<NN>.parquet`, `hsic_frames_slopes.csv` | ~1 GPU-h |
| `analyze_snr_profiles.py` | III-A fits, Table I noise rows, noise-set independence, `--bootstrap` hierarchical CIs | SNR chunks | `fits_<model>_snr.csv`, `summary_snr.csv`, `per_noise_beta_<model>.csv`, `noise_set_independence_<model>.csv`, `hierarchical_bootstrap_snr.csv` | CPU minutes (bootstrap: hours) |
| `analyze_c50_profiles.py` | III-C / III-F six-arm C50 profiles, Table I C50 rows, pre-vs-FT shift, Demucs bottleneck | reverb shards | `c50_per_layer.csv`, `mean_curves_c50.csv`, `summary_c50.csv`, `finetune_shift.csv`, `demucs_bottleneck.csv`, `per_speaker_auc.csv` | CPU minutes |
| `analyze_random_init.py` | III-E random-init statistics | random-init + trained chunks | `random_init_per_layer.csv`, `random_init_summary.json` | CPU minutes |
| `analyze_emergence.py` | III-E convergence to the final epoch, concentration of the change | emergence tree + pretrained chunks | `profile_snr_*.csv`, `emergence_convergence.csv`, `emergence_summary.json` | CPU minutes |
| `analyze_perceptual.py` | III-B per-layer associations, per-speaker and speaker-level intervals | joined metrics parquet (+ the 780 table) | `perceptual_per_layer.csv`, `speaker_*.csv` | CPU minutes |
| `analyze_diffusion.py` | III-G per-block ordering, arc-length ratio, across-layer group distances | psi parquets | `diffusion_per_block.csv`, `diffusion_across_layers.csv`, `diffusion_summary.json` | seconds |
| `analyze_freeze_arms.py` | supplementary profile-guided freezing re-probe (not in the manuscript) | per-arm C50 probes | `freeze_arms_{per_layer,contrasts,stage_summary}.csv` | CPU minutes |
| `build_demo_subset.py` | the ~200 KB `results_demo/` tables; `--snr-chunks-dir` / `--reverb-arm-shards` accept the 824-scale outputs | `results_df/` or chunk dirs | `results_demo/*.parquet` | seconds |
| `hf_upload_artifacts.py` | upload plan for the large aggregates and checkpoints (dry run unless `--yes`) | aggregated dir, checkpoints dir | HuggingFace repos | network |

Shared modules: `_clean_sources.py` (VoiceBank-DEMAND test split loader), `_grid.py`
(extraction loop, CKA rows, resumable chunks, model/arm loaders) and `_tables.py`
(chunk readers and per-chunk reductions).

## Recommended order

1. `setup.py` (checkpoints); fine-tune MP-SENet / Demucs with `training/` if their
   dereverberation arms are needed.
2. `run_snr_grid.py` for the three models, then `analyze_snr_profiles.py --bootstrap`.
3. `run_reverb_grid.py` for the six arms (use `--verify` on a full shard), then
   `analyze_c50_profiles.py`.
4. `run_random_init_grid.py` for seeds 0, 1, 2 and `run_emergence_grid.py`, then the two
   `analyze_*` scripts (both read the pretrained MUSE chunks of step 2).
5. `run_perceptual_grid.py` for the three models, `join_perceptual.py`, `analyze_perceptual.py`.
6. `run_diffusion_centroids.py`, `build_diffusion_psi.py`, `analyze_diffusion.py`.
7. `run_hsic_ablation.py`.
8. `build_demo_subset.py` to refresh `results_demo/`, `hf_upload_artifacts.py` to publish.

The grids above use the paper's full sizes by default; every driver takes
`--n-utts`, `--snrs` and `--noises` for a smoke test before submitting the full job.
