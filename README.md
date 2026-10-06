# SE-Probe

📄 **Paper:** *Probing Layer-Wise Robustness and Sensitivity of Speech Enhancement Models Under Noise and Reverberation* (Amar, Ivry, Cohen; IEEE/ACM Transactions on Audio, Speech and Language Processing) &nbsp;·&nbsp; preprint [arXiv:2512.00482](https://arxiv.org/abs/2512.00482) &nbsp;·&nbsp; 📖 **Book:** <https://yairamar.github.io/SE-Probe/>

## Background

Public companion code for the TASLP paper (and its earlier preprint *"Where Does Speech Enhancement Adapt? Probing Study Under Controlled Degradation"*, [arXiv:2512.00482](https://arxiv.org/abs/2512.00482)). Speech enhancement networks are treated as black boxes: clean and degraded utterances are pushed through a frozen SE model, activations are extracted layer by layer, clean and degraded representations are compared by linear CKA, and the resulting curves are regressed against degradation severity (SNR or C50) into a per-layer **robustness** (intercept) / **sensitivity** (slope) profile. Diffusion-map distances and downstream quality correlations cross check the picture from a different angle. The analysis is run end to end across MUSE, MP-SENet, and Demucs, on additive noise and on reverberation, with every model also probed at its dereverberation fine-tune.

The full narrative lives in the book linked above. This README is just enough to get the code running.

## What is in the repository

| Path | Contents |
|---|---|
| `se_probe/` | The library: CKA (biased and unbiased-HSIC estimators), activation extraction for the three models (pretrained and fine-tuned checkpoints, random-init control), per-layer profiles and the saturation-spread statistics (`profiles.py`), hierarchical cluster bootstrap (`bootstrap.py`), quality association with speaker-level inference (`perceptual.py`), diffusion maps and their 824-utterance analyses, probed-layer sets and depth order (`layers.py`), paper-style plotting. |
| `notebooks/` | Sixteen book chapters. 01–06 walk the pipeline on the in-tree demo tables; 07–16 reproduce every statistic, table and figure of the TASLP manuscript from the shipped result tables (bootstrap CIs, Table I and the saturation spread, noise-set independence, random-initialisation control, emergence over fine-tuning, dereverberation fine-tuning, speaker-level quality association, 824-utterance diffusion maps, the CKA-estimator ablation, and the supplementary profile-guided freezing experiment). |
| `results_tables/` | The small CSV/JSON/parquet tables behind every number in the paper, copied byte-for-byte from the cluster analyses with a `MANIFEST.csv` (SHA-256 + source path) and a README of conventions and provenance. |
| `results_demo/` | ~200 KB demo subsets of the per-utterance CKA tables so chapters 01–06 run on any laptop. |
| `scripts/` | Cluster-agnostic, resumable drivers that regenerate the raw sweeps (noise grid, reverberation grid, random-init, emergence, perceptual metrics, diffusion centroids, HSIC ablation) and the analysis CLIs that turn them into the tables above; `setup.py` fetches checkpoints and the full HuggingFace data. |
| `training/` | The dereverberation fine-tuning pipelines for MUSE, MP-SENet and Demucs (exact configurations of the paper), the held-out evaluation harnesses and results, and the selective-freezing / profile-guided arms. |
| `tests/` | Pytest suite; the profile tests reproduce Table I and the other headline numbers from `results_tables/` to machine precision. |

## Get going

```bash
git clone https://github.com/YairAmar/SE-Probe.git && cd SE-Probe
pip install -e .            # add [models] for the MP-SENet / Demucs adapters, [metrics] for DNSMOS
python scripts/setup.py
jupyter lab notebooks/
```

`scripts/setup.py` fetches the upstream MUSE pretrained weights, downloads the epoch-48 reverb fine-tuned checkpoint from HuggingFace (`yairamr/SE-Probe-models`), and probes the local device (CUDA, MPS, or CPU; autodetected). Chapters 01–06 read precomputed CKA tables from `results_demo/`; chapters 07–16 read `results_tables/`. Both ship in-tree and run in seconds. Add `--full-data` to `setup.py` to also pull the full per-utterance CKA tables from HuggingFace (`yairamr/SE-Probe-data`); the 824-utterance sweeps of the TASLP revision are documented in `data/README.md`. Chapters 01, 06, 10 and 15 optionally re-run model inference when `SE_PROBE_RUN_INFERENCE=1`.

A ten-line tour of the library:

```python
import pandas as pd
from se_probe import fit_profile, tradeoff_summary, mean_level_curves, two_way_cluster_bootstrap
df = pd.read_parquet("results_demo/cka_snr_muse_demo.parquet")
prof = fit_profile(df, "muse")                      # alpha, beta, R2, c_low, c_high, AUC per probed layer
summ = tradeoff_summary(mean_level_curves(df, "muse"))  # r(alpha,beta), saturation spread f, PC1, floor ...
```

The reverb fine-tuning pipelines that produced the probed checkpoints live under `training/`; see `training/README.md`.

## Citation

```bibtex
@article{amar2026probing,
  title   = {Probing Layer-Wise Robustness and Sensitivity of Speech Enhancement Models Under Noise and Reverberation},
  author  = {Yair Amar and Amir Ivry and Israel Cohen},
  journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year    = {2026},
  note    = {Preprint: arXiv:2512.00482}
}
```

The earlier preprint can be cited as:

```bibtex
@article{amar2026seprobe,
  title         = {Where Does Speech Enhancement Adapt? Probing Study Under Controlled Degradation},
  author        = {Yair Amar and Amir Ivry and Israel Cohen},
  year          = {2026},
  eprint        = {2512.00482},
  archivePrefix = {arXiv},
  primaryClass  = {eess.AS},
  url           = {https://arxiv.org/abs/2512.00482}
}
```

## License

MIT, see `LICENSE`. Open an issue at <https://github.com/YairAmar/SE-Probe/issues>.
