# SE-Probe: Probing Layer-Wise Robustness and Sensitivity of Speech Enhancement Models

📄 **Paper:** *Probing Layer-Wise Robustness and Sensitivity of Speech Enhancement Models Under Noise and Reverberation* (Amar, Ivry, Cohen; IEEE/ACM TASLP), preprint [arXiv:2512.00482](https://arxiv.org/abs/2512.00482) &nbsp;·&nbsp; 💻 **Code:** <https://github.com/YairAmar/SE-Probe>

Public companion code for the paper. Speech enhancement networks are treated as black boxes: clean and degraded utterances are pushed through a frozen SE model, activations are extracted layer by layer, clean and degraded representations are compared by linear CKA, and the resulting curves are regressed against degradation severity (SNR or C50) to give each layer a robustness (intercept) and a sensitivity (slope). Diffusion-map distances and downstream quality correlations cross check the picture from a different angle.

The book has two parts.

**Part I, the pipeline (chapters 01–06),** renders the analysis end to end across MUSE, MP-SENet, and Demucs, using the precomputed demo tables shipped under `results_demo/`. Reading top to bottom traces the arc of the paper: the pipeline walkthrough in chapter 01 sets up the per-layer CKA heatmap in 02, the reverb probe in 06, the cross-architecture scatter in 03, the within-group correlation with PESQ in 04, and the diffusion view in 05. Two chapters (01 and 06) optionally re-run model inference when hardware is available; the published version skips those cells.

**Part II, the TASLP results (chapters 07–16),** reproduces every statistic, table and figure of the journal manuscript from the small result tables shipped under `results_tables/` (each with its cluster provenance in `results_tables/README.md`): the hierarchical bootstrap intervals on the per-layer profile (07), Table I and the saturation spread that scopes the robustness–sensitivity tradeoff (08), the noise-set independence check (09), the random-initialisation control (10), the emergence of the profile over fine-tuning (11), the effect of dereverberation fine-tuning on all three architectures together with their held-out quality (12), the quality association with the speaker as the sampling unit (13), the diffusion-map view at 824 utterances (14), the CKA-estimator and sample-convention ablation (15), and the supplementary profile-guided freezing experiment that is not in the manuscript (16). Every chapter prints the manuscript's number next to the value it recomputes.

Source code, issue tracker, and citation metadata at <https://github.com/YairAmar/SE-Probe>.
