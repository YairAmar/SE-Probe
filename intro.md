# SE-Probe: Where Does Speech Enhancement Adapt?

📄 **Paper:** [arXiv:2512.00482](https://arxiv.org/abs/2512.00482) &nbsp;·&nbsp; 💻 **Code:** <https://github.com/YairAmar/SE-Probe>

Public companion code for *"Where Does Speech Enhancement Adapt? Probing Study Under Controlled Degradation"* (Amar, Ivry, Cohen, 2026). Speech enhancement networks are treated as black boxes: clean and degraded utterances are pushed through a frozen SE model, activations are extracted layer by layer, clean and degraded representations are compared by linear CKA, and the resulting curves are regressed against degradation severity (SNR or C50) to identify the layers most sensitive to each condition. Diffusion map distances and downstream PESQ correlations cross check the picture from a different angle.

The book renders the analysis end to end across MUSE, MP-SENet, and Demucs, using the precomputed CKA tables shipped under `results_demo/`. Two chapters (01 and 06) optionally re-run model inference when hardware is available; the published version skips those cells, and every figure is rebuilt from parquet.

Each chapter opens with a short statement of what it shows. Reading top to bottom traces the same arc as the paper: the pipeline walkthrough in chapter 01 sets up the per-layer CKA heatmap in 02, the reverb probe in 06, the cross-architecture scatter in 03, the within-group correlation with PESQ in 04, and the diffusion view in 05.

Source code, issue tracker, and citation metadata at <https://github.com/YairAmar/SE-Probe>.
