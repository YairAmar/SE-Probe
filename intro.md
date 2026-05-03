# SE-Probe: Where Does Speech Enhancement Adapt?

SE-Probe is the public companion code for *"Where Does Speech Enhancement Adapt? Probing Study Under Controlled Degradation"* (Amar, Ivry, Cohen, 2026). We treat speech-enhancement networks as black boxes and ask which layers actually adapt to a given degradation: clean and noisy or reverberant utterances are pushed through a frozen SE model, activations are harvested layer by layer, clean-vs-degraded representations are compared with linear CKA, and the resulting curves are regressed against degradation severity (SNR or C50) to surface the layers most sensitive to each condition. Diffusion-map distances and downstream PESQ/STOI correlations cross-check the picture from a different angle.

The book renders this analysis end-to-end across three architectures — MUSE, MP-SENet, and Demucs — using the precomputed CKA tables shipped under `results_demo/`. Two of the notebooks (01 and 06) optionally re-run model inference when hardware is available; in the version you are reading those cells are skipped, and every figure is rebuilt from parquet.

## How to read this book

Each chapter opens with a short *what this shows / what to look for / runtime* block, so a reader can dip in at any chapter and quickly tell whether the figure they want is on that page. Reading top to bottom traces the same arc as the paper: the pipeline walkthrough in chapter 01 sets up the per-layer CKA heatmap in 02, the reverb-specific probe in 06, the cross-architecture scatter in 03, the within-group correlation with PESQ in 04, and the diffusion-map view in 05.

## References

The full paper is *"Where Does Speech Enhancement Adapt? Probing Study Under Controlled Degradation"* (Amar, Ivry, Cohen, 2026). Source code, issue tracker, and citation metadata live at <https://github.com/YairAmar/SE-Probe>.
