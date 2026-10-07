# SE-Probe: Probing Layer-Wise Robustness and Sensitivity of Speech Enhancement Models

📄 **Paper:** *Probing Layer-Wise Robustness and Sensitivity of Speech Enhancement Models Under Noise and Reverberation* (Amar, Ivry, Cohen; IEEE/ACM TASLP), preprint [arXiv:2512.00482](https://arxiv.org/abs/2512.00482) &nbsp;·&nbsp; 💻 **Code:** <https://github.com/YairAmar/SE-Probe> &nbsp;·&nbsp; 🎛️ **Interactive demo (ICASSP 2026 Show & Tell):** <https://yairamar.github.io/seint-show-web/>

Public companion code for the paper. Speech enhancement networks are treated as black boxes: clean and degraded utterances are pushed through a frozen SE model, activations are extracted layer by layer, clean and degraded representations are compared by linear CKA, and the resulting curves are regressed against degradation severity (SNR or C50) to give each layer a robustness (intercept) and a sensitivity (slope). Diffusion-map distances and downstream quality correlations cross check the picture from a different angle.

The chapters follow the paper in order, with more figures than the paper has room for. Every number the paper prints is recomputed here next to the printed value, from the small result tables shipped under `results_tables/` (each with its provenance in `results_tables/README.md`); the walkthrough chapters also run on the demo tables under `results_demo/` so that nothing needs a GPU.

1. **Pipeline overview** sets up one utterance end to end: degrade, extract, compare.
2. **CKA per layer** builds the (layer, SNR) heatmap and the per-layer linear fit that defines robustness $\alpha$ and sensitivity $\beta$.
3. **Hierarchical bootstrap confidence intervals** puts utterance × noise-type intervals on that profile and shows why the sampling unit matters.
4. **Cross-architecture profiles** repeats the probe on MP-SENet and Demucs.
5. **Reverb probing** swaps additive noise for room impulse responses at controlled C50.
6. **The saturation spread and Table I** scopes the robustness–sensitivity tradeoff on both axes and all three models.
7. **Noise-set independence** shows the profile does not depend on which DEMAND recordings are used.
8. **Random-initialisation control** shows it is absent in an untrained network.
9. **Emergence during fine-tuning** shows it forming epoch by epoch.
10. **Dereverberation fine-tuning** re-probes all three architectures after fine-tuning, with their held-out quality.
11. **CKA to PESQ** and 12. **Quality association with the speaker as the sampling unit** relate representational change to perceptual gain.
13. **Diffusion** and 14. **Diffusion maps at 824 utterances** give the geometric view of the same activations.
15. **CKA estimator and sample-convention ablation** checks the biased estimator against the unbiased one.
16. **Profile-guided freezing** uses the profile to decide which blocks to fine-tune.

Two chapters (the pipeline overview and reverb probing) optionally re-run model inference when hardware is available; the published version skips those cells.

Source code, issue tracker, and citation metadata at <https://github.com/YairAmar/SE-Probe>.
