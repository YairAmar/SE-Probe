# SE-Probe: Probing Layer-Wise Robustness and Sensitivity of Speech Enhancement Models

📄 **Paper:** *Probing Layer-Wise Robustness and Sensitivity of Speech Enhancement Models Under Noise and Reverberation* (Amar, Ivry, Cohen; IEEE/ACM TASLP), preprint [arXiv:2512.00482](https://arxiv.org/abs/2512.00482) &nbsp;·&nbsp; 💻 **Code:** <https://github.com/YairAmar/SE-Probe> &nbsp;·&nbsp; 🎛️ **Interactive demo (ICASSP 2026 Show & Tell):** <https://yairamar.github.io/seint-show-web/>

Public companion code for the paper. Speech enhancement networks are treated as black boxes: clean and degraded utterances are pushed through a frozen SE model, activations are extracted layer by layer, clean and degraded representations are compared by linear CKA, and the resulting curves are regressed against degradation severity (SNR or C50) to give each layer a robustness (intercept) and a sensitivity (slope). Diffusion-map distances and downstream quality correlations cross check the picture from a different angle.

The eight chapters follow the paper in order. Every number the paper prints is recomputed next to the printed value from the small result tables shipped under `results_tables/` (each with its provenance in `results_tables/README.md`); nothing needs a GPU.

1. **Pipeline overview** sets up one utterance end to end: degrade, extract, compare.
2. **The profile under noise** builds the (layer, SNR) heatmap, the per-layer linear fit that defines robustness $\alpha$ and sensitivity $\beta$, its utterance × noise-type intervals, and shows the profile does not depend on which DEMAND recordings are used.
3. **Across architectures** repeats the probe on MP-SENet and Demucs, and scopes the robustness–sensitivity tradeoff with the saturation spread (Table I).
4. **Reverberation and fine-tuning** swaps noise for room impulse responses at controlled C50, then re-probes all three architectures after dereverberation fine-tuning, with their held-out quality.
5. **Origin in training** shows the profile is absent in an untrained network and forms epoch by epoch.
6. **Perceptual quality** relates representational change to PESQ, STOI, SI-SDR and DNSMOS, with the speaker as the sampling unit.
7. **Geometric view** gives the diffusion-map picture of the same activations, across SNR and across layers.
8. **Checks** swaps the CKA estimator and uses the profile to decide which blocks to fine-tune.

Four chapters (the pipeline overview, reverberation, origin in training, and the checks) optionally re-run model inference when `SE_PROBE_RUN_INFERENCE` is set and hardware is available; the published version skips those cells and reads the shipped tables.

Source code, issue tracker, and citation metadata at <https://github.com/YairAmar/SE-Probe>.
