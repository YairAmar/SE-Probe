# Cross-task agreement of the depth profile, within architecture

**Question.** For each architecture separately, does its per-layer profile on the
noise axis look like its per-layer profile on the reverberation axis? This is a
whole-profile correlation across that model's own layers, computed once for beta
and once for alpha.

**Answer in one line.** It holds strongly for MUSE on both parameters, holds for
MP-SENet on alpha only, and does not hold for Demucs on either. Demucs
nevertheless shares one specific, well-resolved feature across the two tasks: the
LSTM bottleneck is the most robust layer on both axes. It does not share the
shape of the profile away from that point.

---

## 1. Arms, sources, and the two preconditions

| axis | arms | source |
|---|---|---|
| SNR (noise) | pretrained denoising checkpoints, 824 utts x 41 SNR x 18 noises | `tools/fits_824_snr/fits_{muse,mpsenet,demucs}_snr_jobB.csv` |
| C50 (reverberation) | **dereverberation fine-tuned**: `muse_ft`, `mpsenet_ft`, `demucs_ft`, 824 utts x 13 C50 x 88-RIR AIR pool | `tools/c50_824_per_layer_supp.csv`, column `arm` |

The C50 side uses the fine-tuned arms as the author decided today. The prior
computation of this quantity used the pretrained C50 arms and is superseded.
Everything below is a fresh computation.

**Precondition 1, same layer set on both axes.** Verified, not assumed. Set
equality of the `layer` column between each SNR table and its matching `*_ft` C50
rows: MUSE 24 = 24 identical, MP-SENet 8 = 8 identical, Demucs 11 = 11 identical.
No layer appears on one axis only. The correlations below are therefore over
matched layers, not over a merged or truncated set.

**Precondition 2, depth ordering.** For all six tables the stored `depth` column
was asserted equal to architectural order before use, per `LAYER_ORDER_AUDIT.md`.
The architectural key is a re-implementation of `depth_order()`:
`stage_rank` then block then layer index for MUSE with stage order
`encoder_level1, encoder_level2, latent, decoder_level2, decoder_level1,
mag_refinement`; `(block index, time before freq)` for MP-SENet; the explicit
`encoder.0-4, lstm, decoder.0-4` list for Demucs. All six assertions passed.
The assertion also checks `depth` is a permutation of `0..n-1`.

A caveat that matters for how much weight to put on preconditions: the two
whole-profile correlations in Section 2 are invariant to a common permutation of
the layers, so ordering cannot corrupt them. Ordering does matter for the Demucs
distance-from-bottleneck statistic in Section 4, which is depth-indexed, and that
is the one where the audit's warning about Demucs sign inversion applies.

## 2. Main result

Spearman and Pearson between the SNR profile and the C50 fine-tuned profile,
across each model's own layers. CIs are 95 percent percentile intervals from a
**layer-level resample**: layers are drawn with replacement from that model's
layer set, n of them, the paired (SNR, C50) values travel together, and the
correlation is recomputed; 20,000 resamples, seed 20260810, resamples with fewer
than three distinct values on either side discarded. This treats the layers as
the sampling unit, which is the right unit for a question about profile shape,
and it is deliberately conservative at n = 8.

| model | n | profile | Spearman | 95% CI | p | Pearson | 95% CI | p |
|---|---|---|---|---|---|---|---|---|
| MUSE | 24 | beta | **+0.837** | [+0.582, +0.942] | 3.3e-07 | +0.879 | [+0.794, +0.945] | 1.6e-08 |
| MUSE | 24 | alpha | **+0.864** | [+0.664, +0.957] | 5.2e-08 | +0.880 | [+0.795, +0.945] | 1.4e-08 |
| MP-SENet | 8 | beta | +0.476 | [-0.538, +1.000] | 0.233 | +0.678 | [-0.257, +0.968] | 0.064 |
| MP-SENet | 8 | alpha | **+0.833** | [+0.300, +1.000] | 0.010 | +0.776 | [+0.177, +0.977] | 0.024 |
| Demucs | 11 | beta | +0.227 | [-0.421, +0.735] | 0.502 | +0.310 | [-0.283, +0.741] | 0.354 |
| Demucs | 11 | alpha | +0.145 | [-0.626, +0.754] | 0.670 | +0.102 | [-0.717, +0.679] | 0.764 |

Bold marks the rows whose CI excludes zero.

### Verdicts

- **MUSE: the pattern translates.** Both parameters, both correlation types, CIs
  well clear of zero, and the result is not carried by any single layer: the
  leave-one-layer-out Spearman stays in [+0.815, +0.878] for beta and
  [+0.846, +0.896] for alpha. This is the strong case and it is the case the
  paper leads with.
- **MP-SENet: the pattern translates in alpha, and is not established in beta.**
  Alpha is +0.833 with a CI of [+0.300, +1.000], so a positive association is
  supported. Beta at +0.476 has a CI that spans zero and cannot be called either
  way at n = 8. The beta result is fragile in the literal sense: dropping
  `TS1.time` raises it to +0.964, dropping `TS0.time` lowers it to +0.214. The
  proximate cause is visible in the profile. On the noise axis `TS0.freq` is
  already a high-beta layer (0.0176 against 0.0026 for `TS0.time`), whereas after
  dereverberation fine-tuning the whole first block is nearly reverberation-invariant
  (`TS0.time` 0.00020, `TS0.freq` 0.00134), so the early-block ranks disagree.
  Fine-tuning compresses the beta range, and with eight layers a compressed range
  is enough to destabilise the rank correlation.
- **Demucs: the pattern does not translate as a whole profile.** Both CIs contain
  zero by a wide margin, on both parameters and both correlation types. See
  Section 4 for what does survive.

### What changed relative to the superseded pretrained-arm computation

Recomputed here for context only; the fine-tuned column is the live number.

| model | profile | pretrained arm | fine-tuned arm | rho(pre profile, ft profile) |
|---|---|---|---|---|
| MUSE | beta | +0.812 | +0.837 | +0.815 |
| MUSE | alpha | +0.935 | +0.864 | +0.837 |
| MP-SENet | beta | +0.905 | **+0.476** | +0.762 |
| MP-SENet | alpha | +0.976 | +0.833 | +0.905 |
| Demucs | beta | +0.300 | +0.227 | +0.991 |
| Demucs | alpha | -0.173 | +0.145 | +0.873 |

The MUSE and Demucs verdicts are unchanged by the arm swap. **The MP-SENet beta
verdict is changed by it**, from a strong +0.905 to an inconclusive +0.476, and
that is the one number in the manuscript's neighbourhood that the swap actually
moves. The prior triple quoted `+0.812 / +0.905 / +0.227` for MUSE, MP-SENet and
Demucs beta; my recomputation of the pretrained arms gives `+0.812 / +0.905 /
+0.300`, so the two spectral values reproduce and the Demucs value in that triple
appears to have been the fine-tuned one already.

### One caution on reading alpha and beta as two results

They are not independent. Within any single axis, alpha and beta are almost
perfectly anticorrelated across layers: r = -0.952 (MUSE, SNR), -0.993 (MUSE,
C50-ft), -0.989 and -0.999 (MP-SENet), -0.881 and -0.918 (Demucs). A layer that
starts high is a layer that falls slowly. So the alpha row and the beta row of the
table above are close to the same measurement twice, and where they disagree, as
they do for MP-SENet, the disagreement is informative about noise rather than
about two separate phenomena. The paper should present one of them as primary,
not treat agreement on both as double confirmation.

## 3. Depth trends per axis, for context

Not the headline quantity, but useful for reading the table above. These are
depth-indexed and therefore did require the Section 1 ordering assertion.

| model | axis | rho(depth, beta) | rho(depth, alpha) |
|---|---|---|---|
| MUSE | SNR | +0.622 (p = 1.2e-03) | -0.741 (p = 3.5e-05) |
| MUSE | C50-ft | +0.615 (p = 1.4e-03) | -0.593 (p = 2.3e-03) |
| MP-SENet | SNR | +0.833 (p = 0.010) | -0.929 (p = 8.6e-04) |
| MP-SENet | C50-ft | +0.571 (p = 0.139) | -0.762 (p = 0.028) |
| Demucs | SNR | -0.727 (p = 0.011) | +0.573 (p = 0.066) |
| Demucs | C50-ft | +0.009 (p = 0.979) | -0.209 (p = 0.537) |

Both spectral models increase sensitivity with depth on both axes, with matching
sign and similar magnitude. Demucs has opposite-signed depth trends on the two
axes, which is the summary form of its whole-profile failure.

## 4. Demucs, the interesting case

Per-layer values in depth order.

| layer | alpha SNR | alpha C50-ft | beta SNR | beta C50-ft |
|---|---|---|---|---|
| encoder.0 | 0.4980 | 0.5286 | 0.02407 | 0.02473 |
| encoder.1 | 0.4885 | 0.5558 | 0.02439 | 0.02287 |
| encoder.2 | 0.3902 | 0.6648 | 0.02733 | 0.01663 |
| encoder.3 | 0.4229 | 0.7173 | 0.02601 | 0.01338 |
| encoder.4 | 0.4362 | 0.7444 | 0.02084 | 0.01081 |
| **lstm** | **0.8211** | **0.8986** | **0.00447** | **0.00419** |
| decoder.0 | 0.7088 | 0.7262 | 0.00616 | 0.01026 |
| decoder.1 | 0.7413 | 0.6761 | 0.00607 | 0.01233 |
| decoder.2 | 0.6480 | 0.5652 | 0.00604 | 0.01731 |
| decoder.3 | 0.6006 | 0.3522 | 0.00436 | 0.02218 |
| decoder.4 | 0.7052 | 0.3955 | 0.01384 | 0.02661 |

### 4a. Is the LSTM bottleneck the alpha maximum on both axes? Yes, confirmed.

| axis | lstm alpha | 95% CI | runner-up | its CI | overlap |
|---|---|---|---|---|---|
| SNR | 0.8211 | [0.7900, 0.8462] | decoder.1, 0.7413 | [0.6939, 0.7782] | none, margin 0.0118 |
| C50-ft | 0.8986 | [0.8966, 0.9004] | encoder.4, 0.7444 | [0.7399, 0.7492] | none, margin 0.147 |

The prior check is verified: 0.821 on SNR and, on the fine-tuned arm, 0.899 on
C50, with the LSTM ranked 1 of 11 for alpha on both axes and non-overlapping
intervals against the runner-up in both cases. On the SNR axis the margin is
narrow but real. On C50-ft a paired resample of the difference gives
alpha(lstm) - alpha(encoder.4) = +0.154, CI [+0.151, +0.158], P(difference > 0) =
1.000 over 2,000 resamples.

CI provenance differs between the two rows and the widths are not comparable. The
SNR intervals are the released two-way utterance x noise cluster bootstrap
(`tools/hierarchical_bootstrap_snr_cluster.csv`, 1,000 reps; its alpha values
reproduce `fits_demucs_snr_jobB.csv` exactly). The C50-ft intervals I computed for
this analysis, since none existed: an 824-utterance cluster resample, 2,000 reps,
seed 20260810, utterance-first averaging matching
`c50_824_per_layer_supp.py`, point estimates reproducing
`c50_824_per_layer_supp.csv` to five decimals. That is a one-way resample and is
narrower than a two-way resample would be. The conclusion does not depend on the
width: the C50-ft gap is 0.147 against intervals of order 0.005.

### 4b. Is it the beta minimum on both axes? Only on C50.

| axis | beta minimum | value | CI | lstm rank | resolved? |
|---|---|---|---|---|---|
| SNR | decoder.3 | 0.00436 | [0.00212, 0.00658] | 2 of 11, 0.00447 [0.00348, 0.00574] | **no** |
| C50-ft | **lstm** | 0.00419 | [0.00411, 0.00427] | 1 of 11 | **yes** |

On the noise axis the nominal minimum is `decoder.3`, but lstm, decoder.0,
decoder.1, decoder.2 and decoder.3 have mutually overlapping intervals, so the
minimum is not identified: the LSTM is inside a five-layer tie and calling it the
minimum would be reading noise. On the reverberation axis the LSTM is the minimum
outright, with the paired difference against the runner-up decoder.0 equal to
+0.00607, CI [+0.00594, +0.00621], P > 0 = 1.000.

So the correct statement is that the LSTM is the alpha maximum on both axes but
the beta minimum only on one. Since alpha and beta are near-mirror images within
an axis, the reason for the asymmetry is not a contradiction: it is that alpha
separates layers that beta cannot, because on the noise axis the entire
post-bottleneck half sits at the same low beta.

### 4c. Does sensitivity rise with distance from the bottleneck on both axes? No.

r(beta, |depth - 5|), where depth 5 is the LSTM:

| axis | Pearson | 95% CI | p | Spearman |
|---|---|---|---|---|
| SNR | +0.305 | [-0.290, +0.732] | 0.362 | +0.202 (p = 0.551) |
| C50-ft | **+0.990** | [+0.979, +0.997] | 5.5e-09 | +0.989 (p = 1.1e-08) |

Both prior values verified exactly. The reverberation profile is a near-perfect V
centred on the bottleneck. The noise profile is not a V, it is a step.

The mechanism, from the table: mean beta over encoder / lstm / decoder is
0.0245 / 0.0045 / 0.0073 on SNR and 0.0177 / 0.0042 / 0.0177 on C50-ft. On the
noise axis the decoder does **not** recover, it stays at 1.6x the bottleneck; on
the reverberation axis it recovers fully, to 4.2x, the same as the encoder. Within
sub-blocks the contrast is total: on C50-ft the encoder is perfectly monotone down
(rho = -1.000) and the decoder perfectly monotone up (rho = +1.000), while on SNR
both sub-blocks are flat (rho = -0.100 and 0.000) and the cross-axis correlation
within the encoder alone is +0.100 and within the decoder alone 0.000.

### 4d. The honest Demucs statement

Demucs shares a **feature** across the two tasks without sharing the **shape** of
the profile. The shared feature is the bottleneck: the LSTM is the most robust
layer of the network on both the noise axis and the reverberation axis, ranked 1
of 11 in alpha on each, with non-overlapping intervals against the runner-up in
each. What is not shared is everything on either side of it. On reverberation,
sensitivity rises symmetrically with distance from the bottleneck, r = +0.990. On
noise, it does not, r = +0.305, p = 0.362; instead the encoder is uniformly
sensitive, and everything from the bottleneck onward is uniformly robust apart
from the output layer. That is a step function against a V, which is exactly why
the whole-profile correlation is +0.227 with a CI containing zero, and why
rho(depth, beta) is -0.727 on one axis and +0.009 on the other.

This is a real cross-task result, and it is more precise than a correlation
coefficient. It should be reported as a shared bottleneck, not as a shared
profile.

## 5. What the paper can honestly claim

The claim that a depth-dependent pattern exists and that it translates between
enhancement tasks is supportable, but it is one strong case, one partial case and
one narrow case, and writing it as three is not supportable. For MUSE the profile
translates outright: the per-layer sensitivity profile on the noise axis predicts
the profile on the reverberation axis at Spearman +0.837 for beta and +0.864 for
alpha, with layer-level bootstrap CIs of [+0.582, +0.942] and [+0.664, +0.957],
and the result survives dropping any single layer. For MP-SENet the alpha profile
translates, +0.833 with CI [+0.300, +1.000], while the beta profile at +0.476 has
a CI spanning zero and should be reported as not established at eight layers
rather than as a positive finding; the honest reading is that MP-SENet is
consistent with the MUSE pattern but does not independently confirm it, because
eight layers cannot. For Demucs the whole profile does not translate on either
parameter, and the paper should not claim it does. What Demucs contributes instead
is a sharper and arguably more interesting statement: the LSTM bottleneck is the
most robust layer of the network under both distortions, ranked first of eleven in
alpha on each axis with non-overlapping confidence intervals, while the profile
away from the bottleneck is a V on reverberation and a step on noise. So the
defensible sentence is that the depth-dependent pattern translates between
enhancement tasks in the two spectral-domain architectures, most convincingly in
MUSE, and that in the time-domain architecture what translates is the location of
the robust bottleneck rather than the shape of the profile around it. It should
also be said in the same breath that alpha and beta are near-mirror images within
each axis, r between -0.88 and -0.999, so agreement on both is one result and not
two.

## 6. Reproduction

- `/private/tmp/claude-501/-Users-yairamar-Desktop-work/0d76dad8-e8fd-4d1d-a2b7-1347f80ba040/scratchpad/xtask/cross.py`
  main table, layer-set identity check, depth-order assertion, Demucs deep dive
- `/private/tmp/claude-501/-Users-yairamar-Desktop-work/0d76dad8-e8fd-4d1d-a2b7-1347f80ba040/scratchpad/xtask/detail.py`
  per-layer listings, pretrained-arm comparison, within-axis alpha/beta coupling
- `/private/tmp/claude-501/-Users-yairamar-Desktop-work/0d76dad8-e8fd-4d1d-a2b7-1347f80ba040/scratchpad/xtask/jack.py`
  leave-one-layer-out jackknife
- `/private/tmp/claude-501/-Users-yairamar-Desktop-work/0d76dad8-e8fd-4d1d-a2b7-1347f80ba040/scratchpad/xtask/demucs2.py`
  sub-block decomposition
- athena `~/tmp/xtask/demucs_c50ft_boot.py`, output
  `~/tmp/xtask/demucs_c50ft_layer_ci.csv` and per-layer resample draws
  `~/tmp/xtask/boot_{alpha,beta}_*.npy`, new for this analysis

Inputs: `tools/fits_824_snr/fits_{muse,mpsenet,demucs}_snr_jobB.csv`,
`tools/c50_824_per_layer_supp.csv`,
`tools/hierarchical_bootstrap_snr_cluster.csv`, and the six-arm sweep parquets at
athena `~/tmp/c50_ft_arms/results/`.

I have not edited the manuscript or the supplement.
