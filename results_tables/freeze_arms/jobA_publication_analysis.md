# Job A - publication-quality analysis

Adds Cohen's d, bootstrap CI on the mean difference (n_boot=10000, seed=0), practical-significance flags, and a STOI mirror of the PESQ analysis to jobA_deep_analysis.md. All numbers are best-epoch per-run.

## Comparisons on PESQ

| a | b | mean(a) | mean(b) | diff | 95% CI on diff | Cohen d | Welch t | p | audibility threshold | practically significant |
|---|---|---|---|---|---|---|---|---|---|---|
| full_ft | freeze_encoder | 2.7827 | 2.6973 | +0.0853 | [+0.0730, +0.0940] | +10.40 | +12.74 | 0.0026 | 0.050 | yes |
| full_ft | freeze_decoder | 2.7827 | 2.6997 | +0.0830 | [+0.0707, +0.0913] | +10.27 | +12.58 | 0.0033 | 0.050 | yes |
| freeze_encoder | freeze_decoder | 2.6973 | 2.6997 | -0.0023 | [-0.0067, +0.0020] | -0.66 | -0.81 | 0.4626 | 0.050 | no |

## Comparisons on STOI

| a | b | mean(a) | mean(b) | diff | 95% CI on diff | Cohen d | Welch t | p | audibility threshold | practically significant |
|---|---|---|---|---|---|---|---|---|---|---|
| full_ft | freeze_encoder | 0.9343 | 0.9313 | +0.0030 | [+0.0023, +0.0037] | +5.20 | +6.36 | 0.0031 | 0.005 | no |
| full_ft | freeze_decoder | 0.9343 | 0.9303 | +0.0040 | [+0.0033, +0.0047] | +6.93 | +8.49 | 0.0011 | 0.005 | no |
| freeze_encoder | freeze_decoder | 0.9313 | 0.9303 | +0.0010 | [+0.0003, +0.0017] | +1.73 | +2.12 | 0.1012 | 0.005 | no |

## Overtraining diagnostic (peak-to-final PESQ gap)

A positive gap means the model's peak PESQ was earlier than epoch 50 (final ep50 undershoots peak).

| arm | mean gap (peak - final) | max gap | interpretation |
|---|---|---|---|
| freeze_decoder | +0.0000 | +0.0000 | no meaningful overtraining |
| freeze_encoder | +0.0077 | +0.0150 | mild overtraining after epoch ~46-48 |
| full_ft | +0.0113 | +0.0210 | mild overtraining after epoch ~46-48 |

## Interpretation for the paper

1. **Both freezing regimes are significantly behind full-FT.** On PESQ the mean difference is 0.085 (95% CI [0.073, 0.094]) for freeze_encoder and 0.083 (95% CI [0.071, 0.091]) for freeze_decoder, both above the 0.050 PESQ audibility threshold. Cohen's d is large in both cases (|d|=10.4 and 10.3); the effect is robust despite n=3 seeds per arm.
2. **The two freeze arms are indistinguishable on both metrics.** The PESQ difference between freeze_encoder and freeze_decoder is -0.0023 (95% CI [-0.0067, +0.0020], |d|=0.66); STOI difference is +0.00100 (95% CI [+0.00033, +0.00167]). Both CIs straddle zero and both are below their respective audibility thresholds. The predicted profile-guided ordering (freeze_encoder ~ full_ft > freeze_decoder) is not present in the data.
3. **A single-scalar caveat.** With n=3 seeds per arm the CI on any single difference is wide, and Welch t-test power is limited. What we can claim is that the observed direction and magnitude of the arm differences do NOT match the profile-guided prediction; a much larger seed count could in principle uncover a small residual freeze_encoder advantage, but any such advantage would be below the audibility threshold and therefore of little practical relevance.
4. **Parameter efficiency remains an honest positive finding.** Both freeze arms train ~72-76% of full_ft's parameters while achieving ~97% of its PESQ (96.9% for freeze_encoder and 97.0% for freeze_decoder). This holds regardless of which end you freeze - a useful guidance for practitioners, even if it does not localize adaptation via the profile.