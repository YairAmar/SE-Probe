<!-- generated 2026-08-03 20:47:05 by analyze_jobA_probe.py, stage=stage2_full -->

### 6.3 Per-arm profiles

**Probes analysed: 10 of 10.**

| probe | present |
|---|---|
| `pre_ft` | yes |
| `full_ft__seed1234` | yes |
| `full_ft__seed2345` | yes |
| `full_ft__seed3456` | yes |
| `freeze_encoder__seed1234` | yes |
| `freeze_encoder__seed2345` | yes |
| `freeze_encoder__seed3456` | yes |
| `freeze_decoder__seed1234` | yes |
| `freeze_decoder__seed2345` | yes |
| `freeze_decoder__seed3456` | yes |

#### Pipeline validation against the published artifact

**REPRODUCED**

| quantity | value |
|---|---|
| per-layer Pearson *r* (AUC) | 1.0000000000 |
| per-layer Pearson *r* (`c_low`) | 1.0000000000 |
| max abs deviation, AUC | 0.000e+00 |
| max abs deviation, `c_low` | 0.000e+00 |
| max abs deviation, beta | 0.000e+00 |
| utterances (mine / manuscript) | 780 / 780 |

#### Stage-level normalized AUC

| stage | pre-FT | full_ft | freeze_encoder | freeze_decoder |
|---|---|---|---|---|
| `encoder_level1` | 0.9976 | 0.9974 | 0.9985 (frozen) | 0.9972 |
| `encoder_level2` | 0.9670 | 0.9710 | 0.9913 (frozen) | 0.9497 |
| `latent` | 0.9621 | 0.9765 | 0.9759 | 0.9609 |
| `decoder_level2` | 0.9119 | 0.9602 | 0.9514 | 0.9407 (frozen) |
| `decoder_level1` | 0.9063 | 0.9503 | 0.9586 | 0.9639 (frozen) |
| `mag_refinement` | 0.9069 | 0.9651 | 0.9515 | 0.9509 |

#### Layers whose profile differs, by contrast

Count of the 24 probed layers whose paired bootstrap CI on delta-AUC excludes zero.

| contrast | layers differing (of 24) |
|---|---|
| `freeze_decoder__seed1234` vs `pre_ft` | 24 |
| `freeze_decoder__seed3456` vs `pre_ft` | 24 |
| `freeze_encoder__seed3456` vs `pre_ft` | 24 |
| `freeze_decoder__seed3456` vs `full_ft__seed3456` | 24 |
| `freeze_encoder__seed1234` vs `pre_ft` | 24 |
| `freeze_encoder__seed1234` vs `freeze_decoder__seed1234` | 24 |
| `freeze_encoder__seed3456` vs `full_ft__seed3456` | 24 |
| `freeze_encoder__seed3456` vs `freeze_decoder__seed3456` | 24 |
| `freeze_encoder__seed2345` vs `pre_ft` | 24 |
| `freeze_encoder__seed2345` vs `full_ft__seed2345` | 24 |
| `full_ft__seed2345` vs `pre_ft` | 24 |
| `freeze_decoder__seed1234` vs `full_ft__seed1234` | 23 |
| `freeze_decoder__seed2345` vs `full_ft__seed2345` | 23 |
| `freeze_encoder__seed1234` vs `full_ft__seed1234` | 23 |
| `full_ft__seed1234` vs `pre_ft` | 23 |
| `freeze_decoder__seed2345` vs `pre_ft` | 21 |
| `full_ft__seed3456` vs `pre_ft` | 21 |
| `freeze_encoder__seed2345` vs `freeze_decoder__seed2345` | 20 |

#### Seed noise versus arm effect (normalized AUC)

| quantity | value |
|---|---|
| mean within-arm across-seed SD, `freeze_decoder` | 0.00319 |
| mean within-arm across-seed SD, `freeze_encoder` | 0.00278 |
| mean within-arm across-seed SD, `full_ft` | 0.00282 |
| mean across-arm SD per layer | 0.00983 |

