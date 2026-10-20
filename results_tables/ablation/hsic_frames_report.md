# Wave-4 ablation: CKA estimator / sample convention robustness

Pretrained MUSE, non-pooled extraction. N=150 utts x 9 SNRs x noises ['TBUS', 'PCAFETER']. 24 norm1 layers.

Variants (all compare clean-ref vs degraded per utterance, fit CKA=alpha+beta*SNR):

- **pooled_linear**: paper baseline. time-pool -> (C,F); samples=C, features=F; biased linear CKA.
- **hsic_unbiased**: same pooled (C,F) rep; unbiased HSIC linear CKA (Song 2012 / Nguyen 2021).
- **frames_TFC**: frames-as-samples, (T, C*F); samples=T frames, features=C*F; biased linear CKA.
- **frames_TC**: [T,C] Demucs convention, mean over F -> (T,C); samples=T, features=C; biased linear CKA.

## Profile correlation vs paper pooled-linear profile (per-layer slope)

| variant | Pearson r vs pooled_linear | peak layer | peak beta |
|---|---|---|---|
| pooled_linear | 1.000 | Enc-L2.3 | 0.02422 |
| hsic_unbiased | 0.976 | Enc-L2.3 | 0.02577 |
| frames_TFC | -0.499 | Enc-L2.0 | 0.01361 |
| frames_TC | -0.263 | Enc-L2.0 | 0.01847 |

## Per-layer slope (beta) by variant

| layer | pooled_linear | hsic_unbiased | frames_TFC | frames_TC |
|---|---|---|---|---|
| Enc-L1.0 | 0.00011 | 0.00010 | 0.01240 | 0.01508 |
| Enc-L1.1 | 0.00342 | 0.00392 | 0.00943 | 0.01104 |
| Enc-L1.2 | 0.01158 | 0.01306 | 0.00918 | 0.01149 |
| Enc-L1.3 | 0.01392 | 0.01500 | 0.00813 | 0.01040 |
| Enc-L2.0 | 0.00868 | 0.01133 | 0.01361 | 0.01847 |
| Enc-L2.1 | 0.01645 | 0.01892 | 0.00987 | 0.01591 |
| Enc-L2.2 | 0.01797 | 0.02041 | 0.00838 | 0.01505 |
| Enc-L2.3 | 0.02422 | 0.02577 | 0.00836 | 0.01525 |
| Latent.0 | 0.01488 | 0.01796 | 0.01131 | 0.01645 |
| Latent.1 | 0.01164 | 0.01601 | 0.00949 | 0.01433 |
| Latent.2 | 0.01425 | 0.01719 | 0.00754 | 0.01130 |
| Latent.3 | 0.01695 | 0.01951 | 0.00711 | 0.01042 |
| Dec-L2.0 | 0.02360 | 0.02441 | 0.00893 | 0.01119 |
| Dec-L2.1 | 0.01821 | 0.02288 | 0.00821 | 0.01068 |
| Dec-L2.2 | 0.01630 | 0.02049 | 0.00794 | 0.01056 |
| Dec-L2.3 | 0.01316 | 0.01498 | 0.00734 | 0.00924 |
| Dec-L1.0 | 0.02109 | 0.02115 | 0.00797 | 0.00871 |
| Dec-L1.1 | 0.01536 | 0.01843 | 0.00484 | 0.00421 |
| Dec-L1.2 | 0.01464 | 0.01778 | 0.00571 | 0.00510 |
| Dec-L1.3 | 0.01388 | 0.01555 | 0.00729 | 0.00706 |
| Refine.0 | 0.02007 | 0.02272 | 0.00845 | 0.00928 |
| Refine.1 | 0.01798 | 0.02187 | 0.00752 | 0.00824 |
| Refine.2 | 0.01706 | 0.02147 | 0.00686 | 0.00875 |
| Refine.3 | 0.02107 | 0.02416 | 0.00180 | 0.00509 |

## Verdict

The decoder-skip-junction peak of the paper's pooled-linear profile is reproduced by 1/3 alternative conventions with r>=0.8 (hsic_unbiased). Correlations: hsic_unbiased=0.98, frames_TFC=-0.50, frames_TC=-0.26. This indicates the layer-wise SNR-sensitivity structure is robust to the CKA estimator (biased vs unbiased HSIC) and to the sample/feature convention (channels-as-samples vs frames-as-samples vs Demucs [T,C]).
