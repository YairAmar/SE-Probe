# Cross-task agreement

Noise-axis profiles are the 18-noise fits in `snr/fits_*_snr_jobB.csv`; C50 profiles are the pretrained and fine-tuned arms in `c50/c50_824_per_layer_supp.csv`. `cross_task_values.csv` holds the same numbers.

inputs: `snr/`, `c50/c50_824_per_layer_supp.csv`, `snr/hierarchical_bootstrap_snr_cluster_probed18.csv`

| model | C50 arm | n | profile | Spearman | 95% CI | p | Pearson | 95% CI | p | LOO Spearman range |
|---|---|---|---|---|---|---|---|---|---|---|
| MUSE | ft | 24 | beta | +0.837 | [+0.580, +0.941] | 3.5e-07 | +0.877 | [+0.790, +0.944] | 1.8e-08 | [+0.814, +0.876] |
| MUSE | ft | 24 | alpha | +0.864 | [+0.664, +0.957] | 5.2e-08 | +0.881 | [+0.795, +0.946] | 1.4e-08 | [+0.846, +0.896] |
| MP-SENet | ft | 8 | beta | +0.476 | [-0.538, +1.000] | 0.23 | +0.676 | [-0.265, +0.967] | 0.066 | [+0.214, +0.964] |
| MP-SENet | ft | 8 | alpha | +0.833 | [+0.300, +1.000] | 0.01 | +0.776 | [+0.188, +0.978] | 0.024 | [+0.750, +0.964] |
| Demucs | ft | 11 | beta | +0.482 | [-0.224, +0.917] | 0.13 | +0.435 | [-0.244, +0.830] | 0.18 | [+0.309, +0.588] |
| Demucs | ft | 11 | alpha | +0.518 | [-0.228, +0.917] | 0.1 | +0.498 | [-0.426, +0.882] | 0.12 | [+0.358, +0.648] |
| MUSE | pre | 24 | beta | +0.797 | [+0.520, +0.932] | 3.1e-06 | +0.823 | [+0.689, +0.922] | 8.1e-07 | [+0.770, +0.861] |
| MUSE | pre | 24 | alpha | +0.935 | [+0.813, +0.977] | 2.3e-11 | +0.887 | [+0.827, +0.946] | 8.1e-09 | [+0.926, +0.955] |
| MP-SENet | pre | 8 | beta | +0.905 | [+0.241, +1.000] | 0.002 | +0.779 | [+0.599, +0.990] | 0.023 | [+0.857, +0.964] |
| MP-SENet | pre | 8 | alpha | +0.976 | [+0.730, +1.000] | 3.3e-05 | +0.844 | [+0.772, +0.992] | 0.0084 | [+0.964, +1.000] |
| Demucs | pre | 11 | beta | +0.545 | [-0.140, +0.934] | 0.083 | +0.549 | [-0.066, +0.878] | 0.08 | [+0.394, +0.661] |
| Demucs | pre | 11 | alpha | +0.245 | [-0.617, +0.827] | 0.47 | +0.415 | [-0.610, +0.842] | 0.2 | [-0.006, +0.358] |

## Other statistics

- Demucs|SNR|r(beta,|depth-5|) [r, lo, hi, p]: [0.42722691801043394, -0.23227150004830202, 0.8236365790098027, 0.18999639985714423]
- Demucs|SNR|spearman(beta,|depth-5|) [rho, p]: [0.4552024849729859, 0.15946586509159283]
- Demucs|C50-ft|r(beta,|depth-5|) [r, lo, hi, p]: [0.99010392620009, 0.979473241668836, 0.997194330824437, 5.5059540727744065e-09]
- Demucs|C50-ft|spearman(beta,|depth-5|) [rho, p]: [0.9885710532241612, 1.0503169230921145e-08]
- Demucs|C50-pre|r(beta,|depth-5|) [r, lo, hi, p]: [0.977581498733404, 0.9427666243484398, 0.9943053973476207, 2.143605425558591e-07]
- Demucs|C50-pre|spearman(beta,|depth-5|) [rho, p]: [0.9885710532241612, 1.0503169230921145e-08]
- MUSE|SNR|rho(depth,beta) [rho, p]: [0.62, 0.0012308328706616464]
- MUSE|SNR|rho(depth,alpha) [rho, p]: [-0.7408695652173911, 3.4620205532400836e-05]
- MUSE|SNR|r(alpha,beta): -0.9495742684024162
- MUSE|C50-ft|rho(depth,beta) [rho, p]: [0.6147826086956522, 0.0013898820033517755]
- MUSE|C50-ft|rho(depth,alpha) [rho, p]: [-0.5930434782608696, 0.0022562058410387437]
- MUSE|C50-ft|r(alpha,beta): -0.9931814056658324
- MUSE|C50-pre|rho(depth,beta) [rho, p]: [0.7191304347826086, 7.504696733116017e-05]
- MUSE|C50-pre|rho(depth,alpha) [rho, p]: [-0.7999999999999999, 2.70612994392676e-06]
- MUSE|C50-pre|r(alpha,beta): -0.9822638806604808
- MUSE|rho(C50-pre profile, C50-ft profile)|beta: 0.8147826086956521
- MUSE|rho(C50-pre profile, C50-ft profile)|alpha: 0.8365217391304347
- MP-SENet|SNR|rho(depth,beta) [rho, p]: [0.8333333333333335, 0.010175540123456752]
- MP-SENet|SNR|rho(depth,alpha) [rho, p]: [-0.9285714285714287, 0.0008629681828999767]
- MP-SENet|SNR|r(alpha,beta): -0.9882071819383567
- MP-SENet|C50-ft|rho(depth,beta) [rho, p]: [0.5714285714285715, 0.13895995716070672]
- MP-SENet|C50-ft|rho(depth,alpha) [rho, p]: [-0.7619047619047621, 0.028004939153071805]
- MP-SENet|C50-ft|r(alpha,beta): -0.9986608484413826
- MP-SENet|C50-pre|rho(depth,beta) [rho, p]: [0.880952380952381, 0.0038503204637324005]
- MP-SENet|C50-pre|rho(depth,alpha) [rho, p]: [-0.880952380952381, 0.0038503204637324005]
- MP-SENet|C50-pre|r(alpha,beta): -0.9994941731292097
- MP-SENet|rho(C50-pre profile, C50-ft profile)|beta: 0.7619047619047621
- MP-SENet|rho(C50-pre profile, C50-ft profile)|alpha: 0.9047619047619048
- Demucs|SNR|rho(depth,beta) [rho, p]: [-0.6000000000000001, 0.05100326070695074]
- Demucs|SNR|rho(depth,alpha) [rho, p]: [0.3090909090909091, 0.35502843970150694]
- Demucs|SNR|r(alpha,beta): -0.7517433057515291
- Demucs|C50-ft|rho(depth,beta) [rho, p]: [0.009090909090909092, 0.9788365001441937]
- Demucs|C50-ft|rho(depth,alpha) [rho, p]: [-0.20909090909090913, 0.5372209352113227]
- Demucs|C50-ft|r(alpha,beta): -0.9179014334034334
- Demucs|C50-pre|rho(depth,beta) [rho, p]: [-0.045454545454545456, 0.8944269969053026]
- Demucs|C50-pre|rho(depth,alpha) [rho, p]: [-0.5272727272727273, 0.09556521835315425]
- Demucs|C50-pre|r(alpha,beta): -0.7548902024500906
- Demucs|rho(C50-pre profile, C50-ft profile)|beta: 0.990909090909091
- Demucs|rho(C50-pre profile, C50-ft profile)|alpha: 0.8727272727272729
- Demucs|SNR|mean beta enc/lstm/dec: [0.02476348739064048, 0.00429010493009, 0.01194806558066098]
- Demucs|SNR|rho(depth,beta) within encoder / decoder: [-0.09999999999999999, 0.9999999999999999]
- Demucs|C50-ft|mean beta enc/lstm/dec: [0.01768463803282538, 0.0041886405197365, 0.01773988439381764]
- Demucs|C50-ft|rho(depth,beta) within encoder / decoder: [-0.9999999999999999, 0.9999999999999999]
- Demucs|C50-pre|mean beta enc/lstm/dec: [0.019836804755367136, 0.0047496324440407, 0.017744669681958557]
- Demucs|C50-pre|rho(depth,beta) within encoder / decoder: [-0.9999999999999999, 0.9999999999999999]
- Demucs|SNR|alpha max, runner-up [layer, alpha, lo, hi]: [['lstm', 0.8775385130584834, 0.8537044829754642, 0.8980676641343961], ['decoder.0', 0.7145353008207627, 0.6768561190143713, 0.7499204600226385]]
- Demucs|SNR|beta min, lstm rank: ['lstm', 1]
- Demucs|SNR|beta CIs [layer, beta, lo, hi] for the minimum and lstm: [['lstm', 0.00429010493009, 0.0034858359241466, 0.0052640459874182], ['lstm', 0.00429010493009, 0.0034858359241466, 0.0052640459874182]]
