# results_tables: the small tables behind every number in the TASLP manuscript

Every file here is a byte-for-byte copy of an analysis output produced on the Technion
cluster (athena, `/rg/iscohen_prj/yairamr/cluster-2026-07/` and `~/tmp/`) or in the closed
analysis repo (`SE-probing-TASLP`). `MANIFEST.csv` lists each file with its size, SHA-256
and the exact source path it was copied from. Nothing was edited or re-derived; the raw
per-utterance parquets these summarise are several GB and are not shipped in-tree.

The manuscript is *Probing Layer-Wise Robustness and Sensitivity of Speech Enhancement
Models Under Noise and Reverberation* (Amar, Ivry, Cohen; IEEE TASLP). Section and figure
labels below refer to its `main.tex` (`fig:regression`, `tab:tradeoff`, ...).

## Conventions every table follows

* **Probed layers.** MUSE: the 24 `.norm1` hooks (first LayerNorm of each transformer
  layer) out of the 72 raw hooks. MP-SENet: the 8 `.norm1` hooks of
  `TSTransformer.{0..3}.{time,freq}_transformer`. Demucs: the 11 block outputs
  `encoder.0-4`, `lstm`, `decoder.0-4`.
* **Depth order is architectural, never alphabetical.** MUSE: stage rank
  `encoder_level1, encoder_level2, latent, decoder_level2, decoder_level1, mag_refinement`,
  then block, then layer index. MP-SENet: block index, time before freq. Demucs: the explicit
  list above. Sorting layer names alphabetically gives the wrong depth order (rank correlation
  with true depth is -0.50 for Demucs) and produced wrong statistics once; every `depth` column
  here was asserted against the architectural order.
* **Fit.** Mean CKA per (layer, level) over utterances and noises/RIRs (utterance-first
  averaging for the C50 axis, so each utterance's five RIRs are averaged before the
  across-utterance mean), then per-layer OLS `CKA = alpha + beta * s`, `s` in dB. `alpha` is the
  value at 0 dB; `beta` the slope.
* **Summary statistics.** `auc` (or `A`) is the normalized trapezoidal area
  `np.trapezoid(curve, levels) / (levels[-1] - levels[0])`, not the mean of the points.
  `c_low`/`c_high` are the empirical mean CKA at the most severe and the mildest swept level,
  never fitted values. `rng = c_high - c_low`.
* **Saturation spread.** `f = sd(c_high, ddof=1) / sd(alpha, ddof=1)` (column
  `sat_freedom`). `pc1`/`pc2` are SVD variance shares of `M - M.mean(axis=0)`, where `M` is the
  layer x level matrix of mean CKA and the mean is taken over layers at each level (centering
  each layer's own curve is a different matrix and is wrong). `identity_r = r(beta,
  (c_high - alpha) / R)`; `r_alpha_chigh` is the `r_c` of Eq. (identity).
* **Grids.** SNR: 41 integer levels, -10..30 dB. C50: 13 levels `np.arange(-5, 27.5, 2.5)`,
  reference condition C50 = 50 dB. The wide window `linspace(-20, 50, 14)` appears only in the
  window-stability ablation.
* **Utterances.** 824 = the VoiceBank-DEMAND test split, speakers p232 (393 utterances,
  `clean_idx` 0..392) and p257 (431, 393..823). 780 = VCTK p226 (356) + p287 (424), the
  earlier sweep; the two sets differ in speakers, not only in size.
* **Noises.** 18 DEMAND recordings. The held-out five that form the VoiceBank-DEMAND test
  split are `TBUS, SCAFE, DLIVING, OOFFICE, SPSQUARE` (after the SCAFE swap; the earlier five
  had PCAFETER, a training noise, in place of SCAFE). The eight training-split noises are
  `DKITCHEN, OMEETING, PCAFETER, PRESTO, PSTATION, TCAR, TMETRO, STRAFFIC`; the five in
  neither split are `DWASHING, NFIELD, NPARK, NRIVER, OHALLWAY`. Files tagged `old5`/`new5`
  use the PCAFETER/SCAFE five respectively; the manuscript quotes `new5`.
* **RIR pool.** 88 AIR RIRs from six rooms (lecture 32, meeting 24, office 18, corridor 6,
  bathroom 4, kitchen 4), five RIRs drawn per utterance with `default_rng(42)`. The rooms
  `aula_carolina`, `booth`, `stairway` are training-split rooms and never appear.
* **Checkpoints.** MUSE pretrained `g_best` (md5 `8fd0ec3d...`); MUSE dereverb FT epoch 48
  `g_00051852` (`fbac7b79...`); MP-SENet pretrained `JacobLinCool/MP-SENet-DNS`, FT
  `all_v2/epoch_50/g_00295909`; Demucs pretrained `dns64()`, FT v2 `epoch_57/g_00047310`.
* **Seeds.** `0` for the hierarchical utterance x noise bootstrap (one `default_rng(0)` threaded
  through every model and layer in `sorted()` order, 1000 replicates); `20260802` for the
  utterance bootstraps on the C50 axis and the AUC CIs; `20260806` for the FT-arm C50 profile
  CIs; `20260803` for the speaker-level analysis (2000 cluster-bootstrap replicates);
  `20260810` for the layer-resample CIs of the cross-axis correlations.

## snr/ : additive-noise axis (Sec. III-A, Fig. `fig:cka_heatmap`, `fig:regression`, Table `tab:tradeoff` SNR rows)

Source sweep: Job B, 824 utterances x 41 SNR x 18 DEMAND noises, three pretrained models,
re-aggregated over all 18 noises by `tools/reagg18/reagg18.py` (the 17-noise control first
reproduced the earlier shipped fits to 1e-15).

| file | columns | backs |
|---|---|---|
| `fits_{muse,mpsenet,demucs}_snr_jobB.csv` | `layer, depth, alpha, beta, r2, c_low, c_high, rng, auc` | per-layer profile; `R^2 >= 0.96` (MUSE), Demucs min `R^2` 0.441 at `decoder.3`, 0.094 CKA span; Fig. `fig:regression` point estimates; Fig. `fig:scatter`(a) |
| `summary_jobB.csv` | `dataset, model, n_layers, n_levels, r_alpha_beta, sat_freedom, identity_r, pc1, pc2, r_auc_alpha, r_auc_clow, r2_min, r2_median, c_low_min/max, c_high_min/max, auc_mean, peak_auc_layer, rho_depth_range, rho_depth_auc, rho_alpha_beta, r_alpha_chigh` | Table `tab:tradeoff` SNR rows (`-0.950/-0.891/0.275/97.54`, `-0.988/-0.929/0.137/98.48`, `-0.877/-0.782/0.956/76.32`) and the `r_c` values `+0.823, +0.853, -0.327` |
| `hierarchical_bootstrap_snr_cluster_all18.csv` | `model_name, layer, alpha, beta, alpha_ci_lo/hi, beta_ci_lo/hi, c_low, c_high, n_boot, seed` | 168 hooked layers (72 + 56 + 40), hierarchical utterance x noise bootstrap, 1000 reps, seed 0, exact replay of the shipped RNG stream (`reagg18c.py`) |
| `hierarchical_bootstrap_snr_cluster_probed18.csv` | same columns | the same CIs restricted to the probed layers (24 + 8 + 11), as read by the figure scripts; copied from `tools/hierarchical_bootstrap_snr_cluster.csv` |
| `muse_profile_values_18_cluster.csv` | `layer, alpha, alpha_ci_lo/hi, beta, beta_ci_lo/hi, r2` | the 24 MUSE values plotted in Fig. `fig:regression` with their CI bands |
| `depth_profiles_two_models_values.csv` | `model, axis, depth, layer, alpha, alpha_lo/hi, beta, beta_lo/hi` | MP-SENet and Demucs profiles on both axes (`rho(depth, beta) = +0.833` MP-SENet noise, `-0.727` Demucs noise) |
| `cka_heatmap_values_raw_jobB18.csv` | `layer, snr, mean, count` (72 layers x 41 SNR, count 14,832 = 824 x 18) | Fig. `fig:cka_heatmap` (the 24 `.norm1` rows) |
| `per_noise_beta_muse.csv` | index = 24 MUSE layers; one `beta` column per DEMAND noise | the noise-set independence paragraph: averaging the five test-split columns against the eight training-split columns reproduces `r = 0.9812`, `rho = 0.9757`, same peak layer Dec-L2.0, and mean `beta` `0.0145 / 0.0136 / 0.0135` (test / train / neither) exactly |
| `IIIA_independence_18.csv` | `contrast, n_held, n_indist, pearson_r, spearman_rho, mean_beta_held, mean_beta_indist` | the two 5-vs-rest contrasts (`old5_vs_12`: 0.9904; `new5_vs_13`: 0.9880); the manuscript's 5-vs-8 contrast is derived from `per_noise_beta_muse.csv` as above |
| `IIIA_noise_set_independence.json`, `IIIA_repro_5old_vs_12_muse.csv`, `IIIA_swap.json` | cluster-side reproduction of the old 5-vs-12 contrast (72-layer and 24-layer variants) and the SCAFE-swap check on Demucs (`refit_IIIA.py`, `refit_IIIA_swap.py`) | provenance of the independence numbers; run before SCAFE was staged, so they carry no 5-vs-8 row |
| `moved_numbers.csv` | `model, quantity, old17, new18, delta` | every reported quantity before/after adding SCAFE (17 -> 18 noises); nothing moves beyond the third decimal |
| `peak_layer_stability_snr.csv`, `peak_layer_stability_c50.csv` | `axis, model, quantity, layer, wins, frequency` | bootstrap frequency (300 utterance x cluster resamples, seed 0) with which each layer is the peak-`beta` / min-`c_low` / max-`c_high` layer (`scripts/peak_stability.py`); the C50 file is from the superseded 5-RIR Job B2 arms |
| `profile_correlations.csv` | `pearson_r, pearson_p, spearman_r, spearman_p, n, label` | early cross-axis / cross-model profile correlations on the 17-noise aggregate (`scripts/profile_correlations.py`); superseded by `cross_task/` for the paper's numbers |
| `refit_5noise_824/` | `fits_snr_5noise_824.csv` (`model_name, layer, alpha, beta, r2, n_levels`), `hierarchical_bootstrap_snr_5noise_824.csv` (+ CIs, `c_low, c_high, n_boot, seed`), `endpoints_snr_5noise_824.csv` | the same sweep restricted to the old held-out five (PCAFETER), all hooked layers; `migration_824/refit_5noise.py`, 1000 reps, seed 0. Kept for provenance of the 5-noise analyses (random init, emergence, diffusion) |

## c50/ : reverberation axis (Sec. III-C, III-E, Fig. `fig:c50_regression`, `fig:scatter`(b), Table `tab:tradeoff` C50 rows)

Source sweep: job 128939, 824 utterances x 13 target C50 x 5-of-88 AIR RIRs per utterance,
six arms (`{muse,mpsenet,demucs}_{pre,ft}`) all probed under one software stack. Table
`tab:tradeoff`'s C50 rows are the **fine-tuned** arms.

| file | columns | backs |
|---|---|---|
| `c50_824_per_layer_supp.csv` | `arm, model, depth, label, layer, A, A_lo, A_hi, c_low, c_low_lo/hi, c_high, alpha, beta, r2, spk_abs_delta_A` (bootstrap over 824 utterances, 1000 reps, seed 20260802) | Table `tab:tradeoff` C50 rows (recomputing `r(alpha,beta)`, `rho`, `f`, `r_c` from the `*_ft` rows gives `-0.993/-0.996/0.037/+0.860`, `-0.999/-0.905/0.025/+0.918`, `-0.918/-0.927/0.167/+0.584`); Sec. III-E before/after correlations (`beta`: 0.91 / 0.81 / 0.98, `alpha`: 0.85 / 0.84 / 0.90; mean abs change in `beta` 0.0020 / 0.0029 / 0.0015); Demucs LSTM `alpha = 0.8986` vs runner-up `encoder.4` `0.7444`; `spk_abs_delta_A` = |A(p232) - A(p257)| per layer |
| `fig_v5_c50_824_ft_profiles_values.csv` | `model, layer, depth, alpha, alpha_lo/hi, beta, beta_lo/hi, n_utts, n_levels, n_boot, seed` (seed 20260806) | Fig. `fig:c50_regression` (MUSE FT arm with CI bands), Fig. `fig:scatter`(b), the MP-SENet/Demucs C50 panels of `snr/depth_profiles_two_models_values.csv` |
| `fig_v5_c50_824_profiles_values.csv` | same columns (seed 20260802) | the **pretrained** arms of job 128105 (MUSE `g_best`, MP-SENet DNS; its Demucs arm is the FT v2 checkpoint) |
| `per_layer_c50.csv` | `scale, model, layer, depth, auc, auc_lo/hi, auc_se, c_low, c_low_lo/hi, c_low_se, n_utts, n_levels, n_boot, seed` | AUC and `c_low` with utterance-bootstrap CIs at 824 and at 780 (`analyze_c50_824.py`, job 128105 arms) |
| `mean_curves_c50.csv` | `scale, model, layer` + one column per C50 level `-5 ... 25` | the per-layer mean level curves of job 128105 |
| `demucs_bottleneck.csv` | `scale, model, lstm_auc, lstm_lo/hi, runner_up, runner_auc, gap, gap_lo/hi, gap_mean, gap_sd, p_gap_gt0, p_peak_is_lstm, n_boot, seed` | Demucs bottleneck: LSTM is the AUC arg-max in 1000/1000 resamples at both scales (the manuscript quotes the same 1000/1000 on `alpha`/`beta`, gap `+0.154 [+0.151, +0.158]`, computed on the FT arm) |
| `profile_shape_stats_c50.csv` | `scale, model, n_layers, r_alpha_beta, sat_freedom, pc1_var, sd_c_high, sd_alpha, auc_min` | shape statistics of the job 128105 arms at 824 and 780 (Demucs 824 row = the FT arm, `pc1 = 83.58`) |
| `per_speaker_824.csv` | `model, speaker, layer, depth, auc, c_low, n_utts` | AUC and `c_low` recomputed within p232 and within p257 separately |
| `fig_c50_824_finetuning_profile_values.csv` | `model, pre, ft, n_layers, pearson, spearman, mean_abs_delta, max_abs_delta, max_delta_layer, peak_layer_pre/ft, peak_auc_pre/ft, max_ci_halfwidth` | pre vs FT agreement of the AUC profile (Demucs, MP-SENet) |
| `fig_c50_ft_beta_profiles_values.csv` | `model, n_layers, pearson_r_beta, mean_abs_dbeta, max_abs_dbeta, max_dbeta_layer, peak_beta_layer_pre/ft, peak_beta_pre/ft` | pre vs FT agreement of the `beta` profile, all three models (Sec. III-E) |
| `window_stability.csv`, `window_cross_sweep.csv`, `window_s9_rows.csv` | `dataset, statistic, n, pearson, spearman, med_rel_wide_ref, med_rel_narrow_ref` / `row, n, r_A_clow, r_A_alpha, r_A_chigh, rho_A, rho_clow, rho_chigh` | window-stability ablation (wide 14-level `linspace(-20, 50, 14)` vs the paper window), 72 MUSE hooks, `tools/window_ablation_824.py`; supplement Tables S9/S10 |
| `auc_bootstrap_ci.csv` | `dataset, layer, depth, auc, auc_fits, lo, hi, se, n_units, n_levels, n_boot, seed, auc_ref_maxdev` | 780-scale AUC CIs (300 reps, seed 20260802) for Demucs C50 v2/pretrained, Demucs SNR, MP-SENet C50 pre/FT (`tools/bootstrap_auc_ci.py`) |
| `refit_5rir_824/` | `fits_reverb_5rir_824.csv` (`model_name, layer, alpha, beta, r2, n_levels`), `hierarchical_bootstrap_reverb_5rir_824.csv` (+ CIs) | **superseded.** Job B2 probed `MUSE_reverbFT_e48` and `MP-SENet` on five RIRs that all came from `aula_carolina`, a training-split room, and the "MUSE_reverbFT_e48" arm was later found (md5) to be the pretrained `g_best`. Kept only so the earlier numbers can be traced; use `c50_824_per_layer_supp.csv` |

## random_init/ : untrained-MUSE control (Sec. III-D, Fig. `fig:random_init`)

Source: `migration_824/random_init_grid_824.py`, 824 utterances x 41 SNR x 5 held-out noises
x 3 seeds, MUSE instantiated from the paper config with `torch.manual_seed(seed)` and no
checkpoint; refit by `scripts/scafe/refit_IIIE.py`.

| file | columns | backs |
|---|---|---|
| `fig_v6_random_init_824_values.csv` | `depth, beta_s0, beta_s1, beta_s2, beta_mean, beta_sd, cka_min, cka_mean` (SCAFE five; fingerprint `max |beta_s0| = 0.000284817601`) | Fig. `fig:random_init`; per-seed maxima `0.00028, 0.00105, 0.00045` |
| `IIIE_random_init.json` | per noise set (`OLD five (PCAFETER)`, `NEW five (SCAFE)`): `min_cka`, per-seed max `beta` and peak layer, `rho_depth_beta` per seed (Spearman and Pearson with p), `trained_peak_beta`, `factor_trained_over_largest_seed`, profile correlation untrained vs trained | min CKA `0.9802`, trained peak `0.02261` at Dec-L2.0, factor `21.5`, `rho(depth, beta) = +0.43, +0.86, -0.07` (the `NEW five` entry) |

## emergence/ : profile along the MUSE dereverberation fine-tune (Sec. III-D)

Source: `migration_824/emergence_ckpt_grid.py`, epochs 1, 3, 5, 7, 9, 10, 12, 24, 36, 48 of
the MUSE reverb fine-tune probed on the noise axis (824 x 41 x 5 noises); the epoch-48 arm
was re-probed on the correct `g_00051852` checkpoint (`e48_fix`). Refit per noise set by
`scripts/scafe/refit_IIIF.py`.

| file | columns | backs |
|---|---|---|
| `IIIF_per_layer_profiles_{new5,old5}.csv` | `layer, block, beta_pre, alpha_pre, beta_e{E}, alpha_e{E}` for the ten epochs | the per-epoch (`alpha`, `beta`) profiles |
| `IIIF_convergence_{new5,old5}.csv` | `epoch, beta_r_vs_e48, alpha_r_vs_e48` (epoch -1 = pretrained) | `r = 0.76` at epoch 1 rising to `>= 0.99` by epoch 36; pretrained `r = 0.76` (`beta`) and `0.73` (`alpha`) |
| `IIIF_emergence.json` | both noise sets: `convergence`, `concentration` (mean abs change decoder+refinement over Enc-L1: `ratio_dbeta_decref_over_encl1`, `ratio_dalpha_decref_over_encl1`), `headline` | the "3x in `beta`, 5.8x in `alpha`" concentration (new5: 3.40 and 5.77; old5: 3.06 and 5.50) |
| `emergence_824_FIXED_{summary.json, per_layer_profiles.csv, convergence.csv}` | as above on the old five, plus Spearman and `rms_dbeta_vs_e48` | the earlier (PCAFETER) run of the same analysis with the e48 checkpoint fix applied |

## perceptual/ : CKA vs quality improvement (Sec. III-B, Fig. `fig:pesq_corr`, `fig:speaker_level`)

Source: `migration_824/rerun_dnsmos/perceptual_grid_v2.py` (PESQ, STOI, SI-SDR, DNSMOS
sig/bak/ovrl per mixture, 3 models x 824 x 5 noises x 41 SNR, SI-SDR sign-correct) joined onto
the Job B CKA rows; `sec3b_v2.py` for the per-layer correlations and `tools/spk_all_v2.py`
for the speaker-level analysis (2000 utterance-cluster bootstrap replicates, seed 20260803;
speaker-level t-interval on n_spk - 1 degrees of freedom).

| file | columns | backs |
|---|---|---|
| `sec3b_824_v2_perlayer.csv` | `model, metric, layer_idx, layer, pearson, p, n` (n = 168,920 per layer) | per-layer Pearson between within-(layer, SNR)-centred CKA and centred delta-metric, six metrics x three models |
| `sec3b_824_v2_summary.csv`, `sec3b_824_v2.md` | `model, metric, min_r, max_r, sign, n_layers, n_per_layer` | per-metric ranges (MUSE SI-SDR all positive `[+0.396, +0.756]`, STOI all negative) |
| `spk_v2__per_speaker_all.csv` | `sweep, model, metric, layer_idx, layer, label, speaker, r, n` | per-speaker correlations (p226/p287 from the 780 sweep, p232/p257 from 824) |
| `spk_v2__speaker_level_all.csv` | `scope, model, metric, layer_idx, label, mean_r, sd, spk_lo, spk_hi, n_spk, per_spk, excludes_zero` | Fig. `fig:speaker_level` (scope `pooled4` = four speakers); Dec-L2.3 PESQ: signs split three to one, mean `-0.21`, spread `[-0.46, +0.05]`; first encoder layer mean `+0.29` `[+0.20, +0.38]`; delta SI-SDR positive in all four speakers at all 24 layers (`0.17` to `0.60` at Dec-L1.3) |
| `spk_v2__utterance_level_all.csv` | `sweep, model, metric, layer_idx, layer, label, r_utt, spearman, utt_lo, utt_hi, n, n_utt` | utterance-cluster bootstrap CIs; the width-`0.041` interval at Dec-L2.3 contrasted with the width-`0.505` speaker interval |
| `spk_v2__dnsmos_level_check.csv`, `spk_v2__fill_check.csv` | `metric, snr, mean_780, mean_824, diff` / per-(model, metric) non-null counts | checks that the 780 and 824 DNSMOS paths agree on levels and that every metric column is 100% populated |
| `perceptual_4metric_perlayer.csv` | `metric, layer_idx, layer, block, pearson, spearman, n` (n = 159,900) | the 780-sweep per-layer correlations (`revision_workspace/scripts/perceptual_multimetric.py`) |
| `fig_v5_pesq_corr_values.csv` | `layer_idx, layer, block, pearson, spearman, n` | Fig. `fig:pesq_corr` (the PESQ rows of the file above, 780 sweep) |

## diffusion/ : diffusion-map view (Sec. III-F, Fig. `fig:diffusion`, `fig:layer_distance_matrices`)

Source: `scripts/confine_780/diffusion_centroids24.py` (824-utterance centroids of all 24
`.norm1` layers per (noise, SNR)), embedded by `results_cluster_2026_08/diffusion_824/build_psi.py`
(per-layer view: all 5 x 41 (noise, SNR) centroids of a block embedded jointly, t = 0.5,
cutoff 0.99, psi then averaged over the five noises per SNR; architecture view: centroids
averaged over the five noises first, then the 16 encoder/decoder layers embedded per SNR,
t = 5). Refit per noise set by `scripts/scafe/refit_IIIG2.py` and `refit_IIIG3.py`.

| file | columns | backs |
|---|---|---|
| `diffusion_maps_per_layer_t0.5_824_{new5,old5}.parquet` | `layer, noise_name, snr, psi_0..psi_13` (1230 rows = 6 representative layers x 5 noises x 41 SNR) | Fig. `fig:diffusion` (the `new5` table is the manuscript's) |
| `diffusion_maps_architecture_t5_824.parquet` | `snr, layer, layer_idx, n_components, eigenvalues, psi_0..psi_12` (656 rows = 16 layers x 41 SNR) | Fig. `fig:layer_distance_matrices` (each panel min-max normalised on its own) |
| `IIIG2_per_block_{new5,old5}.csv`, `IIIG2_per_block.json` | `block, dim, spearman_rho, abs_rho, arc_len_2d, max_dist_from_30dB`; json adds encoder/decoder mean arc lengths and the ratio | `|rho| = 1.00` in all six blocks; arc length `3.03x` larger for decoder (Dec-L2, Dec-L1) than encoder (Enc-L1, Enc-L2) blocks (`0.193` vs `0.064`, `new5`) |
| `IIIG3_across_layers_{new5,old5}.csv`, `IIIG3_across_layers.json` | `snr, dim, norm_between, norm_within_enc, norm_within_dec, between_over_within, raw_between` per SNR | encoder/decoder clustering at low SNR and the shrinking between-group distance with SNR on the per-panel-normalised matrices |
| `diffusion_824_summary.json`, `diffusion_layers_summary.json`, `diffusion_layers_mobility.csv` | per diffusion time t: min `|rho|`, arc lengths, within/between-group distances, `corr(beta, mobility)`; mobility = range over SNR of a layer's mean distance to the other 15 | the 824-vs-780 confinement check of the diffusion claims (old five noises) |

Three files from the same cluster directory (`IIIG_per_block_*.csv.BAD`,
`IIIG_across_layers_*.csv.BAD`, `IIIG_diffusion.json.BAD`) are deliberately **not** shipped.
They came from a script that averaged the centroids over noises *before* embedding, which
collapses the embedding dimension to 3 and gives a decoder/encoder arc ratio of 1.02x instead
of 3.03x. The `IIIG2_*`/`IIIG3_*` files are the correct recipe.

## ablation/ : CKA estimator and sample-convention ablation (Sec. II-D)

Source: `revision_workspace/scripts/ablation_hsic_frames.py`, pretrained MUSE, 150
utterances x 9 SNR x 2 noises (TBUS, PCAFETER), non-pooled first-segment activations.

| file | columns | backs |
|---|---|---|
| `hsic_frames_slopes.csv` | `layer_short, pooled_linear, hsic_unbiased, frames_TFC, frames_TC` (per-layer `beta`) | the unbiased-HSIC check: slope-profile correlation `0.98` with the biased estimator and the same peak layer; the two frame-as-sample conventions are also reported (`-0.50`, `-0.26`) |
| `hsic_frames_report.md` | the analysis report (its templated closing verdict overclaims; the manuscript uses only the HSIC row) | |

## freeze_arms/ : profile-guided fine-tuning arms (not in the manuscript)

Source: Job A (`jobA/jobA_train.py`), MUSE dereverberation fine-tuning under three arms
(`full_ft`, `freeze_encoder`, `freeze_decoder`) x three seeds (1234, 2345, 3456), 50 epochs,
and the per-arm C50 re-probe `jobA_probe/analyze_jobA_probe.py` on the 88-RIR pool (780
utterances, 300 utterance-bootstrap replicates, seed 20260802). The arm checkpoints are
public on HuggingFace (`yairamr/SE-Probe-models`, `cluster-2026-07/<arm>__seed<seed>/g_00040000`).
This experiment was scoped out of the paper; it is shipped as a supplementary null result.

| file | columns | backs |
|---|---|---|
| `jobA_stats.csv`, `jobA_per_run_stats.csv`, `jobA_run_summary.csv` | per-arm and per-run best-epoch PESQ/STOI | `full_ft 2.783 +/- 0.011`, `freeze_encoder 2.697 +/- 0.004`, `freeze_decoder 2.700 +/- 0.003` |
| `jobA_publication_stats.csv`, `jobA_publication_analysis.md` | `metric, arm_a, arm_b, mean_a, mean_b, diff, diff_ci_lo/hi, cohens_d, welch_t, welch_p, practical_significance_threshold, practically_significant` | freeze_encoder - freeze_decoder = `-0.002` PESQ, CI `[-0.007, +0.002]` |
| `jobA_probe_per_layer.csv`, `jobA_probe_arm_summary.csv`, `jobA_probe_contrasts.csv`, `jobA_probe_stage_summary.csv`, `jobA_probe_pre_ft_vs_manuscript.csv`, `RESULTS_SECTION__stage2_full.md` | per-arm per-layer `auc, c_low, c_high, alpha, beta, r2` with CIs; paired contrasts; stage roll-ups; reproduction check against the published pre-FT artifact (`r = 1.0`, max deviation 0) | the per-arm C50 profiles |

## cross_task/ : noise-axis vs reverberation-axis agreement within architecture (Sec. III-C)

`CROSS_TASK.md` holds the whole-profile Spearman/Pearson correlations between each model's
SNR profile (`snr/fits_*_snr_jobB.csv`) and its fine-tuned C50 profile
(`c50/c50_824_per_layer_supp.csv`, `*_ft` arms) with 95% layer-resample CIs (20,000 draws,
seed 20260810): MUSE `beta` `+0.837 [+0.582, +0.942]`, `alpha` `+0.864 [+0.664, +0.957]`;
MP-SENet `alpha` `+0.833 [+0.300, +1.000]`; Demucs neither, but the LSTM is the most robust
layer on both axes. The scratch scripts the note names (`xtask/cross.py` and friends) were
temporary and no longer exist; the two inputs it reads are the two files above, both shipped.

## What is not here

* The raw per-utterance CKA parquets (Job B 96M rows, the C50 shards, the random-init and
  emergence grids, the perceptual join) and the 24-layer centroids. They live on the cluster
  under `cluster-2026-07/` and are the intended contents of the HuggingFace dataset
  `yairamr/SE-Probe-data` under `cluster-2026-07/`.
* The mean level curves of the MUSE and MP-SENet **fine-tuned** C50 arms (only their fitted
  `alpha, beta, A, c_low, c_high` per layer are shipped), so the C50-row PC1 values of Table
  `tab:tradeoff` for those two arms (`96.52`, `99.23`) cannot be recomputed from this
  directory; the Demucs C50 PC1 (`83.58`) can, from `c50/profile_shape_stats_c50.csv`.
* The GTCRN tables (`tools/fits_824_gtcrn/`) and the hinge-fit exploration; neither enters the
  manuscript.


## Added for this release: `c50/mean_curves_c50_six_arms.csv`

Mean CKA per (arm, probed layer, target C50 level) for all six C50 arms of job 128939
(`muse_pre`, `muse_ft`, `mpsenet_pre`, `mpsenet_ft`, `demucs_pre`, `demucs_ft`; 824 utterances,
88-RIR six-room AIR pool, utterance-first averaging over each utterance's five RIRs). Columns:
`scale, arm, model, layer, depth, -5, -2.5, ..., 25`. Computed on athena (job 170742,
2026-10-06) from the retained shards `home_offload/home_tmp/c50_ft_arms/results/shard_<arm>_*`
with `se_probe.profiles`-equivalent code; `fit_curves` on these rows reproduces every
`alpha, beta, A` of `c50_824_per_layer_supp.csv` to 1e-15, and `tradeoff_summary` reproduces the
three C50 rows of Table I (`r, rho, f, PC1`), which the other shipped files could not provide.
