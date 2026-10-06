"""Tests for :mod:`se_probe.bootstrap`.

The tests are organised around the four properties the hierarchical bootstrap
has to have for the paper's intervals to mean what they say:

1. it is reproducible (same seed -> same interval, different seed -> different);
2. it agrees with the analytic answer on the one case where an analytic answer
   exists (a mean of i.i.d. normal draws with a single flat level);
3. the sampling unit changes the answer, in the direction and by the order of
   magnitude the paper reports;
4. it reproduces a published number.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

from se_probe.bootstrap import (
    NestedDesign,
    PseudoReplicationWarning,
    half_width,
    hierarchical_bootstrap,
    hierarchical_resample_indices,
    hierarchical_resample_weights,
    ols_alpha_beta,
    percentile_ci,
    t_interval,
    weighted_mean,
    weighted_pearson_r,
)

# --------------------------------------------------------------------------- #
# Synthetic designs
# --------------------------------------------------------------------------- #
NORMAL_Z = 1.959963984540054  # two-sided 95% standard normal quantile


def nested_speaker_data(n_speakers=20, n_utts=40, sd_between=1.0, sd_within=0.3, seed=11):
    """Utterances nested in speakers, with a large between-speaker component.

    Returns ``(values, levels, sd_between, sd_within)``. Every utterance carries a
    single observation, so the only thing separating utterance-level from
    speaker-level resampling is which level the resampling starts at.
    """
    rng = np.random.default_rng(seed)
    speaker = np.repeat(np.arange(n_speakers), n_utts)
    utterance = np.arange(n_speakers * n_utts)
    values = np.repeat(rng.normal(0.0, sd_between, n_speakers), n_utts) + rng.normal(
        0.0, sd_within, n_speakers * n_utts
    )
    levels = {"speaker": speaker, "utterance": utterance}
    return values, levels, sd_between, sd_within


def mean_statistic(values):
    """A weight-respecting statistic in the ``statistic(indices, weights)`` form."""

    def statistic(indices, weights):
        return weighted_mean(values[indices], weights)

    return statistic


# --------------------------------------------------------------------------- #
# 1. Determinism
# --------------------------------------------------------------------------- #
def test_same_seed_gives_identical_intervals():
    values, levels, _, _ = nested_speaker_data()
    stat = mean_statistic(values)
    kwargs = dict(unit="speaker", n_boot=400)

    a = hierarchical_bootstrap(stat, levels, seed=1234, **kwargs)
    b = hierarchical_bootstrap(stat, levels, seed=1234, **kwargs)

    assert a.ci_low == b.ci_low
    assert a.ci_high == b.ci_high
    np.testing.assert_array_equal(a.replicates, b.replicates)


def test_different_seeds_give_different_intervals():
    values, levels, _, _ = nested_speaker_data()
    stat = mean_statistic(values)
    kwargs = dict(unit="speaker", n_boot=400)

    a = hierarchical_bootstrap(stat, levels, seed=1234, **kwargs)
    c = hierarchical_bootstrap(stat, levels, seed=5678, **kwargs)

    assert a.ci_low != c.ci_low
    assert a.ci_high != c.ci_high
    # The point estimate is seed-independent: it is computed on the observed data.
    assert a.estimate == c.estimate


def test_resamplers_are_deterministic_given_a_seed():
    _, levels, _, _ = nested_speaker_data()
    w1 = hierarchical_resample_weights(levels, unit="speaker", rng=7)
    w2 = hierarchical_resample_weights(levels, unit="speaker", rng=7)
    w3 = hierarchical_resample_weights(levels, unit="speaker", rng=8)
    np.testing.assert_array_equal(w1, w2)
    assert not np.array_equal(w1, w3)


# --------------------------------------------------------------------------- #
# 2. Known-answer sanity: one flat level, i.i.d. normal, mean
# --------------------------------------------------------------------------- #
def test_matches_analytic_normal_interval_for_a_mean():
    """With one level of nesting the bootstrap must recover the textbook CI.

    Tolerance: 8% on the half-width. Two error sources dominate.

    * Monte-Carlo error in the percentile estimate. For a normal bootstrap
      distribution with sd ``s`` and ``B`` replicates, the standard error of the
      2.5th percentile is ``s * sqrt(0.025 * 0.975 / B) / phi(1.96)``. At
      ``B = 4000`` that is ``0.042 * s`` per endpoint, i.e. about 2% of the
      ``1.96 * s`` half-width.
    * The bootstrap uses the plug-in (``ddof = 0``) variance, a factor
      ``sqrt((n - 1) / n)`` = 0.999 at ``n = 600``; negligible here.

    8% leaves roughly a threefold margin over the combined error, so the test is
    sensitive to a real scale mistake (a missing ``sqrt(n)``, resampling rows
    instead of clusters) but not to Monte-Carlo noise.
    """
    n = 600
    rng = np.random.default_rng(3)
    values = rng.normal(0.0, 1.0, n)
    levels = {"utterance": np.arange(n)}

    result = hierarchical_bootstrap(
        mean_statistic(values), levels, unit="utterance", n_boot=4000, seed=3
    )

    analytic_half_width = NORMAL_Z * values.std(ddof=1) / np.sqrt(n)
    assert result.estimate == pytest.approx(values.mean(), rel=1e-12)
    assert result.half_width == pytest.approx(analytic_half_width, rel=0.08)
    # And the interval is centred where it should be.
    assert result.ci_low < values.mean() < result.ci_high


# --------------------------------------------------------------------------- #
# 3. The sampling unit matters
# --------------------------------------------------------------------------- #
def test_speaker_unit_gives_materially_wider_intervals_than_utterance_unit():
    """The paper's Section III-B finding, reproduced on data with a known answer.

    With utterances nested in speakers and a between-speaker sd much larger than
    the within-speaker sd, resampling utterances treats correlated observations
    as independent (pseudo-replication) and shrinks the interval by roughly
    ``sqrt(n_utts_per_speaker)``. Resampling speakers propagates the component
    that actually dominates.

    Predicted variance of the grand mean:

    * speaker unit    ``(sd_b**2 + sd_w**2 / n_utts) / n_speakers``
    * utterance unit  ``(sd_b**2 + sd_w**2) / n_total``

    so the width ratio should be about the square root of their quotient.
    """
    n_speakers, n_utts = 20, 40
    values, levels, sd_b, sd_w = nested_speaker_data(
        n_speakers=n_speakers, n_utts=n_utts, sd_between=1.0, sd_within=0.3
    )
    n_total = n_speakers * n_utts
    stat = mean_statistic(values)

    with pytest.warns(PseudoReplicationWarning):
        utterance_ci = hierarchical_bootstrap(
            stat, levels, unit="utterance", n_boot=4000, seed=0
        )
    speaker_ci = hierarchical_bootstrap(stat, levels, unit="speaker", n_boot=4000, seed=0)

    ratio = float(speaker_ci.width / utterance_ci.width)
    var_speaker = (sd_b**2 + sd_w**2 / n_utts) / n_speakers
    var_utterance = (sd_b**2 + sd_w**2) / n_total
    predicted = np.sqrt(var_speaker / var_utterance)

    # The widening is large, and it is the mechanism the theory predicts.
    assert ratio > 3.0, f"speaker/utterance width ratio {ratio:.2f} shows no widening"
    assert 0.4 * predicted < ratio < 2.5 * predicted, (
        f"width ratio {ratio:.2f} is not consistent with the predicted {predicted:.2f}; "
        "the between-speaker component is not being propagated correctly"
    )

    # The consequence the paper draws: an association that looks conclusive
    # under the wrong unit can span zero under the right one.
    assert bool(utterance_ci.excludes(0.0))
    assert not bool(speaker_ci.excludes(0.0))

    # Provenance records why the two differ.
    assert utterance_ci.fixed_levels == ("speaker",)
    assert speaker_ci.fixed_levels == ()
    assert speaker_ci.resampled_levels == ("speaker",)
    assert speaker_ci.carried_levels == ("utterance",)


def test_widening_scales_with_the_within_speaker_correlation():
    """No between-speaker component -> the two units must agree.

    This is the control for the previous test: the widening has to come from the
    within-speaker correlation, not from the act of clustering. If the speaker
    interval were still wider here, the estimator would be inflating variance
    rather than propagating it.

    The ratio is checked over several independent datasets. On any single dataset
    the speaker-unit width is driven by the empirical variance of the speaker
    means, which carries chi-square noise on ``n_speakers - 1`` degrees of
    freedom -- about 16% on the width at 20 speakers -- so one draw is not enough
    to pin an expectation of 1.
    """
    ratios = []
    for data_seed in range(6):
        values, levels, _, _ = nested_speaker_data(
            n_speakers=20, n_utts=40, sd_between=0.0, sd_within=1.0, seed=100 + data_seed
        )
        stat = mean_statistic(values)
        with pytest.warns(PseudoReplicationWarning):
            utterance_ci = hierarchical_bootstrap(
                stat, levels, unit="utterance", n_boot=2000, seed=0
            )
        speaker_ci = hierarchical_bootstrap(stat, levels, unit="speaker", n_boot=2000, seed=0)
        ratios.append(float(speaker_ci.width / utterance_ci.width))

    mean_ratio = float(np.mean(ratios))
    assert mean_ratio == pytest.approx(1.0, rel=0.15), (
        f"with no between-speaker variance the widths should match on average, "
        f"got mean ratio {mean_ratio:.2f} from {[round(r, 2) for r in ratios]}"
    )
    # Even the noisiest single draw stays far below the correlated case (~6x).
    assert max(ratios) < 2.0


def test_resample_within_inflates_variance_and_is_not_the_default():
    """The two-stage variant is available, opt-in, and documented as inflating.

    Redrawing utterances inside each drawn speaker adds a second
    ``sd_within**2 / n_obs`` term. With no between-speaker variance that doubles
    the bootstrap variance, i.e. widens the interval by about sqrt(2). The
    default must not do this.
    """
    values, levels, _, _ = nested_speaker_data(sd_between=0.0, sd_within=1.0, seed=5)
    stat = mean_statistic(values)

    default_ci = hierarchical_bootstrap(stat, levels, unit="speaker", n_boot=4000, seed=0)
    two_stage_ci = hierarchical_bootstrap(
        stat, levels, unit="speaker", n_boot=4000, seed=0, resample_within=True
    )

    ratio = float(two_stage_ci.width / default_ci.width)
    assert ratio == pytest.approx(np.sqrt(2.0), rel=0.25), (
        f"two-stage/cluster width ratio {ratio:.2f} does not match the expected sqrt(2)"
    )
    assert default_ci.resampled_levels == ("speaker",)
    assert two_stage_ci.resampled_levels == ("speaker", "utterance")


# --------------------------------------------------------------------------- #
# 4. Reproduction of a published interval
# --------------------------------------------------------------------------- #
# Per-speaker Pearson r between MUSE's Dec-L2.3 CKA and the utterance-level PESQ
# gain, from the paper's Section III-B speaker-level analysis
# (revision_workspace/results/speaker_level/per_speaker_all.csv).
PUBLISHED_PER_SPEAKER_R = {
    "p226": -0.3195455220247532,
    "p232": -0.2152412701106129,
    "p257": 0.0217480512006277,
    "p287": -0.3094216153150330,
}
# Published speaker-level interval for the same layer (scope "pooled4").
PUBLISHED_SPEAKER_CI = (-0.458118, 0.046888)
# Published utterance-cluster bootstrap interval for the same layer, 780 sweep.
PUBLISHED_UTTERANCE_CI = (-0.3107058814745452, -0.2693843352677698)


def test_reproduces_published_speaker_level_interval():
    """Reproduce the paper's 0.041-vs-0.505 comparison at MUSE Dec-L2.3.

    The paper reports an utterance-cluster interval of width 0.041 and a
    speaker-level interval of width 0.505 at the same layer, "roughly twelve
    times wider and spanning zero". The speaker-level endpoints are recovered
    here from the four published per-speaker correlations.
    """
    r = np.array(list(PUBLISHED_PER_SPEAKER_R.values()))
    mean_r, low, high, n = t_interval(r)

    assert n == 4
    assert low == pytest.approx(PUBLISHED_SPEAKER_CI[0], abs=1e-6)
    assert high == pytest.approx(PUBLISHED_SPEAKER_CI[1], abs=1e-6)
    assert mean_r == pytest.approx(-0.205615, abs=1e-6)

    speaker_width = high - low
    utterance_width = PUBLISHED_UTTERANCE_CI[1] - PUBLISHED_UTTERANCE_CI[0]
    assert speaker_width == pytest.approx(0.505, abs=5e-4)
    assert utterance_width == pytest.approx(0.041, abs=5e-4)
    assert speaker_width / utterance_width == pytest.approx(12.2, abs=0.2)

    # The paper's qualitative conclusion: the speaker-level interval spans zero,
    # the utterance-level one does not.
    assert low < 0.0 < high
    assert PUBLISHED_UTTERANCE_CI[1] < 0.0


def test_percentile_bootstrap_is_degenerate_at_four_speakers():
    """Why the paper uses a t interval, not a percentile bootstrap, over speakers.

    With four units the empirical bootstrap distribution cannot reach past the
    observed extremes, so the percentile interval is biased inward. It must come
    out materially *narrower* than the t interval on the same four values -- the
    wrong direction, which is exactly why it is not the estimator used.
    """
    r = np.array(list(PUBLISHED_PER_SPEAKER_R.values()))
    _, low, high, _ = t_interval(r)
    t_width = high - low

    boot = hierarchical_bootstrap(
        mean_statistic(r), {"speaker": np.arange(r.size)}, unit="speaker", n_boot=5000, seed=0
    )
    assert float(boot.width) < 0.75 * t_width


# --------------------------------------------------------------------------- #
# The unit must be stated explicitly
# --------------------------------------------------------------------------- #
def test_unit_has_no_default_anywhere():
    for func in (
        hierarchical_bootstrap,
        hierarchical_resample_indices,
        hierarchical_resample_weights,
    ):
        param = inspect.signature(func).parameters["unit"]
        assert param.default is inspect.Parameter.empty, f"{func.__name__} gave unit a default"
        assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_missing_unit_is_an_error():
    values, levels, _, _ = nested_speaker_data(n_speakers=4, n_utts=5)
    with pytest.raises(TypeError):
        hierarchical_bootstrap(mean_statistic(values), levels)  # type: ignore[call-arg]


def test_unknown_unit_names_the_valid_choices():
    values, levels, _, _ = nested_speaker_data(n_speakers=4, n_utts=5)
    with pytest.raises(ValueError, match="not one of the declared levels"):
        hierarchical_bootstrap(mean_statistic(values), levels, unit="noise_type", n_boot=10)


def test_pseudoreplication_warning_can_be_acknowledged():
    values, levels, _, _ = nested_speaker_data(n_speakers=4, n_utts=5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hierarchical_bootstrap(
            mean_statistic(values),
            levels,
            unit="utterance",
            n_boot=10,
            acknowledge_pseudoreplication=True,
        )


def test_no_warning_when_the_unit_is_outermost():
    values, levels, _, _ = nested_speaker_data(n_speakers=4, n_utts=5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hierarchical_bootstrap(mean_statistic(values), levels, unit="speaker", n_boot=10)


# --------------------------------------------------------------------------- #
# Resampling mechanics
# --------------------------------------------------------------------------- #
def test_weight_mode_and_index_mode_agree_exactly():
    values, levels, _, _ = nested_speaker_data(n_speakers=6, n_utts=8)
    stat = mean_statistic(values)
    a = hierarchical_bootstrap(stat, levels, unit="speaker", n_boot=200, seed=7)
    b = hierarchical_bootstrap(
        stat, levels, unit="speaker", n_boot=200, seed=7, expand_indices=True
    )
    np.testing.assert_allclose(a.replicates, b.replicates, rtol=0, atol=1e-12)


def test_resample_draws_whole_clusters():
    """An utterance's observations must move together, never be split."""
    n_utts, per_utt = 12, 5
    utterance = np.repeat(np.arange(n_utts), per_utt)
    weights = hierarchical_resample_weights({"utterance": utterance}, unit="utterance", rng=2)

    for u in range(n_utts):
        w = weights[utterance == u]
        assert len(set(w.tolist())) == 1, "observations of one utterance got different weights"
    assert weights.sum() == n_utts * per_utt
    assert np.any(weights == 0.0) and np.any(weights >= 2.0)


def test_indices_are_a_full_resampled_dataset():
    n_utts, per_utt = 12, 5
    utterance = np.repeat(np.arange(n_utts), per_utt)
    idx = hierarchical_resample_indices({"utterance": utterance}, unit="utterance", rng=2)
    assert idx.size == n_utts * per_utt


def test_strata_are_held_fixed():
    """Noise-stratified utterance bootstrap: stratum sizes are preserved exactly."""
    n_utts, noises = 30, 5
    utterance = np.tile(np.arange(n_utts), noises)
    noise = np.repeat(np.arange(noises), n_utts)

    weights = hierarchical_resample_weights(
        {"utterance": utterance}, unit="utterance", rng=4, strata=noise, strata_name="noise_type"
    )
    for k in range(noises):
        assert weights[noise == k].sum() == n_utts, "a stratum changed size"


def test_crossed_margin_adds_variance():
    """Resampling the noise-type margin as well must widen the interval.

    Utterances crossed with noise types, with a large noise-type effect. The
    utterance-only interval sees none of that effect because every noise type is
    present in every resample; adding the crossed margin exposes it.
    """
    rng = np.random.default_rng(21)
    n_utts, n_noises = 200, 6
    utterance = np.tile(np.arange(n_utts), n_noises)
    noise = np.repeat(np.arange(n_noises), n_utts)
    noise_effect = rng.normal(0.0, 1.0, n_noises)
    values = noise_effect[noise] + rng.normal(0.0, 0.2, n_utts * n_noises)
    stat = mean_statistic(values)

    utt_only = hierarchical_bootstrap(
        stat, {"utterance": utterance}, unit="utterance", n_boot=2000, seed=0
    )
    two_way = hierarchical_bootstrap(
        stat,
        {"utterance": utterance},
        unit="utterance",
        crossed={"noise_type": noise},
        n_boot=2000,
        seed=0,
    )

    assert two_way.crossed == ("noise_type",)
    assert float(two_way.width) > 3.0 * float(utt_only.width)


def _reference_two_way_beta_ci(cube, snrs, n_boot, seed):
    """The shipped two-way estimator, transcribed for cross-validation.

    Reimplements the resampling scheme of the paper's cluster script
    (``hierarchical_bootstrap_cluster.py``, replayed in
    ``tools/reagg18/reagg18c.py``): each replicate draws ``n_utt`` utterances and
    ``n_noise`` noise types with replacement, turns both into normalised weight
    vectors, contracts them against the ``[noise, utt, snr]`` cube to get a mean
    degradation curve, and refits OLS. The RNG stream differs from this module's,
    so only the resulting interval width is comparable, not the draws.
    """
    n_noise, n_utt, _ = cube.shape
    x = np.asarray(snrs, dtype=float)
    xm = x.mean()
    sxx = ((x - xm) ** 2).sum()
    rng = np.random.default_rng(seed)
    betas = np.empty(n_boot)
    for r in range(n_boot):
        wu = np.bincount(rng.integers(0, n_utt, n_utt), minlength=n_utt) / n_utt
        wn = np.bincount(rng.integers(0, n_noise, n_noise), minlength=n_noise) / n_noise
        curve = np.tensordot(wn, np.tensordot(wu, cube, axes=([0], [1])), axes=([0], [0]))
        betas[r] = ((curve - curve.mean()) * (x - xm)).sum() / sxx
    return percentile_ci(betas)


def test_matches_the_shipped_two_way_estimator():
    """This module's crossed-margin path must agree with the paper's own code.

    Same design, same statistic, independent implementations and independent RNG
    streams: the interval widths have to agree to within Monte-Carlo error.
    """
    n_noise, n_utt, snrs = 5, 120, np.arange(-10.0, 31.0, 10.0)
    n_snr = snrs.size
    rng = np.random.default_rng(31)
    cube = (
        0.75
        - 0.008 * snrs[None, None, :]
        + rng.normal(0.0, 0.05, (n_noise, 1, 1))
        + rng.normal(0.0, 0.08, (1, n_utt, 1))
        + rng.normal(0.0, 0.02, (n_noise, n_utt, n_snr))
    )

    noise_idx, utt_idx, snr_idx = np.meshgrid(
        np.arange(n_noise), np.arange(n_utt), np.arange(n_snr), indexing="ij"
    )
    flat = cube.ravel()
    noise_lbl, utt_lbl, snr_lbl = noise_idx.ravel(), utt_idx.ravel(), snr_idx.ravel()

    def statistic(indices, weights):
        # Weighted mean CKA at each SNR level, then OLS of that curve on SNR.
        num = np.bincount(snr_lbl[indices], weights=weights * flat[indices], minlength=n_snr)
        den = np.bincount(snr_lbl[indices], weights=weights, minlength=n_snr)
        _, beta = ols_alpha_beta(num / den, snrs)
        return beta

    n_boot = 3000
    mine = hierarchical_bootstrap(
        statistic,
        {"utterance": utt_lbl},
        unit="utterance",
        crossed={"noise_type": noise_lbl},
        n_boot=n_boot,
        seed=0,
    )
    ref_low, ref_high = _reference_two_way_beta_ci(cube, snrs, n_boot=n_boot, seed=12345)

    assert mine.estimate == pytest.approx(-0.008, abs=2e-3)
    assert float(mine.width) == pytest.approx(ref_high - ref_low, rel=0.10)
    assert float(mine.ci_low) == pytest.approx(ref_low, abs=0.10 * (ref_high - ref_low))
    assert float(mine.ci_high) == pytest.approx(ref_high, abs=0.10 * (ref_high - ref_low))


def test_crossed_factor_cannot_also_be_a_level():
    n = 20
    utterance = np.arange(n)
    with pytest.raises(ValueError, match="also declared as nested levels"):
        hierarchical_resample_weights(
            {"utterance": utterance}, unit="utterance", rng=0, crossed={"utterance": utterance}
        )


def test_non_nested_levels_are_rejected():
    """A crossed factor passed as a nested level must be caught, not silently used.

    Each utterance here appears under all three noise types. Silently splitting
    it into three (noise, utterance) clusters would change the estimand without
    the caller ever being told, so it raises instead.
    """
    utterance = np.tile(np.arange(10), 3)
    noise = np.repeat(np.arange(3), 10)
    with pytest.raises(ValueError, match="not nested inside"):
        NestedDesign(levels={"noise": noise, "utterance": utterance}, unit="noise")


def test_locally_numbered_child_labels_are_rejected():
    """Utterances numbered 0..4 inside every speaker are ambiguous, so they raise."""
    speaker = np.repeat(np.arange(3), 5)
    utterance = np.tile(np.arange(5), 3)
    with pytest.raises(ValueError, match="globally unique"):
        NestedDesign(levels={"speaker": speaker, "utterance": utterance}, unit="speaker")


def test_level_lengths_must_match():
    with pytest.raises(ValueError, match="expected"):
        NestedDesign(
            levels={"speaker": np.arange(10), "utterance": np.arange(11)}, unit="speaker"
        )


# --------------------------------------------------------------------------- #
# Interval and statistic helpers
# --------------------------------------------------------------------------- #
def test_percentile_ci_handles_vector_statistics():
    """A statistic returning one value per layer yields one interval per layer."""
    n_utts, n_layers = 80, 4
    rng = np.random.default_rng(9)
    values = rng.normal(0.0, 1.0, (n_utts, n_layers)) + np.arange(n_layers)
    utterance = np.arange(n_utts)

    def statistic(indices, weights):
        return weighted_mean(values[indices], weights, axis=0)

    result = hierarchical_bootstrap(
        statistic, {"utterance": utterance}, unit="utterance", n_boot=800, seed=0
    )
    assert result.estimate.shape == (n_layers,)
    assert result.ci_low.shape == (n_layers,)
    assert result.replicates.shape == (800, n_layers)
    assert np.all(result.ci_low < result.ci_high)
    assert np.all(result.excludes(-1.0))


def test_percentile_ci_and_half_width():
    samples = np.random.default_rng(0).normal(0.0, 1.0, 20000)
    low, high = percentile_ci(samples, ci_level=0.95)
    assert low == pytest.approx(-1.96, abs=0.08)
    assert high == pytest.approx(1.96, abs=0.08)
    assert half_width(samples) == pytest.approx((high - low) / 2)

    low90, high90 = percentile_ci(samples, ci_level=0.90)
    assert high90 - low90 < high - low


def test_percentile_ci_rejects_bad_coverage():
    with pytest.raises(ValueError, match="ci_level"):
        percentile_ci(np.zeros(10), ci_level=1.5)


def test_weighted_pearson_r_matches_numpy_with_unit_weights():
    rng = np.random.default_rng(4)
    x = rng.normal(size=200)
    y = 0.6 * x + rng.normal(size=200)
    assert weighted_pearson_r(x, y, np.ones_like(x)) == pytest.approx(
        np.corrcoef(x, y)[0, 1], rel=1e-10
    )


def test_weighted_pearson_r_respects_duplication():
    """Weight 2 must equal listing the observation twice: the resampling identity."""
    rng = np.random.default_rng(6)
    x = rng.normal(size=50)
    y = rng.normal(size=50)
    w = rng.integers(0, 3, 50).astype(float)
    idx = np.repeat(np.arange(50), w.astype(int))
    assert weighted_pearson_r(x, y, w) == pytest.approx(
        weighted_pearson_r(x[idx], y[idx], np.ones(idx.size)), rel=1e-10
    )


def test_ols_alpha_beta_recovers_a_known_line():
    x = np.arange(-10.0, 31.0)  # the paper's SNR grid, in dB
    alpha, beta = ols_alpha_beta(0.8 - 0.01 * x, x)
    assert alpha == pytest.approx(0.8, abs=1e-12)
    assert beta == pytest.approx(-0.01, abs=1e-12)

    curves = np.stack([0.8 - 0.01 * x, 0.5 + 0.02 * x])  # two layers at once
    alpha, beta = ols_alpha_beta(curves, x)
    np.testing.assert_allclose(alpha, [0.8, 0.5], atol=1e-12)
    np.testing.assert_allclose(beta, [-0.01, 0.02], atol=1e-12)


def test_result_describe_names_the_estimand():
    values, levels, _, _ = nested_speaker_data(n_speakers=4, n_utts=5)
    result = hierarchical_bootstrap(mean_statistic(values), levels, unit="speaker", n_boot=50)
    text = result.describe()
    assert "sampling unit=speaker" in text
    assert "n=4" in text
