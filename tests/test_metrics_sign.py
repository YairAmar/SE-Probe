"""SI-SDR must be positive for a near-perfect estimate (the sign bug of the early helper)."""
import numpy as np

from se_probe.metrics import sisdr


def test_sisdr_sign_and_scale_invariance():
    rng = np.random.default_rng(0)
    clean = rng.standard_normal(16000).astype("float32")
    noisy = clean + 0.01 * rng.standard_normal(16000).astype("float32")
    v = sisdr(clean, noisy)
    assert v > 30.0  # ~40 dB for 1% noise
    assert abs(sisdr(clean, 3.0 * noisy) - v) < 1e-4  # scale invariant (float32 arithmetic)
    assert sisdr(clean, rng.standard_normal(16000).astype("float32")) < 0.0
