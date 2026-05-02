import numpy as np
import librosa
from typing import Dict, Tuple
from scipy.signal import fftconvolve
from se_probe.consts import EARLY_REFLECTION_IDX

__all__ = [
    "load_demand_noise",
    "add_noise_at_snr",
    "compute_ratio_for_target_c50",
    "convolve_audio",
]

# Cache for loaded noise files
_noise_cache: Dict[str, np.ndarray] = {}


def load_demand_noise(noise_name: str, sample_rate: int = 16000) -> np.ndarray:
    """
    Load a noise file from the DEMAND dataset.

    Args:
        noise_name: Name of the noise file (e.g., "SPSQUARE", "DKITCHEN").
        sample_rate: Target sample rate.

    Returns:
        Noise signal as numpy array.
    """
    cache_key = f"{noise_name}_{sample_rate}"
    if cache_key not in _noise_cache:
        from se_probe.consts import DEMAND_NOISE_DIR
        noise_path = f"{DEMAND_NOISE_DIR}/{noise_name}.wav"
        noise, _ = librosa.load(noise_path, sr=sample_rate)
        _noise_cache[cache_key] = noise
    return _noise_cache[cache_key]


def add_noise_at_snr(
    signal: np.ndarray,
    noise: np.ndarray,
    snr: float,
) -> np.ndarray:
    """
    Add noise to a signal at a specified SNR level by scaling the noise.

    Args:
        signal: Input signal (e.g., clean speech).
        noise: Noise signal to add.
        snr: Target SNR in dB.

    Returns:
        Noisy signal (signal + scaled noise).
    """
    # Match noise length to signal length
    if len(noise) < len(signal):
        reps = int(np.ceil(len(signal) / len(noise)))
        noise = np.tile(noise, reps)
    noise = noise[:len(signal)]

    # Calculate signal power and required noise power for target SNR
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr / 10)
    target_noise_power = signal_power / snr_linear

    # Scale noise to achieve target SNR
    current_noise_power = np.mean(noise ** 2) + 1e-12
    scaling_factor = np.sqrt(target_noise_power / current_noise_power)
    scaled_noise = noise * scaling_factor

    return signal + scaled_noise


def compute_ratio_for_target_c50(
    rir: np.ndarray, target_c50: float, sample_rate: int = 16000
) -> float:
    """
    Compute the ratio needed to scale late reflections to achieve a target C50 value.

    C50 = 10 * log10(early_energy / late_energy)
    When we scale late reflections by ratio, new_late_energy = ratio^2 * original_late_energy
    Solving: ratio = sqrt(early_energy / (10^(target_C50/10) * original_late_energy))

    Args:
        rir: Original RIR signal (before any scaling).
        target_c50: Target C50 value in dB.
        sample_rate: Sample rate for computing energy windows.

    Returns:
        Ratio to scale late reflections to achieve target C50.
    """
    rir = np.asarray(rir)
    if rir.ndim > 1:
        rir = rir.squeeze()
    rir = rir.astype(np.float64)
    eps = 1e-10
    sr = int(sample_rate)

    direct_idx = np.argmax(np.abs(rir))
    early_window = int(0.05 * sr)
    early_start = direct_idx
    early_end = min(direct_idx + early_window, len(rir))
    early_energy = np.sum(rir[early_start:early_end] ** 2)

    late_energy = np.sum(rir[early_end:] ** 2)

    if late_energy < eps:
        return eps
    if early_energy < eps:
        return 1e6

    target_c50_linear = 10 ** (target_c50 / 10.0)
    ratio = np.sqrt(early_energy / (target_c50_linear * late_energy + eps))
    ratio = max(eps, min(ratio, 1e6))
    return float(ratio)


def convolve_audio(
    audio: np.ndarray,
    input_rir: np.ndarray,
    target_c50: float,
    sample_rate: int = 16000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convolve audio with an RIR scaled to achieve a target C50 value.

    Steps:
    1. Align RIR from first peak
    2. Compute scaling ratio for target C50
    3. Scale late reflections (after EARLY_REFLECTION_IDX)
    4. FFT convolve and truncate to original length

    Args:
        audio: Input audio signal.
        input_rir: RIR signal.
        target_c50: Target C50 value in dB.
        sample_rate: Sample rate for computing C50.

    Returns:
        Tuple of (reverberant audio, modified RIR).
    """
    rir = input_rir.copy()
    if rir is None or len(rir) == 0:
        return audio, rir

    first_peak_index = int(np.argmax(np.abs(rir)))
    rir_aligned = rir[first_peak_index:].copy()

    ratio = compute_ratio_for_target_c50(rir_aligned, target_c50, sample_rate)
    rir_aligned[EARLY_REFLECTION_IDX:] = rir_aligned[EARLY_REFLECTION_IDX:] * ratio

    reverb_audio = fftconvolve(audio, rir_aligned, mode='full')
    reverb_audio = reverb_audio[:len(audio)]

    return reverb_audio, rir_aligned
