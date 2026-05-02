"""Audio quality metrics (CPU + GPU evaluators).

The ``GPU*Evaluator`` classes depend on optional packages (``torch_stoi``,
``onnx2torch``, ``speechmos``) that are *not* in the SE-Probe core deps —
install them yourself if you want fast batched STOI / DNSMOS computation:

    pip install torch_stoi onnx2torch speechmos

Top-level imports here only reference packages declared in
``pyproject.toml`` so ``from se_probe.metrics import c50`` works on a fresh
install.
"""
import os
from typing import Dict, Optional, Union

import numpy as np
import pesq
import torch
import torchaudio
from pystoi import stoi

from se_probe.device import get_device

__all__ = [
    "c50",
    "drr",
    "sisdr",
    "compute_audio_metrics",
    "gpu_sisdr",
    "GPUSTOIEvaluator",
    "GPUDNSMOSEvaluator",
    "GPUMetricsEvaluator",
]


def c50(rir: np.ndarray, sr: int = 16000) -> float:
    """
    Calculate C50 from a room impulse response (RIR).
    C50: ratio (in dB) of early energy (first 50ms after direct sound) to late energy.

    Args:
        rir: 1D room impulse response.
        sr: Sample rate.

    Returns:
        C50 value in dB.
    """
    rir = np.asarray(rir)
    if rir.ndim > 1:
        rir = rir.squeeze()
    rir = rir.astype(np.float64)
    eps = 1e-10
    sr = int(sr)
    direct_idx = np.argmax(np.abs(rir))
    early_window = int(0.05 * sr)
    early_start = direct_idx
    early_end = min(direct_idx + early_window, len(rir))
    early_energy = np.sum(rir[early_start:early_end] ** 2)
    late_start = early_end
    late_energy = np.sum(rir[late_start:] ** 2)
    return 10 * np.log10((early_energy + eps) / (late_energy + eps))


def drr(rir: np.ndarray, sr: int = 16000) -> float:
    """
    Calculate DRR from a room impulse response (RIR).
    DRR: ratio of direct (1.5ms window) to remaining energy, in dB.

    Args:
        rir: 1D room impulse response.
        sr: Sample rate.

    Returns:
        DRR value in dB.
    """
    rir = np.asarray(rir)
    if rir.ndim > 1:
        rir = rir.squeeze()
    rir = rir.astype(np.float64)
    eps = 1e-10
    sr = int(sr)
    peak_idx = np.argmax(np.abs(rir))
    direct_window = int(0.0015 * sr)
    direct_start = max(0, peak_idx - direct_window // 2)
    direct_stop = min(len(rir), peak_idx + direct_window // 2)
    direct_energy = np.sum(rir[direct_start:direct_stop] ** 2)
    reverb_mask = np.ones(len(rir), dtype=bool)
    reverb_mask[direct_start:direct_stop] = False
    reverberant_energy = np.sum(rir[reverb_mask] ** 2)
    return 10 * np.log10((direct_energy + eps) / (reverberant_energy + eps))


def sisdr(clean: np.ndarray, degraded: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) metric.

    Args:
        clean: Clean reference signal
        degraded: Degraded/processed signal
        eps: Small epsilon for numerical stability

    Returns:
        SI-SDR score in dB (higher is better)
    """
    # Convert to numpy and squeeze to ensure 1D
    clean = np.asarray(clean).squeeze()
    degraded = np.asarray(degraded).squeeze()

    # Check for empty arrays
    if clean.size == 0 or degraded.size == 0:
        return np.nan

    # Scale-invariant: find optimal scaling factor
    alpha = np.dot(degraded, clean) / (np.dot(clean, clean) + eps)
    s_target = alpha * clean

    # Error signal
    e_noise = degraded - s_target

    # SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    s_target_norm_sq = np.dot(s_target, s_target) + eps
    e_noise_norm_sq = np.dot(e_noise, e_noise) + eps

    sisdr = - 10 * np.log10(s_target_norm_sq / e_noise_norm_sq)
    return float(sisdr)


def compute_audio_metrics(
    reference_audio: Union[np.ndarray, torch.Tensor],
    test_audio: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 16000,
    metric_prefix: Optional[str] = None,
) -> Dict[str, float]:
    """
    Computes audio quality metrics (SISDR, STOI, PESQ) between a reference audio and a test audio.

    Args:
        reference_audio: Reference/clean audio signal (np.ndarray or torch.Tensor).
        test_audio: Test audio signal to evaluate (noisy, enhanced, or any other signal).
        sample_rate: Audio sample rate (default: 16000).
        metric_prefix: Optional prefix for metric names (e.g., 'noisy', 'enhanced').
                      If None, metrics are named 'sisdr', 'stoi', 'pesq'.
                      If provided, metrics are named '{prefix}_sisdr', '{prefix}_stoi', '{prefix}_pesq'.

    Returns:
        Dictionary containing computed metrics. Values are np.nan if computation fails.
    """
    # SISDR
    min_len = min(len(reference_audio), len(test_audio))
    ref_speech = reference_audio[:min_len].copy()
    test_speech = test_audio[:min_len].copy()
    ref_squeezed = np.asarray(ref_speech).squeeze()
    test_squeezed = np.asarray(test_speech).squeeze()
    sisdr_val = sisdr(ref_squeezed, test_squeezed)

    # STOI
    stoi_val = np.nan
    try:
        stoi_val = stoi(ref_squeezed, test_squeezed, sample_rate, extended=False)
    except Exception:
        pass

    # PESQ
    pesq_val = np.nan
    try:
        pesq_val = pesq.pesq(sample_rate, ref_squeezed, test_squeezed, 'wb')
    except Exception:
        pass

    # Build metric names with optional prefix
    if metric_prefix:
        return {
            f"{metric_prefix}_sisdr": sisdr_val,
            f"{metric_prefix}_stoi": stoi_val,
            f"{metric_prefix}_pesq": pesq_val,
        }
    else:
        return {
            "sisdr": sisdr_val,
            "stoi": stoi_val,
            "pesq": pesq_val,
        }


# =============================================================================
# GPU-accelerated metrics for fast inference
# =============================================================================

def gpu_sisdr(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-10) -> float:
    """
    GPU-accelerated SI-SDR computation using torch.

    Args:
        reference: Clean reference signal (1D tensor on GPU)
        estimate: Estimated/degraded signal (1D tensor on GPU)
        eps: Small epsilon for numerical stability

    Returns:
        SI-SDR score in dB (higher is better)
    """
    ref = reference.squeeze()
    est = estimate.squeeze()

    # Align lengths
    min_len = min(len(ref), len(est))
    ref = ref[:min_len]
    est = est[:min_len]

    alpha = torch.dot(est, ref) / (torch.dot(ref, ref) + eps)
    s_target = alpha * ref
    e_noise = est - s_target

    sisdr_val = 10 * torch.log10(
        (torch.dot(s_target, s_target) + eps) / (torch.dot(e_noise, e_noise) + eps)
    )
    return float(sisdr_val.item())


class GPUSTOIEvaluator:
    """GPU-accelerated STOI evaluator using torch-stoi."""

    def __init__(self, sample_rate: int = 16000, device: Optional[Union[str, torch.device]] = None):
        from torch_stoi import NegSTOILoss

        device = get_device(device) if not isinstance(device, torch.device) else device
        self.sample_rate = sample_rate
        self.device = device
        self.stoi_loss = NegSTOILoss(sample_rate=sample_rate).to(device)

    def __call__(self, reference: torch.Tensor, estimate: torch.Tensor) -> float:
        """
        Compute STOI between reference and estimate.

        Args:
            reference: Clean reference (1D or 2D tensor)
            estimate: Estimated signal (1D or 2D tensor)

        Returns:
            STOI score (0-1, higher is better)
        """
        ref = reference.to(self.device)
        est = estimate.to(self.device)

        # Ensure 2D [batch, time]
        if ref.dim() == 1:
            ref = ref.unsqueeze(0)
        if est.dim() == 1:
            est = est.unsqueeze(0)

        # Align lengths
        min_len = min(ref.shape[-1], est.shape[-1])
        ref = ref[..., :min_len]
        est = est[..., :min_len]

        with torch.no_grad():
            # NegSTOILoss returns negative STOI, so negate it
            stoi_val = -self.stoi_loss(ref, est).item()
        return stoi_val


class GPUDNSMOSEvaluator:
    """
    GPU-accelerated DNSMOS evaluator using onnx2torch conversion.

    Converts ONNX DNSMOS models to PyTorch for GPU inference.
    ~170x faster than CPU ONNX inference.
    """

    def __init__(self, sample_rate: int = 16000, device: Optional[Union[str, torch.device]] = None):
        import speechmos
        from onnx2torch import convert

        device = get_device(device) if not isinstance(device, torch.device) else device
        self.sample_rate = sample_rate
        self.device = device
        self.INPUT_LENGTH = 9.01
        self.len_samples = int(self.INPUT_LENGTH * sample_rate)

        # Load and convert ONNX models to PyTorch
        pkg_dir = os.path.dirname(speechmos.__file__)
        primary_path = os.path.join(pkg_dir, "dnsmos_models", "sig_bak_ovr.onnx")
        p808_path = os.path.join(pkg_dir, "dnsmos_models", "model_v8.onnx")

        self.primary_model = convert(primary_path).to(device).eval()
        self.p808_model = convert(p808_path).to(device).eval()

        # Mel spectrogram transform for p808 model
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=321,
            hop_length=160,
            n_mels=120,
            power=2.0
        ).to(device)
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=80
        ).to(device)

        # Polynomial coefficients for score mapping
        self._p_ovr = [-0.06766283, 1.11546468, 0.04602535]
        self._p_sig = [-0.08397278, 1.22083953, 0.0052439]
        self._p_bak = [-0.13166888, 1.60915514, -0.39604546]

    def _polyval(self, coeffs: list, x: float) -> float:
        """Evaluate polynomial."""
        result = 0.0
        for c in coeffs:
            result = result * x + c
        return result

    def __call__(self, audio: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
        """
        Compute DNSMOS scores for audio.

        Args:
            audio: Audio signal (numpy array or torch tensor, values in [-1, 1])

        Returns:
            Dict with keys: 'sig_mos', 'bak_mos', 'ovrl_mos', 'p808_mos'
        """
        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio_t = torch.from_numpy(audio).float().to(self.device)
        else:
            audio_t = audio.float().to(self.device)

        audio_t = audio_t.squeeze()

        # Pad audio to required length
        while len(audio_t) < self.len_samples:
            audio_t = torch.cat([audio_t, audio_t])
        audio_t = audio_t[:self.len_samples]

        with torch.no_grad():
            # Primary model: raw audio -> SIG, BAK, OVR
            input_features = audio_t.unsqueeze(0)  # [1, samples]
            primary_out = self.primary_model(input_features)  # [1, 3]
            sig_raw, bak_raw, ovr_raw = primary_out[0].cpu().numpy()

            # P808 model: mel spectrogram -> P808 MOS
            audio_for_mel = audio_t[:-160]
            mel_spec = self.mel_transform(audio_for_mel)
            mel_spec_db = self.amp_to_db(mel_spec)
            mel_spec_norm = (mel_spec_db + 40) / 40
            p808_input = mel_spec_norm.T.unsqueeze(0)  # [1, time, n_mels]
            p808_out = self.p808_model(p808_input)
            p808_mos = p808_out.squeeze().item()

        # Apply polynomial mapping
        sig_mos = self._polyval(self._p_sig, sig_raw)
        bak_mos = self._polyval(self._p_bak, bak_raw)
        ovrl_mos = self._polyval(self._p_ovr, ovr_raw)

        return {
            'sig_mos': float(sig_mos),
            'bak_mos': float(bak_mos),
            'ovrl_mos': float(ovrl_mos),
            'p808_mos': float(p808_mos),
        }


class GPUMetricsEvaluator:
    """
    Unified GPU metrics evaluator for fast audio quality assessment.

    Combines: GPU DNSMOS, GPU STOI, GPU SI-SDR, PESQ.

    Usage:
        evaluator = GPUMetricsEvaluator(sample_rate=16000)  # autodetects CUDA -> MPS -> CPU
        metrics = evaluator(clean_audio, noisy_audio, enhanced_audio)

    Args:
        sample_rate: Audio sample rate (default 16000)
        device: Torch device for GPU metrics. ``None`` autodetects via
                :func:`se_probe.device.get_device`.
    """

    def __init__(self, sample_rate: int = 16000, device: Optional[Union[str, torch.device]] = None):
        device = get_device(device) if not isinstance(device, torch.device) else device
        self.sample_rate = sample_rate
        self.device = device

        self.dnsmos = GPUDNSMOSEvaluator(sample_rate, device)
        self.stoi = GPUSTOIEvaluator(sample_rate, device)

    def __call__(
        self,
        clean_audio: Union[np.ndarray, torch.Tensor],
        noisy_audio: Union[np.ndarray, torch.Tensor],
        enhanced_audio: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Compute all metrics for noisy and enhanced audio.

        Args:
            clean_audio: Clean reference signal
            noisy_audio: Degraded/noisy input signal
            enhanced_audio: Enhanced/processed output signal

        Returns:
            Dict with metrics for both noisy and enhanced signals:
            - SI-SDR, STOI, PESQ, DNSMOS (sig, bak, ovrl) for both noisy and enhanced
        """
        # Convert to tensors on GPU
        def to_tensor(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).float().to(self.device)
            return x.float().to(self.device)

        clean_t = to_tensor(clean_audio).squeeze()
        noisy_t = to_tensor(noisy_audio).squeeze()
        enhanced_t = to_tensor(enhanced_audio).squeeze()

        # Align lengths
        min_len = min(len(clean_t), len(noisy_t), len(enhanced_t))
        clean_t = clean_t[:min_len]
        noisy_t = noisy_t[:min_len]
        enhanced_t = enhanced_t[:min_len]

        # Get numpy arrays for CPU metrics
        clean_np = clean_t.cpu().numpy()
        noisy_np = noisy_t.cpu().numpy()
        enhanced_np = enhanced_t.cpu().numpy()

        metrics = {}

        # Noisy metrics (GPU)
        metrics['noisy_sisdr'] = gpu_sisdr(clean_t, noisy_t)
        metrics['noisy_stoi'] = self.stoi(clean_t, noisy_t)
        noisy_dnsmos = self.dnsmos(noisy_t)
        metrics['noisy_dnsmos_sig'] = noisy_dnsmos['sig_mos']
        metrics['noisy_dnsmos_bak'] = noisy_dnsmos['bak_mos']
        metrics['noisy_dnsmos_ovrl'] = noisy_dnsmos['ovrl_mos']

        # Enhanced metrics (GPU)
        metrics['enhanced_sisdr'] = gpu_sisdr(clean_t, enhanced_t)
        metrics['enhanced_stoi'] = self.stoi(clean_t, enhanced_t)
        enhanced_dnsmos = self.dnsmos(enhanced_t)
        metrics['enhanced_dnsmos_sig'] = enhanced_dnsmos['sig_mos']
        metrics['enhanced_dnsmos_bak'] = enhanced_dnsmos['bak_mos']
        metrics['enhanced_dnsmos_ovrl'] = enhanced_dnsmos['ovrl_mos']

        # PESQ (CPU only)
        try:
            metrics['noisy_pesq'] = pesq.pesq(self.sample_rate, clean_np, noisy_np, 'wb')
        except Exception:
            metrics['noisy_pesq'] = float('nan')
        try:
            metrics['enhanced_pesq'] = pesq.pesq(self.sample_rate, clean_np, enhanced_np, 'wb')
        except Exception:
            metrics['enhanced_pesq'] = float('nan')

        return metrics

    def close(self):
        """Cleanup (no-op, kept for API compatibility)."""
        pass

    def __del__(self):
        self.close()
