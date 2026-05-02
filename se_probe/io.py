import os
import glob
import librosa
import numpy as np
import scipy.io

from se_probe.consts import SAMPLE_RATE, TEST_SPEAKERS, CKA_RIRS, AIR_TEST_ROOMS

__all__ = ["load_clean_wavs", "load_air_rirs", "load_air_test_rirs"]


def load_clean_wavs():
    """
    Load clean speech from VCTK-Corpus for test speakers.
    Downsamples from 48kHz to 16kHz.

    Returns:
        List of numpy arrays containing clean waveforms at 16kHz.
    """
    from se_probe.consts import VCTK_DIR
    wavs = []
    for speaker in TEST_SPEAKERS:
        speaker_dir = os.path.join(VCTK_DIR, speaker)
        if not os.path.isdir(speaker_dir):
            print(f"Warning: Speaker directory not found: {speaker_dir}")
            continue
        wav_files = sorted(glob.glob(os.path.join(speaker_dir, '*.wav')))
        for wav_path in wav_files:
            # Load and downsample from 48kHz to 16kHz
            wav, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
            wavs.append(wav)
    return wavs


def load_air_rirs(filenames=None, sr=16000):
    """
    Load AIR RIRs from .mat files, align from peak, resample 48->16kHz.

    For each file:
    1. Load h_air from .mat and squeeze to 1D
    2. Align from first peak (remove pre-delay) at 48kHz for sub-sample accuracy
    3. Resample from 48kHz to target sample rate

    Args:
        filenames: List of .mat filenames. If None, uses CKA_RIRS from consts.
        sr: Target sample rate.

    Returns:
        Tuple of (rirs, rir_names) where rirs is a list of numpy arrays.
    """
    from se_probe.consts import AIR_RIR_DIR
    if filenames is None:
        filenames = CKA_RIRS

    rirs = []
    rir_names = []
    for fname in filenames:
        path = os.path.join(AIR_RIR_DIR, fname)
        mat = scipy.io.loadmat(path)
        rir = mat['h_air'].squeeze()

        # Align from first peak at native 48kHz for sub-sample accuracy
        peak = np.argmax(np.abs(rir))
        rir = rir[peak:]

        # Resample from 48kHz to target sr
        rir = librosa.resample(rir.astype(np.float32), orig_sr=48000, target_sr=sr)

        rirs.append(rir)
        rir_names.append(fname)

    return rirs, rir_names


def load_air_test_rirs(sr=16000):
    """Load all AIR RIRs from test-set rooms.

    Scans AIR_RIR_DIR for .mat files whose name contains a test room type
    (office, meeting, lecture, bathroom, kitchen, corridor).
    Each RIR is aligned from peak at 48kHz and resampled to target sr.

    Returns:
        Tuple of (rirs, rir_names) — lists of numpy arrays and filenames.
    """
    from se_probe.consts import AIR_RIR_DIR
    all_mat_files = sorted(glob.glob(os.path.join(AIR_RIR_DIR, "*.mat")))
    test_filenames = []
    for fpath in all_mat_files:
        fname = os.path.basename(fpath)
        if any(room in fname for room in AIR_TEST_ROOMS):
            test_filenames.append(fname)

    print(f"Found {len(test_filenames)} test-set RIRs from {len(AIR_TEST_ROOMS)} room types")
    return load_air_rirs(filenames=test_filenames, sr=sr)
