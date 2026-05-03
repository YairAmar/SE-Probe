"""ReverbDataset: on-the-fly reverb generation with chunk indexing.

Convolves clean speech with onset-clipped RIRs during training.
Returns the same 7-tuple as the original Dataset class.

All dataset classes accept:
  - rir_dir: root directory of raw RIRs (e.g. /home/.../rirmega)
  - rir_wav_paths: pre-filtered list of wav_path strings relative to rir_dir

RIRs are loaded eagerly into memory during __init__:
  1. Read raw multi-channel WAV
  2. Extract channel 0
  3. Onset-clip at absolute peak
"""

import os
import random
import numpy as np
import torch
import torch.utils.data
import soundfile as sf
import librosa
from scipy.signal import fftconvolve
from datasets.dataset import mag_pha_stft


def load_rir(rir_dir, wav_path, channel=0):
    """Load a single RIR: read WAV, extract channel, onset-clip at peak.

    Args:
        rir_dir: root directory of RIR dataset
        wav_path: relative path to WAV file within rir_dir
        channel: channel index to extract (default 0)

    Returns:
        float32 1D numpy array (onset-clipped mono RIR)
    """
    full_path = os.path.join(rir_dir, wav_path)
    rir, _ = sf.read(full_path, dtype="float32")
    if rir.ndim > 1:
        ch = min(channel, rir.shape[1] - 1)
        rir = rir[:, ch]
    # Onset clip at absolute peak
    peak_idx = np.argmax(np.abs(rir))
    return rir[peak_idx:]


def load_rirs(rir_dir, rir_wav_paths, label=""):
    """Eagerly load and onset-clip a list of RIRs into memory."""
    rirs = []
    for wp in sorted(rir_wav_paths):
        rirs.append(load_rir(rir_dir, wp))
    print(f"{label}: loaded {len(rirs)} RIRs into memory (onset-clipped, ch0)")
    return rirs


class ReverbDataset(torch.utils.data.Dataset):
    def __init__(self, training_indexes, clean_wavs_dir, rir_dir, rir_wav_paths,
                 segment_size, n_fft, hop_size, win_size, sampling_rate, compress_factor,
                 split=True, shuffle=True, device=None):
        self.audio_indexes = list(training_indexes)
        self.clean_wavs_dir = clean_wavs_dir
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.compress_factor = compress_factor
        self.split = split
        self.device = device

        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_indexes)

        self.rirs = load_rirs(rir_dir, rir_wav_paths, "ReverbDataset")

        # Build flat chunk index: (utterance_idx, chunk_start)
        self.chunk_index = []
        for utt_idx, filename in enumerate(self.audio_indexes):
            wav_path = os.path.join(self.clean_wavs_dir, filename + ".wav")
            info = sf.info(wav_path)
            length = int(info.frames)

            if length <= segment_size:
                self.chunk_index.append((utt_idx, 0))
            else:
                n_chunks = length // segment_size
                for c in range(n_chunks):
                    self.chunk_index.append((utt_idx, c * segment_size))
                remainder = length % segment_size
                if remainder > 0:
                    self.chunk_index.append((utt_idx, length - segment_size))

        print(f"ReverbDataset: {len(self.audio_indexes)} utterances -> "
              f"{len(self.chunk_index)} chunks (segment_size={segment_size})")

    def __getitem__(self, flat_idx):
        utt_idx, chunk_start = self.chunk_index[flat_idx]
        filename = self.audio_indexes[utt_idx]

        # Load only the needed chunk
        wav_path = os.path.join(self.clean_wavs_dir, filename + ".wav")
        clean_audio, _ = librosa.load(
            wav_path, sr=self.sampling_rate,
            offset=chunk_start / self.sampling_rate,
            duration=self.segment_size / self.sampling_rate,
        )

        # Pad if shorter than segment_size
        if len(clean_audio) < self.segment_size:
            clean_audio = np.pad(clean_audio, (0, self.segment_size - len(clean_audio)))

        # Random RIR convolution
        rir = self.rirs[random.randint(0, len(self.rirs) - 1)]
        reverb_audio = fftconvolve(clean_audio, rir, mode="full")[:self.segment_size]

        # Convert to tensors
        clean_audio = torch.FloatTensor(clean_audio)
        reverb_audio = torch.FloatTensor(reverb_audio)

        # Normalize (same as original Dataset)
        norm_factor = torch.sqrt(len(reverb_audio) / torch.sum(reverb_audio ** 2.0))
        clean_audio = (clean_audio * norm_factor).unsqueeze(0)
        reverb_audio = (reverb_audio * norm_factor).unsqueeze(0)

        # STFT
        clean_mag, clean_pha, clean_com = mag_pha_stft(
            clean_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor)
        reverb_mag, reverb_pha, reverb_com = mag_pha_stft(
            reverb_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor)

        return (clean_audio.squeeze(), clean_mag.squeeze(), clean_pha.squeeze(),
                clean_com.squeeze(), reverb_audio.squeeze(), reverb_mag.squeeze(),
                reverb_pha.squeeze())

    def __len__(self):
        return len(self.chunk_index)


class ReverbValDataset(torch.utils.data.Dataset):
    """Validation dataset for reverb mode.

    Returns full-length (clean_audio, reverb_audio) pairs -- same interface as Val_Dataset.
    Uses provided RIR wav_paths with deterministic assignment for reproducibility.
    """

    def __init__(self, validation_indexes, clean_wavs_dir, rir_dir, rir_wav_paths, sampling_rate):
        self.audio_indexes = list(validation_indexes)
        self.clean_wavs_dir = clean_wavs_dir
        self.sampling_rate = sampling_rate

        self.rirs = load_rirs(rir_dir, rir_wav_paths, "ReverbValDataset")

    def __getitem__(self, index):
        filename = self.audio_indexes[index]
        clean_audio, _ = librosa.load(
            os.path.join(self.clean_wavs_dir, filename + ".wav"),
            sr=self.sampling_rate,
        )

        # Deterministic RIR assignment
        rir = self.rirs[index % len(self.rirs)]
        reverb_audio = fftconvolve(clean_audio, rir, mode="full")[:len(clean_audio)]

        clean_audio = torch.FloatTensor(clean_audio)
        reverb_audio = torch.FloatTensor(reverb_audio)

        # Normalize by reverb energy (same as training and Val_Dataset)
        norm_factor = torch.sqrt(len(reverb_audio) / torch.sum(reverb_audio ** 2.0))
        clean_audio = clean_audio * norm_factor
        reverb_audio = reverb_audio * norm_factor

        return clean_audio, reverb_audio

    def __len__(self):
        return len(self.audio_indexes)

