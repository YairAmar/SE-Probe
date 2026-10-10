"""Time-domain reverb datasets for Demucs (denoiser DNS64) dereverberation FT.

Thin, self-contained re-implementation of the RIR-convolution logic used by
``datasets/reverb_dataset.py`` (the MP-SENet / MUSE reverb protocol), but
returning only the two raw waveforms ``(clean_audio, reverb_audio)`` at 16 kHz
that a time-domain model needs. Kept self-contained (no import from
reverb_dataset) so it does not depend on that module's in-progress state.

Reverb-pair generation matches the MUSE protocol exactly:
  1. RIRs loaded eagerly, channel 0, onset-clipped at absolute peak (argmax).
  2. On-the-fly ``fftconvolve(clean, rir, "full")[:segment_size]``.
  3. Energy-normalize by the reverb energy: both clean and reverb are scaled by
     ``sqrt(len / sum(reverb**2))`` so the reverberant signal has unit RMS.
Random RIR per item for training; deterministic (index % n) for validation.
"""

import os
import random

import numpy as np
import torch
import torch.utils.data
import soundfile as sf
import librosa
from scipy.signal import fftconvolve


def load_rir(rir_dir, wav_path, channel=0):
    """Load one RIR: read WAV, take a channel, onset-clip at absolute peak."""
    full_path = os.path.join(rir_dir, wav_path)
    rir, _ = sf.read(full_path, dtype="float32")
    if rir.ndim > 1:
        ch = min(channel, rir.shape[1] - 1)
        rir = rir[:, ch]
    peak_idx = np.argmax(np.abs(rir))
    return rir[peak_idx:]


def load_rirs(rir_dir, rir_wav_paths, label=""):
    """Eagerly load and onset-clip a list of RIRs into memory."""
    rirs = [load_rir(rir_dir, wp) for wp in sorted(rir_wav_paths)]
    print(f"{label}: loaded {len(rirs)} RIRs into memory (onset-clipped, ch0)")
    return rirs


class DemucsReverbDataset(torch.utils.data.Dataset):
    """Training dataset: returns 1D ``(clean_audio, reverb_audio)`` of length
    ``segment_size``. Clean speech is chunk-indexed; a random RIR is convolved
    per item."""

    def __init__(self, training_indexes, clean_wavs_dir, rir_dir, rir_wav_paths,
                 segment_size, sampling_rate, shuffle=True, seed=1234):
        self.audio_indexes = list(training_indexes)
        self.clean_wavs_dir = clean_wavs_dir
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate

        random.seed(seed)
        if shuffle:
            random.shuffle(self.audio_indexes)

        self.rirs = load_rirs(rir_dir, rir_wav_paths, "DemucsReverbDataset")

        # Build flat chunk index: (utterance_idx, chunk_start) -- identical to
        # the MUSE reverb protocol so an "epoch" covers the same chunks.
        self.chunk_index = []
        for utt_idx, filename in enumerate(self.audio_indexes):
            wav_path = os.path.join(self.clean_wavs_dir, filename + ".wav")
            length = int(sf.info(wav_path).frames)
            if length <= segment_size:
                self.chunk_index.append((utt_idx, 0))
            else:
                n_chunks = length // segment_size
                for c in range(n_chunks):
                    self.chunk_index.append((utt_idx, c * segment_size))
                if length % segment_size > 0:
                    self.chunk_index.append((utt_idx, length - segment_size))

        print(f"DemucsReverbDataset: {len(self.audio_indexes)} utterances -> "
              f"{len(self.chunk_index)} chunks (segment_size={segment_size})")

    def __getitem__(self, flat_idx):
        utt_idx, chunk_start = self.chunk_index[flat_idx]
        filename = self.audio_indexes[utt_idx]

        wav_path = os.path.join(self.clean_wavs_dir, filename + ".wav")
        clean_audio, _ = librosa.load(
            wav_path, sr=self.sampling_rate,
            offset=chunk_start / self.sampling_rate,
            duration=self.segment_size / self.sampling_rate,
        )
        if len(clean_audio) < self.segment_size:
            clean_audio = np.pad(clean_audio, (0, self.segment_size - len(clean_audio)))

        rir = self.rirs[random.randint(0, len(self.rirs) - 1)]
        reverb_audio = fftconvolve(clean_audio, rir, mode="full")[:self.segment_size]

        clean_audio = torch.FloatTensor(clean_audio)
        reverb_audio = torch.FloatTensor(reverb_audio)

        # Energy-normalize by reverb energy (same as MUSE protocol).
        norm_factor = torch.sqrt(len(reverb_audio) / torch.sum(reverb_audio ** 2.0))
        clean_audio = clean_audio * norm_factor
        reverb_audio = reverb_audio * norm_factor

        return clean_audio, reverb_audio

    def __len__(self):
        return len(self.chunk_index)


class DemucsReverbValDataset(torch.utils.data.Dataset):
    """Validation dataset: full-length ``(clean_audio, reverb_audio)`` pairs with
    deterministic RIR assignment for reproducibility."""

    def __init__(self, validation_indexes, clean_wavs_dir, rir_dir, rir_wav_paths, sampling_rate):
        self.audio_indexes = list(validation_indexes)
        self.clean_wavs_dir = clean_wavs_dir
        self.sampling_rate = sampling_rate
        self.rirs = load_rirs(rir_dir, rir_wav_paths, "DemucsReverbValDataset")

    def __getitem__(self, index):
        filename = self.audio_indexes[index]
        clean_audio, _ = librosa.load(
            os.path.join(self.clean_wavs_dir, filename + ".wav"),
            sr=self.sampling_rate,
        )

        rir = self.rirs[index % len(self.rirs)]
        reverb_audio = fftconvolve(clean_audio, rir, mode="full")[:len(clean_audio)]

        clean_audio = torch.FloatTensor(clean_audio)
        reverb_audio = torch.FloatTensor(reverb_audio)

        norm_factor = torch.sqrt(len(reverb_audio) / torch.sum(reverb_audio ** 2.0))
        clean_audio = clean_audio * norm_factor
        reverb_audio = reverb_audio * norm_factor

        return clean_audio, reverb_audio

    def __len__(self):
        return len(self.audio_indexes)
