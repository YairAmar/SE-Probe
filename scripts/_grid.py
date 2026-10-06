"""Shared machinery of the GPU probing grids: extraction, CKA rows, resumable chunks.

Every ``run_*_grid.py`` driver reuses these pieces so that the SNR sweep, the
emergence checkpoints, the random-initialisation control and the fine-tuned arms
are probed by one code path. A chunk is one parquet per ``(model, condition,
level)``; existing chunks are skipped so a job can be re-submitted after a
walltime limit and resume where it stopped.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from se_probe.cka import cka

__all__ = ["Heartbeat", "extract_stack", "cka_rows", "pad_or_trim", "chunk_loop",
           "DEMUCS_LEN", "MODEL_NAMES", "load_extractor"]

#: Demucs activations are time-dependent, so utterances are padded/truncated to a
#: fixed 4 s window (covers >95% of VoiceBank-DEMAND test utterances).
DEMUCS_LEN = 64000
MODEL_NAMES = {"muse": "MUSE", "mpsenet": "MP-SENet", "demucs": "Demucs"}


class Heartbeat:
    """Timestamped progress lines, mirrored to a log file when given."""

    def __init__(self, tag: str, log_path: Optional[Path] = None):
        self.tag = tag
        self.log_path = Path(log_path) if log_path else None
        self.t0 = time.time()
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{self.tag}] {msg} | elapsed {(time.time() - self.t0) / 60:.1f} min"
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(line + "\n")
        print(line, flush=True)


def pad_or_trim(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Zero-pad or truncate a waveform to exactly ``target_len`` samples."""
    if len(audio) >= target_len:
        return np.asarray(audio[:target_len], dtype="float32")
    out = np.zeros(target_len, dtype="float32")
    out[:len(audio)] = audio
    return out


def extract_stack(extractor, audios: Sequence[np.ndarray],
                  layers: Optional[Sequence[str]] = None) -> Dict[str, torch.Tensor]:
    """``{layer: (n_utts, ...)}`` CPU tensors of pooled activations.

    Each utterance's activations move to the CPU before stacking, which keeps the
    GPU footprint flat (Demucs would otherwise exceed 40 GB at 824 utterances).
    """
    per: Dict[str, List[torch.Tensor]] = {}
    for a in audios:
        result = extractor(a)
        acts = result[0] if isinstance(result, tuple) else result
        for L, v in acts.items():
            if layers is not None and L not in layers:
                continue
            per.setdefault(L, []).append(v.detach().cpu().unsqueeze(0))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {L: torch.cat(vs, 0) for L, vs in per.items()}


def cka_rows(clean_stack: Dict[str, torch.Tensor], deg_stack: Dict[str, torch.Tensor],
             device: torch.device, base: dict, offset: int = 0) -> List[dict]:
    """Per-utterance CKA rows for every layer of two stacks; ``base`` holds the
    constant columns of the chunk (snr/noise/model...)."""
    rows = []
    for L in clean_stack:
        a = clean_stack[L].to(device)
        b = deg_stack[L].to(device)
        vals = cka(a, b)
        del a, b
        if device.type == "cuda":
            torch.cuda.empty_cache()
        vals_np = vals.detach().cpu().numpy() if isinstance(vals, torch.Tensor) else np.asarray(vals)
        for u, v in enumerate(vals_np.flatten()):
            rows.append({"clean_idx": u + offset, "layer": L, "CKA": float(v), **base})
    return rows


def chunk_loop(conditions: Iterable, chunk_path: Callable[..., Path],
               make_rows: Callable[..., List[dict]], hb: Heartbeat) -> int:
    """Run ``make_rows(*cond)`` for every condition whose chunk does not exist yet
    and write the result to ``chunk_path(*cond)``. Returns the number of chunks
    written."""
    conditions = list(conditions)
    done = sum(chunk_path(*c).exists() for c in conditions)
    written = 0
    hb(f"{done}/{len(conditions)} chunks already present")
    for c in conditions:
        cp = chunk_path(*c)
        if cp.exists():
            continue
        t = time.perf_counter()
        rows = make_rows(*c)
        cp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(cp, index=False)
        done += 1
        written += 1
        hb(f"{cp.name}: {len(rows)} rows in {time.perf_counter() - t:.1f}s | {done}/{len(conditions)} chunks")
    return written


def load_extractor(model: str, device, checkpoint: Optional[str] = None, reverb: bool = False,
                   random_seed: Optional[int] = None):
    """The activation extractor of one model/arm.

    ``reverb=True`` selects the mean-over-segments pooling used on the C50 axis
    (MUSE, MP-SENet); ``checkpoint`` loads a fine-tuned generator;
    ``random_seed`` builds the untrained MUSE control instead.
    """
    if model == "muse":
        from se_probe.muse.model import (
            load_muse_activation_extractor,
            load_muse_activation_extractor_reverb,
            load_random_init_muse_activation_extractor,
        )
        if random_seed is not None:
            return load_random_init_muse_activation_extractor(random_seed, device=device)
        if reverb:
            return load_muse_activation_extractor_reverb(device=device, checkpoint_path=checkpoint)
        return load_muse_activation_extractor(device=device, checkpoint_path=checkpoint)
    if model == "mpsenet":
        from se_probe.mpsenet.model import (
            load_mpsenet_activation_extractor,
            load_mpsenet_activation_extractor_reverb,
        )
        if reverb or checkpoint:
            return load_mpsenet_activation_extractor_reverb(device=device, checkpoint_path=checkpoint)
        return load_mpsenet_activation_extractor(device=device)
    if model == "demucs":
        from se_probe.demucs.model import (
            load_demucs_activation_extractor,
            load_demucs_activation_extractor_reverb,
        )
        if reverb or checkpoint:
            return load_demucs_activation_extractor_reverb(device=device, checkpoint_path=checkpoint)
        return load_demucs_activation_extractor(device=device)
    raise ValueError(model)
