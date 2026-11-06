"""Clean-utterance sources for the probing grids.

The paper's probing material is the VoiceBank-DEMAND *test* split (824 utterances,
speakers p232 then p257, 16 kHz). Two sources are supported, both returning the
same ``(ids, audios)`` ordering (sorted by utterance id, so ``clean_idx`` 0..392
is p232 and 393..823 is p257):

* a directory of 16 kHz mono wavs (the ``clean_test`` folder of VoiceBank-DEMAND);
* the HuggingFace dataset ``JacobLinCool/VoiceBank-DEMAND-16k`` (``--clean hf``),
  read through ``datasets`` or, if that import fails, directly from the cached
  arrow files with pyarrow + soundfile.

Set ``SEPROBE_CLEAN_TEST_DIR`` to avoid repeating the path on every command line.
"""
from __future__ import annotations

import glob
import io
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

HF_DATASET = "JacobLinCool/VoiceBank-DEMAND-16k"
SAMPLE_RATE = 16000

__all__ = ["HF_DATASET", "load_voicebank_test", "resolve_clean_source"]


def resolve_clean_source(arg: Optional[str]) -> str:
    """``arg`` or ``$SEPROBE_CLEAN_TEST_DIR``; ``"hf"`` selects the HuggingFace copy."""
    src = arg or os.environ.get("SEPROBE_CLEAN_TEST_DIR")
    if not src:
        raise SystemExit("clean source required: pass --clean <dir|hf> or set SEPROBE_CLEAN_TEST_DIR")
    return src


def _load_dir(root: str, n_utts: Optional[int]) -> Tuple[List[str], List[np.ndarray]]:
    import soundfile as sf

    files = sorted(glob.glob(os.path.join(root, "*.wav")))
    if not files:
        raise FileNotFoundError(f"no wav files under {root}")
    if n_utts is not None:
        files = files[:n_utts]
    ids, audios = [], []
    for f in files:
        data, sr = sf.read(f, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]
        if sr != SAMPLE_RATE:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE).astype("float32")
        ids.append(Path(f).stem)
        audios.append(np.ascontiguousarray(data, dtype="float32"))
    return ids, audios


def _load_hf_datasets(n_utts: Optional[int]):
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split="test")
    rows = []
    for ex in ds:
        a = ex["clean"]
        arr = np.asarray(a["array"], dtype="float32")
        if int(a["sampling_rate"]) != SAMPLE_RATE:
            raise RuntimeError(f"unexpected sampling rate {a['sampling_rate']}")
        rows.append((ex["id"], arr))
    rows.sort(key=lambda r: r[0])
    if n_utts is not None:
        rows = rows[:n_utts]
    return [r[0] for r in rows], [r[1] for r in rows]


def _load_hf_arrow(n_utts: Optional[int]):
    """Bypass ``datasets`` (which may need torchcodec) and read the cached arrow shards."""
    import pyarrow.ipc as ipc
    import soundfile as sf

    cache = Path(os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets")))
    shards = sorted(cache.glob("JacobLinCool___voice_bank-demand-16k/**/voice_bank-demand-16k-test*.arrow"))
    if not shards:
        raise FileNotFoundError(f"no cached test arrow files under {cache}; run with `datasets` once")
    rows = []
    for ap in shards:
        with open(ap, "rb") as f:
            tbl = ipc.open_stream(f).read_all()
        for i, c in zip(tbl.column("id").to_pylist(), tbl.column("clean").to_pylist()):
            b = c.get("bytes") if isinstance(c, dict) else None
            if b is not None:
                rows.append((i, b))
    rows.sort(key=lambda r: r[0])
    if n_utts is not None:
        rows = rows[:n_utts]
    ids, audios = [], []
    for i, b in rows:
        data, sr = sf.read(io.BytesIO(b), dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]
        assert sr == SAMPLE_RATE, f"unexpected sr={sr} for {i}"
        ids.append(i)
        audios.append(data.astype("float32"))
    return ids, audios


def load_voicebank_test(source: str, n_utts: Optional[int] = None) -> Tuple[List[str], List[np.ndarray]]:
    """Return ``(ids, audios)`` of the VoiceBank-DEMAND test split, sorted by id.

    ``source`` is a wav directory or ``"hf"``.
    """
    if source.lower() == "hf":
        try:
            return _load_hf_datasets(n_utts)
        except Exception as e:  # torchcodec / datasets incompatibilities
            print(f"[clean] `datasets` path failed ({type(e).__name__}: {e}); trying the arrow cache")
            return _load_hf_arrow(n_utts)
    return _load_dir(source, n_utts)
