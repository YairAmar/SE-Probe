"""Small loaders shared by the analysis CLIs: chunk directories or one aggregated parquet."""
from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import pyarrow.parquet as pq

__all__ = ["read_table", "reduce_chunks", "parse_chunk_name"]

CHUNK_RE = re.compile(r"^(?P<model>[a-z0-9_]+?)__(?P<cond>[A-Za-z0-9_+\-.]+)__(?P<level>snr[+-]\d+|c50[+-]?\d+(?:\.\d+)?)\.parquet$")


def parse_chunk_name(name: str) -> Optional[dict]:
    m = CHUNK_RE.match(name)
    return m.groupdict() if m else None


def list_parquets(src: Path, pattern: str = "*.parquet") -> List[Path]:
    src = Path(src)
    if src.is_file():
        return [src]
    return sorted(Path(p) for p in glob.glob(str(src / "**" / pattern), recursive=True))


def read_table(src: Path, columns: Optional[Iterable[str]] = None, pattern: str = "*.parquet",
               filters=None) -> pd.DataFrame:
    """Concatenate one parquet or every parquet under a directory."""
    files = list_parquets(src, pattern)
    if not files:
        raise SystemExit(f"no parquet files under {src}")
    cols = list(columns) if columns is not None else None
    return pd.concat([pq.read_table(f, columns=cols, filters=filters).to_pandas() for f in files],
                     ignore_index=True)


def reduce_chunks(src: Path, by: List[str], pattern: str = "*.parquet", value: str = "CKA") -> pd.DataFrame:
    """Per-chunk ``groupby(by).mean()`` of ``value``, concatenated: keeps memory flat on
    the 96 M-row noise sweep (each chunk is one (model, noise, snr) cell, so the
    per-chunk mean over utterances is exact)."""
    files = list_parquets(src, pattern)
    if not files:
        raise SystemExit(f"no parquet files under {src}")
    parts = []
    for f in files:
        d = pq.read_table(f, columns=list(set(by) | {value})).to_pandas()
        parts.append(d.groupby(by, observed=True)[value].mean().reset_index())
    return pd.concat(parts, ignore_index=True)
