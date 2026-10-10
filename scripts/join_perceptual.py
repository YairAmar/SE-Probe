#!/usr/bin/env python
"""Join the per-mixture quality metrics onto the per-layer CKA rows, with an exact
coverage report (paper Sec. III-B input table).

The CKA chunks store ``model_name`` as ``MUSE / MP-SENet / Demucs`` while the metric
chunks may store ``muse / mpsenet / demucs``; both sides are canonicalised before
the merge (a raw merge once matched zero rows). Coverage is counted over every row
written, never from a sample.

Inputs: a directory of ``perc_*.parquet`` chunks (``run_perceptual_grid.py``) and a
directory of CKA chunks (``run_snr_grid.py``) or one aggregated parquet.
Outputs: ``perceptual_snr.parquet`` (per-mixture) and ``cka_snr_metrics.parquet``
(CKA rows + metric columns, restricted to the noises present in the metric table)
plus ``coverage.md``.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CANON = {"muse": "muse", "mp-senet": "mpsenet", "mpsenet": "mpsenet", "mp_senet": "mpsenet", "demucs": "demucs"}
METRIC_COLS = [f"{p}_{m}" for p in ("noisy", "enhanced")
               for m in ("pesq", "stoi", "sisdr", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl")]
KEYS = ["clean_idx", "snr", "noise_name", "model_name"]


def canon(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().map(lambda x: CANON.get(x, x))


def iter_cka_frames(src: Path, columns):
    files = [src] if src.is_file() else sorted(Path(p) for p in glob.glob(str(src / "**" / "*.parquet"), recursive=True))
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.metadata.num_row_groups):
            yield pf.read_row_groups([rg], columns=columns).to_pandas()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--perceptual-dir", required=True, type=Path)
    p.add_argument("--cka", required=True, type=Path, help="CKA chunk dir or aggregated parquet")
    p.add_argument("--out-dir", required=True, type=Path)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(p) for p in glob.glob(str(a.perceptual_dir / "**" / "perc_*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no perc_*.parquet under {a.perceptual_dir}")
    td = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    td["model_name"] = canon(td["model_name"])
    td = td.drop_duplicates(subset=KEYS).reset_index(drop=True)
    perc_out = a.out_dir / "perceptual_snr.parquet"
    td.to_parquet(perc_out, index=False)
    lines = [f"# Perceptual join coverage\n\nper-mixture rows: {len(td):,} from {len(files)} chunks\n"]
    lines.append("| model | rows | noises | SNRs | utts | " + " | ".join(METRIC_COLS) + " |")
    lines.append("|---" * (len(METRIC_COLS) + 5) + "|")
    for m, sub in td.groupby("model_name"):
        lines.append(f"| {m} | {len(sub):,} | {sub.noise_name.nunique()} | {sub.snr.nunique()} | {sub.clean_idx.nunique()} | "
                     + " | ".join(f"{int(sub[c].notna().sum()):,}" if c in sub else "-" for c in METRIC_COLS) + " |")
    if "noisy_sisdr" in td:
        for m, sub in td.groupby("model_name"):
            r = float(sub[["snr", "noisy_sisdr"]].dropna().corr().iloc[0, 1])
            lines.append(f"\nSI-SDR sign sanity, {m}: corr(noisy_sisdr, snr) = {r:+.4f} (expect ~ +1)")

    noises = set(td["noise_name"].unique())
    metrics = td[KEYS + [c for c in METRIC_COLS if c in td.columns]]
    join_out = a.out_dir / "cka_snr_metrics.parquet"
    writer = None
    total, nonnull = 0, {c: 0 for c in METRIC_COLS}
    for t in iter_cka_frames(a.cka, None):
        t["model_name"] = canon(t["model_name"])
        t = t[t["noise_name"].isin(noises)]
        if not len(t):
            continue
        t = t.drop(columns=[c for c in METRIC_COLS if c in t.columns])
        j = t.merge(metrics, on=KEYS, how="left")
        total += len(j)
        for c in METRIC_COLS:
            if c in j:
                nonnull[c] += int(j[c].notna().sum())
        tbl = pa.Table.from_pandas(j, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(join_out, tbl.schema)
        writer.write_table(tbl)
    if writer is not None:
        writer.close()
    lines.append(f"\n## Joined CKA table\n\n`{join_out}`: {total:,} rows\n")
    lines.append("| column | non-null | % |\n|---|---:|---:|")
    for c in METRIC_COLS:
        lines.append(f"| `{c}` | {nonnull[c]:,} | {100 * nonnull[c] / max(total, 1):.2f}% |")
    (a.out_dir / "coverage.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
