#!/usr/bin/env python3
"""Build the < 50 MB demo data subset shipped under ``results_demo/``.

Reads the full per-model SNR parquets, the epoch-48 reverb parquet, and the
per-layer diffusion-maps parquet from a local seint-style ``results_df/``
directory, filters them to a small, evenly-spaced slice, and writes 5 output
parquets.

Source path is required — pass ``--source-dir <path>`` or set the
``SEPROBE_RESULTS_DF`` env var. There is no built-in default (it would leak
the maintainer's local layout).

Usage:
    python scripts/build_demo_subset.py --source-dir /path/to/results_df
    SEPROBE_RESULTS_DF=/path/to/results_df python scripts/build_demo_subset.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

# Make the in-repo se_probe package importable when running this script
# directly from a checkout (no install needed).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_TARGET = _REPO_ROOT / "results_demo"
DEFAULT_BUDGET_MB = 50.0

DEFAULT_SNRS = [-10, 0, 10, 20, 30]
DEFAULT_NOISE = "TBUS"
DEFAULT_UTTS_PER_CELL = 10

SNR_FILES = {
    "muse": "snr/cka_snr_muse.parquet",
    "mpsenet": "snr/cka_snr_mpsenet.parquet",
    "demucs": "snr/cka_snr_demucs.parquet",
}
REVERB_FILE = "reverb/cka_reverb_muse_epoch_48.parquet"
DIFFUSION_FILE = "diffusion_maps/diffusion_maps_per_layer_t0.5.parquet"
DIFFUSION_ARCH_FILE = "diffusion_maps/diffusion_maps_architecture_t5.parquet"


def _filter_snr(df: pd.DataFrame, snrs: Iterable[int], noise: str, utts: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df[df["snr"].isin(list(snrs)) & (df["noise_name"] == noise)].copy()
    parts: List[pd.DataFrame] = []
    for (snr, n), grp in df.groupby(["snr", "noise_name"], sort=False):
        unique_clean = grp["clean_idx"].unique()
        if len(unique_clean) > utts:
            chosen = rng.choice(unique_clean, size=utts, replace=False)
            grp = grp[grp["clean_idx"].isin(chosen)]
        parts.append(grp)
    return pd.concat(parts, ignore_index=True) if parts else df


def _load_muse_norm1_layers() -> List[str]:
    """Load ``NORM1_LAYERS`` from ``se_probe/muse/consts.py`` without importing
    the full ``se_probe`` package (which pulls torch). This keeps the script
    runnable in a torch-less environment while still treating the package
    constants as the source of truth."""
    import importlib.util

    consts_path = _REPO_ROOT / "se_probe" / "muse" / "consts.py"
    spec = importlib.util.spec_from_file_location("_muse_consts_iso", consts_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.NORM1_LAYERS)


def _filter_diffusion(df: pd.DataFrame, snrs: Iterable[int], noise: str) -> pd.DataFrame:
    """Subset the per-layer diffusion-maps parquet to MUSE norm1 layers."""
    norm1_layers = _load_muse_norm1_layers()
    return df[
        df["layer"].isin(norm1_layers)
        & df["snr"].isin(list(snrs))
        & (df["noise_name"] == noise)
    ].copy()


def _filter_diffusion_arch(df: pd.DataFrame, snrs: Iterable[int]) -> pd.DataFrame:
    """Subset architecture-level diffusion maps to demo SNRs. No noise column
    in this parquet — the architecture-level diffusion is computed across
    noise types."""
    return df[df["snr"].isin([float(s) for s in snrs])].copy()


def _filter_reverb(df: pd.DataFrame, utts: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if "target_c50" not in df.columns:
        raise KeyError("reverb parquet missing 'target_c50' column")
    c50_levels = sorted(df["target_c50"].unique())
    chosen_c50 = c50_levels[:: max(1, len(c50_levels) // 5)][:5]
    df = df[df["target_c50"].isin(chosen_c50)].copy()

    rir_col = "rir_name" if "rir_name" in df.columns else None
    if rir_col is not None:
        first_rir = sorted(df[rir_col].unique())[0]
        df = df[df[rir_col] == first_rir]

    parts: List[pd.DataFrame] = []
    for c50, grp in df.groupby("target_c50", sort=False):
        unique_clean = grp["clean_idx"].unique()
        if len(unique_clean) > utts:
            chosen = rng.choice(unique_clean, size=utts, replace=False)
            grp = grp[grp["clean_idx"].isin(chosen)]
        parts.append(grp)
    return pd.concat(parts, ignore_index=True) if parts else df


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source-dir", type=Path, default=None,
                        help="Source results_df/ directory (required; overrides $SEPROBE_RESULTS_DF).")
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET,
                        help=f"Output dir (default: {DEFAULT_TARGET})")
    parser.add_argument("--noise", default=DEFAULT_NOISE,
                        help=f"Noise to keep (default: {DEFAULT_NOISE})")
    parser.add_argument("--utts-per-cell", type=int, default=DEFAULT_UTTS_PER_CELL,
                        help=f"Utterances per (snr, noise) cell (default: {DEFAULT_UTTS_PER_CELL})")
    parser.add_argument("--budget-mb", type=float, default=DEFAULT_BUDGET_MB,
                        help=f"Total byte budget in MB (default: {DEFAULT_BUDGET_MB})")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source_dir = args.source_dir or (
        Path(os.environ["SEPROBE_RESULTS_DF"])
        if "SEPROBE_RESULTS_DF" in os.environ
        else None
    )
    if source_dir is None:
        print(
            "ERROR: source results_df/ directory is required.\n"
            "       Pass --source-dir <path> or set $SEPROBE_RESULTS_DF.",
            file=sys.stderr,
        )
        return 2

    src = source_dir.expanduser().resolve()
    dst = args.target_dir.expanduser().resolve()
    if not src.exists():
        print(f"ERROR: source dir not found: {src}", file=sys.stderr)
        return 2

    dst.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for model, rel in SNR_FILES.items():
        in_path = src / rel
        if not in_path.exists():
            print(f"ERROR: missing input parquet: {in_path}", file=sys.stderr)
            return 2
        df = pd.read_parquet(in_path)
        sub = _filter_snr(df, DEFAULT_SNRS, args.noise, args.utts_per_cell, args.seed)
        out_path = dst / f"cka_snr_{model}_demo.parquet"
        sub.to_parquet(out_path, index=False)
        written.append(out_path)
        print(f"  {out_path.name}: {len(sub):>7} rows  ({_human_bytes(out_path.stat().st_size)})")

    reverb_in = src / REVERB_FILE
    if not reverb_in.exists():
        print(f"ERROR: missing reverb parquet: {reverb_in}", file=sys.stderr)
        return 2
    rev_df = pd.read_parquet(reverb_in)
    rev_sub = _filter_reverb(rev_df, args.utts_per_cell, args.seed)
    rev_out = dst / "cka_reverb_muse_demo.parquet"
    rev_sub.to_parquet(rev_out, index=False)
    written.append(rev_out)
    print(f"  {rev_out.name}: {len(rev_sub):>7} rows  ({_human_bytes(rev_out.stat().st_size)})")

    diff_in = src / DIFFUSION_FILE
    if not diff_in.exists():
        print(f"ERROR: missing diffusion-maps parquet: {diff_in}", file=sys.stderr)
        return 2
    diff_df = pd.read_parquet(diff_in)
    diff_sub = _filter_diffusion(diff_df, DEFAULT_SNRS, args.noise)
    diff_out = dst / "diffusion_maps_per_layer_demo.parquet"
    diff_sub.to_parquet(diff_out, index=False)
    written.append(diff_out)
    print(f"  {diff_out.name}: {len(diff_sub):>7} rows  ({_human_bytes(diff_out.stat().st_size)})")

    arch_in = src / DIFFUSION_ARCH_FILE
    if not arch_in.exists():
        print(f"ERROR: missing diffusion-arch parquet: {arch_in}", file=sys.stderr)
        return 2
    arch_df = pd.read_parquet(arch_in)
    arch_sub = _filter_diffusion_arch(arch_df, DEFAULT_SNRS)
    arch_out = dst / "diffusion_maps_architecture_demo.parquet"
    arch_sub.to_parquet(arch_out, index=False)
    written.append(arch_out)
    print(f"  {arch_out.name}: {len(arch_sub):>7} rows  ({_human_bytes(arch_out.stat().st_size)})")

    total = sum(p.stat().st_size for p in written)
    budget = int(args.budget_mb * 1024 * 1024)
    print(f"\nTotal: {_human_bytes(total)} (budget: {args.budget_mb} MB)")
    if total > budget:
        print(
            f"ERROR: demo data ({_human_bytes(total)}) exceeds budget ({args.budget_mb} MB).\n"
            "       Tighten via --utts-per-cell (try 5).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
