#!/usr/bin/env python
"""Quality-association tables (paper Sec. III-B "Connection to Perceptual Quality").

Input: the joined CKA + metrics parquet of ``join_perceptual.py`` (or the earlier
per-utterance sweep tables that carry the metric columns). Writes, per model:

* ``perceptual_per_layer.csv``: per-layer Pearson and Spearman correlation between
  within-(layer, SNR)-centred CKA and the centred improvement of every metric
  present (PESQ, STOI, SI-SDR, DNSMOS SIG/BAK/OVRL);
* ``speaker_per_speaker.csv``, ``speaker_utterance_level.csv`` (with utterance-cluster
  bootstrap intervals) and ``speaker_level.csv`` (Student-t intervals over the
  speakers of the sweep) from ``se_probe.perceptual.speaker_level_analysis``;
* with two sweeps (``--src-780``), ``speaker_level_pooled.csv``: the four-speaker
  pooled intervals the paper reports for MUSE.

The stored SI-SDR sign is checked per table and corrected if an old helper negated it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from _tables import read_table  # noqa: E402

from se_probe.layers import probed_layers  # noqa: E402
from se_probe.perceptual import (  # noqa: E402
    PAPER_N_BOOT_SPEAKER,
    PAPER_SEED_SPEAKER,
    available_metrics,
    per_layer_correlations,
    sisdr_sign_convention,
    speaker_level_analysis,
    speaker_summary_from_published,
)

CANON = {"muse": "muse", "mp-senet": "mpsenet", "mpsenet": "mpsenet", "demucs": "demucs"}


def canon(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().map(lambda x: CANON.get(x, x))


def analyze_sweep(df: pd.DataFrame, sweep: str, n_boot: int, seed: int) -> dict:
    """Per-layer and speaker-level tables for every model in ``df``."""
    df = df.copy()
    df["model_name"] = canon(df["model_name"])
    if "noisy_sisdr" in df.columns:
        conv = sisdr_sign_convention(df)
        if conv["needs_negation"]:
            print(f"[{sweep}] stored SI-SDR is negated (corr={conv['corr']:+.3f}); correcting")
            for c in ("noisy_sisdr", "enhanced_sisdr"):
                df[c] = -df[c]
    per_layer, utt, per_spk, spk = [], [], [], []
    for model, d in df.groupby("model_name"):
        metrics = available_metrics(d)
        if not metrics:
            print(f"[{sweep}] {model}: no populated metric columns, skipped")
            continue
        layers = probed_layers(model, d["layer"].unique())
        t = per_layer_correlations(d, layers, metrics, model=model)
        t.insert(0, "sweep", sweep)
        t.insert(1, "model", model)
        per_layer.append(t)
        r = speaker_level_analysis(d, model, sweep=sweep, metrics=metrics, layers=layers, n_boot=n_boot, seed=seed)
        utt.append(r["utterance"])
        per_spk.append(r["per_speaker"])
        spk.append(r["speaker"])
        print(f"[{sweep}] {model}: {len(layers)} layers, metrics {metrics}")
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()  # noqa: E731
    return {"per_layer": cat(per_layer), "utterance": cat(utt), "per_speaker": cat(per_spk), "speaker": cat(spk)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, type=Path, help="824-sweep joined parquet (or chunk dir)")
    p.add_argument("--src-780", type=Path, default=None, help="optional 780-sweep table (p226/p287)")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--n-boot", type=int, default=PAPER_N_BOOT_SPEAKER)
    p.add_argument("--seed", type=int, default=PAPER_SEED_SPEAKER)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    res = [analyze_sweep(read_table(a.src), "824", a.n_boot, a.seed)]
    if a.src_780 is not None:
        res.append(analyze_sweep(read_table(a.src_780), "780", a.n_boot, a.seed))
    for key, name in (("per_layer", "perceptual_per_layer.csv"), ("utterance", "speaker_utterance_level.csv"),
                      ("per_speaker", "speaker_per_speaker.csv"), ("speaker", "speaker_level.csv")):
        pd.concat([r[key] for r in res], ignore_index=True).to_csv(a.out_dir / name, index=False)
    ps = pd.concat([r["per_speaker"] for r in res], ignore_index=True)
    if len(res) > 1 and len(ps):
        pooled = speaker_summary_from_published(ps[ps["model"] == "muse"], scope="pooled4")
        pooled.to_csv(a.out_dir / "speaker_level_pooled.csv", index=False)
        print(pooled.groupby("metric")["excludes_zero"].agg(["sum", "count"]).to_string())
    print("wrote", a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
