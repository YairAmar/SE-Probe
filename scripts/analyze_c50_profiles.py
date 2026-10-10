#!/usr/bin/env python
"""Per-layer profiles on the reverberation axis for the six C50 arms (paper Secs.
III-C "Generalization Across Degradation Types" and III-F "Effect of
Dereverberation Fine-Tuning").

Input: a directory of ``shard_<arm>_<i>_<j>.parquet`` files from ``run_reverb_grid.py``
(arms ``{muse,mpsenet,demucs}_{pre,ft}``), or any parquet with ``model_name,
clean_idx, target_c50, layer, CKA``. Per arm, on the utterance-first mean level
curve: ``A`` (normalised AUC), ``c_low``, ``c_high``, ``alpha``, ``beta``, ``r2`` with
percentile intervals over whole-utterance resamples (``--seed`` 20260802 for
A/c_low as in the supplement, 20260806 for the fine-tuned alpha/beta profile
figure). Also written: ``mean_curves_c50.csv`` (scale, arm, model, layer, depth, one
column per level, the input of every Table I C50 statistic), ``summary_c50.csv``
(tradeoff summaries), ``finetune_shift.csv`` (pre vs ft profile agreement per model),
``demucs_bottleneck.csv`` (LSTM gap to the runner-up) and ``per_speaker_auc.csv``
(A per probing speaker from the ``utt_id`` prefix).
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _tables import read_table  # noqa: E402

from se_probe.layers import C50_GRID, probed_layers, short_labels  # noqa: E402
from se_probe.profiles import (  # noqa: E402
    PAPER_N_BOOT,
    PAPER_SEED_UTTERANCE,
    bottleneck_gap,
    build_cube,
    finetune_shift,
    fit_curves,
    tradeoff_summary,
    utterance_bootstrap,
)

ARMS = ["muse_pre", "muse_ft", "mpsenet_pre", "mpsenet_ft", "demucs_pre", "demucs_ft"]


def load_arm(src: Path, arm: str) -> pd.DataFrame:
    shards = sorted(glob.glob(str(src / f"shard_{arm}_[0-9]*_[0-9]*.parquet")))
    shards = [s for s in shards if Path(s).name[len(f"shard_{arm}_"):-8].replace("_", "").isdigit()]
    if shards:
        return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    df = read_table(src)
    df = df[df["model_name"].astype(str) == arm]
    if df.empty:
        raise SystemExit(f"no rows for arm {arm} under {src}")
    return df


def analyze_arm(df: pd.DataFrame, arm: str, n_boot: int, seed: int) -> dict:
    model = arm.split("_")[0]
    layers = probed_layers(model, df["layer"].unique())
    cube, units, levels = build_cube(df, layers, level_col="target_c50", unit_col="clean_idx")
    if np.isnan(cube).any():
        raise SystemExit(f"{arm}: holes in the layer x utterance x level cube")
    boot = utterance_bootstrap(cube, levels, n_boot=n_boot, seed=seed,
                               stats_keys=("alpha", "beta", "auc", "c_low", "c_high"))
    curves = pd.DataFrame(np.nanmean(cube, axis=1), index=pd.Index(layers, name="layer"), columns=levels)
    f = fit_curves(curves)
    per_layer = pd.DataFrame({"arm": arm, "model": model, "depth": np.arange(len(layers)),
                              "label": short_labels(model, layers), "layer": layers,
                              "A": f["auc"], "A_lo": boot["auc_lo"], "A_hi": boot["auc_hi"],
                              "c_low": f["c_low"], "c_low_lo": boot["c_low_lo"], "c_low_hi": boot["c_low_hi"],
                              "c_high": f["c_high"], "alpha": f["alpha"], "alpha_lo": boot["alpha_lo"], "alpha_hi": boot["alpha_hi"],
                              "beta": f["beta"], "beta_lo": boot["beta_lo"], "beta_hi": boot["beta_hi"], "r2": f["r2"],
                              "n_utts": len(units), "n_levels": len(levels), "n_boot": n_boot, "seed": seed})
    mc = pd.DataFrame({"scale": len(units), "arm": arm, "model": model, "layer": layers, "depth": np.arange(len(layers))})
    for v in levels:
        mc[f"{v:g}"] = curves[v].to_numpy()
    out = {"per_layer": per_layer, "mean_curves": mc, "summary": {"arm": arm, "model": model, **tradeoff_summary(curves, f)},
           "replicates": boot["replicates"], "layers": layers}
    if "utt_id" in df.columns:
        spk = df["utt_id"].astype(str).str.split("_").str[0]
        rows = []
        for s, sub in df.assign(speaker=spk).groupby("speaker"):
            c, u, lv = build_cube(sub, layers, level_col="target_c50")
            fs = fit_curves(np.nanmean(c, axis=1), lv)
            rows.append(pd.DataFrame({"arm": arm, "model": model, "speaker": s, "layer": layers,
                                      "depth": np.arange(len(layers)), "auc": fs["auc"], "c_low": fs["c_low"], "n_utts": len(u)}))
        out["per_speaker"] = pd.concat(rows, ignore_index=True)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, type=Path, help="directory of shard_<arm>_* parquets (or one parquet)")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--arms", nargs="+", default=ARMS)
    p.add_argument("--n-boot", type=int, default=PAPER_N_BOOT)
    p.add_argument("--seed", type=int, default=PAPER_SEED_UTTERANCE, help="20260802 (supplement) or 20260806 (FT profile figure)")
    p.add_argument("--levels", nargs="+", type=float, default=[float(v) for v in C50_GRID])
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    per_layer, curves, summ, spk, results = [], [], [], [], {}
    for arm in a.arms:
        try:
            df = load_arm(a.src, arm)
        except SystemExit as e:
            print(f"[skip] {e}")
            continue
        df = df[df["target_c50"].astype(float).isin(a.levels)]
        r = analyze_arm(df, arm, a.n_boot, a.seed)
        results[arm] = r
        per_layer.append(r["per_layer"])
        curves.append(r["mean_curves"])
        summ.append(r["summary"])
        if "per_speaker" in r:
            spk.append(r["per_speaker"])
        s = r["summary"]
        print(f"{arm:12s} n={s['n_layers']} r(alpha,beta)={s['r_alpha_beta']:+.3f} rho={s['rho_alpha_beta']:+.3f} "
              f"f={s['sat_freedom']:.3f} PC1={s['pc1']:.2f} A[{r['per_layer'].A.min():.3f},{r['per_layer'].A.max():.3f}]")
    if not results:
        raise SystemExit("no arm could be analysed")
    pd.concat(per_layer, ignore_index=True).to_csv(a.out_dir / "c50_per_layer.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(a.out_dir / "mean_curves_c50.csv", index=False)
    pd.DataFrame(summ).to_csv(a.out_dir / "summary_c50.csv", index=False)
    if spk:
        pd.concat(spk, ignore_index=True).to_csv(a.out_dir / "per_speaker_auc.csv", index=False)
    shifts = []
    for model in ("muse", "mpsenet", "demucs"):
        pre, ft = results.get(f"{model}_pre"), results.get(f"{model}_ft")
        if pre and ft:
            sh = finetune_shift(pre["per_layer"].rename(columns={"A": "auc"}), ft["per_layer"].rename(columns={"A": "auc"}))
            shifts.append({"model": model, **sh})
            print(f"{model}: pre vs ft beta r={sh['beta_r']:.3f} alpha r={sh['alpha_r']:.3f} mean|dbeta|={sh['beta_mean_abs_delta']:.4f}")
    if shifts:
        pd.DataFrame(shifts).to_csv(a.out_dir / "finetune_shift.csv", index=False)
    gaps = []
    for arm in ("demucs_pre", "demucs_ft"):
        r = results.get(arm)
        if r and "lstm" in r["layers"]:
            for key in ("auc", "alpha"):
                g = bottleneck_gap(r["replicates"][key], r["per_layer"]["A" if key == "auc" else "alpha"].to_numpy(), r["layers"])
                gaps.append({"arm": arm, "statistic": key, **g})
    if gaps:
        pd.DataFrame(gaps).to_csv(a.out_dir / "demucs_bottleneck.csv", index=False)
    print("wrote", a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
