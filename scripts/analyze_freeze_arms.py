#!/usr/bin/env python
"""Per-arm C50 re-probe of the profile-guided freezing experiment (supplementary;
not in the manuscript, see ``results_tables/freeze_arms/``).

Input: a directory of ``cka_reverb_muse_<tag>.parquet`` probes over the 88-RIR AIR
pool, tags ``pre_ft``, ``full_ft__seed<S>``, ``freeze_encoder__seed<S>``,
``freeze_decoder__seed<S>``. Per probe and probed layer: normalised AUC, ``c_low``,
``c_high``, ``alpha``, ``beta``, ``r2`` with utterance-bootstrap intervals (300 draws,
seed 20260802); paired contrasts (same utterance resample on both members) between
each arm and ``pre_ft`` / the matching ``full_ft`` seed; stage roll-ups. A pool gate
refuses any probe that does not span the published 88-RIR six-room pool.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from run_reverb_grid import verify_pool  # noqa: E402

from se_probe.layers import MUSE_STAGES, muse_stage_of, probed_layers  # noqa: E402
from se_probe.profiles import build_cube, fit_curves  # noqa: E402

N_BOOT, SEED = 300, 20260802
FROZEN_STAGES = {"full_ft": set(), "pre_ft": set(),
                 "freeze_encoder": {"encoder_level1", "encoder_level2"},
                 "freeze_decoder": {"decoder_level2", "decoder_level1"}}


def arm_of(tag: str) -> str:
    return tag if tag == "pre_ft" else tag.split("__")[0]


def stats_of(cube: np.ndarray, levels: np.ndarray) -> dict:
    f = fit_curves(np.nanmean(cube, axis=1), levels)
    return {k: f[k] for k in ("auc", "c_low", "c_high", "alpha", "beta", "r2")}


def per_probe_stats(cubes: dict, n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    rows = []
    for tag, (layers, units, levels, cube) in cubes.items():
        st = stats_of(cube, levels)
        rng = np.random.default_rng(seed)
        draws = {k: np.empty((n_boot, len(layers))) for k in ("auc", "c_low", "beta")}
        for b in range(n_boot):
            sub = stats_of(cube[:, rng.integers(0, len(units), len(units)), :], levels)
            for k in draws:
                draws[k][b] = sub[k]
        for i, lay in enumerate(layers):
            r = dict(tag=tag, arm=arm_of(tag), seed=("" if tag == "pre_ft" else tag.split("seed")[-1]),
                     layer=lay, depth=i, stage=muse_stage_of(lay),
                     frozen_in_arm=muse_stage_of(lay) in FROZEN_STAGES.get(arm_of(tag), set()), n_units=len(units))
            r.update({k: float(v[i]) for k, v in st.items()})
            for k in draws:
                lo, hi = np.percentile(draws[k][:, i], [2.5, 97.5])
                r[f"{k}_lo"], r[f"{k}_hi"], r[f"{k}_se"] = float(lo), float(hi), float(draws[k][:, i].std(ddof=1))
            rows.append(r)
    return pd.DataFrame(rows)


def paired_contrasts(cubes: dict, n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    pairs = []
    for tag in cubes:
        if tag == "pre_ft":
            continue
        if "pre_ft" in cubes:
            pairs.append((tag, "pre_ft"))
        if arm_of(tag) != "full_ft":
            other = f"full_ft__seed{tag.split('seed')[-1]}"
            if other in cubes:
                pairs.append((tag, other))
    rows = []
    for ta, tb in pairs:
        La, Ua, Va, Ca = cubes[ta]
        Lb, Ub, Vb, Cb = cubes[tb]
        common = np.intersect1d(Ua, Ub)
        Ca_, Cb_ = Ca[:, np.searchsorted(Ua, common), :], Cb[:, np.searchsorted(Ub, common), :]
        sa, sb = stats_of(Ca_, Va), stats_of(Cb_, Va)
        rng = np.random.default_rng(seed)
        d = {k: np.empty((n_boot, len(La))) for k in ("auc", "c_low", "beta")}
        for b in range(n_boot):
            j = rng.integers(0, len(common), len(common))
            xa, xb = stats_of(Ca_[:, j, :], Va), stats_of(Cb_[:, j, :], Va)
            for k in d:
                d[k][b] = xa[k] - xb[k]
        for i, lay in enumerate(La):
            r = dict(tag_a=ta, tag_b=tb, arm_a=arm_of(ta), arm_b=arm_of(tb), layer=lay, depth=i,
                     stage=muse_stage_of(lay), n_units=len(common))
            for k in d:
                lo, hi = np.percentile(d[k][:, i], [2.5, 97.5])
                r[f"d_{k}"] = float(sa[k][i] - sb[k][i])
                r[f"d_{k}_lo"], r[f"d_{k}_hi"], r[f"d_{k}_sig"] = float(lo), float(hi), bool(lo > 0 or hi < 0)
            rows.append(r)
    return pd.DataFrame(rows)


def stage_rollup(per_layer: pd.DataFrame) -> pd.DataFrame:
    ps = per_layer[per_layer["tag"] != "pre_ft"]
    agg = (ps.groupby(["arm", "stage"], as_index=False)
             .agg(auc=("auc", "mean"), beta=("beta", "mean"), c_low=("c_low", "mean"), n_layers=("layer", "nunique"),
                  frozen=("frozen_in_arm", "any")))
    if (per_layer["tag"] == "pre_ft").any():
        pre = per_layer[per_layer["tag"] == "pre_ft"].groupby("stage")["auc"].mean().rename("pre_auc")
        agg = agg.merge(pre, on="stage", how="left")
        agg["auc_minus_pre"] = agg["auc"] - agg["pre_auc"]
    agg["stage_rank"] = agg["stage"].map({s: i for i, s in enumerate(MUSE_STAGES)})
    return agg.sort_values(["arm", "stage_rank"]).reset_index(drop=True)


def load_probe(path: Path):
    all_layers = pq.read_table(path, columns=["layer"])["layer"].to_pandas().unique()
    layers = probed_layers("muse", all_layers)
    df = pq.read_table(path, columns=["clean_idx", "rir_name", "target_c50", "layer", "CKA"],
                       filters=[("layer", "in", set(layers))]).to_pandas()
    verify_pool(df, n_utts=df["clean_idx"].nunique())
    cube, units, levels = build_cube(df, layers, level_col="target_c50")
    return layers, units, levels, cube


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probes-dir", required=True, type=Path, help="dir of cka_reverb_muse_<tag>.parquet")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--seed", type=int, default=SEED)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    cubes = {}
    for p in sorted(a.probes_dir.glob("cka_reverb_muse_*.parquet")):
        tag = p.stem[len("cka_reverb_muse_"):]
        cubes[tag] = load_probe(p)
        print(f"{tag}: {len(cubes[tag][0])} layers x {len(cubes[tag][1])} utts x {len(cubes[tag][2])} levels")
    if not cubes:
        raise SystemExit(f"no cka_reverb_muse_*.parquet under {a.probes_dir}")
    per_layer = per_probe_stats(cubes, a.n_boot, a.seed)
    per_layer.to_csv(a.out_dir / "freeze_arms_per_layer.csv", index=False)
    con = paired_contrasts(cubes, a.n_boot, a.seed)
    con.to_csv(a.out_dir / "freeze_arms_contrasts.csv", index=False)
    stg = stage_rollup(per_layer)
    stg.to_csv(a.out_dir / "freeze_arms_stage_summary.csv", index=False)
    print(stg.pivot(index="stage", columns="arm", values="auc").to_string(float_format=lambda v: f"{v:.4f}"))
    if len(con):
        print(con.groupby(["tag_a", "tag_b"])["d_auc_sig"].sum().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
