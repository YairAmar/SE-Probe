#!/usr/bin/env python
"""CKA estimator and sample-convention ablation (paper Sec. II-D, the unbiased-HSIC
check of the "Centered Kernel Alignment" method).

Pretrained MUSE on a reduced grid (paper: 150 utterances x 9 SNRs x 2 DEMAND noises),
with the non-pooled first-segment activation ``(C, T, F)`` of each of the 24 ``.norm1``
layers turned into four CKA variants (``se_probe.cka.cka_variants``): the paper's
pooled biased linear estimator, the unbiased-HSIC estimator on the same pooled
representation, and two frames-as-samples conventions. The finalize step fits the
per-layer slope profile of every variant and reports its Pearson correlation with
the paper's profile (the paper quotes 0.98 for the unbiased estimator).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _clean_sources import load_voicebank_test, resolve_clean_source  # noqa: E402
from _grid import Heartbeat, chunk_loop  # noqa: E402

from se_probe.cka import CKA_VARIANTS, cka_variants  # noqa: E402
from se_probe.data_generation import add_noise_at_snr, load_demand_noise  # noqa: E402
from se_probe.device import device_info, get_device  # noqa: E402
from se_probe.layers import probed_layers, short_labels  # noqa: E402
from se_probe.muse.consts import NORM1_LAYERS  # noqa: E402
from se_probe.profiles import fit_curves, mean_level_curves, profile_agreement  # noqa: E402

PAPER_SNRS = [-10, -5, 0, 5, 10, 15, 20, 25, 30]


def finalize(chunk_dir: Path, out_csv: Path) -> pd.DataFrame:
    files = sorted(chunk_dir.glob("ablation__*.parquet"))
    if not files:
        raise SystemExit(f"no ablation chunks under {chunk_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    layers = probed_layers("muse", df["layer"].unique())
    betas = {}
    for var in CKA_VARIANTS:
        d = df[df["variant"] == var].rename(columns={"cka": "CKA"})
        curves = mean_level_curves(d, layers=layers, level_col="snr", unit_col="clean_idx")
        betas[var] = fit_curves(curves)["beta"]
    tab = pd.DataFrame({"layer": layers, "label": short_labels("muse", layers), **betas})
    tab.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")
    for var in CKA_VARIANTS:
        ag = profile_agreement(betas["pooled_linear"], betas[var])
        print(f"  {var:14s} r vs pooled_linear = {ag['pearson_r']:+.3f}  rho = {ag['spearman_rho']:+.3f}  "
              f"peak {tab.label[int(np.argmax(betas[var]))]}")
    return tab


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clean", default=None)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--noises", nargs="+", default=["TBUS", "SCAFE"])
    p.add_argument("--snrs", nargs="+", type=int, default=PAPER_SNRS)
    p.add_argument("--n-utts", type=int, default=150)
    p.add_argument("--finalize-only", action="store_true")
    p.add_argument("--log", type=Path, default=None)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    hb = Heartbeat("hsic-ablation", a.log)
    if not a.finalize_only:
        from se_probe.muse.model import load_muse_activation_extractor

        device = get_device()
        hb(f"START {device_info(device)} n_utts={a.n_utts} snrs={a.snrs} noises={a.noises}")
        _, clean_all = load_voicebank_test(resolve_clean_source(a.clean), a.n_utts)
        ext = load_muse_activation_extractor(device=device, with_pooling=False)

        def chunk_path(noise, snr):
            return a.out_dir / f"ablation__{noise}__snr{snr:+03d}.parquet"

        clean_acts = [None] * len(clean_all)

        def make_rows(noise_name, snr):
            noise = load_demand_noise(noise_name)
            rows = []
            for u, clean in enumerate(clean_all):
                if clean_acts[u] is None:
                    acts_c, _ = ext(clean)
                    clean_acts[u] = {L: acts_c[L].detach() for L in NORM1_LAYERS}
                acts_n, _ = ext(add_noise_at_snr(clean, noise, snr).astype("float32"))
                for L in NORM1_LAYERS:
                    for var, v in cka_variants(clean_acts[u][L], acts_n[L].detach()).items():
                        rows.append({"noise_name": noise_name, "snr": int(snr), "clean_idx": u,
                                     "layer": L, "variant": var, "cka": float(v)})
            return rows

        chunk_loop([(n, s) for n in a.noises for s in a.snrs], chunk_path, make_rows, hb)
    finalize(a.out_dir, a.out_dir / "hsic_frames_slopes.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
