#!/usr/bin/env python
"""Manifest-driven upload of the 824-scale artifacts to the HuggingFace repos.

Dry run by default: the script prints the upload plan (local path -> repo path,
size) and exits. Nothing is sent unless ``--yes`` is given. The default manifest
mirrors the layout referenced by the paper's Data Availability statement:

* dataset ``yairamr/SE-Probe-data``, prefix ``cluster-2026-07/``: the aggregated noise
  sweep parquets (``cka_snr_all_<model>.parquet``), the six C50 arm aggregates, the
  random-init, emergence and perceptual aggregates, the diffusion centroids and the
  per-layer fit tables;
* model repo ``yairamr/SE-Probe-models``: the fine-tuned generators (MUSE epoch 48,
  MP-SENet epoch 50, Demucs epoch 57 and 45) and the freeze-arm checkpoints.

Write your own manifest as a CSV with columns ``local_path,repo_id,repo_type,path_in_repo``
or point ``--aggregated-dir`` / ``--checkpoints-dir`` at the directories to scan.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

DATASET_REPO = "yairamr/SE-Probe-data"
MODEL_REPO = "yairamr/SE-Probe-models"
PREFIX = "cluster-2026-07"


def plan_from_dirs(aggregated_dir: Path | None, checkpoints_dir: Path | None) -> list[dict]:
    plan = []
    if aggregated_dir is not None:
        for f in sorted(aggregated_dir.rglob("*")):
            if f.is_file() and f.suffix in (".parquet", ".csv", ".json"):
                plan.append(dict(local_path=str(f), repo_id=DATASET_REPO, repo_type="dataset",
                                 path_in_repo=f"{PREFIX}/{f.relative_to(aggregated_dir)}"))
    if checkpoints_dir is not None:
        for f in sorted(checkpoints_dir.rglob("*")):
            if f.is_file() and (f.name.startswith("g_") or f.suffix in (".pt", ".pth", ".th")):
                plan.append(dict(local_path=str(f), repo_id=MODEL_REPO, repo_type="model",
                                 path_in_repo=f"{PREFIX}/{f.relative_to(checkpoints_dir)}"))
    return plan


def plan_from_manifest(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    need = {"local_path", "repo_id", "repo_type", "path_in_repo"}
    if rows and not need.issubset(rows[0]):
        raise SystemExit(f"manifest must have columns {sorted(need)}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--aggregated-dir", type=Path, default=None)
    p.add_argument("--checkpoints-dir", type=Path, default=None)
    p.add_argument("--yes", action="store_true", help="actually upload (default: print the plan only)")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    plan = plan_from_manifest(a.manifest) if a.manifest else plan_from_dirs(a.aggregated_dir, a.checkpoints_dir)
    if not plan:
        raise SystemExit("nothing to upload: give --manifest or --aggregated-dir / --checkpoints-dir")
    total = 0
    for item in plan:
        size = Path(item["local_path"]).stat().st_size
        total += size
        print(f"{item['local_path']} -> {item['repo_type']}:{item['repo_id']}/{item['path_in_repo']}  ({size / 1e6:.1f} MB)")
    print(f"{len(plan)} files, {total / 1e9:.2f} GB total")
    if not a.yes:
        print("dry run: re-run with --yes to upload")
        return 0
    from huggingface_hub import HfApi

    api = HfApi()
    for item in plan:
        api.upload_file(path_or_fileobj=item["local_path"], path_in_repo=item["path_in_repo"],
                        repo_id=item["repo_id"], repo_type=item["repo_type"])
        print(f"uploaded {item['path_in_repo']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
