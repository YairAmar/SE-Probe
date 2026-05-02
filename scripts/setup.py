#!/usr/bin/env python3
"""SE-Probe unified setup helper.

Run this **after** ``pip install -e .`` to fetch the artefacts the demo
notebooks need:

1. Reverb-FT MUSE checkpoint from HuggingFace Hub
   (default repo: ``YairAmar/SE-Probe-models``; override via the
   ``SEPROBE_HF_CKPT`` env var). Skip with ``--no-reverb-ckpt``.

2. Upstream noise-only MUSE pretrained (``g_best`` + ``config.json``) from
   the upstream MUSE-Speech-Enhancement repo. Cloned shallowly and
   discarded after copying the two files. Skip with ``--no-muse-pretrained``.

3. (Optional, ``--full-data``) Full HuggingFace dataset snapshot
   (default repo: ``YairAmar/SE-Probe-data``; override via
   ``SEPROBE_HF_DATA``). ~3 GB.

4. Device probe via :mod:`se_probe.device`.

Each step is independently try/except so a failure in one (e.g. HF repo
not yet uploaded) does not break the others. A final summary lists which
steps succeeded vs failed.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_DIR = REPO_ROOT / "pretrained_models" / "muse"
RESULTS_FULL_DIR = REPO_ROOT / "results_df"

DEFAULT_HF_CKPT = "YairAmar/SE-Probe-models"
DEFAULT_HF_DATA = "YairAmar/SE-Probe-data"
DEFAULT_HF_CKPT_FILE = "muse_reverb_e48.pt"
UPSTREAM_MUSE_REPO = "https://github.com/huaidanquede/MUSE-Speech-Enhancement"


@dataclass
class StepResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Summary:
    results: List[StepResult] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.results.append(StepResult(name, ok, detail))

    def print(self) -> None:
        print("\n=== setup summary ===")
        for r in self.results:
            tag = "OK  " if r.ok else "FAIL"
            print(f"  [{tag}] {r.name}{(' — ' + r.detail) if r.detail else ''}")
        if all(r.ok for r in self.results):
            print("\nAll steps complete. Next: jupyter lab notebooks/")
        else:
            print("\nSome steps failed; see messages above.")


def _step_reverb_ckpt(summary: Summary) -> None:
    name = "reverb FT checkpoint"
    repo_id = os.environ.get("SEPROBE_HF_CKPT", DEFAULT_HF_CKPT)
    filename = os.environ.get("SEPROBE_HF_CKPT_FILE", DEFAULT_HF_CKPT_FILE)
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[1/4] Fetching reverb FT checkpoint from {repo_id}:{filename} ...")
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (
            HfHubHTTPError,
            RepositoryNotFoundError,
        )

        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(PRETRAINED_DIR),
            )
            summary.add(name, True, f"downloaded to {path}")
            print(f"      -> {path}")
        except (RepositoryNotFoundError, HfHubHTTPError) as e:
            msg = (
                f"HF repo not reachable ({repo_id}). "
                f"Skipping; notebook 06 inference cells will warn until this is uploaded. "
                f"({type(e).__name__})"
            )
            print(f"      WARN: {msg}")
            summary.add(name, False, msg)
    except Exception as e:  # pragma: no cover - defensive
        print(f"      ERROR: {e}")
        summary.add(name, False, str(e))


def _step_muse_pretrained(summary: Summary) -> None:
    name = "upstream MUSE pretrained (g_best + config.json)"
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[2/4] Cloning {UPSTREAM_MUSE_REPO} for g_best + config.json ...")
    with tempfile.TemporaryDirectory(prefix="se-probe-muse-") as tmp:
        tmp_path = Path(tmp) / "MUSE"
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", UPSTREAM_MUSE_REPO, str(tmp_path)],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            msg = (
                f"git clone failed ({type(e).__name__}). "
                f"Fetch g_best + config.json manually from {UPSTREAM_MUSE_REPO} "
                f"and place them in {PRETRAINED_DIR}/."
            )
            print(f"      WARN: {msg}")
            summary.add(name, False, msg)
            return

        copied: List[str] = []
        missing: List[str] = []
        for fname in ("g_best", "config.json"):
            matches = list(tmp_path.rglob(fname))
            if not matches:
                missing.append(fname)
                continue
            shutil.copy2(matches[0], PRETRAINED_DIR / fname)
            copied.append(fname)
            print(f"      -> {PRETRAINED_DIR / fname}")
        if missing:
            msg = (
                f"copied {copied}; could not find {missing} in upstream repo. "
                f"Upstream layout may have changed; fetch manually from "
                f"{UPSTREAM_MUSE_REPO}."
            )
            print(f"      WARN: {msg}")
            summary.add(name, False, msg)
        else:
            summary.add(name, True, "g_best + config.json placed")


def _step_full_data(summary: Summary, enabled: bool) -> None:
    name = "full HF dataset snapshot"
    if not enabled:
        print("\n[3/4] Full HF dataset (skipped — pass --full-data to enable).")
        summary.add(name, True, "skipped (default)")
        return
    repo_id = os.environ.get("SEPROBE_HF_DATA", DEFAULT_HF_DATA)
    print(f"\n[3/4] Downloading full dataset from {repo_id} ...")
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import (
            HfHubHTTPError,
            RepositoryNotFoundError,
        )

        try:
            path = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(RESULTS_FULL_DIR),
            )
            summary.add(name, True, f"downloaded to {path}")
            print(f"      -> {path}")
        except (RepositoryNotFoundError, HfHubHTTPError) as e:
            msg = f"HF dataset not reachable ({repo_id}); {type(e).__name__}"
            print(f"      WARN: {msg}")
            summary.add(name, False, msg)
    except Exception as e:  # pragma: no cover - defensive
        print(f"      ERROR: {e}")
        summary.add(name, False, str(e))


def _step_device_probe(summary: Summary, prefer: Optional[str]) -> None:
    name = "device probe"
    print("\n[4/4] Probing torch device ...")
    try:
        from se_probe.device import device_info, get_device

        device = get_device(prefer if prefer not in (None, "auto") else None)
        info = device_info(device)
        print(f"      {info}")
        summary.add(name, True, info)
    except Exception:  # pragma: no cover - defensive
        traceback.print_exc()
        summary.add(name, False, "see traceback")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-data", action="store_true",
                        help="Also download the full HF dataset (~3 GB).")
    parser.add_argument("--no-reverb-ckpt", action="store_true",
                        help="Skip the reverb FT checkpoint download.")
    parser.add_argument("--no-muse-pretrained", action="store_true",
                        help="Skip cloning the upstream MUSE repo for g_best + config.json.")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                        help="Force device for the probe step (default: auto-detect).")
    args = parser.parse_args()

    summary = Summary()

    if args.no_reverb_ckpt:
        print("[1/4] Reverb FT checkpoint (skipped via --no-reverb-ckpt).")
        summary.add("reverb FT checkpoint", True, "skipped")
    else:
        _step_reverb_ckpt(summary)

    if args.no_muse_pretrained:
        print("[2/4] Upstream MUSE pretrained (skipped via --no-muse-pretrained).")
        summary.add("upstream MUSE pretrained (g_best + config.json)", True, "skipped")
    else:
        _step_muse_pretrained(summary)

    _step_full_data(summary, enabled=args.full_data)
    _step_device_probe(summary, prefer=args.device)

    summary.print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
