"""Tests for ``scripts/setup.py``.

Test level: **INTEGRATION**.
Rationale: ``setup.py`` is a CLI orchestration script that imports
``huggingface_hub``, shells out to ``git``, prints user-facing strings,
and ends with a summary. A unit test that monkeypatches each downloader
would tell us nothing about whether the script actually exits 0 or
prints the device-info line. We invoke it as a subprocess against the
real interpreter and assert on its observable outputs (exit code,
stdout content). We pass ``--no-reverb-ckpt --no-muse-pretrained`` so
the test does not require network access — the device-probe and summary
steps still run, which is what matters for the smoke contract.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP_PY = REPO_ROOT / "scripts" / "setup.py"


def _run_setup(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Invoke setup.py as a subprocess with the running pytest interpreter."""
    return subprocess.run(
        [sys.executable, str(SETUP_PY), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
    )


class TestSetupScriptOfflineFlags:
    """``setup.py`` runs without network when both download steps are skipped."""

    def test_setup_script_exists(self):
        assert SETUP_PY.exists(), (
            f"scripts/setup.py must exist at {SETUP_PY}; the unified setup "
            "is the documented one-command UX (D18)"
        )

    def test_help_runs(self):
        proc = _run_setup("--help")
        assert proc.returncode == 0, (
            f"setup.py --help should exit 0; stderr={proc.stderr!r}"
        )
        # argparse echoes the doc and the four advertised flags
        assert "--full-data" in proc.stdout
        assert "--no-reverb-ckpt" in proc.stdout
        assert "--no-muse-pretrained" in proc.stdout
        assert "--device" in proc.stdout

    def test_offline_run_exits_zero(self):
        proc = _run_setup("--no-reverb-ckpt", "--no-muse-pretrained")
        assert proc.returncode == 0, (
            f"setup.py --no-reverb-ckpt --no-muse-pretrained should exit 0 "
            f"(offline path: only device probe runs). "
            f"returncode={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )

    def test_offline_run_prints_device_line(self):
        proc = _run_setup("--no-reverb-ckpt", "--no-muse-pretrained")
        assert "Detected device:" in proc.stdout, (
            f"setup.py must print the device-info line via se_probe.device.device_info; "
            f"stdout={proc.stdout!r}"
        )
        # The device line must mention one of the three supported backends.
        assert any(b in proc.stdout for b in ("cuda", "mps", "cpu")), (
            f"device-info line should name a backend (cuda|mps|cpu); "
            f"stdout={proc.stdout!r}"
        )

    def test_offline_run_prints_summary(self):
        proc = _run_setup("--no-reverb-ckpt", "--no-muse-pretrained")
        assert "=== setup summary ===" in proc.stdout, (
            "setup.py must end with a '=== setup summary ===' block per "
            "the unified-script spec (D18). Without it, partial-failure "
            f"users have no audit trail. stdout={proc.stdout!r}"
        )
        # Skipped steps must be reported so the user knows what *didn't*
        # run, not just what did.
        assert "skipped" in proc.stdout.lower(), (
            f"summary should mark the two --no-* flags as skipped; "
            f"stdout={proc.stdout!r}"
        )

    def test_offline_run_does_not_create_pretrained_artifacts(self, tmp_path):
        # When both download flags are off, the script should not create
        # any new files in pretrained_models/muse/ that weren't there
        # already. We snapshot before/after and diff.
        muse_dir = REPO_ROOT / "pretrained_models" / "muse"
        before = {p.name for p in muse_dir.iterdir()} if muse_dir.exists() else set()
        proc = _run_setup("--no-reverb-ckpt", "--no-muse-pretrained")
        assert proc.returncode == 0
        after = {p.name for p in muse_dir.iterdir()} if muse_dir.exists() else set()
        new = after - before
        # The setup script's _step_reverb_ckpt always mkdir-p's
        # pretrained_models/muse/ even when skipped — that's fine. But it
        # must not create any file artifacts when both download steps are
        # disabled.
        assert not new, (
            f"setup.py with both --no-* flags should not write any files "
            f"into pretrained_models/muse/; new entries: {new}"
        )


class TestSetupDeviceFlag:
    """``--device cpu`` forces the probe step to report CPU."""

    def test_device_cpu_reported(self):
        proc = _run_setup(
            "--no-reverb-ckpt", "--no-muse-pretrained", "--device", "cpu",
        )
        assert proc.returncode == 0, (
            f"--device cpu offline run should exit 0; stderr={proc.stderr}"
        )
        assert "Detected device: cpu" in proc.stdout, (
            f"--device cpu should force CPU in the probe step; "
            f"stdout={proc.stdout!r}"
        )
