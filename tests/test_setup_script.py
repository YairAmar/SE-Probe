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

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP_PY = REPO_ROOT / "scripts" / "setup.py"


def _run_setup(
    *args: str,
    timeout: int = 60,
    env: "dict[str, str] | None" = None,
) -> subprocess.CompletedProcess:
    """Invoke setup.py as a subprocess with the running pytest interpreter.

    ``env`` overrides specific env vars for the subprocess (default: inherit).
    """
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, str(SETUP_PY), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
        env=full_env,
    )


def _network_available(host: str = "huggingface.co", port: int = 443) -> bool:
    """Quick TCP probe so the WARN-path test self-skips when offline (CI sandbox)."""
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except OSError:
        return False


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


class TestSetupWarnAndContinue:
    """The unified setup-script's contract is that a download failure in any
    one step (e.g., HF repo not yet uploaded — the live default for v0.1.0
    until the user creates the placeholder repos) WARNs and continues so
    that the device-probe step and the summary still run.

    Reviewer specifically asked us to verify this path because the placeholder
    repo IDs ``YairAmar/SE-Probe-{models,data}`` may not be publicly reachable
    when a fresh user runs ``python scripts/setup.py``.
    """

    def test_unreachable_hf_ckpt_warns_but_exits_zero(self):
        # Need to actually reach huggingface.co to confirm the repo doesn't
        # exist — otherwise we can't distinguish "WARN-and-continue worked"
        # from "no network at all". Skip cleanly on offline CI.
        if not _network_available():
            pytest.skip(
                "huggingface.co not reachable; skipping HF-WARN integration test"
            )
        # Invoke setup with the reverb-ckpt step ENABLED but pointed at a
        # nonsense repo, while skipping the git-clone step (which would
        # also need network and would muddy the assertion).
        proc = _run_setup(
            "--no-muse-pretrained",
            "--device", "cpu",
            env={"SEPROBE_HF_CKPT": "se-probe-tester-nonexistent-org/no-such-repo-xxxx"},
            timeout=120,  # HF DNS + 404 round-trip
        )
        assert proc.returncode == 0, (
            "setup.py must exit 0 even when the reverb-ckpt download fails — "
            "WARN-and-continue is the documented contract (Task 3.4 / D18). "
            f"returncode={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )
        assert "WARN" in proc.stdout, (
            f"failed HF download should print a WARN line; stdout={proc.stdout!r}"
        )
        # Device probe must still run after the failed step — this is the
        # whole point of try/except per-step.
        assert "Detected device: cpu" in proc.stdout, (
            "device-probe step must still run after a failed HF step; "
            f"stdout={proc.stdout!r}"
        )
        # And the summary must show the failure visibly so users can audit.
        assert "=== setup summary ===" in proc.stdout, proc.stdout
        assert "FAIL" in proc.stdout, (
            "summary should mark the failed step with [FAIL] so the user "
            f"sees what didn't run; stdout={proc.stdout!r}"
        )
