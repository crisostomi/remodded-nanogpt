"""Subprocess + logging helpers for the search runner."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_subprocess_tee(cmd: list[str], log_path: str | Path, env: dict | None = None) -> int:
    """Run ``cmd``, streaming combined stdout/stderr to both console and ``log_path``.

    Returns the subprocess return code.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        return proc.wait()
