"""Parse the metrics the training script prints to its raw log."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# step:<step>/<train_steps> val_loss:<loss> train_time:<ms>ms step_avg:<ms>
_VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<train_steps>\d+)\s+"
    r"val_loss:(?P<val_loss>[\d.]+)\s+"
    r"train_time:(?P<train_time>\d+)ms\s+"
    r"step_avg:(?P<step_avg>[\d.]+)ms"
)
# peak memory allocated: <mib> MiB reserved: <mib> MiB
_MEM_RE = re.compile(
    r"peak memory allocated:\s*(?P<alloc>\d+)\s*MiB\s+reserved:\s*(?P<reserved>\d+)\s*MiB"
)


def parse_log_text(text: str) -> dict[str, Any]:
    """Parse raw log text into the final-metrics summary.

    Uses the *last* ``val_loss`` line (the final validation) and the peak-memory
    line. Missing fields are returned as ``None``.
    """
    summary: dict[str, Any] = {
        "final_step": None,
        "train_steps": None,
        "val_loss": None,
        "train_time_ms": None,
        "step_avg_ms": None,
        "peak_memory_allocated_mib": None,
        "peak_memory_reserved_mib": None,
    }

    val_matches = list(_VAL_RE.finditer(text))
    if val_matches:
        m = val_matches[-1]
        summary["final_step"] = int(m["step"])
        summary["train_steps"] = int(m["train_steps"])
        summary["val_loss"] = float(m["val_loss"])
        summary["train_time_ms"] = int(m["train_time"])
        summary["step_avg_ms"] = float(m["step_avg"])

    mem = None
    for mem in _MEM_RE.finditer(text):
        pass  # keep last
    if mem is not None:
        summary["peak_memory_allocated_mib"] = int(mem["alloc"])
        summary["peak_memory_reserved_mib"] = int(mem["reserved"])

    return summary


def parse_log(path: str | Path) -> dict[str, Any]:
    return parse_log_text(Path(path).read_text(errors="replace"))
