"""Local run manifest: the GitHub-reproducible record of an experiment.

A run is reproducible from GitHub (code + configs) plus the local artifacts; the
manifest ties them together. Flywheel is only a mirror -- never required to
reconstruct a run.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nano.builder.context import BuildContext
from nano.config.base import baseline_sha, repo_git_info
from nano.config.schema import FeatureSet


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def code_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def software_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": None,
        "cuda": None,
        "triton": None,
    }
    try:  # torch is intentionally absent on dev machines
        import torch

        info["torch"] = torch.version.__version__
        info["cuda"] = torch.version.cuda
    except Exception:
        pass
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception:
        pass
    return info


def create_initial_manifest(
    ctx: BuildContext,
    feature_set: FeatureSet,
    *,
    run_id: str,
    generated_script: str | Path,
    n_gpus: int = 8,
    world_size: int | None = None,
    disabled_features: list[str] | None = None,
    gpu_type: str = "H100",
    project: str | None = None,
) -> dict[str, Any]:
    """Build the initial (status=pending) manifest for a run."""
    generated_script = Path(generated_script)
    world_size = world_size if world_size is not None else n_gpus
    tracking = ctx.metadata.get("tracking", feature_set.tracking) or {}
    project = project or tracking.get("project")

    return {
        "run_id": run_id,
        "project": project,
        "feature_set": feature_set.name,
        "enabled_features": sorted(ctx.enabled_features),
        "disabled_features": sorted(disabled_features or feature_set.disable),
        "base_preset": feature_set.base or feature_set.name,
        "git": repo_git_info(),
        "baseline": {"source_sha": baseline_sha()},
        "generated": {
            "train_script": generated_script.name,
            "sha256": code_sha256(generated_script),
        },
        "hardware": {
            "gpu_type": gpu_type,
            "n_gpus": n_gpus,
            "world_size": world_size,
        },
        "software": software_info(),
        "schedule": {
            "num_scheduled_iterations": ctx.schedule.num_scheduled_iterations,
            "num_extension_iterations": ctx.schedule.num_extension_iterations,
            "total_steps": ctx.schedule.total_steps,
            "cooldown_frac": ctx.schedule.cooldown_frac,
        },
        "metrics": {
            "train_time_ms": None,
            "step_avg_ms": None,
            "val_loss": None,
            "peak_memory_allocated_mib": None,
            "peak_memory_reserved_mib": None,
            "p_value_vs_3_28": None,
        },
        "tracking": tracking,
        "status": "pending",
        "created_at": utc_now_iso(),
    }


def update_manifest_with_summary(
    manifest: dict[str, Any],
    summary: dict[str, Any] | None,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    """Fold parsed log metrics + a final status into a manifest (in place)."""
    if summary:
        m = manifest.setdefault("metrics", {})
        for key in (
            "val_loss",
            "train_time_ms",
            "step_avg_ms",
            "peak_memory_allocated_mib",
            "peak_memory_reserved_mib",
        ):
            if summary.get(key) is not None:
                m[key] = summary[key]
        manifest["final_step"] = summary.get("final_step")
    manifest["status"] = status
    manifest["finished_at"] = utc_now_iso()
    if error:
        manifest["error"] = error
    return manifest
