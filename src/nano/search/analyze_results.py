"""Collect run manifests and print a comparison table."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from nano.config.base import EXPERIMENTS_DIR
from nano.runtime.manifest import read_json


def load_runs(runs_dir: str | Path = EXPERIMENTS_DIR) -> list[dict[str, Any]]:
    runs_dir = Path(runs_dir)
    runs: list[dict[str, Any]] = []
    for manifest_path in sorted(runs_dir.glob("*/manifest.json")):
        try:
            runs.append(read_json(manifest_path))
        except Exception:
            continue
    return runs


def summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for m in runs:
        metrics = m.get("metrics", {})
        rows.append({
            "run_id": m.get("run_id"),
            "feature_set": m.get("feature_set"),
            "status": m.get("status"),
            "val_loss": metrics.get("val_loss"),
            "step_avg_ms": metrics.get("step_avg_ms"),
            "train_time_ms": metrics.get("train_time_ms"),
        })
    rows.sort(key=lambda r: (r["val_loss"] is None, r["val_loss"] if r["val_loss"] is not None else 0.0))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize run manifests.")
    parser.add_argument("--runs-dir", default=str(EXPERIMENTS_DIR))
    args = parser.parse_args(argv)

    rows = summarize(load_runs(args.runs_dir))
    if not rows:
        print("No runs found.")
        return 0
    header = f"{'run_id':<40} {'status':<10} {'val_loss':>9} {'step_avg_ms':>12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        vl = f"{r['val_loss']:.4f}" if r["val_loss"] is not None else "-"
        sa = f"{r['step_avg_ms']:.2f}" if r["step_avg_ms"] is not None else "-"
        print(f"{str(r['run_id']):<40} {str(r['status']):<10} {vl:>9} {sa:>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
