"""CLI: resolve a feature combo, generate a run, launch it, log a manifest.

    python -m nano.search.run_combo --preset current_record \
        --disable sparse_attention_gate --enable xsa_lowering_rewrite \
        --run-name current_minus_sag_xsa_rewrite --nproc-per-node 8

``--dry-run`` builds, validates and writes all artifacts but does not launch
torchrun. Local artifacts are always written; a Flywheel upload failure never
marks the training run itself as failed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from nano.builder.codegen import generate
from nano.builder.validate import FeatureValidationError
from nano.config.base import EXPERIMENTS_DIR
from nano.config.presets import resolve_feature_set
from nano.runtime.logging import run_subprocess_tee
from nano.runtime.manifest import (
    read_json,
    update_manifest_with_summary,
    write_json,
)
from nano.runtime.parse_logs import parse_log
from nano.runtime.tracking import upload_run


def make_run_dir(run_name: str) -> Path:
    run_dir = EXPERIMENTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a feature combo and log a manifest.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--preset")
    src.add_argument("--feature-set")
    parser.add_argument("--enable", action="append", default=[])
    parser.add_argument("--disable", action="append", default=[])
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--tracking-backend", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        feature_set = resolve_feature_set(
            preset=args.preset,
            feature_set_file=args.feature_set,
            enable=args.enable,
            disable=args.disable,
        )
        run_name = args.run_name or feature_set.name
        run_dir = make_run_dir(run_name)
        disabled = sorted(set(args.disable) | set(feature_set.disable))

        # resolve -> validate -> generate static script + sidecar artifacts
        result = generate(
            feature_set,
            run_dir / "train_generated.py",
            run_id=run_name,
            features_path=run_dir / "features.yaml",
            manifest_path=run_dir / "manifest.json",
            n_gpus=args.nproc_per_node,
            world_size=args.nproc_per_node,
            disabled_features=disabled,
        )
    except (FeatureValidationError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    manifest_path = run_dir / "manifest.json"
    print(f"Run dir: {run_dir}")
    print(f"Enabled {len(result['ctx'].enabled_features)} features; "
          f"{len(result['ctx'].optim.param_table)} optimizer params.")
    for w in result["ctx"].warnings:
        print(f"  warning: {w}")

    if args.dry_run:
        manifest = read_json(manifest_path)
        manifest["status"] = "dry_run"
        write_json(manifest, manifest_path)
        print("Dry run: validated and generated artifacts; not launching torchrun.")
        return 0

    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        str(run_dir / "train_generated.py"),
    ]
    env = dict(os.environ)
    if args.data_path:
        env["DATA_PATH"] = args.data_path

    print("Launching:", " ".join(cmd))
    returncode = run_subprocess_tee(cmd, run_dir / "raw.log", env=env)

    summary = parse_log(run_dir / "raw.log")
    write_json(summary, run_dir / "summary.json")
    status = "completed" if returncode == 0 else "failed"
    manifest = read_json(manifest_path)
    update_manifest_with_summary(
        manifest, summary, status,
        error=None if returncode == 0 else f"torchrun exited with code {returncode}",
    )
    # Flush the metrics/status to disk BEFORE uploading so the tracking backend
    # (which re-reads the manifest from disk) records real metrics, not 'pending'.
    write_json(manifest, manifest_path)

    if args.upload:
        upload = upload_run(manifest_path, backend_name=args.tracking_backend)
        # Flywheel failure must NOT mark the training run failed.
        manifest["upload"] = upload
        if not upload["ok"]:
            print(f"  upload failed (training run unaffected): {upload['error']}")
        write_json(manifest, manifest_path)  # persist the upload result
    print(f"Status: {status}; val_loss={summary.get('val_loss')}")
    return 0 if returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
