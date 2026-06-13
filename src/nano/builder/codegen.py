"""CLI: build a static training script from a feature set.

    python -m nano.builder.codegen --preset current_record --out generated/train_current_record.py
    python -m nano.builder.codegen --preset current_record \
        --disable sparse_attention_gate --enable xsa_lowering_rewrite \
        --out generated/train_current_record_minus_sag.py

Writes the train script plus ``<stem>.features.yaml`` and
``<stem>.manifest.json`` sidecars, and copies ``triton_kernels.py`` next to the
script so it is runnable in place.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

from nano.builder.context import BuildContext
from nano.builder.render import build_context, render_train_script
from nano.builder.validate import FeatureValidationError
from nano.config.base import BASELINE_DIR, baseline_sha, repo_git_info
from nano.config.presets import resolve_feature_set
from nano.config.schema import FeatureSet
from nano.runtime.manifest import create_initial_manifest, utc_now_iso, write_json


def build_header(
    feature_set: FeatureSet, ctx: BuildContext, *, disabled_features: list[str]
) -> dict[str, Any]:
    return {
        "feature_set": feature_set.name,
        "base_preset": feature_set.base or feature_set.name,
        "source_sha": repo_git_info().get("sha"),
        "baseline_sha": baseline_sha(),
        "generated_at": utc_now_iso(),
        "enabled_features": sorted(ctx.enabled_features),
        "disabled_features": sorted(disabled_features),
    }


def write_features_yaml(feature_set: FeatureSet, ctx: BuildContext, path: str | Path) -> Path:
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "name": feature_set.name,
        "base": feature_set.base,
        "description": feature_set.description,
        "enabled": sorted(ctx.enabled_features),
        "disable": sorted(feature_set.disable),
        "overrides": feature_set.overrides,
        "tracking": feature_set.tracking,
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def copy_kernels(dest_dir: str | Path) -> Path | None:
    src = BASELINE_DIR / "triton_kernels.py"
    if not src.exists():
        return None
    dest = Path(dest_dir) / "triton_kernels.py"
    shutil.copyfile(src, dest)
    return dest


def generate(
    feature_set: FeatureSet,
    out_script: str | Path,
    *,
    run_id: str | None = None,
    features_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    n_gpus: int = 8,
    world_size: int | None = None,
    disabled_features: list[str] | None = None,
    copy_triton_kernels: bool = True,
) -> dict[str, Any]:
    """Build context, render the script, and write sidecar artifacts.

    Returns a dict with the built context, written paths and the manifest.
    """
    out_script = Path(out_script)
    disabled_features = list(disabled_features or feature_set.disable)

    ctx = build_context(
        feature_set.enabled,
        overrides=feature_set.overrides,
        metadata={"tracking": feature_set.tracking},
    )

    header = build_header(feature_set, ctx, disabled_features=disabled_features)
    render_train_script(ctx, out_script, header)

    if copy_triton_kernels:
        copy_kernels(out_script.parent)

    result: dict[str, Any] = {
        "ctx": ctx,
        "script_path": out_script,
        "features_path": None,
        "manifest_path": None,
        "manifest": None,
    }

    if features_path is not None:
        result["features_path"] = write_features_yaml(feature_set, ctx, features_path)

    if manifest_path is not None:
        manifest = create_initial_manifest(
            ctx,
            feature_set,
            run_id=run_id or feature_set.name,
            generated_script=out_script,
            n_gpus=n_gpus,
            world_size=world_size,
            disabled_features=disabled_features,
        )
        write_json(manifest, manifest_path)
        result["manifest"] = manifest
        result["manifest_path"] = Path(manifest_path)

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a static modded-nanogpt train script.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--preset", help="Named preset under configs/feature_sets/")
    src.add_argument("--feature-set", help="Path to a feature-set YAML file")
    parser.add_argument("--enable", action="append", default=[], help="Enable a feature (repeatable)")
    parser.add_argument("--disable", action="append", default=[], help="Disable a feature (repeatable)")
    parser.add_argument("--out", required=True, help="Output path for the generated train script")
    parser.add_argument("--n-gpus", type=int, default=8)
    args = parser.parse_args(argv)

    out = Path(args.out)
    stem = out.parent / out.stem
    features_path = Path(f"{stem}.features.yaml")
    manifest_path = Path(f"{stem}.manifest.json")

    try:
        feature_set = resolve_feature_set(
            preset=args.preset,
            feature_set_file=args.feature_set,
            enable=args.enable,
            disable=args.disable,
        )
        result = generate(
            feature_set,
            out,
            features_path=features_path,
            manifest_path=manifest_path,
            n_gpus=args.n_gpus,
            disabled_features=sorted(set(args.disable) | set(feature_set.disable)),
        )
    except (FeatureValidationError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {result['script_path']}")
    print(f"Wrote {result['features_path']}")
    print(f"Wrote {result['manifest_path']}")
    print(f"Enabled {len(result['ctx'].enabled_features)} features; "
          f"{len(result['ctx'].optim.param_table)} optimizer params.")
    if result["ctx"].warnings:
        for w in result["ctx"].warnings:
            print(f"  warning: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
