"""Resolve a concrete feature set from preset / file / CLI toggles."""

from __future__ import annotations

from pathlib import Path

from nano.config.base import FEATURE_SETS_DIR
from nano.config.schema import (
    FeatureSet,
    load_preset_dict,
    load_yaml,
    resolve_dict,
)


def list_presets() -> list[str]:
    if not FEATURE_SETS_DIR.exists():
        return []
    return sorted(p.stem for p in FEATURE_SETS_DIR.glob("*.yaml"))


def resolve_feature_set(
    *,
    preset: str | None = None,
    feature_set_file: str | Path | None = None,
    enable: list[str] | None = None,
    disable: list[str] | None = None,
    name: str | None = None,
) -> FeatureSet:
    """Resolve the final :class:`FeatureSet`.

    Precedence: a base preset / file establishes the starting enabled set and
    overrides; CLI ``--enable`` / ``--disable`` are then layered on top.
    """
    if feature_set_file is not None:
        raw = load_yaml(feature_set_file)
        source = str(feature_set_file)
    elif preset is not None:
        raw = load_preset_dict(preset)
        source = preset
    else:
        raw = {}
        source = None

    enabled, overrides, tracking = resolve_dict(raw)
    enabled |= set(enable or [])
    enabled -= set(disable or [])

    return FeatureSet(
        name=name or raw.get("name") or preset or "custom",
        enabled=enabled,
        overrides=overrides,
        base=raw.get("base"),
        enable=list(raw.get("enable", []) or []),
        disable=list(raw.get("disable", []) or []),
        tracking=tracking,
        description=raw.get("description", ""),
        source=source,
    )
