"""Feature-set schema and YAML loading.

A *feature set* is a named collection of features plus optional schedule/model
overrides and tracking metadata. Feature sets may extend a ``base`` preset; the
final enabled set is ``base`` ∪ ``enable`` − ``disable``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nano.config.base import FEATURE_SETS_DIR


@dataclass
class FeatureSet:
    name: str
    enabled: set[str]
    overrides: dict[str, Any] = field(default_factory=dict)
    base: str | None = None
    enable: list[str] = field(default_factory=list)
    disable: list[str] = field(default_factory=list)
    tracking: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    source: str | None = None


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature-set file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Feature-set file {path} must contain a mapping")
    return data


def preset_path(name: str) -> Path:
    return FEATURE_SETS_DIR / f"{name}.yaml"


def load_preset_dict(name: str) -> dict[str, Any]:
    path = preset_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Unknown preset {name!r} (expected {path}). "
            f"Available: {', '.join(sorted(p.stem for p in FEATURE_SETS_DIR.glob('*.yaml')))}"
        )
    return load_yaml(path)


#: Top-level keys a feature-set mapping may contain. Anything else (e.g. a
#: stray `optim:`/`loss:`/`data:` section or a typo) is rejected rather than
#: silently dropped.
KNOWN_FEATURE_SET_KEYS = {
    "name", "base", "description", "enable", "disable", "schedule", "model", "optim", "tracking",
}


def _merge_overrides(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for section in ("schedule", "model", "optim"):
        if section in src:
            dst.setdefault(section, {}).update(src[section] or {})


def resolve_dict(
    raw: dict[str, Any], _seen: set[str] | None = None
) -> tuple[set[str], dict[str, Any], dict[str, Any]]:
    """Resolve a raw feature-set mapping into (enabled, overrides, tracking)."""
    _seen = _seen or set()
    unknown_keys = set(raw) - KNOWN_FEATURE_SET_KEYS
    if unknown_keys:
        raise ValueError(
            f"Unknown feature-set key(s): {sorted(unknown_keys)}; "
            f"allowed: {sorted(KNOWN_FEATURE_SET_KEYS)}"
        )
    enabled: set[str] = set()
    overrides: dict[str, Any] = {}
    tracking: dict[str, Any] = {}

    base = raw.get("base")
    if base:
        if base in _seen:
            raise ValueError(f"Cyclic feature-set base reference: {base}")
        b_enabled, b_over, b_track = resolve_dict(load_preset_dict(base), _seen | {base})
        enabled |= b_enabled
        _merge_overrides(overrides, b_over)
        tracking.update(b_track)

    enabled |= set(raw.get("enable", []) or [])
    enabled -= set(raw.get("disable", []) or [])
    _merge_overrides(overrides, raw)
    tracking.update(raw.get("tracking", {}) or {})

    return enabled, overrides, tracking
