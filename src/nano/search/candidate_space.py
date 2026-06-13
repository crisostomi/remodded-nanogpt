"""Simple candidate-feature-set generators for ablation and tuning sweeps."""

from __future__ import annotations

import copy
import itertools
from typing import Any

from nano.config.schema import FeatureSet


def _fs(name: str, enabled: set[str], *, enable: list[str], disable: list[str]) -> FeatureSet:
    return FeatureSet(
        name=name,
        enabled=set(enabled),
        enable=enable,
        disable=disable,
        description="generated candidate",
    )


def leave_one_out(base_features: set[str], candidates: list[str]) -> list[FeatureSet]:
    """One feature set per candidate, with that candidate removed from the base."""
    out: list[FeatureSet] = []
    for c in candidates:
        if c in base_features:
            out.append(_fs(f"minus_{c}", set(base_features) - {c}, enable=[], disable=[c]))
    return out


def enable_one(base_features: set[str], candidates: list[str]) -> list[FeatureSet]:
    """One feature set per candidate, with that candidate added to the base."""
    out: list[FeatureSet] = []
    for c in candidates:
        if c not in base_features:
            out.append(_fs(f"plus_{c}", set(base_features) | {c}, enable=[c], disable=[]))
    return out


def pairwise_toggles(base_features: set[str], pairs: list[tuple[str, str]]) -> list[FeatureSet]:
    """One feature set per pair, flipping both features' membership vs the base."""
    out: list[FeatureSet] = []
    for a, b in pairs:
        enabled = set(base_features)
        enable, disable = [], []
        for feat in (a, b):
            if feat in enabled:
                enabled.discard(feat)
                disable.append(feat)
            else:
                enabled.add(feat)
                enable.append(feat)
        out.append(_fs(f"toggle_{a}_{b}", enabled, enable=enable, disable=disable))
    return out


# ---------------------------------------------------------------------------
# Tuning-gene search (numeric / hyperparameter sweeps)
# ---------------------------------------------------------------------------

def _set_dotted(d: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``d["a"]["b"]["c"] = value`` from the dotted key ``"a.b.c"``."""
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _fmt(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def hyperparameter_sweep(
    base_features: set[str],
    grid: dict[str, list[Any]],
    *,
    base_overrides: dict[str, Any] | None = None,
    name_prefix: str = "sweep",
) -> list[FeatureSet]:
    """Cartesian-product sweep over numeric genes (the tuning search dimension).

    ``grid`` maps dotted override paths to candidate values, e.g.::

        {
            "optim.adam.lr": [0.007, 0.008, 0.009],
            "optim.normuon.momentum": [0.94, 0.95],
            "schedule.cooldown_frac": [0.55, 0.60],
        }

    Each combination becomes a :class:`FeatureSet` with the same enabled genes
    and the swept overrides merged onto ``base_overrides``.
    """
    keys = list(grid)
    out: list[FeatureSet] = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        overrides = copy.deepcopy(base_overrides or {})
        tag_parts = []
        for key, value in zip(keys, combo):
            _set_dotted(overrides, key, value)
            tag_parts.append(f"{key.split('.')[-1]}{_fmt(value)}")
        out.append(
            FeatureSet(
                name=f"{name_prefix}__" + "__".join(tag_parts),
                enabled=set(base_features),
                overrides=overrides,
                description="hyperparameter sweep candidate",
            )
        )
    return out
