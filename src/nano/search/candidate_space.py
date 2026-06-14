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


# ---------------------------------------------------------------------------
# Curriculum search (the batch / seq-len / window ramps live in TRAINING_STAGES)
# ---------------------------------------------------------------------------

#: Per-stage curriculum knobs that ``curriculum_sweep`` can ramp. ``duration`` and
#: the ``mtp_weights_*`` schedules are deliberately excluded -- tune those via a
#: direct ``schedule.training_stages`` override (their shapes don't ramp cleanly).
CURRICULUM_RAMP_FIELDS = {"batch_size", "train_max_seq_len", "window_sizes", "lr_mul"}


def _baseline_stages() -> list[dict[str, Any]]:
    from nano.builder.context import BASELINE_TRAINING_STAGES

    return BASELINE_TRAINING_STAGES


def _is_window_pair(x: Any) -> bool:
    """True for a single ``(lo, hi)`` sliding-window pair of plain ints."""
    return (
        isinstance(x, (list, tuple))
        and len(x) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) for v in x)
    )


def _expand_ramp(field: str, candidate: Any, num_stages: int) -> list[Any]:
    """Normalise a ramp candidate into one value per stage.

    A scalar is broadcast to every stage; a per-stage list is used positionally.
    ``window_sizes`` is special-cased: a single ``(lo, hi)`` pair broadcasts, a
    list of ``num_stages`` pairs is per-stage.
    """
    if field == "window_sizes":
        if _is_window_pair(candidate):
            return [list(candidate)] * num_stages
        ramp = list(candidate)
        if len(ramp) != num_stages or not all(_is_window_pair(p) for p in ramp):
            raise ValueError(
                f"window_sizes ramp must be a (lo, hi) pair or a list of "
                f"{num_stages} such pairs; got {candidate!r}"
            )
        return [list(p) for p in ramp]

    if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
        return [candidate] * num_stages
    ramp = list(candidate)
    if len(ramp) != num_stages:
        raise ValueError(
            f"{field} ramp must be a scalar or a list of {num_stages} values "
            f"(one per stage); got {candidate!r}"
        )
    return ramp


def make_curriculum(
    field_ramps: dict[str, Any], *, base_stages: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Apply per-stage ramps onto a copy of a base curriculum.

    ``field_ramps`` maps a sweepable stage field (see
    :data:`CURRICULUM_RAMP_FIELDS`) to a *ramp*: a single value broadcast to
    every stage, or a per-stage list. Returns a fresh ``training_stages`` list,
    ready to drop into a ``schedule.training_stages`` override. ``base_stages``
    defaults to the record's ``BASELINE_TRAINING_STAGES``.
    """
    stages = copy.deepcopy(base_stages if base_stages is not None else _baseline_stages())
    for field, candidate in field_ramps.items():
        if field not in CURRICULUM_RAMP_FIELDS:
            raise ValueError(
                f"curriculum field {field!r} is not sweepable; allowed: "
                f"{sorted(CURRICULUM_RAMP_FIELDS)}. Tune duration / mtp_weights "
                f"via a direct schedule.training_stages override."
            )
        for st, value in zip(stages, _expand_ramp(field, candidate, len(stages))):
            st[field] = value
    return stages


def _ramp_tag(idx: int, candidate: Any) -> str:
    """A compact, collision-free name fragment for one ramp option."""
    if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
        return _fmt(candidate)  # readable for the common constant-per-curriculum sweep
    return f"r{idx}"  # whole ramps don't repr compactly -> tag by option index


def curriculum_sweep(
    base_features: set[str],
    ramps: dict[str, list[Any]],
    *,
    base_stages: list[dict[str, Any]] | None = None,
    base_overrides: dict[str, Any] | None = None,
    name_prefix: str = "curriculum",
) -> list[FeatureSet]:
    """Cartesian-product sweep over curriculum ramps (the curriculum search dim).

    ``ramps`` maps a sweepable stage field to a list of candidate ramps, each a
    scalar (broadcast across stages) or a per-stage list, e.g.::

        {
            "batch_size":   [8 * 2048 * 8, 16 * 2048 * 8],            # constant-per-curriculum
            "window_sizes": [[(1, 3), (3, 7), (5, 11), (6, 13)],      # two whole ramps
                             [(2, 4), (4, 8), (6, 12), (7, 14)]],
        }

    Each combination becomes a :class:`FeatureSet` sharing ``base_features``
    (curriculum is config, not genes) and carrying a ``schedule.training_stages``
    override -- the resolved stages merged onto ``base_overrides``.
    """
    if not ramps:
        raise ValueError("curriculum_sweep needs at least one ramp field to sweep")
    fields = list(ramps)
    indexed = [list(enumerate(ramps[f])) for f in fields]
    out: list[FeatureSet] = []
    for combo in itertools.product(*indexed):
        field_ramps = {f: opt for f, (_, opt) in zip(fields, combo)}
        stages = make_curriculum(field_ramps, base_stages=base_stages)
        overrides = copy.deepcopy(base_overrides or {})
        overrides.setdefault("schedule", {})["training_stages"] = stages
        tag = "__".join(f"{f}{_ramp_tag(idx, opt)}" for f, (idx, opt) in zip(fields, combo))
        out.append(
            FeatureSet(
                name=f"{name_prefix}__{tag}",
                enabled=set(base_features),
                overrides=overrides,
                description="curriculum sweep candidate",
            )
        )
    return out
