"""Turn an enabled feature set into a ``BuildContext`` and render a train script."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nano.builder.context import MUDD_PARAMS, BuildContext
from nano.builder.validate import (
    validate_context,
    validate_dependencies,
    validate_feature_names,
    validate_ownership,
    validate_renderable,
    validate_semantics,
)
from nano.features.registry import FEATURES, get_feature

TEMPLATES_DIR = Path(__file__).parent / "templates"


# ---------------------------------------------------------------------------
# Build context
# ---------------------------------------------------------------------------

def topo_sort_features(names: set[str]) -> list[str]:
    """Deterministic topological order respecting ``requires`` edges.

    Dependencies are guaranteed present (validated beforehand). Ties are broken
    alphabetically so application order is reproducible.
    """
    remaining = set(names)
    resolved: list[str] = []
    while remaining:
        ready = sorted(
            n for n in remaining
            if all(dep not in remaining for dep in get_feature(n).spec.requires)
        )
        if not ready:
            raise ValueError(f"Cyclic feature dependencies among: {sorted(remaining)}")
        for n in ready:
            resolved.append(n)
            remaining.discard(n)
    return resolved


def _param_owner_map() -> dict[str, str]:
    """Map every owned param -> its owning feature (global, across the registry)."""
    owners: dict[str, str] = {}
    for feat in FEATURES.values():
        for param in feat.spec.owns_params:
            owners[param] = feat.spec.name
    return owners


def prune_disabled_params(ctx: BuildContext) -> None:
    """Remove param_table / work_order entries whose owning feature is disabled.

    Backbone params (no owner) are always kept. After pruning, ``scatter_order``
    is recomputed as ``list(param_table)`` to mirror the generated optimizer.
    """
    owners = _param_owner_map()
    for label in list(ctx.optim.param_table):
        owner = owners.get(label)
        if owner is not None and owner not in ctx.enabled_features:
            del ctx.optim.param_table[label]
    ctx.optim.work_order = [
        label for label in ctx.optim.work_order if label in ctx.optim.param_table
    ]
    ctx.optim.scatter_order = list(ctx.optim.param_table)


ALLOWED_OVERRIDE_SECTIONS = {"schedule", "model", "optim", "tracking"}

# Per-parameter optimizer fields that may be tuned via optim.params.<label>.
TUNABLE_PARAM_FIELDS = {"lr_mul", "wd_mul", "adam_betas", "eps", "comms", "optim"}


def _apply_optim_overrides(ctx: BuildContext, optim_over: dict[str, Any]) -> None:
    """Tune the optimizer's numeric genes: defaults and per-parameter fields.

    Shape::

        optim:
          adam:    {lr: 0.009, weight_decay: 0.006}
          normuon: {lr: 0.025, momentum: 0.94}
          params:
            lm_head:      {wd_mul: 120}
            bigram_embed: {adam_betas: [0.8, 0.95]}
    """
    unknown = set(optim_over) - {"adam", "normuon", "params"}
    if unknown:
        raise ValueError(f"Unknown optim override key(s): {sorted(unknown)}; allowed: adam, normuon, params")

    for key, defaults in (("adam", ctx.optim.adam_defaults), ("normuon", ctx.optim.normuon_defaults)):
        for k, v in (optim_over.get(key) or {}).items():
            if k not in defaults:
                raise ValueError(f"Unknown optim.{key} field {k!r}; allowed: {sorted(defaults)}")
            defaults[k] = v

    for label, fields in (optim_over.get("params") or {}).items():
        if label not in ctx.optim.param_table:
            raise ValueError(
                f"optim.params.{label}: no such optimizer parameter "
                f"(is its feature enabled?). Known: {sorted(ctx.optim.param_table)}"
            )
        for k, v in (fields or {}).items():
            if k not in TUNABLE_PARAM_FIELDS:
                raise ValueError(f"optim.params.{label}.{k} is not tunable; allowed: {sorted(TUNABLE_PARAM_FIELDS)}")
            ctx.optim.param_table[label][k] = v


def apply_overrides(ctx: BuildContext, overrides: dict[str, Any] | None) -> None:
    """Apply YAML overrides (schedule/model/optim/tracking) onto the context.

    ``model.use_*`` feature toggles may NOT be overridden directly: feature
    membership is the single source of truth (it drives both model construction
    and the optimizer param table). Allowing an override to set a ``use_*`` flag
    out of step with ``enable``/``disable`` would pass local validation but blow
    up at optimizer init on the GPU. Unknown sections are an error, not a no-op.
    """
    if not overrides:
        return
    unknown_sections = set(overrides) - ALLOWED_OVERRIDE_SECTIONS
    if unknown_sections:
        raise ValueError(
            f"Unknown override section(s): {sorted(unknown_sections)}; "
            f"allowed: {sorted(ALLOWED_OVERRIDE_SECTIONS)}"
        )
    for section in ("schedule", "model"):
        cfg = getattr(ctx, section)
        for key, value in (overrides.get(section) or {}).items():
            if section == "model" and key.startswith("use_"):
                raise ValueError(
                    f"model.{key} is a feature toggle; control it via "
                    f"enable/disable, not a model override"
                )
            if not hasattr(cfg, key):
                raise ValueError(f"Unknown override {section}.{key}")
            setattr(cfg, key, value)
    if "optim" in overrides:
        _apply_optim_overrides(ctx, overrides["optim"] or {})
    if "tracking" in overrides:
        ctx.metadata["tracking"] = overrides["tracking"]


def build_context(
    feature_names: set[str],
    overrides: dict[str, Any] | None = None,
    *,
    metadata: dict[str, Any] | None = None,
) -> BuildContext:
    """Validate, apply features in dependency order, prune and re-validate."""
    feature_names = set(feature_names)
    validate_feature_names(feature_names)
    warnings = validate_dependencies(feature_names)
    validate_ownership(feature_names)

    ctx = BuildContext(enabled_features=set(feature_names)).seed_baseline()
    ctx.warnings.extend(warnings)
    if metadata:
        ctx.metadata.update(metadata)

    for name in topo_sort_features(feature_names):
        get_feature(name).apply(ctx)

    apply_overrides(ctx, overrides)
    prune_disabled_params(ctx)
    validate_context(ctx)
    validate_semantics(ctx)
    return ctx


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def _pyval(value: Any) -> str:
    """Format a param-table value as Python source."""
    if value is None or isinstance(value, bool):
        return repr(value)  # None / True / False
    if isinstance(value, list):
        return "[" + ", ".join(_pyval(v) for v in value) + "]"
    return repr(value)


def _format_param_row(label: str, row: dict[str, Any]) -> str:
    body = ", ".join(f'"{k}": {_pyval(v)}' for k, v in row.items())
    return f'            "{label}": {{{body}}},'


def format_param_table(ctx: BuildContext) -> str:
    """Render ``self.param_table = {...}`` (+ MUDD update) from the context.

    Backbone + non-MUDD entries form the base dict; MUDD entries (if present)
    are emitted as a trailing ``.update({...})`` to mirror the record script, so
    that ``scatter_order = list(self.param_table)`` keeps the same order.
    """
    base = [label for label in ctx.optim.param_table if label not in MUDD_PARAMS]
    mudd = [label for label in ctx.optim.param_table if label in MUDD_PARAMS]

    lines = ["        self.param_table = {"]
    lines += [_format_param_row(l, ctx.optim.param_table[l]) for l in base]
    lines.append("        }")
    if mudd:
        lines.append("")
        lines.append("        # ---- MUDD parameter overrides ----")
        lines.append("        self.param_table.update({")
        lines += [_format_param_row(l, ctx.optim.param_table[l]) for l in mudd]
        lines.append("        })")
    return "\n".join(lines)


def format_work_order(ctx: BuildContext) -> str:
    """Render ``self.work_order = [...]`` from the context (order preserved)."""
    items = ", ".join(f'"{label}"' for label in ctx.optim.work_order)
    return f"        self.work_order = [\n            {items},\n        ]"


def format_defaults(name: str, defaults: dict[str, Any]) -> str:
    """Render ``<name> = dict(k=v, ...)`` from a defaults dict (8-space indent)."""
    lines = [f"        {name} = dict("]
    lines += [f"            {k}={_pyval(v)}," for k, v in defaults.items()]
    lines.append("        )")
    return "\n".join(lines)


def _render_header(header: dict[str, Any]) -> str:
    enabled = header.get("enabled_features", [])
    disabled = header.get("disabled_features", [])
    lines = [
        "# Generated by nano.builder.codegen -- DO NOT EDIT BY HAND.",
        f"# Feature set: {header.get('feature_set', '<unnamed>')}",
        f"# Base preset: {header.get('base_preset', '-')}",
        f"# Source git sha: {header.get('source_sha', '-')}",
        f"# Baseline sha: {header.get('baseline_sha', '-')}",
        f"# Generated at: {header.get('generated_at', '-')}",
        "# Enabled features:",
        *[f"#   - {name}" for name in enabled],
    ]
    if disabled:
        lines.append("# Disabled features:")
        lines.extend(f"#   - {name}" for name in disabled)
    return "\n".join(lines)


def render_train_script_text(ctx: BuildContext, header: dict[str, Any] | None = None) -> str:
    """Render the Jinja template for ``ctx`` and return the source text."""
    import jinja2

    validate_renderable(ctx.enabled_features)

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
    )
    template = env.get_template(ctx.render.template_name)
    rendered = template.render(
        model=ctx.model,
        schedule=ctx.schedule,
        optim=ctx.optim,
        data=ctx.data,
        distributed=ctx.distributed,
        loss=ctx.loss,
        render=ctx.render,
        metadata=ctx.metadata,
        enabled_features=sorted(ctx.enabled_features),
        header_comment=_render_header(header or {}),
        param_table_block=format_param_table(ctx),
        work_order_block=format_work_order(ctx),
        adam_defaults_block=format_defaults("adam_defaults", ctx.optim.adam_defaults),
        normuon_defaults_block=format_defaults("normuon_defaults", ctx.optim.normuon_defaults),
    )
    return rendered


def render_train_script(
    ctx: BuildContext, out_path: str | Path, header: dict[str, Any] | None = None
) -> Path:
    """Render and write the training script to ``out_path``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_train_script_text(ctx, header))
    return out_path
