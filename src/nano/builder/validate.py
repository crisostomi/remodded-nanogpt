"""Feature- and context-level validation.

Catches the obvious problems *before* a GPU run is launched. The generated
optimizer still asserts param-table consistency at runtime; we mirror that style
here so an invalid feature set fails fast on a laptop instead of on 8xH100.
"""

from __future__ import annotations

from nano.builder.context import CORE_PARAMS, BuildContext
from nano.features.registry import FEATURES, get_feature


class FeatureValidationError(ValueError):
    """Raised when an enabled feature set or built context is inconsistent."""


# ---------------------------------------------------------------------------
# Feature-set level checks (operate on the set of enabled names)
# ---------------------------------------------------------------------------

def validate_feature_names(names: set[str]) -> None:
    """Rule 1: every enabled feature must exist in the registry."""
    unknown = sorted(n for n in names if n not in FEATURES)
    if unknown:
        raise FeatureValidationError(
            f"Unknown feature(s): {', '.join(unknown)}. "
            f"Known features: {', '.join(sorted(FEATURES))}"
        )


def validate_dependencies(names: set[str]) -> list[str]:
    """Rules 2-4. Raises on missing requires / hard conflicts; returns warnings.

    Soft conflicts produce warning strings rather than errors.
    """
    warnings: list[str] = []
    for name in sorted(names):
        spec = get_feature(name).spec

        # Rule 2: requires must be enabled.
        for dep in spec.requires:
            if dep not in names:
                raise FeatureValidationError(f"{name} requires {dep}")

        # Rule 3: no hard-conflicting pair may both be enabled.
        for other in spec.conflicts:
            if other in names:
                raise FeatureValidationError(
                    f"{name} conflicts with {other}; both are enabled"
                )

        # Rule 4: soft conflicts -> warning only.
        for other in spec.soft_conflicts:
            if other in names and other > name:  # emit once per unordered pair
                warnings.append(
                    f"{name} soft-conflicts with {other}; both are enabled"
                )
    return warnings


def validate_ownership(names: set[str]) -> None:
    """Rules 5-6: no two enabled features may own the same param or buffer."""
    for kind, attr in (("parameter", "owns_params"), ("buffer", "owns_buffers")):
        owners: dict[str, str] = {}
        for name in sorted(names):
            spec = get_feature(name).spec
            for owned in getattr(spec, attr):
                if owned in owners:
                    raise FeatureValidationError(
                        f"{kind} {owned!r} is owned by both "
                        f"{owners[owned]!r} and {name!r}"
                    )
                owners[owned] = name


def validate_renderable(names: set[str]) -> None:
    """A feature set is renderable only if every structural (non-toggleable)
    feature is enabled -- the template's static forward path depends on them.

    This is intentionally *not* part of :func:`validate_context` so that partial
    feature sets can still be built and inspected; it is enforced by the renderer
    before emitting a training script.
    """
    missing = sorted(
        f.spec.name
        for f in FEATURES.values()
        if not f.spec.template_toggleable and f.spec.name not in names
    )
    if missing:
        raise FeatureValidationError(
            "Cannot render a training script: the following structural features "
            f"are not yet template-toggleable and must be enabled: {', '.join(missing)}. "
            "Base your feature set on `current_record` and only toggle the "
            "template-toggleable features."
        )


# ---------------------------------------------------------------------------
# Context-level checks (operate on the fully-applied BuildContext)
# ---------------------------------------------------------------------------

def expected_model_params(names: set[str]) -> set[str]:
    """The params the generated model is expected to create: the core backbone
    plus every param owned by an enabled feature."""
    expected = set(CORE_PARAMS)
    for name in names:
        expected.update(get_feature(name).spec.owns_params)
    return expected


def validate_semantics(ctx: BuildContext) -> None:
    """Cross-field semantic checks that aren't captured by requires/conflicts.

    Notably: a bigram embedding narrower than the model dim must be injected via
    a residual *slice*; the full-residual-add path only type-checks when
    ``bigram_dim == model_dim``.
    """
    m = ctx.model
    if (
        m.use_bigram_hash
        and m.bigram_dim is not None
        and m.bigram_dim < m.model_dim
        and not m.use_residual_slice_bigram_injection
    ):
        raise FeatureValidationError(
            f"bigram_dim ({m.bigram_dim}) < model_dim ({m.model_dim}) requires "
            f"residual_slice_bigram_injection (the full-residual add only works "
            f"when bigram_dim == model_dim)"
        )


def validate_context(ctx: BuildContext) -> None:
    """Rules 7-9: optimizer param-table consistency.

    7. Every expected model parameter must have a param_table entry.
    8. Every param_table entry must correspond to an expected model parameter.
    9. work_order and scatter_order must contain exactly the param_table labels.
    """
    table = set(ctx.optim.param_table)
    expected = expected_model_params(ctx.enabled_features)

    # Rule 7
    missing = sorted(expected - table)
    if missing:
        raise FeatureValidationError(
            f"Parameter {missing[0]} exists but is missing from optimizer "
            f"param_table" + (f" (and {len(missing) - 1} more)" if len(missing) > 1 else "")
        )

    # Rule 8
    extra = sorted(table - expected)
    if extra:
        raise FeatureValidationError(
            f"Optimizer param_table entry {extra[0]} has no corresponding model "
            f"parameter" + (f" (and {len(extra) - 1} more)" if len(extra) > 1 else "")
        )

    # Rule 9
    for order_name, order in (("work_order", ctx.optim.work_order),
                              ("scatter_order", ctx.optim.scatter_order)):
        order_set = set(order)
        if len(order) != len(order_set):
            raise FeatureValidationError(f"{order_name} contains duplicate labels")
        if order_set != table:
            missing_o = sorted(table - order_set)
            extra_o = sorted(order_set - table)
            raise FeatureValidationError(
                f"{order_name} must match param_table labels exactly; "
                f"missing={missing_o} extra={extra_o}"
            )
