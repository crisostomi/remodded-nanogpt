"""Global feature registry.

The registry is the one piece of intentional global state in the system (per the
spec's coding-style guidance). Feature modules register their :class:`Feature`
objects at import time; :mod:`nano.features` imports every module so that simply
importing the package populates :data:`FEATURES`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from nano.features.base import Feature, FeatureSpec, FunctionFeature

if TYPE_CHECKING:
    from nano.builder.context import BuildContext

FEATURES: dict[str, Feature] = {}


def register(feature: Feature) -> Feature:
    """Register a feature by its spec name. Raises on duplicates."""
    name = feature.spec.name
    if name in FEATURES:
        raise ValueError(f"Duplicate feature: {name}")
    FEATURES[name] = feature
    return feature


def get_feature(name: str) -> Feature:
    """Look up a registered feature, raising ``KeyError`` with a clear message."""
    try:
        return FEATURES[name]
    except KeyError:
        raise KeyError(f"Unknown feature: {name}") from None


def feature(
    spec: FeatureSpec,
) -> Callable[[Callable[["BuildContext"], None]], FunctionFeature]:
    """Decorator that wraps an ``apply`` function into a registered feature.

    Usage::

        @feature(FeatureSpec(name="xsa", ...))
        def xsa(ctx: BuildContext) -> None:
            ctx.model.use_xsa = True
    """

    def decorator(fn: Callable[["BuildContext"], None]) -> FunctionFeature:
        wrapped = FunctionFeature(spec=spec, _apply=fn)
        register(wrapped)
        return wrapped

    return decorator


def all_feature_names() -> set[str]:
    return set(FEATURES)
