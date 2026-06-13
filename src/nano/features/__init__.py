"""Feature package.

Importing this package imports every feature module, which registers all
features into :data:`nano.features.registry.FEATURES` as a side effect.
"""

from __future__ import annotations

from nano.features import (  # noqa: F401  (imported for registration side effects)
    bigram,
    kernels,
    mudd,
    optimizer,
    schedule,
    sparse_attention_gate,
    xsa,
)
from nano.features.base import Feature, FeatureSpec, FunctionFeature
from nano.features.registry import (
    FEATURES,
    all_feature_names,
    feature,
    get_feature,
    register,
)

__all__ = [
    "Feature",
    "FeatureSpec",
    "FunctionFeature",
    "FEATURES",
    "all_feature_names",
    "feature",
    "get_feature",
    "register",
]
