"""Config: feature-set schema, loading and preset resolution."""

from __future__ import annotations

from nano.config.presets import list_presets, resolve_feature_set
from nano.config.schema import FeatureSet, load_yaml, resolve_dict

__all__ = [
    "FeatureSet",
    "resolve_feature_set",
    "list_presets",
    "load_yaml",
    "resolve_dict",
]
