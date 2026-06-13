"""Builder: build a context from features and render a static train script."""

from __future__ import annotations

from nano.builder.context import BuildContext
from nano.builder.render import (
    build_context,
    render_train_script,
    render_train_script_text,
)
from nano.builder.validate import FeatureValidationError, validate_context

__all__ = [
    "BuildContext",
    "build_context",
    "render_train_script",
    "render_train_script_text",
    "validate_context",
    "FeatureValidationError",
]
