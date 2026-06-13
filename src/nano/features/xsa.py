"""XSA: gated cross-head value subtraction (arXiv:2603.09078).

PR #264 added zero-initialized per-(layer, head) XSA gates on the non-paired
attention layers. PR #317 proposed a cheaper lowering of the same operation
(and argued the sparse attention gate may be redundant with XSA -- hence the
soft conflict). Both are fully template-toggleable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nano.features.base import FeatureSpec
from nano.features.registry import feature

if TYPE_CHECKING:
    from nano.builder.context import BuildContext


@feature(FeatureSpec(
    name="xsa",
    description="Per-(layer, head) learnable XSA gates on non-paired attention layers (PR #264).",
    owns_params=("xsa_alphas",),
    soft_conflicts=("sparse_attention_gate",),
    modifies_model=True,
    modifies_forward=True,
    modifies_optimizer=True,
    template_toggleable=True,
))
def xsa(ctx: "BuildContext") -> None:
    ctx.model.use_xsa = True


@feature(FeatureSpec(
    name="xsa_lowering_rewrite",
    description="Cheaper XSA lowering: dot/denom * v instead of normalize+proj (PR #317).",
    requires=("xsa",),
    modifies_forward=True,
    template_toggleable=True,
))
def xsa_lowering_rewrite(ctx: "BuildContext") -> None:
    ctx.model.use_xsa_lowering_rewrite = True
