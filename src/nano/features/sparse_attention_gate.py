"""Sparse attention gate: per-(layer, head) sigmoid gate on attention output.

Multiplies each attention layer's output by ``sigmoid(linear(x[..., :12]))`` via
the ``attn_gate_bank`` parameter. This is the first ablation target -- PR #317
argues it may be redundant with XSA -- so it is fully template-toggleable and
soft-conflicts with ``xsa``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nano.features.base import FeatureSpec
from nano.features.registry import feature

if TYPE_CHECKING:
    from nano.builder.context import BuildContext


@feature(FeatureSpec(
    name="sparse_attention_gate",
    description="Per-(layer, head) sigmoid gate on attention output via attn_gate_bank.",
    owns_params=("attn_gate_bank",),
    soft_conflicts=("xsa",),
    modifies_model=True,
    modifies_forward=True,
    modifies_optimizer=True,
    template_toggleable=True,
))
def sparse_attention_gate(ctx: "BuildContext") -> None:
    ctx.model.use_sparse_attention_gate = True
