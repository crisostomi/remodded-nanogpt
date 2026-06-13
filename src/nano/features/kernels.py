"""Misc model features: value embeddings, gates, smear, paired heads, key offset.

All structural for the MVP (always rendered in the record script). They own the
params they introduce so the optimizer table stays consistent, and flip context
flags read by the manifest and the template's construction-time guards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nano.features.base import FeatureSpec
from nano.features.registry import feature

if TYPE_CHECKING:
    from nano.builder.context import BuildContext


@feature(FeatureSpec(
    name="value_embeds",
    description="Token value embeddings injected as auxiliary V (by @KoszarskyB).",
    owns_params=("value_embeds",),
    modifies_model=True,
    modifies_forward=True,
))
def value_embeds(ctx: "BuildContext") -> None:
    ctx.model.use_value_embeds = True


@feature(FeatureSpec(
    name="value_embed_gates",
    description="Learnable per-(layer, head) gates on the value embeddings (ve_gate_bank).",
    requires=("value_embeds",),
    owns_params=("ve_gate_bank",),
    modifies_model=True,
    modifies_forward=True,
))
def value_embed_gates(ctx: "BuildContext") -> None:
    ctx.model.use_value_embed_gates = True


@feature(FeatureSpec(
    name="smear",
    description="Smear token embedding forward one position via a learned gate.",
    owns_params=("smear_gate",),
    modifies_model=True,
    modifies_forward=True,
    template_toggleable=True,
))
def smear(ctx: "BuildContext") -> None:
    ctx.model.use_smear = True


@feature(FeatureSpec(
    name="skip_gate",
    description="Learned skip connection feeding cache[3] into the attention-free layer 6.",
    owns_params=("skip_gate",),
    modifies_model=True,
    modifies_forward=True,
    template_toggleable=True,
))
def skip_gate(ctx: "BuildContext") -> None:
    ctx.model.use_skip_gate = True


@feature(FeatureSpec(
    name="paired_head_attention",
    description="Paired-head attention layers {0,2,5,9}: adjacent heads share keys.",
    modifies_model=True,
    modifies_forward=True,
    template_toggleable=True,
))
def paired_head_attention(ctx: "BuildContext") -> None:
    ctx.model.use_paired_head_attention = True


@feature(FeatureSpec(
    name="partial_key_offset",
    description="Shift keys forward on stationary head dims of long windows (1-layer induction).",
    modifies_forward=True,
    template_toggleable=True,
))
def partial_key_offset(ctx: "BuildContext") -> None:
    ctx.model.use_partial_key_offset = True
