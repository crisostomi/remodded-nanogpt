"""MUDD: Multiway Dynamic Dense connections (arXiv:2502.12170), trimmed.

PR #259 added trimmed MUDD connections plus extra params, optimizer entries,
dynamic residual/value routing and a schedule reduction. MUDD is now
template-toggleable: its last-layer branch (dynamic value + residual
recombination) and the post-loop residual mixer are guarded, and the off-state
falls back to the normal per-layer residual path used by every other layer.

MUDD's last-layer branch fuses the value embeddings into ``aux_v`` and its
post-loop mixer reads the layer-1 value-embed table, so it ``requires``
``value_embeds``; the bigram injection it performs (``mu[11] * x0_bigram``) is
guarded independently so MUDD composes with ``bigram_hash`` being toggled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nano.features.base import FeatureSpec
from nano.features.registry import feature

if TYPE_CHECKING:
    from nano.builder.context import BuildContext


@feature(FeatureSpec(
    name="mudd_last_layers",
    description="Trimmed MUDD dynamic dense connections on the last layer + post-loop (PR #259).",
    requires=("value_embeds",),
    owns_params=("mudd_w1", "mudd_w2", "mudd_b2"),
    modifies_model=True,
    modifies_forward=True,
    modifies_optimizer=True,
    modifies_schedule=True,
    template_toggleable=True,
))
def mudd_last_layers(ctx: "BuildContext") -> None:
    ctx.model.use_mudd = True
