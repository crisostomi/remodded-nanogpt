"""MUDD: Multiway Dynamic Dense connections (arXiv:2502.12170), trimmed.

PR #259 added trimmed MUDD connections plus extra params, optimizer entries,
dynamic residual/value routing and a schedule reduction. For the MVP MUDD is
kept structural (intact behind a static generation switch, per the spec) rather
than decomposed -- so it is *not* yet template-toggleable, but it does own its
params so the optimizer table stays consistent.
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
    owns_params=("mudd_w1", "mudd_w2", "mudd_b2"),
    modifies_model=True,
    modifies_forward=True,
    modifies_optimizer=True,
    modifies_schedule=True,
    template_toggleable=False,
))
def mudd_last_layers(ctx: "BuildContext") -> None:
    ctx.model.use_mudd = True
