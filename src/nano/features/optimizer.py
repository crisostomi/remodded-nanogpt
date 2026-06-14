"""Optimizer / precision features.

These are structural for the MVP: the record optimizer (combined NorMuon+Adam
with Polar Express orthogonalization, cautious weight decay and even/odd Adam
stepping) and the fp8 lm-head are always rendered. Their ``apply`` methods only
flip context flags so the enabled-feature list and manifest stay faithful.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nano.features.base import FeatureSpec
from nano.features.registry import feature

if TYPE_CHECKING:
    from nano.builder.context import BuildContext


@feature(FeatureSpec(
    name="fp8_lm_head",
    description="FP8 matmul for the (transposed) lm_head, by @YouJiacheng.",
    modifies_model=True,
    modifies_loss=True,
    template_toggleable=True,
))
def fp8_lm_head(ctx: "BuildContext") -> None:
    ctx.model.use_fp8_lm_head = True


# ---------------------------------------------------------------------------
# Orthogonalizer allele slot: exactly one of {polar_express, newton_schulz}.
# Both share the same fused-momentum / bf16 / Triton-kernel scaffold (the
# ``polar_express`` function); they differ ONLY in the odd-polynomial iteration
# coefficient table. polar_express uses 5 per-step tuned (a, b, c) rows; Newton-
# Schulz uses the canonical Muon quintic (3.4445, -4.7750, 2.0315) every step.
# Numeric behaviour of the newton_schulz allele is GPU-validation-pending.
# ---------------------------------------------------------------------------

@feature(FeatureSpec(
    name="polar_express",
    description="Polar Express orthogonalization (per-step tuned coefficients).",
    modifies_optimizer=True,
    template_toggleable=True,
    allele_group="orthogonalizer",
))
def polar_express(ctx: "BuildContext") -> None:
    ctx.optim.use_polar_express = True
    ctx.optim.orthogonalizer = "polar_express"


@feature(FeatureSpec(
    name="newton_schulz",
    description="Newton-Schulz orthogonalization (canonical Muon quintic 3.4445/-4.7750/2.0315).",
    modifies_optimizer=True,
    template_toggleable=True,
    allele_group="orthogonalizer",
))
def newton_schulz(ctx: "BuildContext") -> None:
    ctx.optim.orthogonalizer = "newton_schulz"


@feature(FeatureSpec(
    name="normuon",
    description="NorMuon: Muon with a low-rank Adafactor-style variance estimator.",
    modifies_optimizer=True,
))
def normuon(ctx: "BuildContext") -> None:
    ctx.optim.use_normuon = True


@feature(FeatureSpec(
    name="cautious_weight_decay",
    description="Gated (cautious) decoupled weight decay for Adam and NorMuon.",
    modifies_optimizer=True,
    template_toggleable=True,
))
def cautious_weight_decay(ctx: "BuildContext") -> None:
    ctx.optim.use_cautious_weight_decay = True


@feature(FeatureSpec(
    name="adam_every_other_step",
    description="Adam params are only updated on odd steps, by @classiclarryd.",
    modifies_optimizer=True,
    modifies_schedule=True,
    template_toggleable=True,
))
def adam_every_other_step(ctx: "BuildContext") -> None:
    ctx.optim.use_adam_every_other_step = True
