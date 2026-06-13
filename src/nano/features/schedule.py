"""Loss / schedule features.

PR #178 bundled multi-token prediction, untie-embed-at-2/3 and LR-schedule
changes; we keep them as separate features (the spec's "PR is not the
primitive" rule). All are structural for the MVP -- the record schedule already
performs the window / batch-size / seq-len / yarn updates -- so ``apply`` flips
context flags that the manifest and validation read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nano.features.base import FeatureSpec
from nano.features.registry import feature

if TYPE_CHECKING:
    from nano.builder.context import BuildContext


@feature(FeatureSpec(
    name="mtp_loss",
    description="Multi-token prediction loss with a per-stage MTP weight schedule.",
    modifies_loss=True,
    modifies_schedule=True,
))
def mtp_loss(ctx: "BuildContext") -> None:
    ctx.loss.use_mtp = True
    ctx.schedule.use_mtp_schedule = True


@feature(FeatureSpec(
    name="untie_embed_at_2_3",
    description="Start with embed tied to lm_head; copy state and untie at 2/3 of training.",
    modifies_optimizer=True,
    modifies_schedule=True,
    template_toggleable=True,
))
def untie_embed_at_2_3(ctx: "BuildContext") -> None:
    ctx.schedule.use_untie_embed = True


@feature(FeatureSpec(
    name="yarn_window_schedule",
    description="YaRN RoPE updates on sliding-window size changes.",
    modifies_schedule=True,
))
def yarn_window_schedule(ctx: "BuildContext") -> None:
    ctx.schedule.use_yarn_window_schedule = True


@feature(FeatureSpec(
    name="batch_size_schedule",
    description="Batch-size schedule of 8 -> 16 -> 24 across stages.",
    modifies_schedule=True,
    modifies_data=True,
))
def batch_size_schedule(ctx: "BuildContext") -> None:
    ctx.schedule.use_batch_size_schedule = True


@feature(FeatureSpec(
    name="max_seq_len_schedule",
    description="Max sequence length schedule (896 -> 2048 at 1/3 of training).",
    modifies_schedule=True,
    modifies_data=True,
))
def max_seq_len_schedule(ctx: "BuildContext") -> None:
    ctx.schedule.use_max_seq_len_schedule = True
