"""Loss / schedule features.

PR #178 bundled multi-token prediction, untie-embed-at-2/3 and LR-schedule
changes; we keep them as separate features (the spec's "PR is not the
primitive" rule).

The batch-size / max-seq-len / sliding-window ramps that earlier records shipped
as standalone genes are **not** modelled here: they are points in the curriculum
config space, fully captured by the searchable ``TRAINING_STAGES`` (see
``BASELINE_TRAINING_STAGES`` in ``builder/context.py``). Search them with the
``schedule.training_stages`` override directly, or ergonomically via
``nano.search.candidate_space.curriculum_sweep``.

``mtp_loss`` is toggleable as a *config* gene. On the E15 substrate multi-token
prediction is purely the loss objective: the fused softcapped-CE kernel reads
``n_predict = mtp_weights.shape[0]`` future tokens per position from the same
plain next-token ``target_seq``. So its off-state is single-token prediction --
``mtp_weights == [1.0]`` for every stage (already the record's last two stages,
so the n_predict=1 path runs every record). The renderer collapses the schedule
when it is disabled; no template branch is needed (see
``render._finalize_loss_schedule``).
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
    template_toggleable=True,
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
