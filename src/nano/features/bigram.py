"""Bigram-hash embedding family.

PR #201 introduced bigram-hash embeddings; PR #299 added the sign-trick
compression, a wider vocab, a smaller bigram dim and the sliced residual
injection. These are decomposed into independent, searchable features.

``bigram_hash`` itself is kept structural for the MVP (the bigram data path /
forward is not fully guarded for total removal), but every sub-feature on top of
it is template-toggleable so the first ablations can move them independently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nano.features.base import FeatureSpec
from nano.features.registry import feature

if TYPE_CHECKING:
    from nano.builder.context import BuildContext

# next_multiple_of_n(50257, 128) -- the padded GPT-2 vocab used for bigram hashing.
PADDED_VOCAB = 50304


@feature(FeatureSpec(
    name="bigram_hash",
    description="Per-position bigram-hash embeddings injected into the residual stream (PR #201).",
    owns_params=("bigram_embed", "x0_lambdas", "bigram_lambdas"),
    modifies_model=True,
    modifies_forward=True,
    modifies_optimizer=True,
    modifies_data=True,
))
def bigram_hash(ctx: "BuildContext") -> None:
    ctx.model.use_bigram_hash = True
    # Defaults; overridden by bigram_vocab_15x / bigram_dim_192 when enabled.
    if ctx.model.bigram_vocab_size is None:
        ctx.model.bigram_vocab_size = PADDED_VOCAB * 5
    if ctx.model.bigram_dim is None:
        ctx.model.bigram_dim = ctx.model.model_dim
    ctx.data.needs_bigram_inputs = True


@feature(FeatureSpec(
    name="bigram_sign_trick",
    description="Sign-table trick to compress multiple bigrams into shared rows (PR #299).",
    requires=("bigram_hash",),
    owns_buffers=("bigram_sign_table",),
    modifies_model=True,
    modifies_forward=True,
    modifies_distributed=True,
    template_toggleable=True,
))
def bigram_sign_trick(ctx: "BuildContext") -> None:
    ctx.model.use_bigram_sign_trick = True
    if ctx.model.bigram_sign_table_rows is None:
        ctx.model.bigram_sign_table_rows = 8192
    if "bigram_sign_table" not in ctx.distributed.broadcast_buffers:
        ctx.distributed.broadcast_buffers.append("bigram_sign_table")


@feature(FeatureSpec(
    name="bigram_vocab_15x",
    description="Widen the bigram-hash vocab to 15x the padded token vocab.",
    requires=("bigram_hash",),
    modifies_model=True,
    template_toggleable=True,
))
def bigram_vocab_15x(ctx: "BuildContext") -> None:
    ctx.model.bigram_vocab_size = PADDED_VOCAB * 15


@feature(FeatureSpec(
    name="bigram_dim_192",
    description="Shrink the bigram embedding dim to 192 (< model_dim).",
    requires=("bigram_hash",),
    modifies_model=True,
    template_toggleable=True,
))
def bigram_dim_192(ctx: "BuildContext") -> None:
    ctx.model.bigram_dim = 192


@feature(FeatureSpec(
    name="residual_slice_bigram_injection",
    description="Inject bigram embeddings into a leading slice of the residual stream (PR #299).",
    requires=("bigram_hash", "bigram_dim_192"),
    modifies_forward=True,
    template_toggleable=True,
))
def residual_slice_bigram_injection(ctx: "BuildContext") -> None:
    ctx.model.use_residual_slice_bigram_injection = True


@feature(FeatureSpec(
    name="sparse_bigram_comms",
    description="Sparse reduce-scatter of bigram-embed gradients (active at world_size==8).",
    requires=("bigram_hash",),
    modifies_optimizer=True,
    modifies_distributed=True,
    template_toggleable=True,
))
def sparse_bigram_comms(ctx: "BuildContext") -> None:
    ctx.model.use_sparse_bigram_comms = True
    # Upgrade the bigram-embed gradient comms from dense to sparse.
    if "bigram_embed" in ctx.optim.param_table:
        ctx.optim.param_table["bigram_embed"]["comms"] = "sharded_sparse"
