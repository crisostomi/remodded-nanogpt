"""Dependency / conflict / optimizer-consistency validation tests."""

from __future__ import annotations

import pytest

from nano.builder.render import build_context
from nano.builder.validate import (
    FeatureValidationError,
    validate_context,
    validate_renderable,
)
from nano.config.presets import resolve_feature_set


def current_record() -> set[str]:
    return resolve_feature_set(preset="current_record").enabled


def test_sign_trick_requires_bigram_hash():
    with pytest.raises(FeatureValidationError, match="bigram_sign_trick requires bigram_hash"):
        build_context({"bigram_sign_trick"})


def test_xsa_lowering_requires_xsa():
    with pytest.raises(FeatureValidationError, match="xsa_lowering_rewrite requires xsa"):
        build_context({"xsa_lowering_rewrite"})


def test_unknown_feature_rejected():
    with pytest.raises(FeatureValidationError, match="Unknown feature"):
        build_context({"not_a_real_feature"})


def test_sparse_attention_gate_soft_conflicts_with_xsa():
    # current_record enables both sparse_attention_gate and xsa -> soft-conflict warning.
    ctx = build_context(current_record())
    assert any("soft-conflict" in w for w in ctx.warnings)


def test_disable_sparse_attention_gate_removes_attn_gate_bank():
    enabled = current_record() - {"sparse_attention_gate"}
    ctx = build_context(enabled)
    assert "attn_gate_bank" not in ctx.optim.param_table
    assert "attn_gate_bank" not in ctx.optim.work_order
    assert "attn_gate_bank" not in ctx.optim.scatter_order


def test_enable_xsa_adds_xsa_alphas_optimizer_entry():
    ctx = build_context({"xsa"})
    assert "xsa_alphas" in ctx.optim.param_table
    # and disabling it removes the entry
    ctx2 = build_context(current_record() - {"xsa"})
    assert "xsa_alphas" not in ctx2.optim.param_table


def test_optimizer_consistency_missing_param_table_entry():
    # A param that exists but is missing from the optimizer table must fail.
    ctx = build_context({"xsa"})
    del ctx.optim.param_table["xsa_alphas"]
    with pytest.raises(FeatureValidationError, match="missing from optimizer param_table"):
        validate_context(ctx)


def test_optimizer_consistency_extra_param_table_entry():
    ctx = build_context({"xsa"})
    ctx.optim.param_table["ghost_param"] = {"optim": "adam", "comms": "replicated", "adam_betas": None}
    ctx.optim.work_order.append("ghost_param")
    ctx.optim.scatter_order.append("ghost_param")
    with pytest.raises(FeatureValidationError, match="no corresponding model parameter"):
        validate_context(ctx)


def test_renderable_rejects_partial_feature_set():
    # A bare bigram collection lacks the structural features the template needs.
    with pytest.raises(FeatureValidationError, match="structural features"):
        validate_renderable({"bigram_hash", "bigram_sign_trick"})


def test_renderable_accepts_current_record():
    validate_renderable(current_record())  # must not raise


def test_narrow_bigram_dim_requires_residual_slice():
    # Disabling the sliced injection while keeping the 192-dim bigram is invalid:
    # the full-residual add only type-checks when bigram_dim == model_dim.
    with pytest.raises(FeatureValidationError, match="requires residual_slice_bigram_injection"):
        build_context(current_record() - {"residual_slice_bigram_injection"})


def test_disabling_dim192_and_slice_together_is_valid():
    ctx = build_context(current_record() - {"bigram_dim_192", "residual_slice_bigram_injection"})
    assert ctx.model.bigram_dim == ctx.model.model_dim  # full-width bigram, full add is fine


def test_disabling_dim192_but_keeping_slice_is_rejected():
    with pytest.raises(FeatureValidationError, match="requires bigram_dim_192"):
        build_context(current_record() - {"bigram_dim_192"})
