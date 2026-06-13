"""BuildContext assembly tests."""

from __future__ import annotations

import pytest

from nano.builder.render import build_context, topo_sort_features
from nano.config.presets import resolve_feature_set
from nano.config.schema import resolve_dict


def current_record() -> set[str]:
    return resolve_feature_set(preset="current_record").enabled


def test_current_record_param_table_matches_baseline_shape():
    fs = resolve_feature_set(preset="current_record")
    ctx = build_context(fs.enabled, overrides=fs.overrides)
    # 17 base + 3 MUDD entries in the record optimizer table.
    assert len(ctx.optim.param_table) == 20
    assert set(ctx.optim.work_order) == set(ctx.optim.param_table)
    assert ctx.optim.scatter_order == list(ctx.optim.param_table)


def test_current_record_bigram_dims_and_comms():
    ctx = build_context(current_record())
    assert ctx.model.bigram_vocab_size == 50304 * 15
    assert ctx.model.bigram_dim == 192
    assert ctx.model.bigram_sign_table_rows == 8192
    assert ctx.optim.param_table["bigram_embed"]["comms"] == "sharded_sparse"


def test_bigram_defaults_when_subfeatures_disabled():
    ctx = build_context({"bigram_hash"})
    assert ctx.model.bigram_vocab_size == 50304 * 5  # default, no 15x
    assert ctx.model.bigram_dim == ctx.model.model_dim  # default, no dim_192
    # without sparse_bigram_comms the bigram-embed gradient comms stay dense
    assert ctx.optim.param_table["bigram_embed"]["comms"] == "sharded"


def test_schedule_overrides_applied():
    fs = resolve_feature_set(preset="current_record")
    ctx = build_context(fs.enabled, overrides={"schedule": {"num_scheduled_iterations": 1200}})
    assert ctx.schedule.num_scheduled_iterations == 1200
    assert ctx.schedule.total_steps == 1200 + ctx.schedule.num_extension_iterations


def test_topo_sort_respects_requires():
    order = topo_sort_features({"xsa", "xsa_lowering_rewrite"})
    assert order.index("xsa") < order.index("xsa_lowering_rewrite")


def test_topo_sort_bigram_chain():
    order = topo_sort_features(
        {"bigram_hash", "bigram_dim_192", "residual_slice_bigram_injection"}
    )
    assert order.index("bigram_hash") < order.index("bigram_dim_192")
    assert order.index("bigram_dim_192") < order.index("residual_slice_bigram_injection")


def test_disabling_xsa_lowering_keeps_xsa():
    ctx = build_context(current_record())  # current record uses original lowering
    assert ctx.model.use_xsa is True
    assert ctx.model.use_xsa_lowering_rewrite is False


# ---- override-safety regressions (from the adversarial review) ----

def test_model_use_flag_override_is_rejected():
    # A model.use_* override could desync from enable/disable and crash at
    # optimizer init on the GPU; it must be rejected at build time.
    with pytest.raises(ValueError, match="feature toggle"):
        build_context(current_record(), overrides={"model": {"use_xsa": True}})


def test_unknown_override_section_is_rejected():
    with pytest.raises(ValueError, match="Unknown override section"):
        build_context(current_record(), overrides={"loss": {"use_mtp": False}})


def test_legit_model_override_still_applies():
    ctx = build_context(current_record(), overrides={"model": {"num_heads": 8}})
    assert ctx.model.num_heads == 8


def test_unknown_feature_set_key_is_rejected():
    # A stray loss:/data: section (or typo) must error, not silently no-op.
    with pytest.raises(ValueError, match="Unknown feature-set key"):
        resolve_dict({"enable": ["xsa"], "loss": {"use_mtp": False}})


# ---- tuning genes (optim override search dimension) ----

def test_optim_defaults_override_applies():
    ctx = build_context(
        current_record(),
        overrides={"optim": {"adam": {"lr": 0.009}, "normuon": {"momentum": 0.94}}},
    )
    assert ctx.optim.adam_defaults["lr"] == 0.009
    assert ctx.optim.normuon_defaults["momentum"] == 0.94


def test_optim_per_param_override_applies():
    ctx = build_context(
        current_record(),
        overrides={"optim": {"params": {"lm_head": {"wd_mul": 120.0}}}},
    )
    assert ctx.optim.param_table["lm_head"]["wd_mul"] == 120.0


def test_optim_unknown_field_rejected():
    with pytest.raises(ValueError, match="Unknown optim"):
        build_context(current_record(), overrides={"optim": {"adam": {"lbrate": 0.1}}})


def test_optim_unknown_param_rejected():
    with pytest.raises(ValueError, match="no such optimizer parameter"):
        build_context(current_record(), overrides={"optim": {"params": {"nope": {"lr_mul": 1.0}}}})


# ---- newly-toggleable architectural genes ----

def test_fp8_and_partial_key_offset_are_toggleable():
    ctx = build_context(current_record() - {"fp8_lm_head", "partial_key_offset"})
    assert ctx.model.use_fp8_lm_head is False
    assert ctx.model.use_partial_key_offset is False
