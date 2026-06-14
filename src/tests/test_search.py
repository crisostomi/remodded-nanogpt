"""Candidate-space search: ablations and tuning sweeps produce valid scripts."""

from __future__ import annotations

import pytest

from nano.builder.render import build_context, render_train_script_text
from nano.config.presets import resolve_feature_set
from nano.search.candidate_space import (
    curriculum_sweep,
    hyperparameter_sweep,
    leave_one_out,
    make_curriculum,
    pairwise_toggles,
)


def current_record() -> set[str]:
    return resolve_feature_set(preset="current_record").enabled


def _build_and_compile(fs) -> str:
    ctx = build_context(fs.enabled, overrides=fs.overrides)
    text = render_train_script_text(ctx, header={"feature_set": fs.name, "enabled_features": sorted(fs.enabled)})
    compile(text, f"<{fs.name}>", "exec")
    return text


def test_leave_one_out_over_toggleable_genes_all_render():
    base = current_record()
    candidates = ["sparse_attention_gate", "xsa", "bigram_sign_trick", "fp8_lm_head", "partial_key_offset"]
    sets = leave_one_out(base, candidates)
    assert len(sets) == len(candidates)
    for fs in sets:
        _build_and_compile(fs)  # each ablation is a valid, distinct script


def test_leave_one_out_over_entangled_architecture_genes():
    # mudd_last_layers and value_embed_gates are independently droppable; both
    # produce a coherent, compilable script.
    base = current_record()
    for fs in leave_one_out(base, ["mudd_last_layers", "value_embed_gates"]):
        _build_and_compile(fs)


def test_dropping_value_embeds_cluster_renders():
    # value_embeds requires dropping its dependents together; the resulting
    # value-embed-free script still compiles.
    fs = resolve_feature_set(
        preset="current_record",
        disable=["value_embeds", "value_embed_gates", "mudd_last_layers"],
    )
    _build_and_compile(fs)


def test_hyperparameter_sweep_grid_size_and_validity():
    base = current_record()
    grid = {
        "optim.adam.lr": [0.007, 0.008, 0.009],
        "optim.normuon.momentum": [0.94, 0.95],
        "schedule.cooldown_frac": [0.55, 0.60],
    }
    sets = hyperparameter_sweep(base, grid)
    assert len(sets) == 3 * 2 * 2  # cartesian product
    # names are unique
    assert len({fs.name for fs in sets}) == len(sets)

    # spot-check a couple actually thread the swept values into the script
    fs = sets[0]
    text = _build_and_compile(fs)
    assert "lr=" in text  # rendered optimizer defaults present


def test_sweep_values_reach_generated_script():
    base = current_record()
    sets = hyperparameter_sweep(base, {"optim.adam.lr": [0.0123]})
    text = _build_and_compile(sets[0])
    assert "lr=0.0123" in text


def test_pairwise_toggles_render():
    base = current_record()
    for fs in pairwise_toggles(base, [("sparse_attention_gate", "xsa")]):
        _build_and_compile(fs)


# ---------------------------------------------------------------------------
# Curriculum sweep (the batch / seq-len / window ramps -> training_stages)
# ---------------------------------------------------------------------------

def _stages(fs):
    ctx = build_context(fs.enabled, overrides=fs.overrides)
    return ctx.schedule.training_stages


def test_curriculum_sweep_grid_size_uniqueness_and_validity():
    base = current_record()
    ramps = {
        "batch_size":   [8 * 2048 * 8, 16 * 2048 * 8],
        "window_sizes": [[(1, 3), (3, 7), (5, 11), (6, 13)],
                         [(2, 4), (4, 8), (6, 12), (7, 14)]],
    }
    sets = curriculum_sweep(base, ramps)
    assert len(sets) == 2 * 2                         # cartesian product
    assert len({fs.name for fs in sets}) == len(sets)  # unique names
    for fs in sets:
        assert fs.enabled == base                    # curriculum is config, not genes
        _build_and_compile(fs)                       # each is a valid, distinct script


def test_curriculum_sweep_scalar_broadcasts_across_stages():
    base = current_record()
    fs = curriculum_sweep(base, {"batch_size": [12 * 2048 * 8]})[0]
    assert [s["batch_size"] for s in _stages(fs)] == [12 * 2048 * 8] * 4


def test_curriculum_sweep_values_reach_generated_script():
    base = current_record()
    fs = curriculum_sweep(base, {"window_sizes": [[(2, 4), (4, 8), (6, 12), (7, 14)]]})[0]
    text = _build_and_compile(fs)
    assert "window_sizes=(2, 4)" in text and "window_sizes=(7, 14)" in text
    assert "window_sizes=(1, 3)" not in text         # baseline windows overwritten


def test_curriculum_sweep_per_stage_ramp_threads_through():
    base = current_record()
    ramp = [8 * 2048 * 8, 16 * 2048 * 8, 24 * 2048 * 8, 32 * 2048 * 8]
    fs = curriculum_sweep(base, {"batch_size": [ramp]})[0]
    assert [s["batch_size"] for s in _stages(fs)] == ramp


def test_curriculum_sweep_preserves_base_overrides():
    base = current_record()
    sets = curriculum_sweep(base, {"lr_mul": [1.0, 1.1]},
                            base_overrides={"schedule": {"cooldown_frac": 0.55}})
    sched = sets[0].overrides["schedule"]
    assert sched["cooldown_frac"] == 0.55            # untouched sibling key survives
    assert "training_stages" in sched
    _build_and_compile(sets[0])


def test_curriculum_sweep_requires_a_field():
    with pytest.raises(ValueError, match="at least one ramp"):
        curriculum_sweep(current_record(), {})


def test_make_curriculum_defaults_to_baseline_without_mutating_it():
    from nano.builder.context import BASELINE_TRAINING_STAGES

    stages = make_curriculum({"lr_mul": 2.0})
    assert all(s["lr_mul"] == 2.0 for s in stages)
    assert BASELINE_TRAINING_STAGES[0]["lr_mul"] == 1.0  # module constant intact


def test_make_curriculum_rejects_unsweepable_and_malformed_ramps():
    with pytest.raises(ValueError, match="not sweepable"):
        make_curriculum({"duration": [0.5, 0.5, 0.5, 0.5]})
    with pytest.raises(ValueError, match="list of 4 values"):
        make_curriculum({"batch_size": [1, 2, 3]})       # wrong length
    with pytest.raises(ValueError, match="pair or a list of 4"):
        make_curriculum({"window_sizes": [(1, 3), (3, 7), (5, 11)]})
