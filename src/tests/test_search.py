"""Candidate-space search: ablations and tuning sweeps produce valid scripts."""

from __future__ import annotations

from nano.builder.render import build_context, render_train_script_text
from nano.config.presets import resolve_feature_set
from nano.search.candidate_space import (
    hyperparameter_sweep,
    leave_one_out,
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
