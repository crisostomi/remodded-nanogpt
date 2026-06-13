"""Manifest creation / update tests."""

from __future__ import annotations

from nano.builder.render import build_context
from nano.config.presets import resolve_feature_set
from nano.runtime.manifest import (
    code_sha256,
    create_initial_manifest,
    update_manifest_with_summary,
)


def _manifest(tmp_path):
    fs = resolve_feature_set(preset="current_record")
    ctx = build_context(fs.enabled, overrides=fs.overrides)
    script = tmp_path / "train_generated.py"
    script.write_text("print('hello')\n")
    return create_initial_manifest(
        ctx, fs, run_id="run_x", generated_script=script, n_gpus=8, world_size=8
    ), script


def test_create_initial_manifest_fields(tmp_path):
    manifest, _ = _manifest(tmp_path)
    assert manifest["run_id"] == "run_x"
    assert manifest["feature_set"] == "current_record"
    assert manifest["status"] == "pending"
    assert manifest["hardware"] == {"gpu_type": "H100", "n_gpus": 8, "world_size": 8}
    assert manifest["schedule"]["total_steps"] == 1385
    assert manifest["schedule"]["num_scheduled_iterations"] == 1375
    assert set(manifest["metrics"]) >= {"val_loss", "train_time_ms", "step_avg_ms"}
    assert all(v is None for v in manifest["metrics"].values())
    assert "baseline" in manifest and manifest["baseline"]["source_sha"]


def test_manifest_code_hash_matches(tmp_path):
    manifest, script = _manifest(tmp_path)
    assert manifest["generated"]["sha256"] == code_sha256(script)


def test_update_manifest_with_summary(tmp_path):
    manifest, _ = _manifest(tmp_path)
    summary = {
        "final_step": 1385,
        "train_steps": 1385,
        "val_loss": 3.2791,
        "train_time_ms": 84429,
        "step_avg_ms": 60.96,
        "peak_memory_allocated_mib": 12345,
        "peak_memory_reserved_mib": 23456,
    }
    update_manifest_with_summary(manifest, summary, "completed")
    assert manifest["status"] == "completed"
    assert manifest["metrics"]["val_loss"] == 3.2791
    assert manifest["metrics"]["peak_memory_allocated_mib"] == 12345
    assert manifest["final_step"] == 1385


def test_update_manifest_failed_records_error(tmp_path):
    manifest, _ = _manifest(tmp_path)
    update_manifest_with_summary(manifest, None, "failed", error="boom")
    assert manifest["status"] == "failed"
    assert manifest["error"] == "boom"
