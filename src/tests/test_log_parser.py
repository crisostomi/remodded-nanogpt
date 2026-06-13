"""Log-parser tests."""

from __future__ import annotations

from nano.runtime.parse_logs import parse_log, parse_log_text

SAMPLE = """\
Compiling model and warming up kernels (~7 minutes on first execution)
step:0/1385 val_loss:10.8312 train_time:0ms step_avg:0.00ms
step:250/1385 val_loss:3.9011 train_time:15000ms step_avg:60.00ms
step:1250/1385 val_loss:3.3001 train_time:75000ms step_avg:60.00ms
step:1385/1385 val_loss:3.2791 train_time:84429ms step_avg:60.96ms
peak memory allocated: 12345 MiB reserved: 23456 MiB
"""


def test_parse_final_val_loss_from_log():
    summary = parse_log_text(SAMPLE)
    assert summary["final_step"] == 1385
    assert summary["train_steps"] == 1385
    assert summary["val_loss"] == 3.2791
    assert summary["train_time_ms"] == 84429
    assert summary["step_avg_ms"] == 60.96


def test_parse_peak_memory():
    summary = parse_log_text(SAMPLE)
    assert summary["peak_memory_allocated_mib"] == 12345
    assert summary["peak_memory_reserved_mib"] == 23456


def test_parse_picks_last_val_line():
    # an earlier intermediate val_loss must not win
    summary = parse_log_text(SAMPLE)
    assert summary["val_loss"] != 3.3001


def test_parse_missing_fields_are_none():
    summary = parse_log_text("nothing useful here\n")
    assert summary["val_loss"] is None
    assert summary["peak_memory_allocated_mib"] is None


def test_parse_log_from_file(tmp_path):
    p = tmp_path / "raw.log"
    p.write_text(SAMPLE)
    summary = parse_log(p)
    assert summary["val_loss"] == 3.2791
