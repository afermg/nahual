from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "summarize_real_gpu.py"
spec = importlib.util.spec_from_file_location("real_gpu_accounting_test", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write_jsonl(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (tmp_path / "clock.json").write_text(
        json.dumps({"gpu_timestamp_utc_offset_seconds": 0})
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"records": [{"ordinal": 0}, {"ordinal": 1}]}))
    phase = "shm_session_cached/repeat0"
    report = {
        "status": "complete",
        "validation": [{"mode": "shm_session_cached", "passed": True, "exact": True}],
        "config": {
            "label": "test",
            "manifest": str(manifest),
            "limit": None,
            "replicas": 1,
            "decode": "preloaded",
            "batch_size": 1,
            "batch_inference": False,
        },
        "phases": [
            {
                "phase": phase,
                "mode": "shm_session_cached",
                "start_unix": 1700000000,
                "end_unix": 1700000001,
                "wall_seconds": 1.0,
                "sites": 2,
                "model_outputs": 10,
                "sites_per_second_all_five_models": 2.0,
                "median_request_ms": 50.0,
            }
        ],
    }
    (root / "report.json").write_text(json.dumps(report))
    calls = []
    for slot in ["dino_pair", "morphem", "openphenom", "subcell"]:
        worker = slot + "-0"
        calls.extend(
            {
                "phase": phase,
                "slot": slot,
                "worker": worker,
                "ordinals": [index],
                "sites": 1,
                "seconds": 0.05,
            }
            for index in [0, 1]
        )
        write_jsonl(
            root / f"{worker}-metrics.jsonl",
            [
                {
                    "phase": phase,
                    "sites": 1,
                    "processor_seconds": 0.04,
                    "peak_gpu_allocated_bytes": 1024**3,
                }
                for _ in [0, 1]
            ],
        )
    write_jsonl(root / "calls.jsonl", calls)
    stamp = datetime.fromtimestamp(1700000000.5, timezone.utc).strftime(
        "%Y/%m/%d %H:%M:%S.%f"
    )[:-3]
    (root / "gpu.csv").write_text(
        "".join(
            f"{stamp}, {index}, GPU-{index}, 80, 25, 2048, 200, 1500\n"
            for index in range(4)
        )
    )
    return root


def test_accounting_and_explicit_gpu_timezone(run):
    result = module.summarize(run)
    row = result["summary"][0]
    assert row["validation_passed"] and row["validation_exact"]
    assert row["median_sites_per_second_all_five_models"] == 2.0
    assert row["median_nonprocessor_request_ms_mean"] == pytest.approx(10)
    assert row["gpu_mean_utilization_percent"] == {str(index): 80 for index in range(4)}
    assert result["accounting"] == [
        {
            "phase": "shm_session_cached/repeat0",
            "requests": 8,
            "model_outputs": 10,
            "passed": True,
        }
    ]


@pytest.mark.parametrize("change", ["missing", "duplicate", "entire-slot"])
def test_missing_duplicate_or_missing_slot_cannot_pass(run, change):
    calls = module.jsonl(run / "calls.jsonl")
    if change == "missing":
        calls.pop()
    elif change == "duplicate":
        calls.append(calls[0])
    else:
        calls = [call for call in calls if call["slot"] != "subcell"]
    write_jsonl(run / "calls.jsonl", calls)
    with pytest.raises(RuntimeError, match="missing|duplicate"):
        module.summarize(run)


def test_unobserved_server_retry_is_rejected(run):
    path = run / "dino_pair-0-metrics.jsonl"
    server = module.jsonl(path)
    write_jsonl(path, server + [server[0]])
    with pytest.raises(RuntimeError, match="request count mismatch"):
        module.summarize(run)


def test_incomplete_run_has_no_success_summary(run):
    path = run / "report.json"
    report = json.loads(path.read_text())
    report["status"] = "failed"
    report["error"] = "injected"
    path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="incomplete run"):
        module.summarize(run)
