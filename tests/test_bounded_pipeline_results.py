from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import statistics

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "bounded_summary", ROOT / "benchmarks/summarize_bounded_pipeline.py"
)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def test_recorded_results_accounting():
    report = json.loads(
        (ROOT / "benchmarks/results/bounded_pipeline_20261102/summary.json").read_text()
    )
    assert len(report["runs"]) == 2
    for run in report["runs"]:
        assert run["validation"] == dict(
            historical=18,
            streamed_sites=64,
            passed=82,
            exact=82,
            maximum_absolute_difference=0.0,
        )
        assert run["recipe_files_verified"] == 87
        assert run["config"]["gpu_indices"] == [1, 2, 3]
        assert len(run["workers"]) == 3
        assert all(len(worker["loaded_state_sha256"]) == 5 for worker in run["workers"])
        assert len(run["phases"]) == 10
        for phase in run["phases"]:
            assert phase["sites"] == 64
            assert phase["sites_per_second"] == pytest.approx(64 / phase["seconds"])
            if phase["kind"] == "bounded":
                stats = phase["stats"]
                assert stats["peak_in_flight"] <= 12
                assert stats["peak_reserved_input_bytes"] <= 256 * 1024**2
                assert (
                    stats["yielded"]
                    == stats["processed"]
                    == stats["loaded"]
                    == stats["admitted"]
                    == 64
                )
                assert stats["in_flight"] == stats["reserved_input_bytes"] == 0
        for kind, evidence in run["kinds"].items():
            rates = [
                phase["sites_per_second"]
                for phase in run["phases"]
                if phase["kind"] == kind
            ]
            assert evidence["median_sites_per_second"] == statistics.median(rates)
            assert all(gpu["samples"] > 0 for gpu in evidence["gpus"].values())


def test_summary_timezone_and_validation_gate(tmp_path):
    epoch = datetime(2026, 11, 2, 20, 0, 0, tzinfo=timezone.utc).timestamp()
    validation = dict(passed=True, exact=True, max_abs=0.0)
    report = dict(
        status="ok",
        config={"gpu_indices": [1]},
        historical_validation=[validation],
        stream_validation=[validation],
        phases=[
            dict(kind=kind, sites_per_second=rate, start_unix=epoch, end_unix=epoch + 1)
            for kind, rate in [("no_prefetch", 10), ("bounded", 16)]
        ],
        recipe_files_verified=1,
        workers=[],
        manifest_sha256="manifest",
        reference_sha256="reference",
        files_sha256={},
    )
    (tmp_path / "report.json").write_text(json.dumps(report))
    (tmp_path / "gpu.csv").write_text(
        "2026/11/02 15:00:00.500, 1, GPU-id, 80, 4275, 150\n"
    )
    result = summary.summarize(tmp_path, -18000)
    assert result["ratio_of_medians"] == 1.6
    assert result["kinds"]["bounded"]["gpus"]["1"]["mean_utilization_percent"] == 80
    with pytest.raises(ValueError, match="no telemetry"):
        summary.summarize(tmp_path, 0)
    report["status"] = "failed"
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="failed/incomplete"):
        summary.summarize(tmp_path, -18000)
    report["status"] = "ok"
    report["stream_validation"][0]["passed"] = False
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="parity"):
        summary.summarize(tmp_path, -18000)
    report["stream_validation"][0]["passed"] = True
    report["historical_validation"] = []
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="parity"):
        summary.summarize(tmp_path, -18000)


def test_historical_reference_coverage(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks"))
    runner_spec = importlib.util.spec_from_file_location(
        "bounded_runner_test", ROOT / "benchmarks/run_bounded_pipeline.py"
    )
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    records = [dict(ordinal=i, shape=[5, 224 * (i + 1), 224]) for i in range(2)]
    complete = {
        f"{i:03d}__{slot}": None
        for i in range(2)
        for slot in ("dino_pair", "morphem", "openphenom", "subcell")
    }
    assert runner.historical_canaries(records, complete) == records
    for reference in [
        {},
        {key: value for key, value in complete.items() if key.startswith("000")},
        {key: value for key, value in complete.items() if key != "001__morphem"},
    ]:
        with pytest.raises(ValueError, match="historical"):
            runner.historical_canaries(records, reference)
