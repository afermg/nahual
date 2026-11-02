from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

from benchmarks import summarize_bounded_pipeline as summary
from benchmarks.run_bounded_pipeline import historical_canaries


@pytest.fixture
def run(tmp_path):
    """Small synthetic run, independent of scientific data or a particular host."""
    epoch = datetime(2026, 1, 1, 20, 0, 0, tzinfo=timezone.utc).timestamp()
    phases = []
    telemetry = []
    for kind, rates in [("no_prefetch", [10, 30, 20]), ("bounded", [20, 60, 40])]:
        for rate in rates:
            start = epoch + 2 * len(phases)
            phases.append(
                dict(
                    kind=kind,
                    sites_per_second=rate,
                    start_unix=start,
                    end_unix=start + 1,
                )
            )
            stamp = datetime.fromtimestamp(start + 0.5 - 18000, timezone.utc).strftime(
                "%Y/%m/%d %H:%M:%S.%f"
            )
            for gpu in [1, 2]:
                telemetry.append(f"{stamp}, {gpu}, GPU-{gpu}, {40 + gpu}, 1024, 100\n")
    report = dict(
        status="ok",
        config={"gpu_indices": [1, 2]},
        historical_validation=[dict(passed=True, exact=True, max_abs=0.0)],
        stream_validation=[dict(passed=True, exact=False, max_abs=1e-6)],
        phases=phases,
        recipe_files_verified=1,
        workers=[],
        manifest_sha256="manifest",
        reference_sha256="reference",
        files_sha256={},
    )
    (tmp_path / "report.json").write_text(json.dumps(report))
    (tmp_path / "gpu.csv").write_text("".join(telemetry))
    return tmp_path


def test_summary_medians_counts_and_gpu_windows(run):
    result = summary.summarize(run, -18000)
    assert result["ratio_of_medians"] == 2.0
    assert result["kinds"]["no_prefetch"]["median_sites_per_second"] == 20
    assert result["kinds"]["bounded"]["median_sites_per_second"] == 40
    assert result["validation"] == dict(
        historical=1,
        streamed_sites=1,
        passed=2,
        exact=1,
        maximum_absolute_difference=1e-6,
    )
    for kind in result["kinds"].values():
        assert kind["repeats"] == 3
        assert kind["gpus"] == {
            str(gpu): dict(
                samples=3,
                mean_utilization_percent=40 + gpu,
                peak_device_memory_mib=1024,
            )
            for gpu in [1, 2]
        }
    assert len(result["report_sha256"]) == 64


def test_summary_requires_explicit_matching_timezone(run):
    with pytest.raises(ValueError, match="no telemetry"):
        summary.summarize(run, 0)


@pytest.mark.parametrize(
    "failure", ["failed", "parity", "missing_history", "missing_stream"]
)
def test_summary_rejects_incomplete_evidence(run, failure):
    path = run / "report.json"
    report = json.loads(path.read_text())
    if failure == "failed":
        report["status"] = "failed"
    elif failure == "parity":
        report["stream_validation"][0]["passed"] = False
    elif failure == "missing_history":
        report["historical_validation"] = []
    else:
        report["stream_validation"] = []
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="failed/incomplete|parity"):
        summary.summarize(run, -18000)


def test_historical_reference_coverage():
    records = [dict(ordinal=i, shape=[5, 224 * (i + 1), 224]) for i in range(2)]
    complete = {
        f"{i:03d}__{slot}": None
        for i in range(2)
        for slot in ("dino_pair", "morphem", "openphenom", "subcell")
    }
    assert historical_canaries(records, complete) == records
    for reference in [
        {},
        {key: value for key, value in complete.items() if key.startswith("000")},
        {key: value for key, value in complete.items() if key != "001__morphem"},
    ]:
        with pytest.raises(ValueError, match="historical"):
            historical_canaries(records, reference)
