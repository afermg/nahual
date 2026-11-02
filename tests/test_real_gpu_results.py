from __future__ import annotations

import json
from collections import Counter
from importlib.resources import files

RESULTS = files("benchmarks").joinpath("results", "jump_lite_20261102")


def test_committed_results_close_accounting_and_keep_rejections():
    result = json.loads((RESULTS / "summary.json").read_text())
    complete = [run for run in result["runs"] if run["status"] == "complete"]
    rejected = [run for run in result["runs"] if run["status"] == "failed"]
    assert len(complete) == result["completed_configurations"] == 14
    assert len(rejected) == result["rejected_configurations"] == 2
    rows = [row for run in complete for row in run["summary"]]
    assert sum(row["repeats"] for row in rows) == result["measured_repeats"] == 110
    assert (
        sum(row["sites_per_repeat"] * row["repeats"] * 5 for row in rows)
        == result["measured_model_site_outputs"]
        == 35200
    )
    assert all(row["validation_passed"] and row["validation_exact"] for row in rows)
    assert result["all_successful_canaries_exact"]
    assert (
        result["gpus_clean_after"]
        and result["all_source_inputs_and_caches_rehashed_after"]
    )
    assert not result["production_mutated"] and not result["manuscript_updated"]
    for run in rejected:
        assert run["timed_repeats"] == 0
        assert "batch parity failure" in run["error"]
        assert run["failed_validation"][0]["max_abs"] > 5e-4
        assert "summary" not in run
    assert (
        sum(row["validation_cases"] for row in rows)
        == result["successful_canary_checks"]
        == 624
    )


def test_raw_and_mq_cohorts_are_matched_but_not_conflated():
    cohorts = json.loads((RESULTS / "cohorts.json").read_text())
    raw, mq = [cohort["records"] for cohort in cohorts]
    assert len(raw) == len(mq) == 64
    assert len({row["site"] for row in raw}) == 64
    assert len({row["source"] for row in raw}) == 8
    assert len({row["plate"] for row in raw}) == 33
    assert len({tuple(row["shape"]) for row in raw}) == 6
    assert Counter(row["dataset"] for row in raw) == {
        "jump_lite_raw": 32,
        "target2_raw": 32,
    }
    for original, compressed in zip(raw, mq):
        assert original["ordinal"] == compressed["ordinal"]
        assert original["site"] == compressed["site"]
        assert original["shape"] == compressed["shape"]
        assert (
            original["channels"]
            == compressed["channels"]
            == ["AGP", "DNA", "ER", "Mito", "RNA"]
        )
        assert original["dtype"] == compressed["dtype"] == "uint16"
        assert original["array_sha256"] != compressed["array_sha256"]
        for record in [original, compressed]:
            assert all(len(item["sha256"]) == 64 for item in record["files"])
