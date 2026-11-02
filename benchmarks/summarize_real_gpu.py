"""Validate call accounting and summarize real-data transport/GPU experiments."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


def jsonl(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def summarize(root):
    report = json.loads((root / "report.json").read_text())
    if report["status"] != "complete":
        raise RuntimeError(f"incomplete run: {root}: {report.get('error')}")
    config = report["config"]
    manifest = json.loads(Path(config["manifest"]).read_text())
    records = (
        manifest["records"][: config["limit"]]
        if config["limit"]
        else manifest["records"]
    )
    expected = Counter(record["ordinal"] for record in records)
    calls = jsonl(root / "calls.jsonl")
    server_calls = {
        path.name.removesuffix("-metrics.jsonl"): jsonl(path)
        for path in root.glob("*-metrics.jsonl")
    }
    clock = json.loads((root.parent / "clock.json").read_text())
    gpu_timezone = timezone(
        timedelta(seconds=clock["gpu_timestamp_utc_offset_seconds"])
    )
    gpu = []
    with (root / "gpu.csv").open() as stream:
        for row in csv.reader(stream):
            if len(row) != 8:
                continue
            (
                stamp,
                index,
                uuid,
                utilization,
                memory_utilization,
                memory,
                power,
                clock,
            ) = [value.strip() for value in row]
            gpu.append(
                {
                    "unix": datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S.%f")
                    .replace(tzinfo=gpu_timezone)
                    .timestamp(),
                    "index": int(index),
                    "uuid": uuid,
                    "utilization": float(utilization),
                    "memory_mib": float(memory),
                    "power_w": float(power),
                    "sm_clock_mhz": float(clock),
                }
            )
    modes = defaultdict(list)
    accounting = []
    for phase in report["phases"]:
        phase_calls = [call for call in calls if call["phase"] == phase["phase"]]
        expected_slots = (
            {"full_stack"}
            if config.get("schedule") == "data_parallel"
            else {"dino_pair", "morphem", "openphenom", "subcell"}
        )
        if {call["slot"] for call in phase_calls} != expected_slots:
            raise RuntimeError(f"missing/foreign model slot: {root}: {phase['phase']}")
        for slot in expected_slots:
            observed = Counter(
                index
                for call in phase_calls
                if call["slot"] == slot
                for index in call["ordinals"]
            )
            if observed != expected:
                raise RuntimeError(
                    f"missing/duplicate sites: {root}: {phase['phase']}: {slot}"
                )
        phase_servers = []
        for worker in {call["worker"] for call in phase_calls}:
            client = [call for call in phase_calls if call["worker"] == worker]
            server = [
                call for call in server_calls[worker] if call["phase"] == phase["phase"]
            ]
            if len(client) != len(server) or sum(
                call["sites"] for call in client
            ) != sum(call["sites"] for call in server):
                raise RuntimeError(
                    f"server/client request count mismatch: {root}: {worker}"
                )
            phase_servers.extend(server)
        accounting.append(
            {
                "phase": phase["phase"],
                "requests": len(phase_calls),
                "model_outputs": phase["model_outputs"],
                "passed": True,
            }
        )
        samples = [
            item
            for item in gpu
            if phase["start_unix"] <= item["unix"] <= phase["end_unix"]
        ]
        enriched = {
            **phase,
            "processor_seconds_sum": sum(
                call["processor_seconds"] for call in phase_servers
            ),
            "nonprocessor_request_ms_mean": (
                sum(call["seconds"] for call in phase_calls)
                - sum(call["processor_seconds"] for call in phase_servers)
            )
            / len(phase_calls)
            * 1000,
            "gpu": {
                str(index): {
                    "samples": sum(item["index"] == index for item in samples),
                    "mean_utilization_percent": statistics.mean(
                        item["utilization"]
                        for item in samples
                        if item["index"] == index
                    )
                    if any(item["index"] == index for item in samples)
                    else None,
                    "peak_memory_mib": max(
                        (
                            item["memory_mib"]
                            for item in samples
                            if item["index"] == index
                        ),
                        default=None,
                    ),
                }
                for index in range(4)
            },
            "peak_pytorch_allocated_gib": max(
                call["peak_gpu_allocated_bytes"] for call in phase_servers
            )
            / 1024**3,
        }
        modes[phase["mode"]].append(enriched)
    summary = []
    for mode, phases in modes.items():
        timings = [phase["wall_seconds"] for phase in phases]
        summary.append(
            {
                "run": root.name,
                "label": config["label"],
                "mode": mode,
                "replicas": config["replicas"],
                "schedule": config.get("schedule", "model_parallel"),
                "decode": config["decode"],
                "batch_size": config["batch_size"],
                "batch_inference": config["batch_inference"],
                "sites_per_repeat": len(records),
                "repeats": len(phases),
                "median_seconds": statistics.median(timings),
                "first_repeat_seconds": timings[0],
                "median_seconds_excluding_first": statistics.median(timings[1:])
                if len(timings) > 1
                else None,
                "median_sites_per_second_excluding_first": statistics.median(
                    phase["sites_per_second_all_five_models"] for phase in phases[1:]
                )
                if len(phases) > 1
                else None,
                "median_sites_per_second_all_five_models": statistics.median(
                    phase["sites_per_second_all_five_models"] for phase in phases
                ),
                "median_sites_per_hour_all_five_models": statistics.median(
                    phase["sites_per_second_all_five_models"] for phase in phases
                )
                * 3600,
                "minimum_seconds": min(timings),
                "maximum_seconds": max(timings),
                "median_request_ms": statistics.median(
                    phase["median_request_ms"] for phase in phases
                ),
                "median_nonprocessor_request_ms_mean": statistics.median(
                    phase["nonprocessor_request_ms_mean"] for phase in phases
                ),
                "gpu_mean_utilization_percent": {
                    str(index): statistics.mean(
                        phase["gpu"][str(index)]["mean_utilization_percent"]
                        for phase in phases
                        if phase["gpu"][str(index)]["mean_utilization_percent"]
                        is not None
                    )
                    for index in range(4)
                },
                "gpu_peak_memory_mib": {
                    str(index): max(
                        phase["gpu"][str(index)]["peak_memory_mib"] or 0
                        for phase in phases
                    )
                    for index in range(4)
                },
                "peak_pytorch_allocated_gib": max(
                    phase["peak_pytorch_allocated_gib"] for phase in phases
                ),
                "validation_cases": sum(
                    item["mode"] == mode for item in report["validation"]
                ),
                "validation_passed": all(
                    item["passed"]
                    for item in report["validation"]
                    if item["mode"] == mode
                ),
                "validation_exact": all(
                    item["exact"]
                    for item in report["validation"]
                    if item["mode"] == mode
                ),
            }
        )
    return {
        "root": str(root),
        "summary": summary,
        "accounting": accounting,
        "per_phase": [phase for phases in modes.values() for phase in phases],
        "clock": clock,
        "canonical_outputs_written": False,
        "observational_shared_host": True,
        "warnings": [
            "Time windows are sequential, not randomized backend crossovers.",
            "GPU utilization is sampled every 200 ms; CUDA event durations are not pure kernel busy time.",
            "Decode cases warm models on preloaded data; the first measured repeat also includes codec/Python cache initialization. Both all-repeat and first-excluded medians are reported.",
            "Legacy raw-report input_decode_cpu_seconds is decoder task wall time, not CPU time; see the current input_decode_task_wall_seconds_sum field.",
            "Nonprocessor latency includes serialization, copies, scheduling, IPC and response handling, not only transport.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-failed", action="store_true")
    args = parser.parse_args()
    result = []
    for root in args.runs:
        report = json.loads((root / "report.json").read_text())
        if report["status"] == "failed" and args.allow_failed:
            result.append(
                {
                    "root": str(root),
                    "status": "failed",
                    "config": report["config"],
                    "error": report.get("error"),
                    "summary": [],
                    "failed_attempt_retained": True,
                }
            )
        else:
            result.append(summarize(root))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("run\tmode\treplicas\tbatch\tdecode\tsites/s\tGPU means\tparity")
    for report in result:
        for row in report["summary"]:
            print(
                f"{row['run']}\t{row['mode']}\t{row['replicas']}\t{row['batch_size']}\t{row['decode']}\t{row['median_sites_per_second_all_five_models']:.2f}\t{row['gpu_mean_utilization_percent']}\t{row['validation_passed']}"
            )


if __name__ == "__main__":
    main()
