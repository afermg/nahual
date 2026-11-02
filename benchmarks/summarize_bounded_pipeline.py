"""Summarize successful bounded-pipeline runs and matched GPU telemetry windows."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


def summarize(folder, utc_offset_seconds):
    report_path = folder / "report.json"
    report = json.loads(report_path.read_text())
    if report["status"] != "ok":
        raise ValueError("cannot summarize a failed/incomplete run as successful")
    validation = report["historical_validation"] + report["stream_validation"]
    if (
        not report["historical_validation"]
        or not report["stream_validation"]
        or not all(row["passed"] for row in validation)
    ):
        raise ValueError("missing or failed numerical parity checks")
    samples = []
    with (folder / "gpu.csv").open() as stream:
        for row in csv.reader(stream):
            timestamp = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            # GPU timestamp is local wall time, not an epoch or necessarily UTC.
            epoch = (
                timestamp.replace(tzinfo=timezone.utc).timestamp() - utc_offset_seconds
            )
            samples.append((epoch, int(row[1]), float(row[3]), float(row[4])))
    kinds = {}
    for kind in ("no_prefetch", "bounded"):
        phases = [phase for phase in report["phases"] if phase["kind"] == kind]
        rates = [phase["sites_per_second"] for phase in phases]
        gpus = {}
        for gpu in report["config"]["gpu_indices"]:
            matched = [
                sample
                for sample in samples
                if sample[1] == gpu
                and any(
                    phase["start_unix"] <= sample[0] <= phase["end_unix"]
                    for phase in phases
                )
            ]
            if not matched:
                raise ValueError("no telemetry in measured phase windows")
            gpus[str(gpu)] = dict(
                samples=len(matched),
                mean_utilization_percent=statistics.mean(row[2] for row in matched),
                peak_device_memory_mib=max(row[3] for row in matched),
            )
        kinds[kind] = dict(
            repeats=len(phases),
            sites_per_second=rates,
            median_sites_per_second=statistics.median(rates),
            gpus=gpus,
        )
    return dict(
        artifact_root=str(folder),
        report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        gpu_timestamp_utc_offset_seconds=utc_offset_seconds,
        config=report["config"],
        kinds=kinds,
        ratio_of_medians=kinds["bounded"]["median_sites_per_second"]
        / kinds["no_prefetch"]["median_sites_per_second"],
        validation=dict(
            historical=len(report["historical_validation"]),
            streamed_sites=len(report["stream_validation"]),
            passed=len(validation),
            exact=sum(row["exact"] for row in validation),
            maximum_absolute_difference=max(row["max_abs"] for row in validation),
        ),
        phases=report["phases"],
        recipe_files_verified=report["recipe_files_verified"],
        workers=report["workers"],
        manifest_sha256=report["manifest_sha256"],
        reference_sha256=report["reference_sha256"],
        files_sha256=report["files_sha256"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--gpu-utc-offset-seconds", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [summarize(folder, args.gpu_utc_offset_seconds) for folder in args.runs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(dict(runs=runs), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
