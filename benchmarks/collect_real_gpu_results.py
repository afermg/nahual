"""Audit local run receipts and export compact results (never images or weights)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    full = json.loads((args.evidence / "summary.json").read_text())
    inventory = (
        (args.evidence / "frozen-recipe-inventory-check.log").read_text().splitlines()
    )
    if not inventory or not all(line.endswith(": OK") for line in inventory):
        raise RuntimeError("frozen recipe inventory did not verify")
    runs, completed, failed, repeats, outputs, canaries = [], 0, 0, 0, 0, 0
    snapshots = [
        args.evidence / name
        for name in ("primary-harness", "tuning-harness", "zmq-harness")
    ]
    for result in full:
        root = Path(result["root"])
        report = json.loads((root / "report.json").read_text())
        code = json.loads((root / "code.json").read_text())
        matching = [
            snapshot.name
            for snapshot in snapshots
            if all(
                (snapshot / name).exists() and sha(snapshot / name) == digest
                for name, digest in code["harness_files"].items()
            )
        ]
        if not matching:
            raise RuntimeError(f"as-run harness snapshot is missing: {root}")
        if report["hardware_after"]["apps"].strip():
            raise RuntimeError(f"GPU processes remained after a run: {root}")
        if report["status"] == "complete":
            if not report["validation"] or not all(
                item["passed"] and item["exact"] for item in report["validation"]
            ):
                raise RuntimeError(f"successful-case parity not exact: {root}")
            if not all(row["passed"] for row in result["accounting"]):
                raise RuntimeError(f"accounting failed: {root}")
            completed += 1
            repeats += len(report["phases"])
            outputs += sum(phase["model_outputs"] for phase in report["phases"])
            canaries += len(report["validation"])
            runs.append(
                {
                    "name": root.name,
                    "status": "complete",
                    "summary": result["summary"],
                    "report_sha256": sha(root / "report.json"),
                    "code": code,
                    "as_run_snapshots": matching,
                    "parallel_model_startup_seconds": report[
                        "parallel_model_startup_seconds"
                    ],
                }
            )
        elif report["status"] == "failed":
            failed += 1
            runs.append(
                {
                    "name": root.name,
                    "status": "failed",
                    "error": report["error"],
                    "timed_repeats": len(report["phases"]),
                    "report_sha256": sha(root / "report.json"),
                    "failed_validation": [
                        entry for entry in report["validation"] if not entry["passed"]
                    ],
                    "as_run_snapshots": matching,
                    "code": code,
                }
            )
        else:
            raise RuntimeError(f"unfinished run: {root}")
    cohorts = []
    for name in ("inputs", "inputs-mq-v2"):
        manifest = args.evidence / name / "manifest.json"
        cohort = json.loads(manifest.read_text())
        for record in cohort["records"]:
            if sha(record["cache_path"]) != record["cache_sha256"]:
                raise RuntimeError("input cache changed after benchmarking")
            for item in record["files"]:
                if sha(item["path"]) != item["sha256"]:
                    raise RuntimeError("source input changed after benchmarking")
        cohorts.append(
            {
                "name": name,
                "manifest_sha256": sha(manifest),
                "seed": cohort["seed"],
                "scope": cohort["scope"],
                "rejected_candidates": cohort["rejected_candidates"],
                "records": [
                    {
                        key: record[key]
                        for key in (
                            "ordinal",
                            "dataset",
                            "site",
                            "source",
                            "plate",
                            "channels",
                            "shape",
                            "dtype",
                            "nbytes",
                            "array_sha256",
                            "cache_sha256",
                            "files",
                        )
                    }
                    for record in cohort["records"]
                ],
            }
        )
    first = json.loads((args.evidence / "master-primary" / "report.json").read_text())

    def get_commit(ref):
        return subprocess.check_output(["git", "rev-parse", ref], text=True).strip()

    final_apps = (
        (args.evidence / "gpu-processes-final.csv").read_text().strip().splitlines()
    )
    if len(final_apps) != 1:
        raise RuntimeError("GPU inventory was not clean after the experiment")
    compact = {
        "date": "2026-11-02",
        "host": "oppy",
        "baseline_commit": get_commit("56aa3fe"),
        "optimized_commit": get_commit("7609bff"),
        "source": str(args.evidence),
        "hardware": (args.evidence / "gpu-hardware-final.csv").read_text(),
        "software": first["hardware_after"]["packages"],
        "pyzmq_trial": {
            "pyzmq": "27.1.0",
            "libzmq": "4.3.5",
            "production_backend_changed": False,
        },
        "loaded_model_state_sha256": first["model_state_expectations"],
        "recipe_inventory_files_verified": len(inventory),
        "clock": json.loads((args.evidence / "clock.json").read_text()),
        "completed_configurations": completed,
        "rejected_configurations": failed,
        "measured_repeats": repeats,
        "measured_model_site_outputs": outputs,
        "successful_canary_checks": canaries,
        "all_successful_canaries_exact": True,
        "all_source_inputs_and_caches_rehashed_after": True,
        "gpus_clean_after": True,
        "production_mutated": False,
        "manuscript_updated": False,
        "runs": runs,
        "scope": "Local decoded-input and decoding-inclusive GPU/IPC measurements; exclude remote downloads, profile-table writes, signing and production publication.",
        "warnings": full[0]["warnings"],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(compact, indent=2) + "\n")
    # One record per line keeps the frozen identity/hash ledger reviewable
    # without thousands of indentation-only lines. This is ordinary JSON.
    cohort_documents = []
    for cohort in cohorts:
        metadata = {key: value for key, value in cohort.items() if key != "records"}
        header = json.dumps(metadata, indent=2).rstrip()[:-1].rstrip()
        rows = ",\n".join(
            "    " + json.dumps(record, sort_keys=True) for record in cohort["records"]
        )
        cohort_documents.append(header + ',\n  "records": [\n' + rows + "\n  ]\n}")
    (args.output / "cohorts.json").write_text(
        "[\n" + ",\n".join(cohort_documents) + "\n]\n"
    )
    print(
        f"Audited {completed} complete + {failed} rejected configurations; {repeats} repeats, {outputs} model/site outputs, {canaries} exact canary checks"
    )


if __name__ == "__main__":
    main()
