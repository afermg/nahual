"""Scratch-only JUMP-lite streaming runner and matched no-prefetch comparison.

Uses existing frozen FP32/per-site worker code. Requires its scientific runtime,
read-only recipe, frozen input manifest and historical validation outputs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import queue
import random
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

from nahual.pipeline import process_map
from nahual.process import Client
from real_gpu_contract import expected_output_shape, full_stack_reference


def historical_canaries(records, reference):
    canaries = [
        record
        for record in records
        if f"{record['ordinal']:03d}__dino_pair" in reference
    ]
    shapes = {tuple(record["shape"]) for record in records}
    if not canaries or {tuple(record["shape"]) for record in canaries} != shapes:
        raise ValueError("historical references must cover every input geometry")
    for record in canaries:
        for slot in ("dino_pair", "morphem", "openphenom", "subcell"):
            if f"{record['ordinal']:03d}__{slot}" not in reference:
                raise ValueError("historical reference is missing a model slot")
    return canaries


def decode_record(record, kind):
    if kind == "cache":
        array = np.load(record["cache_path"], allow_pickle=False)
    elif kind == "tiff":
        import tifffile

        array = np.stack([tifffile.imread(item["path"]) for item in record["files"]])
    else:
        import zarr

        array = zarr.open_array(record["zarr_path"], mode="r")[:]
    array = np.ascontiguousarray(array)
    if (
        list(array.shape) != record["shape"]
        or str(array.dtype) != record["dtype"]
        or array.nbytes != record["nbytes"]
        or hashlib.sha256(memoryview(array)).hexdigest() != record["array_sha256"]
    ):
        raise RuntimeError(f"decoded identity changed: {record['site']}")
    return array


def check_output(record, output):
    if (
        output.dtype != np.float32
        or output.shape != expected_output_shape("full_stack", [record["shape"]])
        or not np.isfinite(output).all()
    ):
        raise RuntimeError(f"invalid output: {record['site']}")


def persist(folder, record, output):
    check_output(record, output)
    path = folder / f"{record['ordinal']:03d}.npy"
    with path.open("xb") as stream:
        np.save(stream, output, allow_pickle=False)
    return dict(
        ordinal=record["ordinal"],
        site=record["site"],
        path=path.name,
        array_sha256=hashlib.sha256(memoryview(output)).hexdigest(),
        shape=list(output.shape),
        dtype=str(output.dtype),
    )


def compare(record, result, target, worker):
    check_output(record, result)
    passed = result.shape == target.shape and np.allclose(
        result, target, rtol=1e-5, atol=5e-4
    )
    evidence = dict(
        ordinal=record["ordinal"],
        worker=worker,
        exact=bool(np.array_equal(result, target)),
        passed=bool(passed),
        max_abs=float(np.max(np.abs(result - target))),
    )
    if not passed:
        raise RuntimeError(f"numerical parity failure: {evidence}")
    return evidence


def run_pass(kind, records, addresses, output, loader, options, phase):
    output.mkdir()
    receipt = []
    setup = dict(phase=phase, validation_seed=None, batch_inference=False)
    with ExitStack() as stack:
        if kind == "bounded":
            results = stack.enter_context(
                process_map(
                    records,
                    addresses,
                    loader=loader,
                    size_of=lambda record: record["nbytes"],
                    setup_parameters=setup,
                    **options,
                )
            )
        else:
            clients = [
                stack.enter_context(Client(address, timeout_ms=120_000))
                for address in addresses
            ]
            for client in clients:
                client.setup(setup)
        started_unix, started = time.time(), time.perf_counter()
        if kind == "bounded":
            for result in results:
                item = persist(output, result.item, result.output)
                item.update(
                    worker=result.address,
                    load_seconds=result.load_seconds,
                    process_seconds=result.process_seconds,
                )
                receipt.append(item)
            stats = results.stats
        else:
            # Equal full-stack replicas, dynamic site assignment, same pooled
            # clients and persistence; decode/infer/write are serial per worker.
            pending = queue.Queue()
            for record in records:
                pending.put(record)

            def work(client, address):
                completed = []
                while True:
                    try:
                        record = pending.get_nowait()
                    except queue.Empty:
                        return completed
                    start = time.perf_counter()
                    array = loader(record)
                    loaded = time.perf_counter()
                    result = client.process(array, shared_memory=True)
                    inferred = time.perf_counter()
                    del array
                    item = persist(output, record, result)
                    del result
                    item.update(
                        worker=address,
                        load_seconds=loaded - start,
                        process_seconds=inferred - loaded,
                    )
                    completed.append(item)

            with ThreadPoolExecutor(max_workers=len(addresses)) as executor:
                futures = [
                    executor.submit(work, client, address)
                    for client, address in zip(clients, addresses)
                ]
                for future in futures:
                    receipt.extend(future.result())
            stats = None
        elapsed, ended_unix = time.perf_counter() - started, time.time()
    expected = sorted(record["ordinal"] for record in records)
    if sorted(item["ordinal"] for item in receipt) != expected:
        raise RuntimeError("missing or duplicated site output")
    (output / "receipt.json").write_text(
        json.dumps(sorted(receipt, key=lambda item: item["ordinal"]), indent=2) + "\n"
    )
    return dict(
        kind=kind,
        phase=phase,
        sites=len(receipt),
        seconds=elapsed,
        sites_per_second=len(receipt) / elapsed,
        start_unix=started_unix,
        end_unix=ended_unix,
        stats=stats,
        output_bytes=sum(path.stat().st_size for path in output.glob("*.npy")),
    )


def main():
    from run_real_gpu import inventory, sha

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--recipe-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-indices", type=int, nargs="+", required=True)
    parser.add_argument("--decode", choices=["tiff", "zarr_mq", "cache"], required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--load-workers", type=int, default=8)
    parser.add_argument("--max-in-flight", type=int, default=12)
    parser.add_argument("--max-input-mib", type=int, default=256)
    args = parser.parse_args()
    if min(args.repeats, args.load_workers, args.max_in_flight, args.max_input_mib) < 1:
        parser.error("counts and byte budget must be positive")
    if not args.output.resolve().is_relative_to(Path("/work/scratch/amunoz")):
        parser.error(
            "this site-specific runner only writes fresh user scratch directories"
        )
    uuids = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).splitlines()
    if len(set(args.gpu_indices)) != len(args.gpu_indices) or any(
        index < 0 or index >= len(uuids) for index in args.gpu_indices
    ):
        parser.error("GPU indices must be unique and available")
    selected = [uuids[index] for index in args.gpu_indices]
    before = inventory()
    if any(uuid in before["apps"] for uuid in selected):
        raise RuntimeError(
            "a selected GPU is occupied; existing jobs are never stopped"
        )
    manifest = json.loads(args.manifest.read_text())
    records = manifest["records"]
    if not records or len({r["ordinal"] for r in records}) != len(records):
        raise ValueError("nonempty unique-ordinal records required")
    if len({r["site"] for r in records}) != len(records):
        raise ValueError("duplicate site identities")
    args.output.mkdir(parents=True, exist_ok=False)
    # Check all frozen recipe files without importing or modifying that recipe.
    verified_recipe = 0
    for line in (
        (args.recipe_root / "DEPLOYMENT-INVENTORY.sha256").read_text().splitlines()
    ):
        digest, relative = line.split(maxsplit=1)
        if sha(args.recipe_root / relative.lstrip("*")) != digest:
            raise RuntimeError(f"frozen recipe changed: {relative}")
        verified_recipe += 1
    if args.decode == "zarr_mq":
        import numcodecs
        from imagecodecs.numcodecs import Jpegxl

        numcodecs.register_codec(Jpegxl)
    for record in records:
        if sha(record["cache_path"]) != record["cache_sha256"]:
            raise RuntimeError("frozen input cache changed")
    expected = json.loads(
        (args.recipe_root / "MODEL-STATE-EXPECTATIONS.json").read_text()
    )
    states = {
        model: digest for slot in expected.values() for model, digest in slot.items()
    }
    options = dict(
        max_in_flight=args.max_in_flight,
        max_input_bytes=args.max_input_mib * 1024**2,
        load_workers=args.load_workers,
        timeout_ms=120_000,
    )
    report = dict(
        status="running",
        config={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        selected_gpus=selected,
        hardware_before=before,
        recipe_files_verified=verified_recipe,
        phases=[],
        historical_validation=[],
        stream_validation=[],
        workers=[],
        manifest_sha256=sha(args.manifest),
        reference_sha256=sha(args.reference),
        files_sha256={
            str(path): sha(path)
            for path in [
                Path(__file__),
                Path(__file__).with_name("real_gpu_worker.py"),
                Path(__file__).with_name("real_gpu_contract.py"),
                *Path(__file__)
                .resolve()
                .parents[1]
                .joinpath("src/nahual")
                .glob("*.py"),
            ]
        },
    )
    processes = []
    try:
        with ExitStack() as stack:
            ipcdir = stack.enter_context(
                tempfile.TemporaryDirectory(prefix="nahual-stream-")
            )
            addresses = []
            readies = []
            for index, uuid in zip(args.gpu_indices, selected):
                address = f"ipc://{ipcdir}/{index}.ipc"
                ready = args.output / f"gpu-{index}-ready.json"
                log = stack.enter_context((args.output / f"gpu-{index}.log").open("x"))
                environment = {**os.environ, "CUDA_VISIBLE_DEVICES": uuid}
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-B",
                        str(Path(__file__).with_name("real_gpu_worker.py")),
                        "--slot",
                        "full_stack",
                        "--address",
                        address,
                        "--ready",
                        str(ready),
                        "--metrics",
                        str(args.output / f"gpu-{index}-metrics.jsonl"),
                    ],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                processes.append(process)
                addresses.append(address)
                readies.append(ready)
            deadline = time.monotonic() + 300
            while not all(path.exists() for path in readies):
                if any(process.poll() is not None for process in processes):
                    raise RuntimeError("worker failed; inspect logs")
                if time.monotonic() > deadline:
                    raise TimeoutError("worker startup deadline")
                time.sleep(0.25)
            report["workers"] = [json.loads(path.read_text()) for path in readies]
            if any(
                worker["loaded_state_sha256"] != states for worker in report["workers"]
            ):
                raise RuntimeError("model state hash mismatch")
            print("models ready; validating all streaming outputs", flush=True)
            monitor_log = stack.enter_context((args.output / "gpu.csv").open("x"))
            monitor = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                    "-lms",
                    "200",
                ],
                stdout=monitor_log,
                stderr=subprocess.STDOUT,
            )
            processes.append(monitor)
            historical = stack.enter_context(
                np.load(args.reference, allow_pickle=False)
            )
            # Never silently skip a missing historical reference cohort.
            canaries = historical_canaries(records, historical)
            for address in addresses:
                with Client(address, timeout_ms=120_000) as client:
                    client.setup(
                        dict(
                            phase="historical-validation",
                            validation_seed=17012026,
                            batch_inference=False,
                        )
                    )
                    for record in canaries:
                        output = client.process(
                            decode_record(record, "cache"), shared_memory=True
                        )
                        target = full_stack_reference(historical, record["ordinal"])
                        report["historical_validation"].append(
                            compare(record, output, target, address)
                        )
            # Seeded golden output for every site; no inference batching or
            # tolerance changes. Timed phases below reset seed policy to None.
            golden_dir = args.output / "golden"
            golden_dir.mkdir()
            with Client(addresses[0], timeout_ms=120_000) as client:
                client.setup(
                    dict(
                        phase="golden", validation_seed=17012026, batch_inference=False
                    )
                )
                for record in records:
                    persist(
                        golden_dir,
                        record,
                        client.process(
                            decode_record(record, "cache"), shared_memory=True
                        ),
                    )

            def loader(record):
                return decode_record(record, args.decode)

            with process_map(
                records,
                addresses,
                loader=loader,
                size_of=lambda record: record["nbytes"],
                setup_parameters=dict(
                    phase="stream-validation",
                    validation_seed=17012026,
                    batch_inference=False,
                ),
                **options,
            ) as results:
                for result in results:
                    target = np.load(
                        golden_dir / f"{result.item['ordinal']:03d}.npy",
                        allow_pickle=False,
                    )
                    report["stream_validation"].append(
                        compare(result.item, result.output, target, result.address)
                    )
            if len(report["stream_validation"]) != len(records):
                raise RuntimeError("incomplete validation")
            for repeat in range(args.repeats):
                order = records.copy()
                random.Random(20261102 + repeat).shuffle(order)
                kinds = (
                    ["no_prefetch", "bounded"]
                    if repeat % 2 == 0
                    else ["bounded", "no_prefetch"]
                )
                for kind in kinds:
                    phase = f"{kind}-{repeat}"
                    evidence = run_pass(
                        kind,
                        order,
                        addresses,
                        args.output / phase,
                        loader,
                        options,
                        phase,
                    )
                    report["phases"].append(evidence)
                    print(
                        phase,
                        round(evidence["sites_per_second"], 3),
                        "sites/s",
                        flush=True,
                    )
            report["medians"] = {
                kind: statistics.median(
                    phase["sites_per_second"]
                    for phase in report["phases"]
                    if phase["kind"] == kind
                )
                for kind in ["no_prefetch", "bounded"]
            }
            report["status"] = "ok"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
        for process in reversed(processes):
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        report["hardware_after"] = inventory()
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
