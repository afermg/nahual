"""Bounded, local-data, four-GPU JUMP-lite transport/throughput experiment.

Outputs are scratch-only. No production services, weights, or data are changed.
Use the frozen JUMP-lite model environment with its read-only recipe imports.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
from benchmarks.real_gpu_contract import expected_output_shape, full_stack_reference

from nahual import process as protocol

SLOTS = ("dino_pair", "morphem", "openphenom", "subcell")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory():
    def query(args):
        return subprocess.check_output(["nvidia-smi", *args], text=True, timeout=15)

    return {
        "gpus": query(
            [
                "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                "--format=csv",
            ]
        ),
        "apps": query(
            ["--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"]
        ),
        "cpu": platform.processor(),
        "lscpu": subprocess.check_output(["lscpu"], text=True, timeout=15),
        "timezone": time.tzname,
        "host": platform.node(),
        "kernel": platform.platform(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "python": sys.version,
        "threads": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "torch",
                "torchvision",
                "transformers",
                "pynng",
                "trio",
                "tifffile",
            )
        },
    }


def callables(address, mode, cache_bytes):
    if mode == "zmq_session":
        from benchmarks.real_gpu_zmq import Session

        client = Session(address)
        return client.setup, client.process, client.close
    if mode in ("wire_once", "shm_once"):

        def setup(parameters):
            return protocol.send_receive_process(
                parameters, "dict", address, timeout_ms=120_000
            )

        def process(array):
            return protocol.send_receive_process(
                array,
                "numpy",
                address,
                shared_memory=mode == "shm_once",
                timeout_ms=120_000,
            )

        return setup, process, lambda: None
    client = protocol.Client(
        address,
        timeout_ms=120_000,
        shared_memory_cache_bytes=0 if mode == "shm_session_uncached" else cache_bytes,
    )
    return (
        client.setup,
        lambda array: client.process(array, shared_memory=mode != "wire_session"),
        client.close,
    )


def batches(order, records, size):
    if size == 1:
        return [[index] for index in order]
    grouped = {}
    for index in order:
        grouped.setdefault(tuple(records[index]["shape"]), []).append(index)
    return [
        items[start : start + size]
        for items in grouped.values()
        for start in range(0, len(items), size)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[
            "wire_once",
            "shm_once",
            "wire_session",
            "shm_session_uncached",
            "shm_session_cached",
            "zmq_session",
        ],
        default=[
            "wire_once",
            "shm_once",
            "wire_session",
            "shm_session_uncached",
            "shm_session_cached",
        ],
    )
    parser.add_argument("--replicas", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--decode", choices=["preloaded", "tiff", "zarr_mq"], default="preloaded"
    )
    parser.add_argument(
        "--schedule",
        choices=["model_parallel", "data_parallel"],
        default="model_parallel",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-inference", action="store_true")
    parser.add_argument("--cache-mib", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    backend = "zmq" if args.modes == ["zmq_session"] else "nng"
    if "zmq_session" in args.modes and len(args.modes) != 1:
        parser.error("run the experimental ZMQ adapter separately from NNG modes")
    slots = ("full_stack",) if args.schedule == "data_parallel" else SLOTS
    if args.schedule == "data_parallel" and (
        args.replicas != 4 or args.batch_size != 1
    ):
        parser.error(
            "data_parallel uses four full-stack workers, one per GPU, and per-site inference"
        )
    if args.decode == "zarr_mq":
        import numcodecs
        import zarr
        from imagecodecs.numcodecs import Jpegxl

        numcodecs.register_codec(Jpegxl)
    if not args.output.resolve().is_relative_to(Path("/work/scratch/amunoz")):
        parser.error("benchmark outputs must remain in user scratch")
    if (
        args.batch_size < 1
        or args.repeats < 1
        or args.cache_mib < 0
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error(
            "positive batch size/repeats/limit and non-negative cache budget required"
        )
    before = inventory()
    if before["apps"].strip():
        raise RuntimeError("GPUs are occupied; refusing to change existing jobs")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    records = manifest["records"][: args.limit] if args.limit else manifest["records"]
    for record in records:
        if sha(record["cache_path"]) != record["cache_sha256"]:
            raise RuntimeError("frozen input cache changed")
    arrays = [np.load(record["cache_path"], allow_pickle=False) for record in records]
    for array, record in zip(arrays, records):
        if hashlib.sha256(array.tobytes()).hexdigest() != record["array_sha256"]:
            raise RuntimeError("decoded input identity changed")
        array.flags.writeable = False
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["backend"] = backend
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "hardware-before.json").write_text(
        json.dumps(before, indent=2) + "\n"
    )
    source_root = Path(protocol.__file__).resolve().parent
    (args.output / "code.json").write_text(
        json.dumps(
            {
                "nahual_root": str(source_root),
                "nahual_files": {
                    p.name: sha(p) for p in sorted(source_root.glob("*.py"))
                },
                "harness_files": {
                    p.name: sha(p)
                    for p in [
                        Path(__file__),
                        Path(__file__).with_name("real_gpu_worker.py"),
                        Path(__file__).with_name("real_gpu_contract.py"),
                        Path(__file__).with_name("real_gpu_zmq.py"),
                    ]
                },
                "manifest_sha256": sha(args.manifest),
            },
            indent=2,
        )
        + "\n"
    )
    uuids = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).splitlines()
    if len(uuids) != 4:
        raise RuntimeError("this experiment requires the four available GPUs")
    workers, logs = [], []
    monitor = None
    report = {"status": "running", "config": config, "phases": [], "validation": []}
    reference = np.load(args.reference, allow_pickle=False) if args.reference else None
    golden = {}
    try:
        with tempfile.TemporaryDirectory(prefix="nahual-real-") as ipcdir:
            started = time.perf_counter()
            for slot_index, slot in enumerate(slots):
                for replica in range(args.replicas):
                    tag = f"{slot}-{replica}"
                    ready = args.output / f"{tag}-ready.json"
                    log = (args.output / f"{tag}.log").open("x")
                    logs.append(log)
                    environment = os.environ.copy()
                    device_index = (
                        replica if args.schedule == "data_parallel" else slot_index
                    )
                    environment["CUDA_VISIBLE_DEVICES"] = uuids[device_index]
                    address = f"ipc://{ipcdir}/{tag}.ipc"
                    command = [
                        sys.executable,
                        "-B",
                        "-m",
                        "benchmarks.real_gpu_worker",
                        "--slot",
                        slot,
                        "--backend",
                        backend,
                        "--address",
                        address,
                        "--metrics",
                        str(args.output / f"{tag}-metrics.jsonl"),
                        "--ready",
                        str(ready),
                    ]
                    process = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        start_new_session=True,
                    )
                    workers.append(
                        {
                            "slot": slot,
                            "replica": replica,
                            "tag": tag,
                            "process": process,
                            "ready": ready,
                            "address": address,
                        }
                    )
            deadline = time.monotonic() + 300
            while not all(worker["ready"].exists() for worker in workers):
                for worker in workers:
                    if worker["process"].poll() is not None:
                        raise RuntimeError(f"{worker['tag']} exited; inspect its log")
                if time.monotonic() > deadline:
                    raise TimeoutError("model startup deadline")
                time.sleep(0.25)
            report["parallel_model_startup_seconds"] = time.perf_counter() - started
            report["workers"] = [
                {**json.loads(w["ready"].read_text()), "replica": w["replica"]}
                for w in workers
            ]
            expected = json.loads(
                Path(
                    "/work/users/amunoz/deployments/JUMP_lite-remaining-preliminary-fast-v1-pipelined4-nonappose-20261021/MODEL-STATE-EXPECTATIONS.json"
                ).read_text()
            )
            report["model_state_expectations"] = expected
            for worker in report["workers"]:
                expected_states = (
                    {
                        model: digest
                        for states in expected.values()
                        for model, digest in states.items()
                    }
                    if worker["slot"] == "full_stack"
                    else expected[worker["slot"]]
                )
                if worker["loaded_state_sha256"] != expected_states:
                    raise RuntimeError(f"loaded model state changed: {worker['slot']}")
            print(
                args.label,
                "models ready",
                report["parallel_model_startup_seconds"],
                flush=True,
            )
            monitor_log = (args.output / "gpu.csv").open("x")
            logs.append(monitor_log)
            monitor = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm",
                    "--format=csv,noheader,nounits",
                    "-lms",
                    "200",
                ],
                stdout=monitor_log,
                stderr=subprocess.STDOUT,
            )
            metrics_log = (args.output / "calls.jsonl").open("x", buffering=1)
            logs.append(metrics_log)
            canaries = []
            seen_shapes = set()
            for index, record in enumerate(records):
                shape = tuple(record["shape"])
                if shape not in seen_shapes:
                    seen_shapes.add(shape)
                    canaries.append(index)
            for mode in args.modes:
                clients = [
                    callables(w["address"], mode, args.cache_mib * 1024**2)
                    for w in workers
                ]
                try:
                    # Compare transports with seeded validation only. All measured
                    # calls use the unchanged, unseeded MorphEM behavior.
                    for worker_index, worker in enumerate(workers):
                        setup, call, _ = clients[worker_index]
                        setup(
                            {"phase": f"validation/{mode}", "validation_seed": 17012026}
                        )
                        for index in canaries:
                            result = call(arrays[index])
                            key = f"{records[index]['ordinal']:03d}__{worker['slot']}"
                            if key not in golden:
                                golden[key] = result.copy()
                            target = (
                                full_stack_reference(
                                    reference, records[index]["ordinal"]
                                )
                                if reference is not None
                                and worker["slot"] == "full_stack"
                                else reference[key]
                                if reference is not None
                                else golden[key]
                            )
                            if (
                                result.shape != target.shape
                                or not np.isfinite(result).all()
                            ):
                                raise RuntimeError(f"invalid validation output: {key}")
                            maximum = float(np.max(np.abs(result - target)))
                            exact = bool(np.array_equal(result, target))
                            passed = bool(
                                np.allclose(result, target, rtol=1e-5, atol=5e-4)
                            )
                            report["validation"].append(
                                {
                                    "mode": mode,
                                    "worker": worker["tag"],
                                    "key": key,
                                    "shape": list(result.shape),
                                    "max_abs": maximum,
                                    "exact": exact,
                                    "passed": passed,
                                }
                            )
                            if not passed:
                                raise RuntimeError(
                                    f"transport parity failure: {key}: {maximum}"
                                )
                    if args.batch_size > 1:
                        for worker_index, worker in enumerate(workers):
                            if worker["slot"] == "morphem":
                                continue  # retain its original per-site RNG boundaries
                            setup, call, _ = clients[worker_index]
                            setup(
                                {
                                    "phase": f"batch-validation/{mode}",
                                    "validation_seed": 17012026,
                                    "batch_inference": args.batch_inference,
                                }
                            )
                            checked_shapes = set()
                            for indices in batches(
                                list(range(len(records))), records, args.batch_size
                            ):
                                shape = tuple(records[indices[0]]["shape"])
                                if len(indices) < 2 or shape in checked_shapes:
                                    continue
                                checked_shapes.add(shape)
                                target = np.concatenate(
                                    [call(arrays[index]) for index in indices]
                                )
                                result = call(
                                    np.ascontiguousarray(
                                        np.stack([arrays[index] for index in indices])
                                    )
                                )
                                maximum = float(np.max(np.abs(result - target)))
                                passed = bool(
                                    np.allclose(result, target, rtol=1e-5, atol=5e-4)
                                )
                                report["validation"].append(
                                    {
                                        "kind": "batch_vs_single",
                                        "mode": mode,
                                        "worker": worker["tag"],
                                        "shape": list(shape),
                                        "sites": len(indices),
                                        "max_abs": maximum,
                                        "exact": bool(np.array_equal(result, target)),
                                        "passed": passed,
                                    }
                                )
                                if not passed:
                                    raise RuntimeError(
                                        f"batch parity failure: {worker['tag']}: {maximum}"
                                    )
                    with ThreadPoolExecutor(max_workers=len(workers)) as executor:

                        def configure(phase):
                            futures = [
                                executor.submit(
                                    client[0],
                                    {
                                        "phase": phase,
                                        "validation_seed": None,
                                        "batch_inference": args.batch_inference,
                                    },
                                )
                                for client in clients
                            ]
                            for future in futures:
                                future.result()

                        configure(f"warmup/{mode}")
                        # Each shape is warm in both model execution and SHM cache.
                        futures = [
                            executor.submit(
                                lambda call=client[1]: [
                                    call(arrays[index]) for index in canaries
                                ]
                            )
                            for client in clients
                        ]
                        for future in futures:
                            future.result()
                        for repeat in range(args.repeats):
                            phase = f"{mode}/repeat{repeat}"
                            configure(phase)
                            order = list(range(len(records)))
                            random.Random(args.seed + repeat).shuffle(order)
                            phase_start = time.time()
                            wall_start = time.perf_counter()
                            with ThreadPoolExecutor(max_workers=8) as decoder:
                                if args.decode != "preloaded":

                                    def decode(index):
                                        begin = time.perf_counter()
                                        if args.decode == "zarr_mq":
                                            array = np.ascontiguousarray(
                                                zarr.open_array(
                                                    records[index]["zarr_path"],
                                                    mode="r",
                                                )[:]
                                            )
                                        else:
                                            array = np.ascontiguousarray(
                                                np.stack(
                                                    [
                                                        tifffile.imread(item["path"])
                                                        for item in records[index][
                                                            "files"
                                                        ]
                                                    ]
                                                )
                                            )
                                        if (
                                            hashlib.sha256(array.tobytes()).hexdigest()
                                            != records[index]["array_sha256"]
                                        ):
                                            raise RuntimeError(
                                                "source image decode changed"
                                            )
                                        array.flags.writeable = False
                                        return array, time.perf_counter() - begin

                                    inputs = {
                                        index: decoder.submit(decode, index)
                                        for index in order
                                    }
                                else:
                                    inputs = None

                                def run_worker(worker_index):
                                    worker = workers[worker_index]
                                    call = clients[worker_index][1]
                                    items = batches(
                                        order[worker["replica"] :: args.replicas],
                                        records,
                                        1
                                        if worker["slot"] == "morphem"
                                        else args.batch_size,
                                    )
                                    summaries = []
                                    for indices in items:
                                        raw = [
                                            inputs[index].result()[0]
                                            if inputs
                                            else arrays[index]
                                            for index in indices
                                        ]
                                        begin = time.perf_counter()
                                        packet = (
                                            raw[0]
                                            if len(raw) == 1
                                            else np.ascontiguousarray(np.stack(raw))
                                        )
                                        stage_seconds = time.perf_counter() - begin
                                        begin_unix = time.time()
                                        begin = time.perf_counter()
                                        result = call(packet)
                                        seconds = time.perf_counter() - begin
                                        expected_shape = expected_output_shape(
                                            worker["slot"],
                                            [
                                                records[index]["shape"]
                                                for index in indices
                                            ],
                                        )
                                        if (
                                            result.shape != expected_shape
                                            or not np.isfinite(result).all()
                                        ):
                                            raise RuntimeError(
                                                f"invalid measured result from {worker['tag']}"
                                            )
                                        entry = {
                                            "phase": phase,
                                            "worker": worker["tag"],
                                            "slot": worker["slot"],
                                            "ordinals": [
                                                records[index]["ordinal"]
                                                for index in indices
                                            ],
                                            "start_unix": begin_unix,
                                            "seconds": seconds,
                                            "batch_stage_seconds": stage_seconds,
                                            "input_bytes": packet.nbytes,
                                            "output_bytes": result.nbytes,
                                            "sites": len(indices),
                                            "rows": expected_shape[0],
                                        }
                                        summaries.append(entry)
                                    return summaries

                                futures = [
                                    executor.submit(run_worker, index)
                                    for index in range(len(workers))
                                ]
                                calls = [
                                    call
                                    for future in futures
                                    for call in future.result()
                                ]
                            wall = time.perf_counter() - wall_start
                            for call in calls:
                                metrics_log.write(json.dumps(call) + "\n")
                            values = [call["seconds"] for call in calls]
                            entry = {
                                "phase": phase,
                                "mode": mode,
                                "repeat": repeat,
                                "start_unix": phase_start,
                                "end_unix": time.time(),
                                "wall_seconds": wall,
                                "sites": len(records),
                                "model_outputs": len(records) * 5,
                                "sites_per_second_all_five_models": len(records) / wall,
                                "median_request_ms": statistics.median(values) * 1000,
                                "p95_request_ms": float(np.percentile(values, 95))
                                * 1000,
                                "slot_seconds": {
                                    slot: sum(
                                        call["seconds"]
                                        for call in calls
                                        if call["slot"] == slot
                                    )
                                    for slot in slots
                                },
                                "input_decode_task_wall_seconds_sum": sum(
                                    f.result()[1] for f in inputs.values()
                                )
                                if inputs
                                else 0,
                            }
                            report["phases"].append(entry)
                            (args.output / "report.json").write_text(
                                json.dumps(report, indent=2) + "\n"
                            )
                            print(
                                args.label,
                                phase,
                                round(wall, 3),
                                "s",
                                round(len(records) / wall, 2),
                                "sites/s",
                                flush=True,
                            )
                finally:
                    for _, _, close in clients:
                        close()
            for array, record in zip(arrays, records):
                if (
                    hashlib.sha256(array.tobytes()).hexdigest()
                    != record["array_sha256"]
                ):
                    raise RuntimeError("borrowed input was mutated")
            np.savez(args.output / "validation-outputs.npz", **golden)
            report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        for worker in workers:
            worker["process"].terminate()
        for worker in workers:
            try:
                worker["process"].wait(timeout=15)
            except subprocess.TimeoutExpired:
                worker["process"].kill()
                worker["process"].wait()
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=10)
        for log in logs:
            log.close()
        report["hardware_after"] = inventory()
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if reference is not None:
            reference.close()


if __name__ == "__main__":
    # Silence only the stateless client's verbose transport prints; progress and
    # exceptions remain visible on stderr and server logs are retained.
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        # Progress uses the original stderr so logs remain useful while running.
        import builtins

        original_print = builtins.print

        def progress(*args, **kwargs):
            if args and isinstance(args[0], str) and args[0].startswith(("REQ:",)):
                return
            original_print(
                *args,
                file=sys.stderr,
                **{k: v for k, v in kwargs.items() if k != "file"},
            )

        builtins.print = progress
        main()
