"""Same-host transport benchmark; no model, disk I/O, or GPU transfers.

Run from a checkout with PYTHONPATH=src. PyZMQ is optional (--zmq); trio is
required to run the real Nahual responder. Results are JSON, timings are ms.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import platform
import select
import statistics
import subprocess
import sys
import tempfile
import time

import numpy

from nahual.process import Client, send_receive_process


def processor_for(mode):
    if mode == "identity":
        return lambda array: array
    return lambda array: numpy.full((1, 1024), array.flat[0], dtype=numpy.float32)


def serve(args):
    processor = processor_for(args.output)
    if args.server == "nng":
        import pynng
        import trio

        from nahual.server import responder

        async def main():
            with pynng.Rep0(listen=args.address) as socket:
                print("READY", flush=True)
                await responder(socket, setup=lambda: None, processor=processor)

        trio.run(main)
    else:
        import zmq

        with zmq.Context() as context, context.socket(zmq.REP) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            socket.bind(args.address)
            print("READY", flush=True)
            while True:
                frames = socket.recv_multipart(copy=False)
                metadata = json.loads(bytes(frames[0]))
                array = numpy.frombuffer(
                    frames[1].buffer, dtype=metadata["dtype"]
                ).reshape(metadata["shape"])
                array.flags.writeable = False
                result = numpy.ascontiguousarray(processor(array))
                header = json.dumps(
                    {"dtype": result.dtype.str, "shape": result.shape}
                ).encode()
                tracker = socket.send_multipart(
                    [header, memoryview(result)], copy=False, track=True
                )
                # Preserve all buffer owners until the send completes.
                tracker.wait()


@contextlib.contextmanager
def server(kind, output, directory):
    address = f"ipc://{directory}/{kind}-{output}.ipc"
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "benchmarks.transport",
            "--server",
            kind,
            "--address",
            address,
            "--output",
            output,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready, _, _ = select.select([process.stdout], [], [], 15)
        if not ready or process.stdout.readline().strip() != "READY":
            raise RuntimeError(f"{kind} benchmark server failed to start")
        yield address
    finally:
        process.terminate()
        try:
            _, errors = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            _, errors = process.communicate()
        if errors:
            raise RuntimeError(errors)


@contextlib.contextmanager
def zmq_client(address):
    import zmq

    with zmq.Context() as context, context.socket(zmq.REQ) as socket:
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, 10_000)
        socket.setsockopt(zmq.RCVTIMEO, 10_000)
        socket.connect(address)

        def call(array):
            header = json.dumps(
                {"dtype": array.dtype.str, "shape": array.shape}
            ).encode()
            tracker = socket.send_multipart(
                [header, memoryview(array)], copy=False, track=True
            )
            frames = socket.recv_multipart(copy=False)
            tracker.wait()
            metadata = json.loads(bytes(frames[0]))
            result = numpy.frombuffer(
                frames[1].buffer, dtype=metadata["dtype"]
            ).reshape(metadata["shape"])
            # The buffer's memoryview retains its Frame owner.
            result.flags.writeable = False
            return result

        yield call


def measure(call, expected, iterations, warmup):
    output = call()
    numpy.testing.assert_array_equal(output, expected)
    assert output.dtype == expected.dtype
    del output
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        output = call()
        samples.append((time.perf_counter_ns() - start) / 1e6)
        del output
    # Validate another call after the timed run, outside the timer.
    numpy.testing.assert_array_equal(call(), expected)
    return {
        "median_ms": round(statistics.median(samples), 4),
        "p95_ms": round(float(numpy.percentile(samples, 95)), 4),
    }


def benchmark(args):
    if args.zmq:
        import zmq  # noqa: F401 -- fail before launching workers if absent

    results = []
    with tempfile.TemporaryDirectory(prefix="nahual-bench-") as directory:
        for mode in ["identity", "embedding"]:
            with contextlib.ExitStack() as stack:
                address = stack.enter_context(server("nng", mode, directory))
                zmq_address = (
                    stack.enter_context(server("zmq", mode, directory))
                    if args.zmq
                    else None
                )
                for size in args.sizes:
                    if size % 4 or size < 4:
                        raise ValueError(
                            "sizes must be positive multiples of four bytes"
                        )
                    data = (
                        numpy.ones((size // 4096, 1024), dtype=numpy.float32)
                        if size % 4096 == 0
                        else numpy.ones(size // 4, dtype=numpy.float32)
                    )
                    expected = processor_for(mode)(data)
                    with contextlib.ExitStack() as clients:
                        wire = clients.enter_context(Client(address, timeout_ms=10_000))
                        uncached = clients.enter_context(
                            Client(
                                address, timeout_ms=10_000, shared_memory_cache_bytes=0
                            )
                        )
                        cached = clients.enter_context(
                            Client(
                                address,
                                timeout_ms=10_000,
                                shared_memory_cache_bytes=max(size, 1),
                            )
                        )
                        calls = {
                            "wire_once": lambda: send_receive_process(
                                data, "numpy", address, timeout_ms=10_000
                            ),
                            "shm_once": lambda: send_receive_process(
                                data,
                                "numpy",
                                address,
                                shared_memory=True,
                                timeout_ms=10_000,
                            ),
                            "wire_session": lambda: wire.process(data),
                            "shm_session_uncached": lambda: uncached.process(
                                data, shared_memory=True
                            ),
                            "shm_session_cached": lambda: cached.process(
                                data, shared_memory=True
                            ),
                        }
                        if zmq_address:
                            zmq_call = clients.enter_context(zmq_client(zmq_address))
                            calls["zmq_multipart_session"] = lambda: zmq_call(data)
                        row = {
                            "input_bytes": size,
                            "output": mode,
                            "output_bytes": expected.nbytes,
                        }
                        # Suppress legacy request logging, not server errors.
                        with (
                            open(os.devnull, "w") as sink,
                            contextlib.redirect_stdout(sink),
                        ):
                            for name, call in calls.items():
                                row[name] = measure(
                                    call, expected, args.iterations, args.warmup
                                )
                        results.append(row)
    packages = ["numpy", "pynng", "trio"] + (["pyzmq"] if args.zmq else [])
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "iterations": args.iterations,
        "warmup": args.warmup,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[4096, 1048576, 16777216]
    )
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--zmq",
        action="store_true",
        help="include experimental PyZMQ multipart baseline",
    )
    parser.add_argument("--server", choices=["nng", "zmq"], help=argparse.SUPPRESS)
    parser.add_argument("--address", help=argparse.SUPPRESS)
    parser.add_argument(
        "--output",
        choices=["identity", "embedding"],
        default="identity",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive; warmup must be non-negative")
    if any(size < 4 or size % 4 for size in args.sizes):
        parser.error("sizes must be positive multiples of four bytes")
    if args.server:
        serve(args)
    else:
        print(json.dumps(benchmark(args), indent=2))


if __name__ == "__main__":
    main()
