"""Benchmark-only responder reusing frozen JUMP-lite CUDA model functions.

Never invokes campaign/coordinator/publication entry points. Scientific
preprocessing, weights and FP32 inference are unchanged in per-site mode.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import pynng
import torch
import trio
from benchmarks.real_gpu_contract import SLOT_MODELS as SLOTS
from benchmarks.real_gpu_contract import pack_full_stack

from nahual.server import responder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", choices=list(SLOTS), required=True)
    parser.add_argument("--address", required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--backend", choices=["nng", "zmq"], default="nng")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    from rebuttal.broader_jump_ranking_feasibility.dl_successor import gpu_canary
    from rebuttal.broader_jump_ranking_feasibility.dl_successor.annotated_v3_execution import (
        DINOV2_GENERATION3_SNAPSHOT,
    )
    from rebuttal.broader_jump_ranking_feasibility.dl_successor.cuda_preprocessing import (
        prepare_cuda_model_tiles,
    )
    from rebuttal.broader_jump_ranking_feasibility.dl_successor.model_evidence import (
        _state_for_hash,
    )
    from rebuttal.broader_jump_ranking_feasibility.dl_successor.vit_state_hash import (
        vit_state_sha256,
    )

    gpu_canary.DINOV2_REPOSITORY = DINOV2_GENERATION3_SNAPSHOT
    device = torch.device("cuda:0")
    started = time.perf_counter()
    models, states = {}, {}
    for model_id in SLOTS[args.slot]:
        model = gpu_canary._load_model(model_id)
        states[model_id] = vit_state_sha256(_state_for_hash(model.state_dict()))
        models[model_id] = model.to(device).eval()
    torch.cuda.synchronize()
    evidence = {
        "slot": args.slot,
        "backend": args.backend,
        "models": list(models),
        "loaded_state_sha256": states,
        "pid": os.getpid(),
        "device": torch.cuda.get_device_name(0),
        "visible_device_uuid": os.environ["CUDA_VISIBLE_DEVICES"],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
        "load_seconds": time.perf_counter() - started,
        "model_gpu_bytes": torch.cuda.memory_allocated(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }
    state = {"phase": "startup", "seed": None, "batch_inference": False}
    metrics = args.metrics.open("x", buffering=1)

    def processor(array):
        if (
            array.dtype != np.uint16
            or array.ndim not in (3, 4)
            or not array.flags.c_contiguous
        ):
            raise ValueError("expected contiguous uint16 CYX or NCYX input")
        raws = [array] if array.ndim == 3 else list(array)
        if any(raw.shape[0] != 5 for raw in raws):
            raise ValueError("five channels required")
        if state["seed"] is not None:
            # Validation only. Timed phases do not reset MorphEM's RNG.
            torch.manual_seed(state["seed"])
            torch.cuda.manual_seed_all(state["seed"])
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        unix_start = time.time()
        outputs = [[] for _ in raws]
        timings = []
        for model_id, model in models.items():
            first, prepared, inferred = [
                torch.cuda.Event(enable_timing=True) for _ in range(3)
            ]
            first.record()
            tiles = [
                prepare_cuda_model_tiles(model_id, list(raw), device=device)
                for raw in raws
            ]
            prepared.record()
            if state["batch_inference"] and model_id != "morphem" and len(tiles) > 1:
                # Experimental throughput tuning; never used for transport parity.
                counts = [len(item) for item in tiles]
                embeddings = gpu_canary._infer_cuda(
                    model_id, model, torch.cat(tiles), device
                )
                chunks = np.split(embeddings, np.cumsum(counts)[:-1])
            else:
                chunks = [
                    gpu_canary._infer_cuda(model_id, model, item, device)
                    for item in tiles
                ]
            inferred.record()
            torch.cuda.synchronize()
            timings.append(
                {
                    "model": model_id,
                    "tiles": sum(len(item) for item in tiles),
                    "prepare_cuda_ms": first.elapsed_time(prepared),
                    "inference_cuda_ms": prepared.elapsed_time(inferred),
                }
            )
            for output, chunk in zip(outputs, chunks):
                output.append(chunk)
            del tiles, chunks
        if args.slot == "full_stack":
            result = pack_full_stack([part for parts in outputs for part in parts])
        else:
            result = np.ascontiguousarray(
                np.concatenate(
                    [np.concatenate(parts, axis=1) for parts in outputs], axis=0
                )
            )
        metrics.write(
            json.dumps(
                {
                    "phase": state["phase"],
                    "start_unix": unix_start,
                    "end_unix": time.time(),
                    "processor_seconds": time.perf_counter() - wall_start,
                    "processor_cpu_seconds": time.process_time() - cpu_start,
                    "peak_host_rss_bytes": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss
                    * 1024,
                    "sites": len(raws),
                    "input_bytes": array.nbytes,
                    "output_bytes": result.nbytes,
                    "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "models": timings,
                }
            )
            + "\n"
        )
        return result

    def setup(phase="benchmark", validation_seed=None, batch_inference=False):
        state.update(phase=phase, seed=validation_seed, batch_inference=batch_inference)
        torch.cuda.reset_peak_memory_stats()
        return processor, evidence

    async def serve():
        with pynng.Rep0(listen=args.address, recv_timeout=300_000) as socket:
            args.ready.write_text(json.dumps(evidence, indent=2) + "\n")
            await responder(socket, setup=setup, processor=processor)

    try:
        if args.backend == "zmq":
            from benchmarks.real_gpu_zmq import serve as serve_zmq

            serve_zmq(
                args.address,
                setup,
                processor,
                lambda: args.ready.write_text(json.dumps(evidence, indent=2) + "\n"),
            )
        else:
            trio.run(serve)
    finally:
        metrics.close()


if __name__ == "__main__":
    main()
