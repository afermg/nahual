"""Freeze matched existing JUMP-lite/Target-2 MQ Zarr inputs, without recompression."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numcodecs
import numpy as np
import zarr
from imagecodecs.numcodecs import Jpegxl

numcodecs.register_codec(Jpegxl)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path("/work/scratch/amunoz")):
        parser.error("scratch outputs only")
    args.output.mkdir(parents=True, exist_ok=False)
    original = json.loads(args.raw_manifest.read_text())
    roots = {
        "jump_lite_raw": Path(
            "/work/datasets/jump_lite/images/compressed/compressed_test/jump_lite_updated/jpegxl_lossy_mq.zarr"
        ),
        "target2_raw": Path(
            "/work/datasets/jump_lite/images/compressed/jump_target2_4plate/jpegxl_lossy_mq.zarr"
        ),
    }
    records = []
    for record in original["records"]:
        source = roots[record["dataset"]] / record["site"]
        if not source.is_dir():
            raise FileNotFoundError(source)
        started = time.perf_counter()
        array = np.ascontiguousarray(zarr.open_array(str(source), mode="r")[:])
        seconds = time.perf_counter() - started
        if array.dtype != np.uint16 or list(array.shape) != record["shape"]:
            raise ValueError(f"MQ shape/dtype mismatch: {source}")
        raw = np.load(record["cache_path"], allow_pickle=False)
        sample_raw = raw[:, ::16, ::16].reshape(5, -1).astype(np.float64)
        sample_mq = array[:, ::16, ::16].reshape(5, -1).astype(np.float64)
        correlation_matrix = np.corrcoef(sample_raw, sample_mq)[:5, 5:]
        correlations = np.diag(correlation_matrix).tolist()
        # MQ is lossy: high raw/MQ pixel correlation is not an admission
        # requirement. Check channel identity by the best raw/MQ match instead
        # of confusing expected compression error with a channel permutation.
        if not np.array_equal(np.argmax(correlation_matrix, axis=0), np.arange(5)):
            raise ValueError(
                f"MQ channel identity mismatch: {source}: {correlation_matrix.tolist()}"
            )
        destination = args.output / f"{record['ordinal']:03d}.npy"
        with destination.open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
        files = sorted(path for path in source.rglob("*") if path.is_file())
        records.append(
            {
                **record,
                "dataset": record["dataset"].replace("_raw", "_mq"),
                "decode_kind": "zarr_mq",
                "zarr_path": str(source),
                "raw_array_sha256": record["array_sha256"],
                "array_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                "cache_path": str(destination),
                "cache_sha256": sha(destination),
                "files": [
                    {
                        "path": str(path),
                        "sha256": sha(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in files
                ],
                "initial_decode_seconds": seconds,
                "sample_channel_pearson_raw_vs_mq": correlations,
                "sample_cross_channel_pearson_raw_vs_mq": correlation_matrix.tolist(),
            }
        )
        print(record["ordinal"], record["site"], seconds, flush=True)
    manifest = {
        **original,
        "created_unix": time.time(),
        "raw_manifest_sha256": sha(args.raw_manifest),
        "scope": "Same site identities as raw cohort, decoded from existing local lossy MQ JPEG-XL Zarr stores. MQ and raw are separate inference inputs, never assumed feature-identical.",
        "records": records,
        "rejected_candidates": [],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
