"""Freeze a local-only, source-balanced JUMP-lite / Target-2 GPU benchmark cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tifffile

CHANNELS = ("AGP", "DNA", "ER", "Mito", "RNA")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-dataset", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path("/work/scratch/amunoz")):
        parser.error("benchmark outputs must be in user scratch")
    args.output.mkdir(parents=True, exist_ok=False)
    index = Path(
        "/work/datasets/jump_lite/cpg_release/metadata/jump_lite_site_index.parquet"
    )
    cols = [
        "Metadata_Site_Key",
        "Metadata_Source",
        "Metadata_Batch",
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_Site",
    ]
    rows = pq.read_table(index, columns=cols).to_pylist()
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    raw_lite = Path("/work/datasets/jump_lite/images/raw/jump_lite")
    grouped = {}
    for row in rows:
        grouped.setdefault(row["Metadata_Source"], []).append(row)
    lite_candidates = []
    while grouped:
        for source in sorted(list(grouped)):
            if grouped[source]:
                row = grouped[source].pop()
                paths = [
                    raw_lite / f"{row['Metadata_Site_Key']}__{channel}.tif"
                    for channel in CHANNELS
                ]
                lite_candidates.append((row["Metadata_Site_Key"], paths))
            else:
                del grouped[source]
        if len(lite_candidates) > 1024:
            break
    target_root = Path("/work/datasets/jump_lite/images/raw/jump_target2_4plate")
    target_files = sorted(target_root.glob("*__DNA__*__Orig.tif"))
    rng.shuffle(target_files)
    target_groups = {}
    for path in target_files:
        parts = path.name.removesuffix(".tif").split("__")
        source, batch, plate, well, _, site, _ = parts
        key = "__".join((source, batch, plate, well, site))
        paths = [
            target_root
            / "__".join((source, batch, plate, well, channel, site, "Orig.tif"))
            for channel in CHANNELS
        ]
        target_groups.setdefault((source, batch, plate), []).append((key, paths))
    target_candidates = []
    while target_groups:
        for plate in sorted(list(target_groups)):
            if target_groups[plate]:
                target_candidates.append(target_groups[plate].pop())
            else:
                del target_groups[plate]
    records, failures = [], []
    for dataset, candidates in [
        ("jump_lite_raw", lite_candidates),
        ("target2_raw", target_candidates),
    ]:
        admitted = 0
        for key, paths in candidates:
            if admitted >= args.per_dataset:
                break
            try:
                if not all(path.is_file() for path in paths):
                    raise FileNotFoundError("missing local channel")
                started = time.perf_counter()
                planes = [tifffile.imread(path) for path in paths]
                decode_seconds = time.perf_counter() - started
                if any(
                    p.dtype != np.uint16 or p.ndim != 2 or not np.any(p) for p in planes
                ):
                    raise ValueError("channel is not nonzero uint16 YX")
                if len({plane.shape for plane in planes}) != 1:
                    raise ValueError("unequal channel shapes")
                array = np.ascontiguousarray(np.stack(planes))
                if min(array.shape[1:]) < 448:
                    raise ValueError("too small for SubCell")
                ordinal = len(records)
                destination = args.output / f"{ordinal:03d}.npy"
                with destination.open("xb") as stream:
                    np.save(stream, array, allow_pickle=False)
                records.append(
                    {
                        "ordinal": ordinal,
                        "dataset": dataset,
                        "site": key,
                        "source": key.split("__")[0],
                        "plate": "__".join(key.split("__")[:3]),
                        "channels": list(CHANNELS),
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                        "nbytes": array.nbytes,
                        "array_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                        "cache_path": str(destination),
                        "cache_sha256": sha(destination),
                        "files": [
                            {
                                "path": str(path),
                                "sha256": sha(path),
                                "bytes": path.stat().st_size,
                            }
                            for path in paths
                        ],
                        "initial_decode_seconds": decode_seconds,
                    }
                )
                admitted += 1
                print(dataset, admitted, key, array.shape, flush=True)
            except Exception as error:
                failures.append(
                    {
                        "dataset": dataset,
                        "site": key,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        if admitted != args.per_dataset:
            raise RuntimeError(f"only {admitted} valid local sites in {dataset}")
    manifest = {
        "created_unix": time.time(),
        "seed": args.seed,
        "site_index": str(index),
        "site_index_sha256": sha(index),
        "scope": "Local raw TIFFs only; source/plate round-robin deterministic selection. Not a biological evaluation cohort or object-density-stratified sample.",
        "records": records,
        "rejected_candidates": failures,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("manifest", args.output / "manifest.json", len(records), flush=True)


if __name__ == "__main__":
    main()
