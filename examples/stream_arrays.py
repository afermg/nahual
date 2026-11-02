"""Stream a JSONL manifest of NumPy inputs through existing worker replicas.

Each line: {"id": "site-key", "path": "/absolute/input.npy", "nbytes": 12345}.
The decoded byte count must be an upper bound, not the compressed file size.
No worker launch, inference batching, automatic retries or resume is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nahual.pipeline import process_map


def records(path):
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            if not isinstance(record["id"], str):
                raise ValueError("id must be a string")
            if not Path(record["path"]).is_absolute():
                raise ValueError("input paths must be absolute")
            yield record


def load(record):
    return np.load(record["path"], allow_pickle=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--address", action="append", required=True)
    parser.add_argument("--setup-json", type=Path)
    parser.add_argument("--load-workers", type=int, default=4)
    parser.add_argument("--max-in-flight", type=int)
    parser.add_argument("--max-input-mib", type=int, default=256)
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument(
        "--wire", action="store_true", help="disable local shared memory"
    )
    args = parser.parse_args()
    setup = json.loads(args.setup_json.read_text()) if args.setup_json else None
    # Refuse to overwrite prior outputs. A failed run has no SUCCESS.json;
    # partial output and receipts remain available for inspection, not resume.
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "receipt.jsonl").open("x", buffering=1) as receipt:
        with process_map(
            records(args.manifest),
            args.address,
            loader=load,
            size_of=lambda record: record["nbytes"],
            max_input_bytes=args.max_input_mib * 1024**2,
            max_in_flight=args.max_in_flight,
            load_workers=args.load_workers,
            setup_parameters=setup,
            shared_memory=not args.wire,
            timeout_ms=args.timeout_ms,
        ) as results:
            for result in results:
                filename = f"{result.index:09d}.npy"
                with (args.output / filename).open("xb") as stream:
                    np.save(stream, result.output, allow_pickle=False)
                receipt.write(
                    json.dumps(
                        dict(
                            index=result.index,
                            id=result.item["id"],
                            input_path=result.item["path"],
                            output=filename,
                            worker=result.address,
                            input_nbytes=result.input_nbytes,
                        )
                    )
                    + "\n"
                )
        stats = results.stats
    (args.output / "SUCCESS.json").write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
