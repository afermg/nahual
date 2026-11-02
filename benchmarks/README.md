# Transport benchmarks

Run from the repository root with NumPy, pynng and trio installed. Benchmark
modules are checkout-only developer tools, not part of the installed Nahual
library. Use `python -m benchmarks.<name>` for their CLI entry points.
Keep new run artifacts in external scratch storage (or the ignored
`benchmarks/results/` directory); tests should use synthetic fixtures, not dated
machine-specific measurements.

For example:

```bash
PYTHONPATH=src python -m benchmarks.transport --iterations 30 > /tmp/nahual-results.json
```

To include an **experimental, benchmark-only** alternative:

```bash
pip install pyzmq
PYTHONPATH=src python -m benchmarks.transport --zmq --iterations 30 > /tmp/nahual-zmq-results.json
```

No production backend or dependency changes are made. The PyZMQ prototype uses
public multipart APIs with `copy=False` and tracked sends, keeping buffer owners
alive until sending finishes. Small frames can still be copied by PyZMQ's
threshold, and OS/socket copies remain. It is not a drop-in Nahual server: it
omits setup, typed errors, version negotiation, lifecycle/recovery checks and
shared memory. Its synchronous server also differs from the real asynchronous
Nahual responder used by the NNG cases. Treat it as a direction to investigate,
not a fully controlled backend comparison or production implementation.

The benchmark launches separate processes over local Unix sockets, validates
outputs outside the timing window, and reports median/p95 call latency. It
compares fresh and persistent NNG sockets, per-call shared-memory allocation,
and bounded shared-memory reuse. Identity output stresses both directions;
embedding-shaped output returns 4 KiB from any input. There is no model, disk
I/O, preprocessing or GPU transfer. Inputs are C-contiguous float32 arrays.

Defaults: 4 KiB, 1 MiB and 16 MiB inputs, three warmup calls and 25 timed calls
per case. Override with `--sizes` (bytes), `--iterations`, and `--warmup`.
Large sizes must fit the legacy wire shape limits and available `/dev/shm`.
Shared memory must be available to both subprocesses on the same Linux host.

## Example local result

[`results-local.json`](results-local.json) records one 30-call run on Linux,
Python 3.12.11, NumPy 2.5.4, pynng 0.9.0, trio 0.35.0 and PyZMQ 27.2.0.
Medians in milliseconds:

| Input / output | NNG fresh wire | NNG session wire | Session SHM, uncached | Session SHM, cached | PyZMQ prototype |
|---|---:|---:|---:|---:|---:|
| 4 KiB / identity | 0.615 | 0.255 | 0.667 | 0.510 | 0.208 |
| 1 MiB / identity | 6.548 | 5.636 | 2.365 | 0.667 | 0.580 |
| 16 MiB / identity | 88.076 | 55.246 | 11.830 | 1.337 | 5.545 |
| 4 KiB / embedding | 0.725 | 0.385 | 0.780 | 0.684 | 0.222 |
| 1 MiB / embedding | 2.400 | 2.100 | 2.804 | 0.842 | 0.486 |
| 16 MiB / embedding | 41.325 | 34.952 | 11.555 | 1.161 | 3.773 |

These are host-specific synthetic measurements, not inference speedups or
performance guarantees. The cases run sequentially, so load and cache state
can affect comparisons. Repeat on the target machine with realistic shapes,
dtypes and inference before choosing a backend or batch size. The useful
local conclusions are that small calls benefit from persistent wire sockets,
large repeated calls benefit strongly from reusing shared-memory allocations,
and a wholesale backend replacement is not necessary for those gains.
