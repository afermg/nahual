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

## Historical experiments

Machine-specific GPU campaign scripts, input inventories, and measured results
are archived in Git history rather than maintained on the default branch. The
[November 2026 experiments](https://github.com/afermg/nahual/tree/9ae74676f9a31584c3a7c0981c01b98975872fdc/benchmarks)
include the transport, full-stack GPU scheduling, and bounded-streaming reports.
They are historical evidence, not portable launch instructions or performance
guarantees. Keep new measurements outside Git; `benchmarks/results/` is ignored.

For the supported streaming API and a portable example, see
[bounded streaming](BOUNDED_PIPELINE.md). Regression tests use synthetic inputs
and do not require historical measurements or scientific datasets.
