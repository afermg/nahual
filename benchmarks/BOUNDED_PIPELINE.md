# Bounded decode/inference/output streaming

`nahual.pipeline.process_map` provides bounded streaming: one persistent NNG client with a bounded SHM cache per endpoint, site
assignment to available replicas, bounded concurrent loading, and incremental
result consumption. It has no Torch, TIFF, Zarr or other new runtime dependency.

It does **not** deploy models, select GPUs, merge inference batches, change
precision, retry failed sites, or resume scientific campaigns. An endpoint must
already host an equivalent independent worker. Replicate the same model or
complete model stack at every endpoint. Do not point this scheduler at
heterogeneous single-model endpoints.

## Runnable generic example

`examples/stream_arrays.py` consumes JSONL descriptors without preloading all
arrays. Each record has an ID, absolute `.npy` path, and an upper bound on decoded
NumPy bytes (not compressed file size):

```json
{"id": "site-1", "path": "/scratch/inputs/site-1.npy", "nbytes": 13824000}
```

After starting equivalent dedicated workers:

```bash
python examples/stream_arrays.py inputs.jsonl /scratch/new-results \
  --address ipc:///tmp/gpu1.ipc --address ipc:///tmp/gpu2.ipc \
  --load-workers 4 --max-in-flight 8 --max-input-mib 256
```

Optional `--setup-json parameters.json` applies the same setup to every worker;
omit it to retain the existing model setup. The output directory must not exist.
Outputs use input indices rather than caller IDs as filenames; receipts retain
both. Partial files remain after failure, but `SUCCESS.json` is written only
once the stream and receipt file finish successfully. These are example NumPy
outputs, **not** publication-ready profile tables, crash-durable commits, or
production recovery checkpoints. No automatic resume or overwrite is provided.

## Resource and error contract

- `max_in_flight` counts queued loads, decoded inputs, requests and completed
  results not yet yielded. Default: twice the endpoint count.
- `max_input_bytes` requires a cheap `size_of(record)` upper bound and reserves
  it before scheduling a load. A single oversized input fails; an underestimate
  fails before inference. A loader may already have allocated too much memory
  before that check, so this is not a hard allocator/RSS limit.
- Reservations are conservatively retained through result delivery. Source
  iteration may retain one additional cheap descriptor as lookahead.
- SHM cache budgets (default 64 MiB/eight idle segments **per client**), active
  staging, loader temporaries, outputs and consumer-retained objects are extra.
  Counting results does not bound their byte sizes. Do not use `list(results)`
  for large production runs if bounded consumer memory is required.
- Loaders run concurrently in threads; they must return independent NumPy
  arrays and must not reuse a mutable scratch buffer while requests are active.
  Caller-owned arrays likewise must not be mutated before result delivery.
- Results are available-completion order by default; `ordered=True` applies
  backpressure behind a slow head rather than buffering an unlimited tail.
- `PipelineError` carries stage, input index (when applicable) and endpoint (for
  setup/process failures), with the original exception chained. Errors terminate
  the iterator even if caught inside the `with` block. No retries are performed.
- Exit cancels queued work, drains active requests/loads, then closes clients and
  unlinks their caches. A timeout does not cancel server computation; uncertain
  shared storage is discarded, never recycled. Finite request timeouts are
  recommended. Blocking loader callbacks cannot be forcibly cancelled.
- Consume/close on the creating thread and process. Existing stateless helpers,
  `Client`, server wrappers and wire protocols are unchanged.

## Single-worker use

One endpoint is supported too: concurrent loaders can prepare upcoming inputs
while that worker processes the current request. Persistent connections and
cached shared-memory buffers still apply. Additional loaders do not parallelize
inference within a worker; tune prefetch limits to available host memory and
measure your own workload.
