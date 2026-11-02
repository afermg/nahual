# Bounded decode/inference/output streaming

`nahual.pipeline.process_map` makes the previously benchmarked scheduling pattern
reusable: one persistent NNG client with a bounded SHM cache per endpoint, site
assignment to available replicas, bounded concurrent loading, and incremental
result consumption. It has no Torch, TIFF, Zarr or other new runtime dependency.

It does **not** deploy models, select GPUs, merge inference batches, change
precision, retry failed sites, or resume scientific campaigns. An endpoint must
already host an equivalent independent worker. For JUMP-lite, the tested layout
is the complete five-model stack on each GPU, not different models at different
endpoints. Do not point this scheduler at heterogeneous single-model endpoints.

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

## Matched local-data validation: 2026-11-02

Evidence: [`results/bounded_pipeline_20261102/summary.json`](results/bounded_pipeline_20261102/summary.json).
Full scratch evidence: `/work/scratch/amunoz/nahual-bounded-pipeline-20261102/`.
Exact benchmark/library sources are archived in its `as-run-source/` directory.
The final harness additionally rejects incomplete historical-reference coverage;
that guard was checked against both real reference files after timing. The core
pipeline and inference code are byte-identical to the measured versions.

Same 64 sites and six geometries as the [earlier real-GPU study](REAL_GPU.md):
32 JUMP-lite and 32 Target-2 sites, eight sources, 33 plates. Raw TIFF and existing
lossy MQ JPEG-XL Zarr inputs are separate, hash-bound cohorts. Each site runs
DINOv2, checkpointed random DINOv2, MorphEM, OpenPhenom and SubCell, preserving
frozen CUDA preprocessing and FP32/per-site inference. All 87 frozen recipe
inventory entries and every loaded model-state hash pass.

Only **GPUs 1–3** were used (H100 NVL, one full stack per GPU). GPU 0 had another
experiment and was left untouched. No service, canonical dataset, checkpoint,
model weight, manuscript result or historical benchmark output was modified.

### Numerical validation

For each codec:

1. All three workers match six historical single-slot canaries (18 comparisons).
2. Generate a seeded, per-site full-stack reference for all 64 frozen cached
   inputs, then compare **every** decoded streaming output to its reference.
3. All 82 comparisons are **bit-exact**, or **164/164 across both cohorts**.
   Acceptance remains `rtol=1e-5, atol=5e-4`; no tolerance was relaxed.

Seed `17012026` is used only for comparisons to control stochastic MorphEM.
Measured phases do not reset its RNG. No merged batches, autocast or BF16.

### Throughput including decoding and NumPy output writes

Five matched repeats per configuration/codec, alternating execution order;
identical shuffled site order within each pair. Baseline: three dynamically
sharded workers, each serially decoding, inferring with pooled SHM and writing
its output. Candidate: eight load threads, three inference workers, at most
12 admitted sites and a 256 MiB decoded-input reservation budget. Both hash-check
decoded inputs, check output shape/finiteness, and write uncompressed `.npy`
outputs. Client setup is outside timing; input consumption, output writes and
stream draining are inside. Candidate iterator teardown is included; baseline
client teardown and both per-run JSON receipt serializations are outside timing.
Caches start empty each pass; model weights remain loaded and warmed.

| Input | No prefetch, sites/s | Bounded, sites/s | Ratio of medians |
|---|---:|---:|---:|
| Raw TIFF | 15.51 | 15.37 | 0.99× |
| MQ JPEG-XL Zarr | 10.70 | 17.14 | **1.60×** |

MQ mean sampled GPU utilization rises from **48–49% to 77–79%** across the three
GPUs. TIFF utilization remains approximately 68–73%; there is **no TIFF speedup**
in this run. Candidate peak reservations are at most **146,181,840 bytes
(139.4 MiB)** and 12 sites, independent of total input count. All runs yield
exactly 64 outputs and drain their queues. Twenty measured passes produced
1,280 site outputs / 6,400 model-site outputs; validation work is additional.

These short repeated windows on a shared host are descriptive, not confidence
intervals. OS page caches are warm, output writes are not `fsync`-durable, and
GPU utilization is sampled every 200 ms (local offset explicitly recorded as
`-18000` seconds). Do not directly compare these three-GPU numbers with the
previous four-GPU experiment, which used different output/timing boundaries.
Downloads, production profile-table generation, signing and publication remain
excluded. More decode threads do not help every codec: tune against matched
inputs rather than treating prefetch as a universal speedup.

### Reproduce on the recorded site

Use the same frozen scientific runtime/PYTHONPATH described in
`launch_real_gpu.sh`, but invoke `run_bounded_pipeline.py` directly. It refuses
occupied selected GPUs and existing output directories; its site-specific
output guard requires `/work/scratch/amunoz/`. It never stops other jobs.

```bash
python -B benchmarks/run_bounded_pipeline.py \
  --manifest "$PRIOR/inputs-mq-v2/manifest.json" \
  --reference "$PRIOR/mq-baseline/validation-outputs.npz" \
  --recipe-root "$RECIPE" --decode zarr_mq --gpu-indices 1 2 3 \
  --output /work/scratch/amunoz/new-bounded-mq-run
```

For TIFF use `inputs/manifest.json`,
`baseline-before/validation-outputs.npz`, and `--decode tiff`. `--decode cache`
is also supported for reading the frozen `.npy` inputs. The optional scientific
packages are not required by the core library or generic example.

```bash
python benchmarks/summarize_bounded_pipeline.py RUN_TIFF RUN_MQ \
  --gpu-utc-offset-seconds -18000 --output new-summary.json
```

The benchmark reuses `real_gpu_worker.py` and its benchmark-only packed
full-stack framing; it is not a new production wire schema. The standalone
runner saves separate scratch outputs and has no access to campaign checkpoint
logic. Production adoption should be a separately versioned, explicitly
validated deployment, not a patch to an immutable scientific recipe.
