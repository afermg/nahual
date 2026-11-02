# Local JUMP-lite GPU validation — 2026-11-02

This is a **local operator benchmark**, not a downloadable model suite, a new
supported backend, a manuscript timing update, or a production deployment.
The scripts require the already installed JUMP-lite environments and data paths
on `oppy`. No weights, images, production profiles, receipts, services or model
wrappers are modified. Only a new scratch namespace is written.

## Inputs and scientific contract

- 64 distinct sites: 32 local JUMP-lite sites and 32 local Target-2 sites.
- Eight sources, 33 physical plates, six image geometries, five uint16 channels
  in AGP/DNA/ER/Mito/RNA order. Geometries range from 970×970 to 1080×1280.
- The same site identities are also frozen from existing local MQ JPEG-XL Zarr
  stores. Raw and lossy MQ are **different input arrays**, with separate hashes
  and separate reference outputs; their features are not assumed identical.
- Selection is deterministic and source/plate round-robin, seed 20261102. It is
  not a biological evaluation cohort or an object-density-stratified sample.
- The immutable October 21 JUMP-lite deployment supplies the exact model-loading,
  CUDA preprocessing and FP32 inference functions for pretrained DINOv2-S/14,
  checkpointed random DINOv2, MorphEM, OpenPhenom and SubCell.
- Loaded tensor-state hashes must match `MODEL-STATE-EXPECTATIONS.json` before
  inference. Normalization, channel selection and tiling remain on CUDA. TIFF /
  JPEG-XL container decoding remains on CPU. There is no autocast, BF16,
  compilation, re-training or precision change.

The transport comparison uses previous Nahual master `56aa3fe` versus optimized
master `7609bff`, with otherwise identical frozen model code. The former is
**not** a claim to reproduce the custom vendored transport in the live full-JUMP
production service.

## Protocol

Four initially idle H100 NVL GPUs (95,830 MiB each) are UUID-isolated. Each model
is loaded once per worker. Model-parallel scheduling places the DINO pair,
MorphEM, OpenPhenom and SubCell on one GPU each. Data-parallel scheduling instead
loads all five models on each GPU and shards sites across four full-stack workers.
The latter changes scheduling and input fanout, not scientific model functions;
its improvement must not be attributed solely to a transport PR. Single-site
inference is preserved; no inference-batch merging or timed-call RNG reset is
introduced in that schedule.

Each case has five measured repeats over all 64 sites, with the same deterministic
per-repeat shuffle. Every input geometry is warmed first. Preloaded cases exclude
image decoding. Decoding-inclusive cases use eight CPU prefetch workers and include
source reads, decoding, array assembly and decoded-pixel integrity checks.
Decode warmups use preloaded arrays: the first measured decoding repeat also
contains codec/Python cache initialization. It is retained. The JSON reports
both the all-five-repeat median and the median excluding the first repeat.

CUDA is synchronized at processor timing boundaries. Per-call CUDA-event spans,
processor wall/CPU time, request latency, sampled GPU utilization/power/memory,
worker identities, state hashes, software versions and source hashes are retained.
GPU telemetry is sampled every 200 ms. `clock.json` records its local-wall-time
UTC offset so offline summaries do not depend on the summarizing host's timezone.

Six geometry canaries validate transports against previous-master outputs. Only
validation calls reset the CUDA seed (17012026), so stochastic MorphEM can be
compared. Timed calls retain unseeded MorphEM behavior. Inputs must remain unchanged,
outputs must have the expected shape and finite values, and client/server request
counts must match. Each site must occur exactly once per model slot per repeat.

Batch inference has a separate predeclared `atol=5e-4, rtol=1e-5` gate against
single-site inference. MorphEM retains one site per request to preserve its RNG
boundaries. Failed trials remain in the evidence and receive no throughput score.

## Results

The compact machine-readable result is
[`results/jump_lite_20261102/summary.json`](results/jump_lite_20261102/summary.json).
All rates below are median **sites/second including all five models**, across four
GPUs—not single-model image rates.

### Matched transport comparison, inputs already decoded

| Transport | Sites/s |
|---|---:|
| Previous master, fresh wire connection | 7.37 before / 7.31 after |
| Previous master, per-call shared memory | 11.86 before / 11.54 after |
| Optimized master, fresh wire connection | 13.27 |
| Optimized master, persistent wire connection | 13.17 |
| Optimized master, uncached session SHM | 11.96 |
| Optimized master, cached session SHM | 14.43 |

Removing packet copies helped materially. Persistent wire connections alone did
not improve this large-input, inference-heavy workload. Cached SHM was about
22–25% faster than previous-master per-call SHM, not the ~10× gain seen in a
transport-only synthetic test. Previous-master wire reruns before/after the
optimized cases differed by less than 1%; SHM varied by about 3%.

### Scheduling and decoding

| Configuration | Input mode | Sites/s | Mean utilization by GPU |
|---|---|---:|---|
| Model slots, one worker/model slot, cached SHM | Preloaded raw | 14.43 | 26%, 50%, 82%, 40% |
| Two workers/model slot, cached SHM | Preloaded raw | 15.34 | 31%, 59%, 97%, 46% |
| Four workers/model slot, cached SHM | Preloaded raw | 15.32 | 30%, 60%, 100%, 47% |
| One full-stack worker/GPU, cached SHM | Preloaded raw | 24.23 | 82%, 83%, 83%, 84% |
| One full-stack worker/GPU, cached SHM | TIFF decode + inference | 15.39 | 57%, 56%, 56%, 59% |
| Previous master model slots, fresh wire | MQ decode + inference | 7.01 | 13%, 24%, 38%, 19% |
| Previous master model slots, per-call SHM | MQ decode + inference | 11.33 | 21%, 41%, 63%, 30% |
| Optimized model slots, cached SHM | MQ decode + inference | 13.58 | 24%, 48%, 78%, 37% |
| One full-stack worker/GPU, cached SHM | MQ decode + inference | 22.72 | 70%, 70%, 70%, 71% |

Full-stack sharding improved load balance substantially while retaining exact
canary outputs. It used approximately 4.2 GiB reported device memory per GPU;
VRAM capacity was not the limiting resource. TIFF decoding-inclusive repeats
were variable (about 15–22 sites/s). More model replicas did not fix the
model-parallel OpenPhenom bottleneck. The useful next target is decoding /
prefetch / output-pipeline overlap, not indiscriminately adding GPU processes.

### Real-inference backend trial

| Schedule / input | Cached NNG SHM | PyZMQ multipart session |
|---|---:|---:|
| Model slots / preloaded raw | 14.43 | 14.28 |
| Full stack per GPU / preloaded raw | 24.23 | 24.18 |
| Model slots / MQ decode | 13.58 | 13.90 |
| Full stack per GPU / MQ decode | 22.72 | 20.69 |

The preloaded comparisons were effectively tied. Decode-inclusive cases were
more variable and do not establish a causal backend difference. All PyZMQ
canaries were exact too, but this prototype supplies no compelling real-workload
reason to replace NNG plus SHM. Keep the existing transport and focus first on
load balance and input overlap. PyZMQ in this environment was 27.1.0 / libzmq 4.3.5.

### Numerical and operational limits

- Fourteen completed configurations, 110 measured repeats and **35,200 measured
  model/site outputs** passed finite-value and accounting checks. Successful
  transport/scheduling canaries (including PyZMQ) were bit-identical to their
  matching raw or MQ baseline. Two rejected batching configurations are retained.
- Four-site and eight-site inference batches failed the existing strict gate:
  DINO-pair maximum differences were 0.00390697 and 0.00450843 respectively.
  They were rejected before timed measurement; tolerances were not loosened.
- The first MQ preparation attempt incorrectly required raw/MQ pixel correlation
  ≥0.95. Expected lossy-codec differences triggered it before GPU execution.
  The retained second preparation checks source identity, shape/dtype and best
  cross-channel correspondence instead; inference parity tolerances are unchanged.
- Historical `input_decode_cpu_seconds` fields in early raw reports contain summed
  **decoder-task wall time**, not CPU time. Current code names this correctly as
  `input_decode_task_wall_seconds_sum`; do not interpret the old field as CPU usage.

## Scope and caveats

These are warm, local-data decode/preprocess/inference/IPC measurements on a
shared host with CPU production jobs still running. They exclude remote downloads,
long-format profile serialization, receipt signing, production publication and
model startup from steady-state rates. Startup is recorded separately but is not
a cold-cache benchmark. They do **not** establish a production throughput or ETA.
Cases run in sequential windows, not randomized backend crossovers. Treat small
differences cautiously; no universal backend or speedup guarantee is claimed.

The optional PyZMQ adapter uses public multipart buffers and tracked sends with
the same CUDA processor. It is benchmark-only: no supported backend API, production
recovery policy, authentication or compatibility promise is introduced. Its server
loop is synchronous, unlike Nahual's async responder. See the result JSON for its
real-inference measurements; a backend decision must consider those limitations.

## Replaying locally

The staging launcher pins the existing environment rather than installing into it:

```bash
# Freeze a NEW cohort directory; never overwrite an earlier run.
# Run prepare_real_inputs.py / prepare_mq_inputs.py in the installed JUMP-lite
# environment (pyarrow, tifffile, zarr, imagecodecs and NumPy are required).

bash benchmarks/launch_real_gpu.sh \
  --output /work/scratch/amunoz/my-new-real-gpu-run \
  --label my-new-run --modes shm_session_cached --repeats 5

# Best tested scheduling; no changes to per-site inference or precision.
bash benchmarks/launch_real_gpu.sh \
  --output /work/scratch/amunoz/my-new-full-stack-run \
  --label my-new-full-stack --modes shm_session_cached \
  --schedule data_parallel --replicas 4 --repeats 5
```

Pass a matched MQ `--manifest` and `--decode zarr_mq` to measure actual MQ decode;
pass `--decode tiff` for the raw cohort. `--modes zmq_session` runs the optional
prototype separately. `NAHUAL_CODE_ROOT` selects an isolated baseline source tree.
The launcher and recipe paths are site-specific by design; inspect them before
replaying. Fresh output directories and initially idle GPUs are mandatory.

CPU-only CI tests validate framing, accounting, frozen-result consistency and
optional PyZMQ buffer lifetime/error handling. CI does not rerun the H100/data
experiment. Evidence includes a successful 87-file immutable recipe inventory
check. The collector verifies as-run harness snapshots, exact canary admission,
count closure and post-run input hashes.

Full local evidence, as-run harness snapshots, manifests, source/image hashes,
validation arrays, per-call/per-repeat reports, failures and GPU CSVs:

`/work/scratch/amunoz/nahual-real-gpu-20261102/`
