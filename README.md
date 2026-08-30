<div align="center">
<img src="./logo.svg" width="150px">
</div>

# Nahual: Communication layer to send and transform data across environments and/or processes.

The problem: When trying to train, compare and deploy many different models (deep learning or otherwise), the number of dependencies in one Python environment can get out of control very quickly (e.g., one model requires PyTorch 2.1 and another one 2.7). 

Potential solution: I figured that if we can move parameters and numpy arrays between environments, we can isolate each model and having them process our data on-demand. 

Thus the goal of this tool is provide a way to deploy model(s) in one (or many) environments, and access them from another one, usually an orchestrator.

## Available models and tools

All wraps are deployed with [Nix](https://nixos.org/) and run on GPU (`cuda:0` or `GPU:0`). Launch any of them with `nix run github:afermg/<repo> -- ipc:///tmp/<name>.ipc`. With `nahual >= 0.0.9` a single server can host multiple `setup()` calls — re-call setup with a new dict to swap models without restarting.

### Embeddings / feature extraction

| Model | Output | Notes |
|---|---|---|
| [DINOv2](https://github.com/afermg/dinov2) | `(N, D · ⌈C/3⌉)` cls token — D = 384 (S/14), 768 (B/14), 1024 (L/14), 1536 (G/14) | Generalist self-supervised visual features. Rigid 3-channel ImageNet backbone; inputs with C ≠ 3 run through `⌈C/3⌉` passes (recycling leading channels in the trailing chunk) and per-chunk cls tokens are concatenated. |
| [DINOv3](https://github.com/afermg/dinov3) | `(N, D · ⌈C/3⌉)` cls token — D = 384 (S/16), 768 (B/16), 1024 (L/16), 1536 (G/16) | Latest iteration of DINO. Same multi-pass channel handling as DINOv2. Direct factory imports (skips `torch.hub.load`). |
| [ViT](https://github.com/afermg/nahual_vit) | `(N, 384)` (OpenPhenom); `(N, 384 × C)` (MorphEm, per-channel cls concatenated) | HuggingFace ViTs incl. [OpenPhenom](https://huggingface.co/recursionpharma/OpenPhenom) and [MorphEM](https://huggingface.co/CaicedoLab/MorphEm). |
| [SubCell](https://github.com/afermg/SubCellPortable) | `(N, 1536)` | Single-cell morphology + protein-localisation encoder. |
| [scDINO](https://github.com/afermg/scDINO) | `(N, 384)` | Self-supervised ViT-S/B for multi-channel single-cell images. |
| [uniDINO](https://github.com/afermg/uniDINO) | `(N, 384 × C)` | Assay-independent fluorescence-microscopy ViT. Each channel is embedded independently with the shared single-channel backbone, then the cls tokens are concatenated in channel order. Official weights are separately licensed CC BY-NC-ND 4.0 (non-commercial, no derivatives), Pooch-cached on request or available through an opt-in Nix app, and excluded from the default closure. |
| [ChannelSFormer](https://github.com/afermg/ChannelSFormer) | `(N, 384)` | Channel-agnostic ViT for cell-painting (insitro). |
| [DeepProfiler (CPCNNv1)](https://github.com/afermg/DeepProfiler) | `(N, 2048 · ⌈C/3⌉)` | TensorFlow ResNet50V2 ImageNet morphological profiler. Same multi-pass channel handling as DINOv2/v3. |

### Segmentation

| Model | Output | Notes |
|---|---|---|
| [BABY](https://github.com/afermg/baby) | yeast labels + lineage | Budding-yeast seg, tracking, lineage. |
| [Cellpose](https://github.com/afermg/cellpose) | `(H, W)` instance mask | Generalist segmentation. |
| [StarDist](https://github.com/afermg/stardist) | `(N, H, W)` int32 | Star-convex polygon segmentation, TF backend. |
| [EmbedSeg](https://github.com/afermg/EmbedSeg) | `(N, H, W)` int32 | Embedding-based instance segmentation (PyTorch). |
| [InstanSeg](https://github.com/afermg/instanseg) | `(N, H, W)` int32 | Fast cell segmentation across biomarkers. |
| [MegaSeg](https://github.com/afermg/allencell-segmenter-ml) | `(N, 1, Z, Y, X)` uint8 | Allen Institute MegaSegmenter — 3-D, Hydra/napari-free inference. |
| [micro-sam](https://github.com/afermg/micro-sam) | `(N, H, W)` int32 | SAM tuned for microscopy. All conda-only deps (vigra, nifty, affogato, torch_em, python-elf) packaged as proper Nix derivations. Cold-cache build ~30 min — pre-warm with `nix develop --impure --command true`. |
| [CellSAM](https://github.com/afermg/cellSAM) | `(N, H, W)` int32 | ONNX-only foundation model, no auth. Backed by [keejkrej/cellsam-onnx](https://huggingface.co/keejkrej/cellsam-onnx); license is *Modified Apache 2.0, academic-only*. The original DeepCell-auth PyTorch path is preserved on the [`nahual-wrap-deepcell`](https://github.com/afermg/cellSAM/tree/nahual-wrap-deepcell) branch. |
| [Spotiflow](https://github.com/afermg/spotiflow/tree/nahual-wrap) | `(N, H, W)` int32 label mask, one disk per spot | Fluorescence-puncta detector ([Weigert lab](https://github.com/weigertlab/spotiflow)). Server rasterises Spotiflow's `(N, 2)` `(y, x)` centroid output into per-spot disks (configurable radius, default 3 px) so the result drops into any downstream cp_measure / skimage.measure pipeline that already consumes cellpose-style label masks. Default checkpoint: `general`. Pass `signature=("dict", "numpy")` to `dispatch_setup_process`. |

### Tracking

| Model | Output | Notes |
|---|---|---|
| [Trackastra](https://github.com/afermg/trackastra) | track IDs across timepoints | Transformer-based tracking. |
| [Ultrack](https://github.com/afermg/ultrack) | `(T, Z, Y, X)` int32 | ILP-based tracking + segmentation. Tracking core CPU-bound (CBC/CLP solver); optional torch detection nets are GPU-capable. |

### Generic loaders

| Model | Output | Notes |
|---|---|---|
| [BioImage Model Zoo](https://github.com/afermg/nahual_bioimageio) | depends on RDF | One server, any RDF identifier (DOI / Zenodo URL / nickname like `affable-shark` / local rdf.yaml). Four GPU-validated flake variants: `apps.default` (ONNX/TorchScript), `apps.with-careamics`, `apps.with-stardist`, `apps.with-monai`. 21 well-known BIMZ models pre-validated; see the repo README for the full table. TF 1.15 SavedModels can't load (bioimageio.core 0.10.2 routes through Keras 3 `TFSMLayer`); RDFs that publish only `pytorch_state_dict` aren't usable through `default` (use a model-specific wrap from above). |

## Wrapped, outside the supported categories

These models are deployable through Nahual today (same `setup` / `process` API,
same Nix-launched server) but don't fit any of the categories above and aren't
covered by the project's supported scope — input/output conventions differ, and
no guarantees are made about keeping them aligned with future API changes.
Listed here so users can find them, not as recommended building blocks.

| Model | Output | Notes |
|---|---|---|
| [CellWhisperer](https://github.com/afermg/CellWhisperer) | `(N_cells, hidden_size)` | Multimodal scRNA-seq + language model — input is `(N_cells, N_genes)`, not NCZYX. Single-cell transcriptomics, not imaging. |

## Considered but not wrapped

- **ilastik** — interactive ML pipeline (Qt-based), not a single-shot inference model.
- **MCMICRO** — Nextflow pipeline orchestrator; doesn't fit the single-server pattern.
- **Cytoself** ([afermg/cytoself](https://github.com/afermg/cytoself)) — VQ-VAE produces spatial token grids `(N, 64, H, W)` rather than flat embeddings.
- **CellDino** ([afermg/CellDino](https://github.com/afermg/CellDino)) — Mask-DINO instance-seg + tracking; upstream has not released pretrained weights, and the inference path needs CUDA-compiled `MSDeformAttn` + mmcv extensions that are non-trivial to package under Nix.
- **Micronucleus detector / CHAMMI-75 / Virtual staining** — discussion-mentioned but no public upstream URL was provided.

## Usage
### Step 1: Deploy server
`cd` to the model you want to deploy. In this case we will test the image embedding model DINOv2.

```bash
git clone https://github.com/afermg/dinov2.git
cd dinov2
nix develop --command bash -c "python server.py ipc:///tmp/dinov2.ipc"
```

### Step 2: Run client
Once the server is running, you can call it from a different python script.
```python
import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("dinov2")
address = "ipc:///tmp/dinov2.ipc"

# %%Load models server-side
parameters = {"repo_or_dir": "facebookresearch/dinov2", "model": "dinov2_vits14_lc"}
response = setup(parameters, address=address)

# %% Define custom data
data = numpy.random.random_sample((1, 3, 420, 420))
result = process(data + 1000, address=address)
```

### Optional local shared-memory data plane

For a client and server on the same Linux host, NumPy process calls can opt into
shared memory while NNG remains the control transport:

```python
result = process(data, address=address, shared_memory=True)
```

This stages an ordinary NumPy input into a per-call client-owned segment. The
server receives a synchronous, read-only borrowed view, and may reuse that
segment for a fitting output. Nahual always returns an owning NumPy result and
cleans up its segment. Setup dictionaries, parameters, model information,
errors, and the small shared-memory descriptor continue to travel over NNG.

When an upstream producer can fill shared memory directly, install the optional
Appose adapter:

```bash
pip install 'nahual[appose]>=0.0.11'
```

Both client and model server must use Nahual 0.0.11 or newer. Nix model
wrappers pinned to an older Nahual revision must update that pin before this
protocol can be used.

```python
import appose

with appose.NDArray("float32", [1, 5, 1, 128, 128]) as shared_input:
    view = shared_input.ndarray()
    acquire_or_preprocess_directly_into(view)
    result = process(
        shared_input,
        address=address,
        shared_memory=True,
        timeout_ms=-1,
    )
```

Nahual borrows an Appose-owned input without staging, and never closes, unlinks,
disposes, or overwrites it. Appose input v1 always returns the output through
the ordinary NNG NumPy wire as an owning array. The caller must keep the input
alive and must not mutate or reuse it until the synchronous call completes;
`timeout_ms=-1` is required so a timeout cannot outlive that exclusive borrow.
Copying an existing NumPy array into `shared_input.ndarray()` is still one full
staging copy, so the largest gain comes when acquisition or preprocessing writes
there directly. See [`examples/appose_local.py`](examples/appose_local.py).

This is a **local shared-memory data plane with optional Appose-owned input
buffers**, not an Appose transport or backend. It initially supports Linux
CPython 3.9–3.13 and only `ipc://` endpoints with NumPy process input/output.
The IPC endpoint and operating-system shared-memory permissions are a trusted
same-user boundary; do not expose this mode to untrusted local clients.
Processors must not retain the borrowed server-side input view. Appose
`NDArray` is host memory, not CUDA memory: GPU models still copy input from host
to device, although they can keep models and intermediate state resident on the
GPU.

You can press `C-c C-c` from the terminal where the server lives to kill it. We will also add a way to kill the server from within the client.

## Design decisions and details
I strive to be as lean as possible (both in dependency count and architectural complexity), it is designed around three layers:

- Server deployment: A collection of functions/tool (we could even call it a "model zoo" if we are trying to sound cool) that we may want to use, (e.g., Cellpose for object segmentation or Trackastra for tracking).
- Transport layer: We need to move the data between environments. I also wrote my own (trivially simple) numpy serializer. Since we have Python at both ends of the connection, we can reuse these functions server-side.
- Orchestration: This can be a script, or my own pipelining framework [aliby](https://github.com/afermg/aliby), massages the data into the desired shape/type, and then hands it over to `nahual`.

This tool is my personal one-stop-shop source for multiple models to process imaging data or their derivatives. Please note that this is work in progress, and very likely to undergo major changes as I develop a better understanding of the main challenges.

To reduce maintenance burden, we support only the necessary data types:
- Dictionaries: To send parameters to deploy and evaluate models/functions.
- Numpy arrays (and numpy-able lists/tuples): The main type of data we deal with.

### Tech stack 
- For model/tool deployment I use [Nix](https://nixos.org/), which gives me unique guarantees of reproducibility while allowing me to use bleeding-edge models and libraries. Implementation of OCI container support is coming.
- Transport layer I use [pynng](github.com/codypiersall/pynng), I like that it is very minimalistic and provides easy-to-reproduce [examples](https://github.com/codypiersall/pynng/tree/7fd3d76573c3cb40c1e5f7e10d4a6091e411b9c2/examples). NNG remains the standard transport even when the optional local shared-memory data plane is enabled. An alternative would have been `gRPC` + `protobuf`, but since I am trying to understand the constraints and tradeoffs I do not want to commit to a big framework unless I have a compelling reason to do so.

## Adding support for new models
Any model requires a thin layer that communicates using [nng](https://github.com/nanomsg/nng). You can see an example of trackastra's [server](https://github.com/afermg/trackastra/blob/main/server.py) and [client](./examples/trackastra.py).
	
## Roadmap
- Formalize supported packet formats: (e.g., numpy arrays, dictionary).
- Document server-side API.
- Integrate into the [aliby](github.com/afermg/aliby) pipelining framework, in a way that is agnostic to which model is being used.
- Implement OCI containers that wrap the Nix derivations.

## Why nahual?
In Mesoamerican folklore, a Nahual is a shaman able to transform into different animals.

