<div align="center">
<img src="./logo.svg" width="150px">
</div>

# Nahual: Communication layer to send and transform data across environments and/or processes.

The problem: When trying to train, compare and deploy many different models (deep learning or otherwise), the number of dependencies in one Python environment can get out of control very quickly (e.g., one model requires PyTorch 2.1 and another one 2.7). 

Potential solution: I figured that if we can move parameters and numpy arrays between environments, we can isolate each model and having them process our data on-demand. 

Thus the goal of this tool is provide a way to deploy model(s) in one (or many) environments, and access them from another one, usually an orchestrator.

## Available models and tools 
I deployed tools using [Nix](https://nixos.org/).

- [BABY](https://github.com/afermg/baby): Segmentation, tracking and lineage assignment for budding yeast.
- [Cellpose](https://github.com/afermg/cellpose): Generalist segmentation model.
- [DINOv2](https://github.com/afermg/dinov2): Generalist self-supervised model to obtain visual features.
- [Trackastra](https://github.com/afermg/trackastra): Transformer-based tracking trained on a multitude of datasets.
- [ViT](https://github.com/afermg/nahual_vit): HuggingFace's Visual Transformers models (e.g., [OpenPhenom](https://huggingface.co/recursionpharma/OpenPhenom)). 
- [SubCell](https://github.com/afermg/SubCellPortable): Encoder of single cell morphology and protein localisation.
- [DINOv3](https://github.com/afermg/dinov3): Generalist self-supervised model, latest iteration.

## Future supported tools
- [DeepProfiler](https://github.com/cytomining/DeepProfiler)
- [scDINO](https://github.com/JacobHanimann/scDINO)

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

You can press `C-c C-c` from the terminal where the server lives to kill it. We will also add a way to kill the server from within the client.

## Design decisions and details
I strive to be as lean as possible (both in dependency count and architectural complexity), it is designed around three layers:

- Server deployment: A collection of functions/tool (we could even call it a "model zoo" if we are trying to sound cool) that we may want to use, (e.g., Cellpose for object segmentation or Trackastra for tracking).
- Transport layer: We need to move the data between environments. I also wrote my own (trivially simple) numpy serializer. Since we have Python at both ends of the connection, we can reuse these functions server-side.
- Orchestration: This can be a script, or my own pipelining framework [aliby](https://github.com/afermg/aliby), massages the data into the desired shape/type, and then hands it over to `nahual`.

This tool is my personal one-stop-shop source for multiple models to process imaging data or their derivatives. Please note that this is work in progress, and very likely to undergo major changes as I understand the core challenges.

To reduce maintenance burden, we support only the necessary data types:
- Dictionaries: To send parameters to deploy and evaluate models/functions.
- Numpy arrays (and numpy-able lists/tuples): The main type of data we deal with.

### Tech stack 
- Model/tool deployment I use [Nix](https://nixos.org/), and at the moment do not plan to support containers. The logic behind  gives me unique guarantees of reproducibility, whilst allowing me to use bleeding edge models and libraries.
- Transport layer I use [pynng](github.com/codypiersall/pynng), I like that it is very minimalistic and provides easy-to-reproduce [https://github.com/codypiersall/pynng/tree/7fd3d76573c3cb40c1e5f7e10d4a6091e411b9c2/examples](examples). An alternative would have been `gRPC` + `protobuf`, but since I am trying to understand the constraints and tradeoffs I do not want to commit to a big framework unless I have a compelling reason to do so.

## Adding support for new models
Any model requires a thin layer that communicates using [nng](https://github.com/nanomsg/nng). You can see an example of trackastra's [server](https://github.com/afermg/trackastra/blob/main/server.py) and [client](./examples/trackastra.py).
	
## Roadmap
- Support multiple instances of a model loaded on memory server-side.
- Formalize supported packet formats: (e.g., numpy arrays, dictionary).
- Increase number of supported models/methods.	
- Document server-side API.
- Integrate into the [aliby](github.com/afermg/aliby) pipelining framework, in a way that is agnostic to which model is being used.
- Support containers that wrap the Nix derivations.

## Why nahual?
In Mesoamerican folklore, a Nahual is a shaman able to transform into different animals.

