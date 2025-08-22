<div align="center">
<img src="./logo.svg" width="150px">
</div>

# Nahual: Deploy and access image and data processing models across environments/processes.

Note that this is early work in progress.

This tool aims to provide a one-stop-shop source for multiple models to process imaging data or their derivatives. You can think of it as a much simpler [ollama](https://github.com/ollama/ollama) but for biological analyses, deep learning-based or otherwise.

## Implemented models and tools 
By default, the models and tools are deployable using [Nix](https://nixos.org/).

- [BABY](https://github.com/afermg/baby): Segmentation, tracking and lineage assignment for budding yeast.
- [Cellpose](https://github.com/afermg/cellpose): Generalist segmentation model.
- [DINOv2](https://github.com/afermg/dinov2): Generalist self-supervised model to obtain visual features.
- [Trackastra](https://github.com/afermg/trackastra): Transformer-based models trained on a multitude of datasets.

## WIP
- [DINOv3](https://github.com/afermg/dinov3): Generalist self-supervised model, latest iteration.

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

from nahual.client.dinov2 import load_model, process_data

address = "ipc:///tmp/example_name.ipc"

# Load models server-side
parameters = {"repo_or_dir": "facebookresearch/dinov2", "model": "dinov2_vits14_lc"}
load_model(parameters, address=address)

# Define custom data
data = numpy.random.random_sample((1, 3, 420, 420))
result = process_data(data, address=address)
```

You can press `C-c C-c` from the terminal where the server lives to kill it. We will also add a way to kill the server from within the client.

## Adding support for new models
Any model requires a thin layer that communicates using [nng](https://github.com/nanomsg/nng). You can see an example of trackastra's [server](https://github.com/afermg/trackastra/blob/main/server.py) and [client](https://github.com/afermg/nahual/blob/master/src/nahual.client/trackastra.py).
	
## Roadmap
- Support multiple instances of a model loaded on memory server-side.
- Formalize supported packet formats: (e.g., numpy arrays, dictionary).
- Increase number of supported models/methods.	
- Document server-side API.
- Integrate into the [aliby](github.com/afermg/aliby) pipelining framework.
- Support containers that wrap the Nix derivations.

## Why nahual?
In Mesoamerican folklore, a Nahual is a shaman able to transform into different animals.

