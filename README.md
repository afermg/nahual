# Nahual: Deploy and access image and data processing models across environments/processes.

Note that this is early work in progress.

This tool aims to provide a one-stop-shop source for multiple models to process imaging data or their derivatives. You can think of it as a much simpler [ollama](https://github.com/ollama/ollama) but for biological analyses, deep learning-based or otherwise.

## Implemented tools 
By default, the models and tools are deployable using [Nix](https://nixos.org/).

- [Baby](https://github.com/afermg/baby): Segmentation, tracking and lineage assignment for budding yeast.

## WIP tools
- [trackastra](https://github.com/afermg/trackastra): Transformer-based models trained on a multitude of datasets.
- [DINOv2](https://github.com/afermg/dinov2): Generalistic self-supervised model to obtain visual features.

## Minimal example for FastAPI-based server+client
	Any model requires a thin layer that communicates using [[https://github.com/nanomsg/nng][nng]].
	
This is the server side
```python
import numpy
import orjson
from fastapi import FastAPI, Request, Response

app = FastAPI()

@app.post("/process")
async def process(request: Request):
    # Convert list to numpy array
    array = numpy.asarray(orjson.loads(await request.body()))
    # Example processing, here is where processing is performed
    result = array * 2
    return Response(
        orjson.dumps(result, option=orjson.OPT_SERIALIZE_NUMPY),
    )
```

This is the client side
```python
import numpy
import orjson
import requests

# Serialize a numpy array using orjson (faster json serialization)
serial_numpy = orjson.dumps(
    numpy.array([[1, 2], [3, 4]]),
    option=orjson.OPT_SERIALIZE_NUMPY,
)
response = requests.post(
    "http://localhost:8000/process",
    serial_numpy,
)
print(orjson.loads(response.content))
# [[2, 4], [6, 8]]

```

## Why nahual?
![logo](logo.svg)

In Mesoamerican folklore, a Nahual is a shaman able to transform into different animals.

