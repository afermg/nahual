# Nahual: Deploy and access image and data processing models across environments/processes.

Note that this is early work in progress.

This tool aims to provide a one-stop-shop source for multiple models to process imaging data or their derivatives. You can think of it as a much simpler [ollama](https://github.com/ollama/ollama) but for biological analyses, deep learning-based or otherwise.

## Implemented tools 
By default, the models and tools are deployable using [Nix](https://nixos.org/).

- [trackastra](https://github.com/afermg/trackastra): Transformer-based models trained on a multitude of datasets.

## WIP tools
- [Baby](https://github.com/afermg/baby): Segmentation, tracking and lineage assignment for budding yeast.
- [DINOv2](https://github.com/afermg/dinov2): Generalistic self-supervised model to obtain visual features.

## Minimal example for FastAPI-based server+client
	Any model requires a thin layer that communicates using [[https://github.com/nanomsg/nng][nng]]. You can see an example of trackastra's [[https://github.com/afermg/trackastra/blob/main/server.py][server]] and [[https://github.com/afermg/nahual/blob/master/src/nahual/clients/trackastra.py][client]].
	
## Future goals
- Support multiple instances of a model loaded on memory server-side.
- Formalize supported packet formats: (e.g., numpy arrays, dictionary).
- Increase number of supported models/methods.	
- Document server-side API
- Integrate into the [[github.com/afermg/aliby][aliby]] pipelining framework.

## Why nahual?
![logo](logo.svg)

In Mesoamerican folklore, a Nahual is a shaman able to transform into different animals.

