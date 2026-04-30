"""
This example uses a server within the environment defined on `https://github.com/afermg/DeepProfiler.git`.

Run `nix run --impure github:afermg/DeepProfiler -- ipc:///tmp/deepprofiler.ipc`
from the root directory of that repository (or `nix develop --impure --command
python server.py ipc:///tmp/deepprofiler.ipc`).

Note: DeepProfiler's CPCNNv1 backbone is a Keras ResNet50V2 (TF 2.13 + tf-keras
2.17 in legacy mode) that emits a 2048-d feature embedding per single-cell crop.
Because "deepprofiler" is not one of the built-in signature names in
nahual.process, we pass ``signature=("dict", "numpy")`` explicitly.
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process(
    "deepprofiler", signature=("dict", "numpy")
)
address = "ipc:///tmp/deepprofiler.ipc"

# %% Load model server-side (server defaults: ResNet50V2 + ImageNet weights).
parameters = {}
response = setup(parameters, address=address)
print(response)

# %% Define custom data — NCZYX, with C=3 (RGB) and Z=1.
tile_size = 224
numpy.random.seed(seed=42)
data = numpy.random.random_sample((2, 3, 1, tile_size, tile_size))
result = process(data, address=address)
print(result.shape)
# Expected: (2, 2048)
