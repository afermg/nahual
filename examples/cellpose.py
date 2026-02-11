# +/usr/bin/env python
"""
This example uses a server within the environment defined on `https://github.com/afermg/dinov2.git`.

Run `nix run github:afermg/cellpose/2146b5ee3b2c7eb2c826efe7a24b3b289432500b -- ipc:///tmp/cellpose.ipc"` from the root directory of that repository.
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("cellpose")
address = "ipc:///tmp/cellpose.ipc"

# %%Load models server-side
parameters = {"device": 1}
response = setup(parameters, address=address)
print(response)
# Loaded model with parameters {'setup': {'device': 'cuda:1', 'gpu': 'True'}, 'execution': {'return_2d': 'True', 'z_axis': '0', 'stitch_threshold': '0.1'}}

# %% Define custom data
tile_size = 1024
numpy.random.seed(seed=42)
data = numpy.random.random_sample((1, tile_size, tile_size))
result = process(data, address=address)
print(f"Shape: {result.shape}, Max: {result.max()}")
# Shape: (1024, 1024), Max: 0
