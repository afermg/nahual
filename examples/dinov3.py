# +/usr/bin/env python
"""
This example uses a server within the environment defined on `https://github.com/afermg/dinov3.git`.

Run `nix run github:afermg/dinov3 -- ipc:///tmp/dinov3.ipc` from any directory,
or `nix develop --command bash -c "python server.py ipc:///tmp/dinov3.ipc"` from
the root of that repository.
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("dinov3", signature=("dict", "numpy"))
address = "ipc:///tmp/dinov3.ipc"

# %%Load models server-side
parameters = {
    "model_name": "dinov3_vits16",
    # optional
    "pretrained": False,
    "device": 0,
}
response = setup(parameters, address=address)

# %% Define custom data
# Added z-dimension
tile_size = 224  # multiples of 16
numpy.random.seed(seed=42)
data = numpy.random.random_sample((2, 3, 1, tile_size, tile_size))
result = process(data, address=address)
print(result.shape)
