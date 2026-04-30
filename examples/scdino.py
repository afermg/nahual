"""
This example uses a server within the environment defined on `https://github.com/afermg/scDINO.git`.

Run `nix run github:afermg/scDINO -- ipc:///tmp/scdino.ipc` from the root directory of that repository.
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("scdino", signature=("dict", "numpy"))
address = "ipc:///tmp/scdino.ipc"

# %% Load model server-side (random init by default; pass `weights` for a checkpoint).
parameters = {
    "arch": "vit_small",
    "patch_size": 16,
    "in_chans": 3,
    # optional
    # "weights": "/path/to/checkpoint.pth",
    # "checkpoint_key": "teacher",
    # "device": 0,
}
response = setup(parameters, address=address)

# %% Define custom data — NCZYX (Z is dropped server-side before forward).
tile_size = 224
numpy.random.seed(seed=42)
data = numpy.random.random_sample((2, 3, 1, tile_size, tile_size))
result = process(data, address=address)
print(result.shape)
# Expected: (2, 384)
