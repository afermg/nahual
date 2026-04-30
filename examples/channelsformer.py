"""
This example uses a server within the environment defined on `https://github.com/afermg/ChannelSFormer.git`.

Run `nix run --impure github:afermg/ChannelSFormer -- ipc:///tmp/channelsformer.ipc` from any directory, or
`nix develop --command bash -c "python server.py ipc:///tmp/channelsformer.ipc"` from the root directory of that repository.
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("channelsformer")
address = "ipc:///tmp/channelsformer.ipc"

# %% Load model server-side
parameters = {
    "img_size": 224,
    "patch_size": 16,
    "in_chans": 5,
    "embed_dim": 384,
    "depth": 12,
    "num_heads": 6,
    # optional
    # "device": 0,
    # "weights": "/path/to/checkpoint.pth",
}
response = setup(parameters, address=address)

# %% Define custom data
# NCZYX, 5 channels, single Z slice, 224x224 tile (multiples of patch_size=16).
tile_size = 224
numpy.random.seed(seed=42)
data = numpy.random.random_sample((2, 5, 1, tile_size, tile_size))
result = process(data, address=address)
print(result.shape)
# Expected: (2, 384)
