"""Run uniDINO through its Nahual server.

Start the server first:

    nix run github:afermg/uniDINO -- ipc:///tmp/unidino.ipc

The released checkpoint is not bundled. Download and extract it from
https://doi.org/10.5281/zenodo.14988837, then pass its absolute path below.
Omit ``weights`` only for a random-weight smoke test.
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("unidino")
address = "ipc:///tmp/unidino.ipc"

# %% Load the single-channel ViT-S/16 server-side.
parameters = {
    # "weights": "/absolute/path/to/checkpoints/uniDINO.pth",
    # "device": 0,
}
response = setup(parameters, address=address)
print(response)

# %% NCZYX float input in [0, 1]. Z must be 1; channel order is preserved.
tile_size = 224
channels = 5
numpy.random.seed(seed=42)
data = numpy.random.random_sample((2, channels, 1, tile_size, tile_size))
result = process(data, address=address)
print(result.shape)
# Expected: (2, 1920), i.e. 384 features per input channel.
