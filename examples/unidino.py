"""Run uniDINO through its Nahual server.

Start the lightweight server first:

    nix run github:afermg/uniDINO -- ipc:///tmp/unidino.ipc

With ``pretrained=True``, the server downloads the official Zenodo checkpoint
once and caches the verified file through Pooch. Alternatively, put the
checkpoint in the Nix store before serving it:

    nix run github:afermg/uniDINO#pretrained -- ipc:///tmp/unidino.ipc
The weights are separately licensed CC BY-NC-ND 4.0.
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("unidino")
address = "ipc:///tmp/unidino.ipc"

# %% Load the official single-channel ViT-S/16 teacher backbone.
# `cache=True` is the default, but is explicit to make the behavior visible.
parameters = {
    "pretrained": True,
    "cache": True,
    # "cache_dir": "/custom/cache/directory",
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
