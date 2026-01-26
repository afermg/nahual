# +/usr/bin/env python
"""
This example uses a server within the environment defined on `https://github.com/afermg/dinov2.git`.

Run `nix develop --command bash -c "python server.py ipc:///tmp/dinov2.ipc"` from the root directory of that repository.
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("dinov2")
address = "ipc:///tmp/dinov2.ipc"

# %%Load models server-side
parameters = {
    "repo_or_dir": "facebookresearch/dinov2",
    "model_name": "dinov2_vits14_lc",
}
response = setup(parameters, address=address)

# %% Define custom data
# Added z-dimension
tile_size = 224  # multiples of 14
data = numpy.random.random_sample((2, 3, 1, tile_size, tile_size))
result = process(data, address=address)
