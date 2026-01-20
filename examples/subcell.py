"""
This example uses a server within the environment defined on `https://github.com/afermg/SubCellPortable.git`. You can read the `server.py` file at the root directory.

Run `nix run github:afermg/subcellportable` from the root directory of that repository.
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("subcell")
address = "ipc:///tmp/subcell.ipc"

# %%Load models server-side
parameters = dict(model_type="mae_contrast_supcon_model", model_channels="rybg")
response = setup(parameters, address=address)

# %% Define custom data
data = numpy.random.random_sample((2, 4, 1, 256, 256))
result = process(data, address=address)
