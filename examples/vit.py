"""
This example uses a server within the environment defined on `https://github.com/afermg/nahual_vit.git`.

Run `nix develop --command bash -c "python server.py ipc:///tmp/openphenom.ipc"` from the root directory of that repository.
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("vit")
address = "ipc:///tmp/vit.ipc"
"""
Run Visual Transformers using the HuggingFace's transformers library.

Run `nix develop --command bash -c "python src/vit/server.py ipc:///tmp/vit.ipc"` from the root directory of https://github.com/afermg/nahual_vit.
"""

# %%Load models server-side
parameters = dict(model_name="recursionpharma/OpenPhenom")
response = setup(parameters, address=address)

# %% Define custom data
tile_size = 256
data = numpy.random.random_sample((2, 6, 1, tile_size, tile_size))
result = process(data, address=address)
