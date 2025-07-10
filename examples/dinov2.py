"""
This example uses a server within the environment defined on `https://github.com/afermg/dinov2.git`.

Run `nix develop --command bash -c "python server.py ipc:///tmp/example_name.ipc"` from the root directory of that repository.
"""

import numpy

from nahual.clients.dinov2 import load_model, process_data

address = "ipc:///tmp/example_name.ipc"

# Load models server-side
parameters = {"repo_or_dir": "facebookresearch/dinov2", "model": "dinov2_vits14_lc"}
load_model(parameters, address=address)

# Define custom data
data = numpy.random.random_sample((1, 3, 420, 420))
result = process_data(data, address=address)
