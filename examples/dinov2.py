"""
This example uses a server within the environment defined on `https://github.com/afermg/dinov2.git`.

Run `nix develop --command bash -c "python server.py ipc:///tmp/dinov2.ipc"` from the root directory of that repository.
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("dinov2")
address = "ipc:///tmp/dinov2.ipc"
i = 0

# %%Load models server-side
parameters = {"repo_or_dir": "facebookresearch/dinov2", "model": "dinov2_vits14_lc"}
response = setup(parameters, address=address)

# %% Define custom data
data = numpy.random.random_sample((1, 3, 420 + 14 * i, 420))
i += 1
result = None
result = process(data + 1000, address=address)
print(result[:10, :10])
if result is not None:
    prev = result.copy()
# %%
