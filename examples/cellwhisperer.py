"""
This example uses a server within the environment defined on `https://github.com/afermg/CellWhisperer.git`.

Run `nix run github:afermg/CellWhisperer -- ipc:///tmp/cellwhisperer.ipc` from the root directory of that repository.

Note: CellWhisperer is a single-cell transcriptomics model. The Nahual interface
here takes a 2-D ``(N_cells, N_genes)`` numpy array (not the 5-D NCZYX images
used by image-based models) and returns a per-cell embedding of shape
``(N_cells, hidden_size)``. Because "cellwhisperer" is not one of the built-in
signature names in nahual.process, we pass ``signature=("dict", "numpy")``
explicitly.
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process(
    "cellwhisperer", signature=("dict", "numpy")
)
address = "ipc:///tmp/cellwhisperer.ipc"

# %% Load model server-side (random-init scaffold encoder by default).
parameters = {
    "hidden_size": 256,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 512,
    "max_position_embeddings": 4096,
    "vocab_size": 25426,
    # optional
    # "weights": "/path/to/cellwhisperer_clip_v1.ckpt",
    # "device": 0,
}
response = setup(parameters, address=address)
print(response)

# %% Define custom data — (N_cells, N_genes), NOT 5-D NCZYX.
numpy.random.seed(seed=42)
data = numpy.random.random_sample((4, 2000)).astype(numpy.float32)
result = process(data, address=address)
print(result.shape)
# Expected: (4, 256)
