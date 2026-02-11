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
    # optional
    "pretrained": "False",
    "device": 1,
}
response = setup(parameters, address=address)
# print(dinov2.blocks[0].attn.qkv.weight[0, :1])

# %% Define custom data
# Added z-dimension
tile_size = 224  # multiples of 14
numpy.random.seed(seed=42)
data = numpy.random.random_sample((2, 3, 1, tile_size, tile_size))
result = process(data, address=address)
print(result)
# result[:10].sum()
# np.float32(26.936646)
# >>> result
# array([[-4.3235917 , -1.2983917 ,  0.5459024 , ..., -7.1253276 ,
#         -1.2375512 ,  0.9614856 ],
#        [-4.1552525 , -0.71603006, -0.09970876, ..., -6.0501924 ,
#         -0.5937261 ,  0.98573315]], shape=(2, 1000), dtype=float32)
