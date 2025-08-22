"""
This client matches https://github.com/afermg/trackastra/blob/main/server.py
"""

import numpy

from nahual.process import dispatch_setup_process

img = numpy.random.randint(100, size=(20, 100, 100))
masks = numpy.zeros_like(img, dtype=int)
masks[:, 20:40, 20:40] = numpy.random.randint(20, size=20) + 1

setup, process = dispatch_setup_process("trackastra")

# %%
address = "ipc:///tmp/trackastra.ipc"
parameters = {"model": "general_2d", "mode": "greedy"}
model_info = setup(parameters, address=address)
# %%
result = process((img, masks), address=address)
# %%
