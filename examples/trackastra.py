"""
This client matches https://github.com/afermg/trackastra/blob/main/server.py
"""

import numpy

import nahual.client.trackastra as tr

img = numpy.random.randint(100, size=(20, 100, 100))
masks = numpy.zeros_like(img, dtype=int)
masks[:, 20:40, 20:40] = numpy.random.randint(20, size=20) + 1

# %%
address = "ipc:///tmp/trackastra.ipc"
parameters = {"model": "general_2d", "mode": "greedy"}
model_info = tr.load_model(parameters, address=address)
# %%
result = tr.process_data((img, masks), address=address)
# %%
