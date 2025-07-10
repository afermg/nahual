"""
This client matches https://github.com/afermg/trackastra/blob/main/server.py
"""

import numpy

from nahual.client.trackastra import load_model, process_data

img = numpy.random.randint(100, size=(20, 100, 100))
masks = numpy.zeros_like(img, dtype=int)
masks[:, 20:40, 20:40] = numpy.random.randint(20, size=20) + 1

address = "ipc:///tmp/example_name.ipc"
parameters = {"model": "general_2d", "mode": "greedy"}
model_info = load_model(parameters, address=address)
result = process_data((img, masks), address=address)
