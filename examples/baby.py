"""
This client matches baby-phone's API (https://github.com/afermg/baby). Run baby phone using `baby-phone` on the CLI.
"""

import numpy

import nahual.client.baby as bb

seed = 42
address = "http://0.0.0.0:5101"  # URL to reach baby-phone
modelset = "yeast-alcatras-brightfield-sCMOS-60x-1z"

rng = numpy.random.default_rng(seed)

running_sessions = bb.list_sessions(address)
session_id = (
    running_sessions[0]
    if len(running_sessions) > 0
    else bb.load_model(address, modelset)
)

# Create suitable N x H x W x Z array
# dtype must be either uint8 or uint16
img = rng.integers(2**8, size=(2, 80, 120, 1), dtype="uint8")

output = bb.process_data(
    img,
    address,
    session_id,
)
