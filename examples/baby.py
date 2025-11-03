"""
This client matches baby-phone's API (https://github.com/afermg/baby). Run baby phone using `baby-phone` on the CLI.

Note that this example is quite different from the rest, due to legacy code reasons.
"""

from pathlib import Path

import numpy
import nahual.client.baby as bb
from PIL import Image

seed = 42
address = "http://0.0.0.0:5101"  # URL to reach baby-phone
modelset = "yeast-alcatras-brightfield-sCMOS-60x-1z"
running_sessions = bb.list_sessions(address)
session_id = (
    running_sessions[0]
    if len(running_sessions) > 0
    else bb.load_model(address, modelset)
)

path = Path("/datastore/alan/aliby/PDR5_GFP_100ugml_flc_25hr_00/PDR5_GFP_001/PDR5_GFP_100ugml_flc_25hr_00_000000_Brightfield_002.tif")
out_path = "example_image.tiff"

window = 120
x=510
y=455
im = numpy.array(Image.open(path))

# Pad it to add tile and z dimensions
img = im[x:x+window, y:y+window][numpy.newaxis,...,numpy.newaxis]
img = ((img / 65536) * 256).astype(numpy.uint8)

# Create suitable N x H x W x Z array
# dtype must be either uint8 or uint16
# img = rng.integers(2**8, size=(2, 80, 120, 1), dtype="uint8")
extra_args = (("refine_outlines", ("", "true")), ("with_edgemasks", ("", "true")), ("with_masks", ("", "true")))

output = bb.process_data(
    img,
    address,
    session_id,
    input_dimorder = "NYXZ",
    extra_args=extra_args,
)
