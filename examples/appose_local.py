"""Use an Appose-owned input buffer with Nahual's local NNG server.

Install it with ``pip install 'nahual[appose]>=0.0.11'``. Both client and
server need Nahual 0.0.11 or newer, so update an older Nix model wrapper's
Nahual pin first. For uniDINO, launch its normal NNG app at the address below.
"""

import appose

from nahual.process import dispatch_setup_process

address = "ipc:///tmp/unidino.ipc"
setup, process = dispatch_setup_process("unidino")

# Parameters and model information still use Nahual's ordinary JSON-over-NNG
# control path. Appose is an input-buffer optimization, not a transport.
info = setup({"pretrained": True, "cache": True}, address=address)
print(info)

shape = [1, 5, 1, 128, 128]  # NCZYX
with appose.NDArray("float32", shape) as shared_input:
    input_view = shared_input.ndarray()

    # Best case: acquisition/preprocessing writes into this view directly.
    # Assigning an existing NumPy array here would add one full staging copy.
    input_view.fill(0)

    embedding = process(
        shared_input,
        address=address,
        shared_memory=True,
        timeout_ms=-1,
    )

# v1 returns an owning NumPy output, so it remains valid after Appose disposes
# the borrowed input allocation.
print(embedding.shape, embedding.dtype, embedding.flags.owndata)
