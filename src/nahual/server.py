"""Shared "responder" method to homogeneise all processing models/tools."""

import json
import time
from typing import Callable

import numpy
from pynng import Timeout
from pynng.nng import Socket

from nahual.serial import deserialize_numpy, serialize_numpy
from nahual.shared_memory import handle_server_request, is_shared_request


def _is_setup_message(payload: bytes) -> bool:
    """Heuristic dispatch on the wire payload.

    The wire format already cleanly distinguishes the two payload shapes:
    - numpy.ndarray packets (see ``nahual.serial.serialize_numpy``) start with
      a single-byte dtype character (e.g. ``b'f'``, ``b'd'``, ``b'B'``)
      followed by an ndim byte and packed shape — never with ``b'{'``.
    - dict packets are produced by ``json.dumps(...).encode()`` and therefore
      always start with the ASCII ``b'{'`` byte.

    So peeking at the first byte is sufficient and does not change the wire
    format at all.
    """
    return len(payload) > 0 and payload[:1] == b"{"


async def responder(sock: Socket, setup: Callable, processor: Callable = None):
    """Asynchronous responder function for handling model setup and data processing.

    This function continuously listens for incoming messages via a socket. It
    dispatches each message by payload type using the existing wire format:

    - dict-shaped (JSON) messages always trigger ``setup(**payload)``,
      replacing the currently-loaded ``processor`` in place. This lets a
      single long-lived server load model A, run it, then load model B and
      run it without restarting the daemon.
    - numpy-shaped messages are forwarded to the currently-loaded
      ``processor``.

    Sending a dict before any setup also works (this is the usual cold-start
    path).

    Parameters
    ----------
        sock: pynng. (object): The socket object used for receiving and sending messages.

    Returns
    -------
        None: This function does not return a value but sends responses via the socket.

    Raises
    ------
        Exception: If an error occurs during message handling or processing.


    Notes:
        - The function uses JSON for parameters serialization.
        - The 'setup' function is called (or re-called) on every dict message.
        - The 'process' function is used to compute results from input data.
    """

    stage = "Model loading"
    while True:
        try:
            msg = await sock.arecv_msg()
            payload = msg.bytes

            if is_shared_request(payload):
                if processor is None:
                    raise RuntimeError(
                        "Received a shared-memory process message but no model has "
                        "been loaded yet. Call setup with a dict first."
                    )
                stage = "Shared-memory data processing"
                response = handle_server_request(payload, msg.pipe.url, processor)
                await sock.asend(response)
            elif payload.startswith(b"\x00"):
                raise ValueError("unknown NUL-prefixed Nahual protocol/version")
            elif _is_setup_message(payload):
                stage = "Model loading"
                processor = await setup_content(payload, sock, setup)
            else:
                if processor is None:
                    raise RuntimeError(
                        "Received a non-dict (numpy) message but no model has "
                        "been loaded yet. Call setup with a dict first."
                    )
                stage = "Data processing"
                await process_content(payload, sock, processor)

        except Timeout as e:
            print(f"Waiting for {stage.split(' ')[0]}: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"{stage} failed: {e}")
            # Send back a typed error envelope so the client can distinguish
            # a real failure from a valid numpy/dict payload. Previously we
            # sent ``json.dumps({}).encode()`` ("empty dict"), which the
            # client deserialized as a malformed numpy header (the famous
            # "unpack requires a buffer of 2 bytes" / "Header: {}" error)
            # and silently treated as data. The 0x21 ("!") prefix is
            # unambiguous against:
            #   * numpy.serialize_numpy: starts with a dtype char (letter)
            #   * setup replies / JSON dicts: start with '{' (0x7B)
            print("Sending error envelope")
            envelope = b"!" + json.dumps({"error": str(e), "stage": stage}).encode()
            await sock.asend(envelope)


async def setup_content(payload: bytes, sock, setup: Callable) -> Callable:
    parameters = json.loads(payload.decode())
    # if "model" in parameters:  # Start
    print("NODE0: RECEIVED REQUEST")
    processor, info = setup(**parameters)
    info_str = f"Loaded model with parameters {info}"
    print(info_str)
    print("Sending model info back")
    await sock.asend(json.dumps(info).encode())
    print("Model loaded. Will wait for data.")
    return processor


async def process_content(payload: bytes, sock, processor) -> None:
    # Receive data
    img = deserialize_numpy(payload)
    # Add data processing here
    result = processor(img)
    # Cover for processes that keep data in the gpu
    if not isinstance(result, numpy.ndarray):
        result = result.cpu().detach().numpy()
    await sock.asend(serialize_numpy(result))
