"""Shared "responder" method to homogeneise all processing models/tools."""

import json
import time
from typing import Callable

import numpy
from pynng import Timeout
from pynng.nng import Socket

from nahual.serial import deserialize_numpy, serialize_numpy


async def responder(sock: Socket, setup: Callable, processor: Callable = None):
    """Asynchronous responder function for handling model setup and data processing.

        This function continuously listens for incoming messages via a socket. It handles two
        modes: initializing a model based on received parameters and processing data using
        an already loaded model.

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
        - The 'setup' function is called to initialize the model.
        - The 'process' function is used to compute results from input data.
    """

    while True:
        try:
            msg = await sock.arecv_msg()

            if len(msg.bytes) == 1:
                print("Exiting")
                break

            if processor is None:
                stage = "Model loading"
                processor = await setup_content(msg, sock, setup)
            else:
                stage = "Data processing"
                await process_content(msg, sock, processor)

        except Timeout as e:
            print(f"Waiting for {stage.split(' ')[0]}: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"{stage} failed: {e}")
            # Send back an empty dictionary if things did not work,
            # to avoid blocking the client.
            print("Sending empty dict")
            await sock.asend(json.dumps({}).encode())


async def setup_content(msg, sock, setup: Callable) -> Callable:
    content = msg.bytes.decode()
    parameters = json.loads(content)
    # if "model" in parameters:  # Start
    print("NODE0: RECEIVED REQUEST")
    processor, info = setup(**parameters)
    info_str = f"Loaded model with parameters {info}"
    print(info_str)
    print("Sending model info back")
    await sock.asend(json.dumps(info).encode())
    print("Model loaded. Will wait for data.")
    return processor


async def process_content(msg, sock, processor) -> None:
    # Receive data
    img = deserialize_numpy(msg.bytes)
    # Add data processing here
    result = processor(img)
    # Cover for processes that keep data in the gpu
    if not isinstance(result, numpy.ndarray):
        result = result.cpu().detach().numpy()
    await sock.asend(serialize_numpy(result))
