import json
import time
from typing import Callable

from nahual.serial import deserialize_numpy, serialize_numpy


async def responder(sock, setup: Callable, processor: Callable = None):
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
        if processor is None:
            try:
                msg = await sock.arecv_msg()
                if len(msg.bytes) == 1:
                    print("Exiting")
                    break
                content = msg.bytes.decode()
                parameters = json.loads(content)
                # if "model" in parameters:  # Start
                try:
                    print("NODE0: RECEIVED REQUEST")
                    processor, info = setup(**parameters)
                    info_str = f"Loaded model with parameters {info}"
                    print(info_str)
                    print("Sending model info back")
                    await sock.asend(json.dumps(info).encode())
                    print("Model loaded. Will wait for data.")
                except Exception as e:
                    print(f"Model loading failed: {e}")
                    # Send back an empty dictionary if things did not work,
                    # to avoid blocking the client.
                    await sock.asend(json.dumps({}).encode())

            except Exception as e:
                print(f"Waiting for parameters: {e}")
                time.sleep(1)
        else:
            try:
                # Receive data
                msg = await sock.arecv_msg()
                if len(msg.bytes) == 1:
                    print("Exiting")
                    break
                img = deserialize_numpy(msg.bytes)
                # Add data processing here
                result = processor(img)
                result_np = result.cpu().detach().numpy()
                await sock.asend(serialize_numpy(result_np))

            except Exception as e:
                print(f"Waiting for data: {e}")
