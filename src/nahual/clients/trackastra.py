"""
This client matches https://github.com/afermg/trackastra/blob/main/server.py.

This functions define how Nahual's trackastra client contacts its server counterpart.
"""

import json

import numpy

from nahual.serial import serialize_numpy
from nahual.transport import request_receive

ADDRESS = "ipc:///tmp/reqrep.ipc"


def load_model(
    parameters: dict = {"model_name": "general_2d"}, address: str = ADDRESS
) -> str:
    """
    Load a model by sending parameters to a specified address.

    Parameters:
    -----------
    parameters : dict, optional
        A dictionary containing model parameters. Defaults to {"model_name": "general_2d"}.
    address : str, optional
        The address to send the request. Defaults to ADDRESS.

    Returns:
    --------
    str
        The decoded response from the server.
    """
    # encode
    packet = json.dumps(parameters).encode()
    # Request->receive
    response = request_receive(packet, address=address)
    # decode
    decoded = response.decode()
    print(f"REQ: RECEIVED {decoded}")

    return json.loads(decoded)


def process_data(data: list | numpy.ndarray, address: str = ADDRESS) -> dict:
    """
    Process data by sending it to a specified address.

    Parameters:
    -----------
    data : list or numpy.ndarray
        The input data to process.
    address : str, optional
        The address to send the request. Defaults to ADDRESS.

    Returns:
    --------
    dict
        The processed data received from the server.
    """
    # format
    original_array = numpy.asarray(data)
    # encode
    packet = serialize_numpy(original_array)
    # Request->receive
    response = request_receive(packet, address=address)
    # decode
    decoded = json.loads(response.decode())
    print(f"REQ: RECEIVED PROCESSED DATA {len(decoded), decoded.keys()}")

    return decoded
