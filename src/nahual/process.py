"""Collects the behaviour of data type serialization and deserialization."""

import json
from functools import partial
from typing import Literal

import numpy

from nahual.serial import deserialize_numpy, serialize_numpy
from nahual.transport import request_receive


def serialize(data: numpy.ndarray | dict) -> bytes:
    if isinstance(data, (numpy.ndarray, list, tuple)):
        original_array = numpy.asarray(data)
        packet = serialize_numpy(original_array)
    elif isinstance(data, dict):
        packet = json.dumps(data).encode()
    else:
        raise Exception(f"Unsupported data type {type(data)}")

    return packet


def deserialize(packet: bytes, dtype: Literal["numpy", "dict"]) -> numpy.ndarray | dict:
    if dtype == "numpy":
        data = deserialize_numpy(packet)
    elif dtype == "dict":
        data = json.loads(packet.decode())
    else:
        raise Exception(f"Unsupported data type {dtype}")

    return data


def send_receive_process(
    data: numpy.ndarray | dict,
    expected_output_dtype: Literal["numpy", "dict"],
    address: str,
    expected_input_dtype: numpy.ndarray | dict | None = None,
):
    # Optional input type-checking
    if expected_input_dtype is not None:
        assert isinstance(
            data, expected_input_dtype
        ), f"Input type {type(data)} does not match expected input type {expected_input_dtype}"

    # encode
    packet = serialize(data)
    # Request -> receive
    response = request_receive(packet, address=address)
    # deserialize
    output = deserialize(response, dtype=expected_output_dtype)
    
    return output


def dispatch_setup_process(name: str):
    # What is the expected output for a given tool
    OUTPUT_SIGNATURES = dict(
        cellpose=("dict", "numpy"),
        dinov2=("dict", "numpy"),
        vit=("dict", "numpy"),
        trackastra=("dict", "dict"),
    )

    """Get the setup function for each supported model."""
    setup, process = [
        partial(send_receive_process, expected_output_dtype=x)
        for x in OUTPUT_SIGNATURES[name]
    ]
    return setup, process
