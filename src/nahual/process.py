"""Collects the behaviour of data type serialization and deserialization."""

from __future__ import annotations

import json
from functools import partial
from typing import Any, Callable, Literal

import numpy

from nahual.serial import deserialize_numpy, serialize_numpy
from nahual.transport import RequestSession, request_receive


def serialize(data: numpy.ndarray | dict | list | tuple) -> bytes:
    """Serialize a NumPy array, dictionary, or list/tuple into bytes.

    This function takes either a NumPy array (or an array-like object such
    as a list or tuple) or a dictionary and converts it into a byte string
    for transmission or storage. NumPy arrays are handled by a specialized
    `serialize_numpy` function, while dictionaries are serialized using JSON.

    Parameters
    ----------
    data : numpy.ndarray or dict or list or tuple
        The data to be serialized. If a list or tuple is provided, it will be
        converted to a NumPy array before serialization.

    Returns
    -------
    bytes
        The serialized data as a byte string.

    Raises
    ------
    Exception
        If the provided `data` is of an unsupported type.

    """
    if isinstance(data, (numpy.ndarray, list, tuple)):
        original_array = numpy.asarray(data)
        packet = serialize_numpy(original_array)
    elif isinstance(data, dict):
        packet = json.dumps(data).encode()
    else:
        raise Exception(f"Unsupported data type {type(data)}")
    return packet


def deserialize(packet: bytes, dtype: Literal["numpy", "dict"]) -> numpy.ndarray | dict:
    """Deserialize a byte packet into a specified data structure.

    Parameters
    ----------
    packet : bytes
        The byte packet to be deserialized.
    dtype : {"numpy", "dict"}
        The target data type for deserialization. Supported values are "numpy"
        for a NumPy array and "dict" for a Python dictionary.

    Returns
    -------
    numpy.ndarray or dict
        The deserialized data, either as a NumPy array or a dictionary,
        depending on the `dtype` argument.

    Raises
    ------
    Exception
        If an unsupported `dtype` is provided.

    """
    if dtype == "numpy":
        data = deserialize_numpy(packet)
    elif dtype == "dict":
        data = json.loads(packet.decode())
    else:
        raise Exception(f"Unsupported data type {dtype}")

    return data


def send_receive_process(
    data: Any,
    expected_output_dtype: Literal["numpy", "dict"],
    address: str,
    expected_input_dtype: numpy.ndarray | dict | None = None,
    shared_memory: bool = False,
    timeout_ms: int | None = None,
):
    """Serialize, send, receive, and deserialize data.

    This function encapsulates the process of sending data to a network
    address, receiving a response, and processing it. It handles serialization
    of the input data, making a request, and deserializing the response
    into a specified format.

    Parameters
    ----------
    data : numpy.ndarray or dict
        The input data to be serialized and sent.
    expected_output_dtype : {"numpy", "dict"}
        The expected data type of the output after deserialization.
    address : str
        The network address (e.g., ZMQ port) to send the data to.
    expected_input_dtype : {numpy.ndarray, dict}, optional
        If provided, asserts that the input `data` is of this type before
        processing. Defaults to None, which skips the check.
    shared_memory : bool, optional
        Use the local shared-memory data plane for a NumPy process call. An
        optional ``appose.NDArray`` input avoids staging when installed through
        ``nahual[appose]``. Setup dictionaries and non-NumPy outputs are not
        supported. Defaults to False.
    timeout_ms : int or None, optional
        Per-call NNG timeout in milliseconds. None preserves the configured
        default. Appose-owned input requires an explicit -1 (unbounded) timeout
        to keep the external buffer borrowed until the server completes.

    Returns
    -------
    numpy.ndarray or dict
        The deserialized data received in response, matching the type
        specified by `expected_output_dtype`.

    Raises
    ------
    AssertionError
        If `expected_input_dtype` is specified and the type of `data`
        does not match.

    """
    # Optional input type-checking
    if expected_input_dtype is not None:
        assert isinstance(data, expected_input_dtype), (
            f"Input type {type(data)} does not match expected input type {expected_input_dtype}"
        )

    return _exchange(
        data, expected_output_dtype, address, shared_memory, timeout_ms, request_receive
    )


def _exchange(
    data, expected_output_dtype, address, shared_memory, timeout_ms, requester
):
    if shared_memory:
        if isinstance(data, dict):
            raise TypeError(
                "shared_memory=True is process-only and cannot be used for setup/dict calls"
            )
        if expected_output_dtype != "numpy":
            raise TypeError("shared_memory=True requires a NumPy output signature")
        # Lazy import keeps the default transport independent of shared-memory
        # implementation details and the optional Appose dependency.
        from nahual.shared_memory import client_process

        return client_process(data, address, timeout_ms=timeout_ms, requester=requester)

    # encode
    packet = serialize(data)
    # Request -> receive
    response = requester(packet, address=address, timeout_ms=timeout_ms)
    # deserialize
    output = deserialize(response, dtype=expected_output_dtype)

    return output


class Client:
    """Context-managed setup/process client with a persistent NNG connection.

    The existing dispatch helpers remain stateless. Use one Client per worker
    and close it explicitly or with ``with``. Calls are synchronous; overlapping
    calls and use after fork are rejected. ``timeout_ms`` on a call overrides
    the client default. A timeout does not cancel server-side processing.
    """

    def __init__(
        self,
        address: str,
        *,
        signature: tuple[str, str] = ("dict", "numpy"),
        timeout_ms: int | None = None,
    ):
        if len(signature) != 2 or any(
            kind not in ("dict", "numpy") for kind in signature
        ):
            raise ValueError("signature must contain two 'dict' or 'numpy' entries")
        self._signature = tuple(signature)
        self._session = RequestSession(address, timeout_ms=timeout_ms)

    def setup(self, parameters: dict, *, timeout_ms: int | None = None):
        if not isinstance(parameters, dict):
            raise TypeError("setup requires a parameter dictionary")
        return self._call(parameters, self._signature[0], False, timeout_ms)

    def process(
        self, data, *, shared_memory: bool = False, timeout_ms: int | None = None
    ):
        if isinstance(data, dict):
            raise TypeError("process does not accept setup dictionaries; use setup()")
        return self._call(data, self._signature[1], shared_memory, timeout_ms)

    def _call(self, data, output_dtype, shared_memory, timeout_ms):
        self._session._check_available()
        effective_timeout = (
            self._session.timeout_ms if timeout_ms is None else timeout_ms
        )
        return _exchange(
            data,
            output_dtype,
            self._session.address,
            shared_memory,
            effective_timeout,
            self._session.request,
        )

    def close(self):
        self._session.close()

    def __enter__(self):
        self._session.__enter__()
        return self

    def __exit__(self, *_exc):
        self.close()


def get_output_signature(name: str) -> tuple[str, str]:
    """Get the type signature for a given tool name.

    This function looks up a predefined dictionary to find the expected
    input and output type signature for a specific tool.

    Parameters
    ----------
    name : str
        The name of the tool.

    Returns
    -------
    tuple[str, str]
        A tuple containing the string representations of the input and
        output types.

    Raises
    ------
    KeyError
        If the tool `name` is not found in the predefined signatures.

    Examples
    --------
    >>> get_output_signature("cellpose")
    ('dict', 'numpy')
    """
    OUTPUT_SIGNATURES = {
        # "CaicedoLab/MorphEm": ("dict", "numpy"),
        "bioimageio": ("dict", "numpy"),
        "cellpose": ("dict", "numpy"),
        "cellsam": ("dict", "numpy"),
        "cellwhisperer": ("dict", "numpy"),
        "channelsformer": ("dict", "numpy"),
        "deepprofiler": ("dict", "numpy"),
        "dinov2": ("dict", "numpy"),
        "dinov3": ("dict", "numpy"),
        "embedseg": ("dict", "numpy"),
        "instanseg": ("dict", "numpy"),
        "megaseg": ("dict", "numpy"),
        "microsam": ("dict", "numpy"),
        "recursionpharma/OpenPhenom": ("dict", "numpy"),
        "scdino": ("dict", "numpy"),
        "stardist": ("dict", "numpy"),
        "subcell": ("dict", "numpy"),
        "trackastra": ("dict", "dict"),
        "ultrack": ("dict", "numpy"),
        "unidino": ("dict", "numpy"),
        "vit": ("dict", "numpy"),
    }

    # IMPORTANT: If not in list dict->numpy, which is the most common
    signature = OUTPUT_SIGNATURES.get(name, ("dict", "numpy"))
    # Temporarily emoved due to noisy output
    # if name not in OUTPUT_SIGNATURES:
    #     print(
    #         f"Model name {name} not explicitly defined as one of the signatures. Assuming (dict->numpy). Available values are {list(OUTPUT_SIGNATURES.keys())}"
    #     )

    return signature


def dispatch_setup_process(
    model_group: str, signature: str | tuple[str] | None = None
) -> tuple[Callable, Callable]:
    """Get the setup and process functions for a given model.

    This function dispatches to the correct setup and process partials
    based on the model's output signature. The signature determines the
    expected data types for the communication process.

    Parameters
    ----------
    model_group : str
        The name of the model group. Used as the default signature if `signature`
        is not provided.
    signature : str | tuple[str] | None, optional
        The output signature of the model. If a string, it is used to look
        up the signature tuple. If None, `name` is used as the signature
        string. The signature should be a tuple of two strings
        representing the data types for the setup and process steps,
        respectively. By default None.

    Returns
    -------
    tuple[Callable, Callable]
        A tuple containing two `functools.partial` objects:
        (`setup`, `process`). These partials are configured from
        `send_receive_process` with the appropriate `expected_output_dtype`
        set from the signature.

    """
    if signature is None:
        signature = model_group

    if isinstance(signature, str):
        # Assumes get_output_signature is defined elsewhere
        signature: tuple[str] = get_output_signature(signature)

    # Assumes send_receive_process is defined elsewhere
    setup, process = [
        partial(send_receive_process, expected_output_dtype=x) for x in signature
    ]
    return setup, process
