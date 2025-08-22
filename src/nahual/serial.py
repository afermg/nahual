import struct
from itertools import chain

import numpy


def serialize_numpy(data: numpy.ndarray) -> bytes:
    """Serialize numpy-like arrays while encoding their dtype and shape.

    Parameters
    ----------
    data : numpy.ndarray
        The input array to be serialized.

    Returns
    -------
    bytes
        The serialized array, containing dtype, shape, and data.

    Notes
    -----
    The serialized format is as follows:
    - Byte 0: Datatype represented as a single character
    - Byte 1: Number of dimensions, encoded as an unsigned 8-bit integer.
    - Bytes 2-n: The shape of every dimension, encoded as an unsigned short.
    - Remaining bytes: The contents of the numpy array.
    """
    array_bytes = data.tobytes()
    dtype_char = data.dtype.char.encode()
    ndim = data.ndim.to_bytes(length=1, byteorder="big")
    shape = data.shape

    shape = bytes()
    for x in data.shape:
        shape = shape + struct.pack(">H", x)
    message = dtype_char + ndim + shape + array_bytes

    return message


def deserialize_numpy(message: bytes) -> numpy.ndarray:
    """Deserialize a NumPy array from a byte stream.

    Parameters
    ----------
    message : bytes
        The byte stream containing the serialized NumPy array.

    Returns
    -------
    numpy.ndarray
        The deserialized NumPy array.

    Notes
    -----
    The byte stream format is as follows:
    - First byte: data type (dtype)
    - Second byte: number of dimensions (ndim)
    - Remaining bytes: shape of the array
    """
    # First byte: dtype char
    in_dtype = numpy.dtype(message[:1].decode())
    # Second byte: ndim
    in_ndim = int.from_bytes(message[1:2], byteorder="big")
    # Third->n: shape
    in_shape = tuple(
        chain.from_iterable(
            [
                struct.unpack(">H", message[i * 2 + 2 : i * 2 + 4])
                for i in range(in_ndim)
            ]
        )
    )

    reshaped_array = numpy.frombuffer(
        message[2 * in_ndim + 2 :], dtype=in_dtype
    ).reshape(in_shape)

    return reshaped_array
