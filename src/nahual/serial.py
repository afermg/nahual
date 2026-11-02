import struct

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
    header = (
        data.dtype.char.encode()
        + data.ndim.to_bytes(length=1, byteorder="big")
        + struct.pack(f">{data.ndim}H", *data.shape)
    )
    # Joining a contiguous buffer copies directly into the final packet, instead
    # of first allocating data.tobytes(). Strided/Fortran inputs still need a
    # C-order staging array to preserve the legacy wire layout.
    contiguous = numpy.ascontiguousarray(data)
    return b"".join((header, memoryview(contiguous)))


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
    view = memoryview(message)
    if len(view) < 2:
        raise ValueError("truncated NumPy packet header")
    try:
        in_dtype = numpy.dtype(bytes(view[:1]).decode("ascii"))
    except (UnicodeDecodeError, TypeError) as error:
        raise ValueError("invalid NumPy packet dtype") from error
    in_ndim = view[1]
    header_size = 2 * in_ndim + 2
    if len(view) < header_size:
        raise ValueError("truncated NumPy packet shape")
    in_shape = struct.unpack_from(f">{in_ndim}H", view, 2)
    # A bytes slice would duplicate the entire payload. NumPy retains this
    # memoryview (and its packet owner), so the returned view remains valid.
    return numpy.frombuffer(view[header_size:], dtype=in_dtype).reshape(in_shape)
