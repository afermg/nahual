import gc
import struct

import numpy
import pytest

from nahual.serial import deserialize_numpy, serialize_numpy


@pytest.mark.parametrize("shape", [(), (0, 3), (10,), (3, 5), (2, 10, 10)])
@pytest.mark.parametrize("dtype", ["bool", "uint16", "int32", "float32", "float64", "complex64"])
def test_serialization(shape, dtype):
    data = numpy.arange(numpy.prod(shape, dtype=int)).reshape(shape).astype(dtype)
    serialized = serialize_numpy(data)
    deserialized = deserialize_numpy(serialized)
    assert data.dtype == deserialized.dtype
    assert data.shape == deserialized.shape
    numpy.testing.assert_array_equal(data, deserialized)
    assert not deserialized.flags.writeable


@pytest.mark.parametrize(
    "data",
    [
        numpy.arange(30, dtype=numpy.float32).reshape(5, 6),
        numpy.arange(30, dtype=numpy.float32).reshape(5, 6)[:, ::2],
        numpy.arange(30, dtype=numpy.float32).reshape(5, 6)[::-1],
        numpy.asfortranarray(numpy.arange(30, dtype=numpy.float32).reshape(5, 6)),
        numpy.broadcast_to(numpy.arange(6, dtype=numpy.uint16), (5, 6)),
    ],
    ids=["contiguous", "strided", "reversed", "fortran", "broadcast"],
)
def test_wire_format_is_unchanged_for_layouts(data):
    legacy = (
        data.dtype.char.encode()
        + data.ndim.to_bytes(length=1, byteorder="big")
        + b"".join(struct.pack(">H", dimension) for dimension in data.shape)
        + data.tobytes(order="C")
    )
    assert serialize_numpy(data) == legacy
    numpy.testing.assert_array_equal(deserialize_numpy(legacy), data)


def test_deserialization_borrows_packet_and_keeps_it_alive():
    data = numpy.arange(64, dtype=numpy.float32).reshape(8, 8)
    packet = serialize_numpy(data)
    output = deserialize_numpy(packet)
    backing = numpy.frombuffer(packet, dtype=numpy.uint8)
    assert numpy.shares_memory(output, backing)
    del packet, backing
    gc.collect()
    numpy.testing.assert_array_equal(output, data)


@pytest.mark.parametrize("packet", [b"", b"f", b"f\x02\x00\x03", b"\xff\x00"])
def test_invalid_header_raises_value_error(packet):
    with pytest.raises(ValueError):
        deserialize_numpy(packet)
