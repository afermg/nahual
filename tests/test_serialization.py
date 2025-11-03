import numpy
import pytest

from nahual.serial import deserialize_numpy, serialize_numpy


@pytest.mark.parametrize("shape", [(10,), (3, 5), (2, 10, 10)])
@pytest.mark.parametrize("dtype", ["uint16", "float32", "float64"])
def test_serialization(shape: tuple[int], dtype: str):
    numpy.random.seed(42)
    data = numpy.random.randint(64, size=(3, 10, 10)).astype(numpy.dtype(dtype))
    serialized = serialize_numpy(data)
    deserialized = deserialize_numpy(serialized)
    assert (
        data.dtype == deserialized.dtype
    ), "Data types are not retained upon serialization"
    assert (
        data.shape == deserialized.shape
    ), "Shapes are not retained upon serialization"
