"""Validation and preprocessing of data."""

import numpy


def validate_input_shape(input_yx: tuple[int], expected_tile_size: tuple[int]):
    assert all((x % expected_tile_size == 0) for x in input_yx), (
        f"Invalid input shape {input_yx}. Last dims should be divisible by {expected_tile_size}"
    )


def pad_channel_dim(pixels: numpy.ndarray, expected_channels: int) -> numpy.ndarray:
    """Pads the channel dimension of a numpy array to a target size.

    This function is designed to work with image data where the channel
    dimension is not the last one. It makes two key assumptions based on the
    original implementation:
    1. The z-stack (third dimension, index 2) is removed by taking the
       first slice.
    2. The channel dimension to be padded is the second dimension (index 1).

    If the number of channels after slicing the z-stack is less than
    `expected_channels`, it is padded with zeros.

    Parameters
    ----------
    pixels : numpy.ndarray
        The input image data, expected to have at least 3 dimensions.
        For example, a shape like (H, W, Z, ...).
    expected_channels : int
        The desired number of channels for the output array.

    Returns
    -------
    numpy.ndarray
        The processed array with the z-stack removed and the channel
        dimension padded to `expected_channels`. If the array already has
        enough channels, it is returned as is after z-stack removal.

    """
    # Note: This logic is preserved from the original implementation.
    # It assumes a specific data layout where the z-stack is the 3rd dimension.
    if pixels.ndim > 2:
        pixels = pixels[:, :, 0]

    input_channels = pixels.shape[1]
    to_pad = expected_channels - input_channels

    if to_pad > 0:
        padding_shape = list(pixels.shape)
        padding_shape[1] = to_pad
        padding = numpy.zeros(padding_shape, dtype=pixels.dtype)
        pixels = numpy.concatenate((pixels, padding), axis=1)

    return pixels
