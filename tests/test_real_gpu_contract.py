from __future__ import annotations

import numpy as np
import pytest

from benchmarks import real_gpu_contract as contract


@pytest.mark.parametrize(
    "shape",
    [
        (5, 1080, 1280),
        (5, 1080, 1080),
        (5, 970, 970),
        (5, 996, 996),
        (5, 998, 998),
        (5, 1024, 1024),
    ],
)
def test_full_stack_layout_matches_individual_slots(shape):
    outputs = {}
    for number, model in enumerate(contract.SLOT_MODELS["full_stack"]):
        size = contract.MODEL_TILE_SIZES[model]
        count = (shape[1] // size) * (shape[2] // size)
        outputs[model] = np.full(
            (count, contract.MODEL_DIMENSIONS[model]), number + 0.25, dtype=np.float32
        )
    paired = np.concatenate([outputs["dinov2"], outputs["dinov2_random"]], axis=1)
    reference = {
        "003__dino_pair": paired,
        "003__morphem": outputs["morphem"],
        "003__openphenom": outputs["openphenom"],
        "003__subcell": outputs["subcell_clip01"],
    }
    packed = contract.pack_full_stack(list(outputs.values()))
    assert packed.dtype == np.float32 and packed.flags.c_contiguous
    assert packed.shape == contract.expected_output_shape("full_stack", [shape])
    assert max(packed.shape) <= 65535
    np.testing.assert_array_equal(packed, contract.full_stack_reference(reference, 3))
    position = 0
    for output in outputs.values():
        np.testing.assert_array_equal(
            packed.reshape(-1)[position : position + output.size].reshape(output.shape),
            output,
        )
        position += output.size


def test_framing_preserves_noncontiguous_values_and_zero_pads():
    first = np.arange(100, dtype=np.float32).reshape(10, 10)[:, ::2]
    second = np.array([123.0], dtype=np.float32)
    packed = contract.pack_full_stack([first, second])
    assert packed.shape == (2, 32)
    np.testing.assert_array_equal(packed.reshape(-1)[:50], first.reshape(-1))
    assert packed.reshape(-1)[50] == 123
    np.testing.assert_array_equal(packed.reshape(-1)[51:], 0)


@pytest.mark.parametrize(
    "slot,rows,columns",
    [
        ("dino_pair", 36, 768),
        ("morphem", 36, 1920),
        ("openphenom", 29, 384),
        ("subcell", 8, 1536),
    ],
)
def test_multisite_output_shape(slot, rows, columns):
    assert contract.expected_output_shape(slot, [(5, 1080, 1280), (5, 996, 996)]) == (
        rows,
        columns,
    )
