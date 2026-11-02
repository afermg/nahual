"""Pure NumPy shape/framing helpers for the local GPU benchmark, not a wire API."""

from __future__ import annotations

import numpy as np

MODEL_DIMENSIONS = {
    "dinov2": 384,
    "dinov2_random": 384,
    "morphem": 1920,
    "openphenom": 384,
    "subcell_clip01": 1536,
}
MODEL_TILE_SIZES = {
    "dinov2": 224,
    "dinov2_random": 224,
    "morphem": 224,
    "openphenom": 256,
    "subcell_clip01": 448,
}
SLOT_MODELS = {
    "dino_pair": ("dinov2", "dinov2_random"),
    "morphem": ("morphem",),
    "openphenom": ("openphenom",),
    "subcell": ("subcell_clip01",),
}
SLOT_MODELS["full_stack"] = tuple(
    model for models in list(SLOT_MODELS.values()) for model in models
)


def pack_full_stack(parts):
    flat = np.concatenate(
        [np.asarray(part, dtype=np.float32).reshape(-1) for part in parts]
    )
    # The legacy NumPy wire has 16-bit axis lengths. Fixed 32-float rows avoid
    # overflowing one dimension while retaining all values and bounded padding.
    padding = (-flat.size) % 32
    if padding:
        flat = np.pad(flat, (0, padding))
    return np.ascontiguousarray(flat.reshape(-1, 32))


def expected_output_shape(slot, shapes):
    if slot == "full_stack":
        count = sum(
            (shape[1] // MODEL_TILE_SIZES[model])
            * (shape[2] // MODEL_TILE_SIZES[model])
            * MODEL_DIMENSIONS[model]
            for shape in shapes
            for model in SLOT_MODELS[slot]
        )
        return ((count + 31) // 32, 32)
    first = SLOT_MODELS[slot][0]
    rows = sum(
        (shape[1] // MODEL_TILE_SIZES[first]) * (shape[2] // MODEL_TILE_SIZES[first])
        for shape in shapes
    )
    return (rows, sum(MODEL_DIMENSIONS[model] for model in SLOT_MODELS[slot]))


def full_stack_reference(reference, ordinal):
    stem = f"{ordinal:03d}__"
    paired = reference[stem + "dino_pair"]
    return pack_full_stack(
        [
            paired[:, :384],
            paired[:, 384:],
            reference[stem + "morphem"],
            reference[stem + "openphenom"],
            reference[stem + "subcell"],
        ]
    )
