"""Pure Python mock kernel for local workflow validation."""

from __future__ import annotations


KERNEL_TYPE = "mock_elementwise"
TARGET_PLATFORM = "mock_platform"


def kernel_fn(**inputs):
    matrix = inputs["x"]
    bias = inputs["bias"]
    out = []
    for row in matrix:
        out.append([value + bias[idx] for idx, value in enumerate(row)])
    return out
