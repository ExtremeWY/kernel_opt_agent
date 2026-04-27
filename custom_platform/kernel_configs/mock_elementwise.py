"""Mock kernel config that avoids external runtime dependencies."""

from __future__ import annotations

import random

import references


def input_generator(size: dict, dtype: str, device: str, seed: int = 42) -> dict:
    rng = random.Random(seed)
    matrix = [[rng.uniform(-1.0, 1.0) for _ in range(size["N"])] for _ in range(size["M"])]
    bias = [rng.uniform(-1.0, 1.0) for _ in range(size["N"])]
    return {"x": matrix, "bias": bias}


def reference_fn(inputs: dict) -> list[list[float]]:
    return references.mock_elementwise_ref(inputs["x"], inputs["bias"])


def flops_fn(size: dict) -> int:
    return size["M"] * size["N"]


def bytes_fn(size: dict, dtype: str) -> int:
    element_bytes = 4 if dtype == "float32" else 8
    return (size["M"] * size["N"] * 2 + size["N"]) * element_bytes
