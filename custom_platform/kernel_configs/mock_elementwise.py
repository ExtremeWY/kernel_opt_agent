"""Mock kernel config that avoids external runtime dependencies."""

from __future__ import annotations

import random

import references


def input_generator(size: dict, dtype: str, device: str, seed: int = 42) -> dict:
    rng = random.Random(seed)
    matrix = [[rng.uniform(-1.0, 1.0) for _ in range(size["N"])] for _ in range(size["M"])]
    bias = [rng.uniform(-1.0, 1.0) for _ in range(size["N"])]
    return {"x": matrix, "bias": bias}


def numerical_stability_cases(
    size: dict,
    dtype: str,
    device: str,
    seed: int = 42,
) -> list[tuple[str, dict]]:
    nominal = input_generator(size, dtype, device, seed=seed)
    low_amplitude = {
        "x": [[value * 0.1 for value in row] for row in nominal["x"]],
        "bias": [value * 0.1 for value in nominal["bias"]],
    }
    high_amplitude = {
        "x": [[value * 4.0 for value in row] for row in nominal["x"]],
        "bias": [value * 4.0 for value in nominal["bias"]],
    }
    zero_input = {
        "x": [[0.0 for _ in row] for row in nominal["x"]],
        "bias": list(nominal["bias"]),
    }
    constant_bias = {
        "x": [list(row) for row in nominal["x"]],
        "bias": [0.5 for _ in nominal["bias"]],
    }
    return [
        ("nominal", nominal),
        ("low_amplitude", low_amplitude),
        ("high_amplitude", high_amplitude),
        ("zero_input", zero_input),
        ("constant_bias", constant_bias),
    ]


def reference_fn(inputs: dict) -> list[list[float]]:
    return references.mock_elementwise_ref(inputs["x"], inputs["bias"])


def flops_fn(size: dict) -> int:
    return size["M"] * size["N"]


def bytes_fn(size: dict, dtype: str) -> int:
    element_bytes = 4 if dtype == "float32" else 8
    return (size["M"] * size["N"] * 2 + size["N"]) * element_bytes
