"""Example kernel config used by the scaffold."""

from __future__ import annotations

import torch

import references

from kernel_configs._utils import dtype_bytes


def input_generator(size: dict, dtype: torch.dtype, device: str, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    x = torch.randn(size["M"], size["N"], dtype=dtype, device=device)
    bias = torch.randn(size["N"], dtype=dtype, device=device)
    return {"x": x, "bias": bias}


def numerical_stability_cases(
    size: dict,
    dtype: torch.dtype,
    device: str,
    seed: int = 42,
) -> list[tuple[str, dict]]:
    nominal = input_generator(size, dtype, device, seed=seed)
    low_amplitude = {
        "x": nominal["x"] * 0.1,
        "bias": nominal["bias"] * 0.1,
    }
    high_amplitude = {
        "x": nominal["x"] * 4.0,
        "bias": nominal["bias"] * 4.0,
    }
    zero_input = {
        "x": torch.zeros_like(nominal["x"]),
        "bias": nominal["bias"],
    }
    constant_bias = {
        "x": nominal["x"],
        "bias": torch.full_like(nominal["bias"], 0.5),
    }
    return [
        ("nominal", nominal),
        ("low_amplitude", low_amplitude),
        ("high_amplitude", high_amplitude),
        ("zero_input", zero_input),
        ("constant_bias", constant_bias),
    ]


def reference_fn(inputs: dict) -> torch.Tensor:
    return references.example_op_ref(inputs["x"], inputs["bias"])


def flops_fn(size: dict) -> int:
    return size["M"] * size["N"]


def bytes_fn(size: dict, dtype: torch.dtype) -> int:
    eb = dtype_bytes(dtype)
    return (size["M"] * size["N"] * 2 + size["N"]) * eb
