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


def reference_fn(inputs: dict) -> torch.Tensor:
    return references.example_op_ref(inputs["x"], inputs["bias"])


def flops_fn(size: dict) -> int:
    return size["M"] * size["N"]


def bytes_fn(size: dict, dtype: torch.dtype) -> int:
    eb = dtype_bytes(dtype)
    return (size["M"] * size["N"] * 2 + size["N"]) * eb

