"""Shared helpers for kernel config modules."""

from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


DTYPE_MAP = {
    "float16": getattr(torch, "float16", "float16"),
    "float32": getattr(torch, "float32", "float32"),
    "float64": getattr(torch, "float64", "float64"),
    "bfloat16": getattr(torch, "bfloat16", "bfloat16"),
    "int8": getattr(torch, "int8", "int8"),
    "int16": getattr(torch, "int16", "int16"),
    "int32": getattr(torch, "int32", "int32"),
    "int64": getattr(torch, "int64", "int64"),
    "uint8": getattr(torch, "uint8", "uint8"),
    "bool": getattr(torch, "bool", "bool"),
}

DTYPE_BYTES = {
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "bfloat16": 2,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "uint8": 1,
    "bool": 1,
}


def dtype_name(dtype) -> str:
    if isinstance(dtype, str):
        return dtype
    if torch is not None:
        for name, mapped in DTYPE_MAP.items():
            if dtype == mapped:
                return name
    return str(dtype)


def dtype_bytes(dtype) -> int:
    name = dtype_name(dtype)
    if name in DTYPE_BYTES:
        return DTYPE_BYTES[name]
    if torch is not None:
        return torch.tensor([], dtype=dtype).element_size()
    raise ValueError(f"Unknown dtype '{dtype}'")
