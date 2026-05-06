"""Benchmark config for Qwen3.5 MoE GDN prefill."""

from __future__ import annotations

import torch

from ._utils import dtype_bytes
from references.qwen35moe_gdn_prefill import qwen35moe_gdn_prefill_ref


D = 128
H_K = 16
H_V = 32

KERNEL_OPT_CHARACTERISTICS = {
    "is_matmul_like": True,
    "supports_tensor_core_candidate": True,
    "mma_tile_m": 16,
    "effective_m": 64,
}


def _filled(shape: tuple[int, ...], scale: float, bias: float, dtype: torch.dtype, device: str) -> torch.Tensor:
    n = 1
    for dim in shape:
        n *= dim
    x = torch.arange(n, device=device, dtype=torch.float32)
    x = ((x % 251) - 125.0) * scale + bias
    return x.reshape(shape).to(dtype)


def input_generator(size: dict, dtype: torch.dtype, device: str, seed: int = 42) -> dict:
    del seed
    batch = int(size["batch"])
    seq_len = int(size["seq_len"])
    return {
        "q": _filled((batch, seq_len, H_K, D), 0.001, 0.0, dtype, device),
        "k": _filled((batch, seq_len, H_K, D), 0.001, 0.0, dtype, device),
        "v": _filled((batch, seq_len, H_V, D), 0.001, 0.0, dtype, device),
        "g": _filled((batch, seq_len, H_V), 0.0001, -0.01, dtype, device),
        "beta": _filled((batch, seq_len, H_V), 0.0001, 0.5, dtype, device),
        "state": _filled((batch, H_V, D, D), 0.0001, 0.0, dtype, device),
        "scale": D ** -0.5,
    }


def numerical_stability_cases(size: dict, dtype: torch.dtype, device: str, seed: int = 42) -> list[tuple[str, dict]]:
    del seed
    small = dict(size)
    small["seq_len"] = min(int(small["seq_len"]), 2048)
    base = input_generator(small, dtype, device)

    zero_state = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in base.items()}
    zero_state["state"] = torch.zeros_like(zero_state["state"])

    softer_gate = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in base.items()}
    softer_gate["g"] = torch.full_like(softer_gate["g"], -0.02)

    return [
        ("baseline", base),
        ("zero_state", zero_state),
        ("softer_gate", softer_gate),
    ]


def reference_fn(inputs: dict) -> list[torch.Tensor]:
    return qwen35moe_gdn_prefill_ref(**inputs)


def flops_fn(size: dict) -> int:
    return 4 * int(size["batch"]) * H_V * int(size["seq_len"]) * D * D


def bytes_fn(size: dict, dtype: torch.dtype) -> int:
    eb = dtype_bytes(dtype)
    batch = int(size["batch"])
    seq_len = int(size["seq_len"])
    qk = 2 * batch * seq_len * H_K * D
    vo = 2 * batch * seq_len * H_V * D
    gb = 2 * batch * seq_len * H_V
    st = 2 * batch * H_V * D * D
    return (qk + vo + gb + st) * eb
