"""FlashAttention-2 forward kernel config."""

from __future__ import annotations

import math

import torch

import references

from ._utils import dtype_bytes


def _rms_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = x.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
    return x / rms


def _make_attention_inputs(
    size: dict,
    dtype: torch.dtype,
    device: str,
    seed: int = 42,
    *,
    q_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> dict:
    torch.manual_seed(seed)
    batch = size["batch"]
    head_num = size["head_num"]
    seq_len = size["seq_len"]
    head_dim = size["head_dim"]

    q = torch.randn(batch, head_num, seq_len, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(batch, head_num, seq_len, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(batch, head_num, seq_len, head_dim, device=device, dtype=torch.float32)

    q = _rms_normalize(q) * q_scale
    k = _rms_normalize(k) * k_scale
    v = v * v_scale

    return {
        "q": q.to(dtype),
        "k": k.to(dtype),
        "v": v.to(dtype),
        "causal": True,
        "sm_scale": 1.0 / math.sqrt(head_dim),
    }


def input_generator(size: dict, dtype: torch.dtype, device: str, seed: int = 42) -> dict:
    return _make_attention_inputs(size, dtype, device, seed=seed)


def numerical_stability_cases(
    size: dict,
    dtype: torch.dtype,
    device: str,
    seed: int = 42,
) -> list[tuple[str, dict]]:
    cases = [
        ("softmax_flat", _make_attention_inputs(size, dtype, device, seed=seed, q_scale=0.5, k_scale=0.5)),
        ("softmax_nominal", _make_attention_inputs(size, dtype, device, seed=seed, q_scale=1.0, k_scale=1.0)),
        ("softmax_sharp", _make_attention_inputs(size, dtype, device, seed=seed, q_scale=2.0, k_scale=2.0)),
        ("qk_imbalance", _make_attention_inputs(size, dtype, device, seed=seed, q_scale=2.0, k_scale=0.5)),
        ("value_amplified", _make_attention_inputs(size, dtype, device, seed=seed, v_scale=2.0)),
    ]

    zero_q = _make_attention_inputs(size, dtype, device, seed=seed)
    zero_q["q"] = torch.zeros_like(zero_q["q"])
    cases.append(("zero_queries", zero_q))

    const_v = _make_attention_inputs(size, dtype, device, seed=seed)
    const_v["v"] = torch.full_like(const_v["v"], 0.5)
    cases.append(("constant_values", const_v))

    return cases


def reference_fn(inputs: dict) -> torch.Tensor:
    return references.flash_attention_2_ref(**inputs)


def flops_fn(size: dict) -> int:
    pairs = size["seq_len"] * (size["seq_len"] + 1) // 2
    return 4 * size["batch"] * size["head_num"] * pairs * size["head_dim"]


def bytes_fn(size: dict, dtype: torch.dtype) -> int:
    eb = dtype_bytes(dtype)
    elems = size["batch"] * size["head_num"] * size["seq_len"] * size["head_dim"]
    return elems * eb * 4
