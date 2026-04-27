"""PyTorch-native reference for FlashAttention-2 style causal attention."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def flash_attention_2_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: float | None = None,
) -> torch.Tensor:
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have shape [batch, heads, seq_len, head_dim]")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("q, k, and v must be bfloat16")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("q, k, and v must be CUDA tensors")
    if q.shape[-1] != 128:
        raise ValueError("head_dim must be 128")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=causal,
        scale=float(sm_scale),
    )
