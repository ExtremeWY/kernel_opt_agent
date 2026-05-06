"""Original FP32 scalar CUDA reference for Qwen3.5 MoE GDN prefill."""

from __future__ import annotations

import torch

from kernels.qwen35moe_gdn_prefill import kernel_fn as _scalar_kernel_fn


def qwen35moe_gdn_prefill_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    scale: float | None = None,
) -> list[torch.Tensor]:
    return _scalar_kernel_fn(q=q, k=k, v=v, g=g, beta=beta, state=state, scale=scale)
