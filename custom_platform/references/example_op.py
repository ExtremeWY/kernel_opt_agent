"""Reference implementation for the scaffold example kernel."""

from __future__ import annotations

import torch


def example_op_ref(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return x + bias

