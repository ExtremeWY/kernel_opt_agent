"""Reference implementation for the mock kernel."""

from __future__ import annotations


def mock_elementwise_ref(x: list[list[float]], bias: list[float]) -> list[list[float]]:
    return [[value + bias[idx] for idx, value in enumerate(row)] for row in x]
