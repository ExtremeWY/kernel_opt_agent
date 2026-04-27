"""Reference implementations for correctness verification."""

from __future__ import annotations

from importlib import import_module

__all__ = ["example_op_ref", "mock_elementwise_ref"]


def __getattr__(name: str):
    if name == "example_op_ref":
        return import_module("references.example_op").example_op_ref
    if name == "mock_elementwise_ref":
        return import_module("references.mock_elementwise").mock_elementwise_ref
    raise AttributeError(name)
