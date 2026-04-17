"""Example placeholder kernel for one concrete hardware platform."""

from __future__ import annotations


KERNEL_TYPE = "example_op"
TARGET_PLATFORM = "custom_platform"


def kernel_fn(**inputs):
    raise NotImplementedError(
        "Replace kernels/example_op.py with a real kernel implementation for your platform."
    )

