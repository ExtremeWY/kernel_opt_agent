"""Working kernel copy used by the experiment loop.

The default copy points at the mock platform example so the full artifact
pipeline can run even without vendor hardware.
"""

from kernels.mock_elementwise import KERNEL_TYPE, TARGET_PLATFORM, kernel_fn
