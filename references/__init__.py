"""Reference implementations for correctness verification.

Each submodule provides a PyTorch-native implementation that the optimized
kernel is checked against. Do NOT modify these files during experiments.
"""

from .matmul import matmul_ref
from .rms_norm import rms_norm_ref
from .swiglu_input_quant import swiglu_input_quant_ref
from .qkv_part_rope import qkv_part_rope_ref
from .dsa_forward import dsa_forward_ref
from .flash_attention_2 import flash_attention_2_ref
from .qwen35moe_gdn_prefill import qwen35moe_gdn_prefill_ref

__all__ = [
    "matmul_ref",
    "rms_norm_ref",
    "swiglu_input_quant_ref",
    "qkv_part_rope_ref",
    "dsa_forward_ref",
    "flash_attention_2_ref",
    "qwen35moe_gdn_prefill_ref",
]
