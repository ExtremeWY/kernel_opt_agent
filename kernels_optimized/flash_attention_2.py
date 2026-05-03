"""Active FlashAttention-2 kernel entrypoint."""

from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import textwrap

import torch


KERNEL_TYPE = "flash_attention_2"
_HEAD_DIM = 128
_DTYPE = torch.bfloat16


def _extension_name(source_path: Path) -> str:
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()[:12]
    return f"flash_attention_2_cuda_{digest}"


def _build_extension() -> object:
    source_path = Path(__file__).with_suffix(".cu")
    build_root = source_path.parent / "workspace" / ".torch_extensions"
    build_root.mkdir(parents=True, exist_ok=True)
    name = _extension_name(source_path)
    build_dir = build_root / name
    build_dir.mkdir(parents=True, exist_ok=True)

    so_candidates = sorted(build_dir.glob(f"{name}*.so"))
    if not so_candidates:
        setup_path = build_dir / "setup.py"
        setup_path.write_text(
            textwrap.dedent(
                f"""
                from setuptools import setup
                from torch.utils.cpp_extension import BuildExtension, CUDAExtension

                setup(
                    name="{name}",
                    ext_modules=[
                        CUDAExtension(
                            name="{name}",
                            sources=[{source_path.as_posix()!r}],
                            extra_compile_args={{
                                "cxx": ["-O3", "-std=c++17", "-w"],
                                "nvcc": [
                                    "-O3",
                                    "-lineinfo",
                                    "-w",
                                    "--use_fast_math",
                                    "-std=c++17",
                                    "--expt-relaxed-constexpr",
                                    "--expt-extended-lambda",
                                    "-U__CUDA_NO_HALF_OPERATORS__",
                                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                                    "-gencode=arch=compute_89,code=sm_89",
                                ],
                            }},
                        ),
                    ],
                    cmdclass={{"build_ext": BuildExtension.with_options(use_ninja=False)}},
                )
                """
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [
                sys.executable,
                str(setup_path),
                "build_ext",
                "--build-lib",
                str(build_dir),
                "--build-temp",
                str(build_dir / "temp"),
            ],
            check=True,
            cwd=build_dir,
        )
        so_candidates = sorted(build_dir.glob(f"{name}*.so"))

    if not so_candidates:
        raise FileNotFoundError(f"Failed to build extension for {source_path}")

    so_path = so_candidates[0]
    spec = importlib.util.spec_from_file_location(name, so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load extension spec from {so_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EXTENSION = None


def _load_extension() -> object:
    global _EXTENSION
    if _EXTENSION is None:
        _EXTENSION = _build_extension()
    return _EXTENSION


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [batch, heads, seq_len, head_dim]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have identical shapes")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("q, k, and v must be CUDA tensors")
    if q.dtype != _DTYPE or k.dtype != _DTYPE or v.dtype != _DTYPE:
        raise ValueError("q, k, and v must be bfloat16 tensors")
    if q.shape[-1] != _HEAD_DIM:
        raise ValueError(f"head_dim must be {_HEAD_DIM}")
    return q.contiguous(), k.contiguous(), v.contiguous()


def kernel_fn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: float | None = None,
) -> torch.Tensor:
    q, k, v = _validate_inputs(q, k, v)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    extension = _load_extension()
    return extension.flash_attention_2_forward(q, k, v, float(sm_scale), bool(causal))


def get_inputs(
    batch: int = 8,
    head_num: int = 8,
    seq_len: int = 2048,
    head_dim: int = _HEAD_DIM,
    dtype: torch.dtype = _DTYPE,
    causal: bool = True,
    seed: int = 42,
) -> dict[str, torch.Tensor | bool | float]:
    torch.manual_seed(seed)
    device = "cuda"
    q = torch.randn(batch, head_num, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, head_num, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, head_num, seq_len, head_dim, device=device, dtype=dtype)
    return {
        "q": q,
        "k": k,
        "v": v,
        "causal": causal,
        "sm_scale": 1.0 / math.sqrt(head_dim),
    }


def get_flops(
    batch: int = 8,
    head_num: int = 8,
    seq_len: int = 2048,
    head_dim: int = _HEAD_DIM,
    causal: bool = True,
) -> int:
    attn_pairs = seq_len * (seq_len + 1) // 2 if causal else seq_len * seq_len
    return 4 * batch * head_num * attn_pairs * head_dim


def get_bytes(
    batch: int = 8,
    head_num: int = 8,
    seq_len: int = 2048,
    head_dim: int = _HEAD_DIM,
    element_size: int = 2,
) -> int:
    tensor_elems = batch * head_num * seq_len * head_dim
    return tensor_elems * element_size * 4
