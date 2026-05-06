"""Qwen3.5 MoE GDN chunked BF16 Tensor Core prefill wrapper."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import textwrap

import torch


KERNEL_TYPE = "qwen35moe_gdn_prefill"
D = 128
H_K = 16
H_V = 32


def _extension_name(source_path: Path) -> str:
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()[:12]
    return f"qwen35moe_gdn_prefill_bf16tc_cuda_{digest}"


def _repo_root_for(source_path: Path) -> Path:
    for parent in (source_path.parent, *source_path.parents):
        if (parent / "program.md").is_file():
            return parent
    return source_path.parent


def _build_extension() -> object:
    source_path = Path(__file__).resolve().with_suffix(".cu")
    build_root = _repo_root_for(source_path) / "workspace" / ".torch_extensions"
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


def _validate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v must be [batch, tokens, heads, dim]")
    if g.ndim != 3 or beta.ndim != 3:
        raise ValueError("g/beta must be [batch, tokens, H_v]")
    if state.ndim != 4:
        raise ValueError("state must be [batch, H_v, D, D] in [v_col, k_row] layout")
    if q.shape[-1] != D or k.shape[-1] != D or v.shape[-1] != D:
        raise ValueError("head dimension must be 128")
    if q.shape[2] != H_K or k.shape[2] != H_K or v.shape[2] != H_V:
        raise ValueError("expected H_k=16 and H_v=32")
    if q.shape != k.shape:
        raise ValueError("q and k shapes must match")
    if v.shape[:2] != q.shape[:2] or g.shape != q.shape[:2] + (H_V,) or beta.shape != g.shape:
        raise ValueError("inconsistent token/head shapes")
    if state.shape != (q.shape[0], H_V, D, D):
        raise ValueError("state shape must be [batch, 32, 128, 128]")
    tensors = (q, k, v, g, beta, state)
    if any(not t.is_cuda for t in tensors):
        raise ValueError("all tensors must be CUDA tensors")
    if any(t.dtype != torch.float32 for t in tensors):
        raise ValueError("llama.cpp GDN interface tensors must be float32")
    return tuple(t.contiguous() for t in tensors)


def kernel_fn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    scale: float | None = None,
) -> list[torch.Tensor]:
    q, k, v, g, beta, state = _validate(q, k, v, g, beta, state)
    if scale is None:
        scale = D ** -0.5
    return _load_extension().qwen35moe_gdn_prefill_bf16tc(q, k, v, g, beta, state, float(scale))


def get_inputs(batch: int = 1, seq_len: int = 4096, dtype: torch.dtype = torch.float32, seed: int = 42) -> dict:
    from kernels.qwen35moe_gdn_prefill import get_inputs as _get_inputs

    return _get_inputs(batch=batch, seq_len=seq_len, dtype=dtype, seed=seed)


def get_flops(batch: int = 1, seq_len: int = 4096) -> int:
    return 4 * batch * H_V * seq_len * D * D


def get_bytes(batch: int = 1, seq_len: int = 4096, element_size: int = 4) -> int:
    qk = 2 * batch * seq_len * H_K * D
    vo = 2 * batch * seq_len * H_V * D
    gb = 2 * batch * seq_len * H_V
    st = 2 * batch * H_V * D * D
    return (qk + vo + gb + st) * element_size
