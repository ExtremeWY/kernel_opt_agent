"""Shared kernel compute-trait heuristics for conditional optimization guidance."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any

DEFAULT_MMA_TILE_M = 16
_TENSOR_CORE_DTYPES = {"torch.float16", "torch.bfloat16", "float16", "bfloat16", "fp16", "bf16"}


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def parse_metric_percent(text: Any) -> float | None:
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    raw = str(text).strip()
    if not raw:
        return None
    raw = raw.replace(",", "")
    if "(" in raw:
        raw = raw.split("(", 1)[-1].rstrip(")")
    if "=" in raw:
        raw = raw.split("=", 1)[-1]
    raw = raw.replace("%", "").strip()
    try:
        return float(raw)
    except ValueError:
        return None


def _top_stall_name(ncu_metrics: dict[str, Any] | None) -> str:
    if not ncu_metrics:
        return ""
    raw = str(ncu_metrics.get("ncu_top_stall") or "").strip().lower()
    if not raw:
        return ""
    return raw.split(" ", 1)[0]


def load_kernel_type_from_file(kernel_file: str | Path) -> str:
    path = Path(kernel_file).resolve()
    spec = importlib.util.spec_from_file_location("_kernel_compute_traits", str(path))
    if spec is None or spec.loader is None:
        return ""
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return str(getattr(mod, "KERNEL_TYPE", "") or "")


def select_primary_size(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    sizes = config.get("test_sizes") or []
    for label, size in sizes:
        if label == "large":
            return label, size
    if sizes:
        label, size = sizes[-1]
        return label, size
    return "", {}


def _resolve_trait_overrides(kernel_type: str, config: dict[str, Any], size: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    static_meta = config.get("kernel_opt_characteristics")
    if isinstance(static_meta, dict):
        overrides.update(static_meta)
    traits_fn = config.get("optimization_traits")
    if callable(traits_fn):
        try:
            dynamic = traits_fn(size)
        except TypeError:
            dynamic = traits_fn(size=size, kernel_type=kernel_type)
        if isinstance(dynamic, dict):
            overrides.update(dynamic)
    return overrides


def _looks_attention_like(kernel_type: str, size: dict[str, Any]) -> bool:
    keys = set(size)
    lowered = kernel_type.lower()
    if {"batch", "head_num", "seq_len", "head_dim"}.issubset(keys):
        return True
    if {"batch", "seq_len_q", "seq_len_kv", "head_dim"}.issubset(keys):
        return True
    return "attention" in lowered or lowered.startswith("flash_")


def _infer_effective_m(size: dict[str, Any], overrides: dict[str, Any]) -> int | None:
    explicit = _coerce_int(overrides.get("effective_m"))
    if explicit is not None and explicit > 0:
        return explicit
    if "M" in size:
        m_val = _coerce_int(size.get("M"))
        if m_val is not None and m_val > 0:
            return m_val
    batch = _coerce_int(size.get("batch"))
    seq_q = _coerce_int(size.get("seq_len_q"))
    seq = _coerce_int(size.get("seq_len"))
    if batch and seq_q:
        return batch * seq_q
    if batch and seq:
        return batch * seq
    if seq_q:
        return seq_q
    if seq:
        return seq
    return None


def _infer_is_matmul_like(kernel_type: str, size: dict[str, Any], overrides: dict[str, Any]) -> bool:
    explicit = _coerce_bool(overrides.get("is_matmul_like"))
    if explicit is not None:
        return explicit
    if {"M", "N", "K"}.issubset(size):
        return True
    if _looks_attention_like(kernel_type, size):
        return True
    lowered = kernel_type.lower()
    return any(token in lowered for token in ("gemm", "matmul", "attention", "mma"))


def _infer_tensor_core_candidate(is_matmul_like: bool, dtype: Any, overrides: dict[str, Any]) -> bool:
    explicit = _coerce_bool(overrides.get("supports_tensor_core_candidate"))
    if explicit is not None:
        return explicit
    if not is_matmul_like:
        return False
    if dtype is None:
        return True
    return str(dtype) in _TENSOR_CORE_DTYPES


def _shape_regime(size: dict[str, Any], effective_m: int | None, decode_like_risk: bool, small_m_risk: bool) -> str:
    if decode_like_risk:
        return "small_m_decode_like"
    if small_m_risk:
        return "small_m"
    if effective_m is not None and effective_m >= 128:
        return "full_m"
    if any(key in size for key in ("seq_len", "seq_len_q", "seq_len_kv")):
        return "attention_like"
    return "generic"


def _infer_decode_like(size: dict[str, Any], overrides: dict[str, Any], effective_m: int | None) -> bool:
    explicit = _coerce_bool(overrides.get("decode_like_risk"))
    if explicit is not None:
        return explicit
    batch = _coerce_int(size.get("batch")) or 0
    seq_q = _coerce_int(size.get("seq_len_q"))
    seq = _coerce_int(size.get("seq_len"))
    seq_kv = _coerce_int(size.get("seq_len_kv"))
    q_tokens = seq_q if seq_q is not None else seq
    if q_tokens is None:
        return False
    if q_tokens <= 4 and batch and batch < 16:
        return True
    if q_tokens <= 8 and seq_kv and seq_kv >= max(64, q_tokens * 8):
        return True
    return bool(effective_m is not None and effective_m <= 16 and batch and batch < 16)


def compute_kernel_traits(
    kernel_type: str,
    config: dict[str, Any],
    size: dict[str, Any] | None,
    *,
    dtype: Any = None,
    bench_metrics: dict[str, Any] | None = None,
    ncu_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    size = dict(size or {})
    overrides = _resolve_trait_overrides(kernel_type, config, size)
    mma_tile_m = max(1, _coerce_int(overrides.get("mma_tile_m")) or DEFAULT_MMA_TILE_M)
    effective_m = _infer_effective_m(size, overrides)
    is_matmul_like = _infer_is_matmul_like(kernel_type, size, overrides)
    supports_tensor_core_candidate = _infer_tensor_core_candidate(is_matmul_like, dtype, overrides)

    padded_m = None
    mma_m_fill_ratio = None
    padding_overhead_ratio = None
    if effective_m is not None and effective_m > 0:
        padded_m = int(math.ceil(effective_m / mma_tile_m) * mma_tile_m)
        mma_m_fill_ratio = effective_m / padded_m if padded_m > 0 else None
        padding_overhead_ratio = padded_m / effective_m - 1.0 if effective_m > 0 else None

    explicit_small_m = _coerce_bool(overrides.get("small_m_risk"))
    if explicit_small_m is not None:
        small_m_risk = explicit_small_m
    else:
        small_m_risk = bool(
            effective_m is not None
            and (
                effective_m < mma_tile_m * 2
                or (mma_m_fill_ratio is not None and mma_m_fill_ratio < 0.75)
                or (padding_overhead_ratio is not None and padding_overhead_ratio > 0.25)
            )
        )

    decode_like_risk = _infer_decode_like(size, overrides, effective_m) if is_matmul_like else False

    if not is_matmul_like or not supports_tensor_core_candidate:
        mma_shape_risk = "high"
    elif decode_like_risk or (padding_overhead_ratio is not None and padding_overhead_ratio > 0.25):
        mma_shape_risk = "high"
    elif small_m_risk or (mma_m_fill_ratio is not None and mma_m_fill_ratio < 0.9):
        mma_shape_risk = "medium"
    else:
        mma_shape_risk = "low"

    bench_bottleneck = str((bench_metrics or {}).get("bottleneck") or "")
    ncu_bottleneck = str((ncu_metrics or {}).get("ncu_bottleneck") or "")
    compute_bound = bench_bottleneck == "compute_bound" or ncu_bottleneck.startswith("compute_bound")
    tensor_core_pct = parse_metric_percent((ncu_metrics or {}).get("ncu_tensor_core_pct"))
    occupancy_pct = parse_metric_percent((ncu_metrics or {}).get("ncu_occupancy"))
    ipc = parse_metric_percent((ncu_metrics or {}).get("ncu_ipc"))
    top_stall = _top_stall_name(ncu_metrics)
    low_tensor_core_pct = tensor_core_pct is not None and tensor_core_pct < 15.0
    strong_mma_fit = is_matmul_like and supports_tensor_core_candidate and mma_shape_risk == "low"

    if not is_matmul_like:
        recommendation = "avoid"
        reasoning = "Kernel does not look matmul/MMA-like; prefer algorithmic simplification, warp-level compute, instruction-mix, or register-pressure work over a tensor-core rewrite."
    elif not supports_tensor_core_candidate:
        recommendation = "avoid"
        reasoning = "Kernel is matmul-like but the active dtype/path is not an obvious tensor-core candidate; compare compute-path simplifications before considering an MMA rewrite."
    elif decode_like_risk or small_m_risk or mma_shape_risk == "high":
        recommendation = "compare_first"
        reasoning = "Kernel is matmul-like, but the current M-shape is a weak MMA fit; compare a CUDA-core path against a tensor-core path with padding/packing before prioritizing tensor cores."
    elif tensor_core_pct is None and strong_mma_fit:
        recommendation = "needs_ncu_evidence"
        reasoning = "Kernel is matmul-like, compute-oriented, and shape-friendly for MMA, but bench-only evidence is not enough; collect NCU tensor-core utilization before prioritizing a tensor-core rewrite."
    elif compute_bound and low_tensor_core_pct and strong_mma_fit:
        recommendation = "recommended"
        reasoning = "Kernel is matmul-like, compute-bound, shape-friendly for MMA, and shows low tensor-core instruction share; a tensor-core/MMA rewrite is a high-priority structural experiment."
    else:
        recommendation = "avoid"
        if tensor_core_pct is not None and tensor_core_pct >= 15.0:
            reasoning = "Tensor-core activity is already present; focus on tile shape, pipeline depth, instruction mix, or register pressure before pushing harder on tensor-core enablement."
        elif compute_bound and is_matmul_like and strong_mma_fit:
            reasoning = "Kernel is matmul-like and MMA-friendly, but without NCU evidence of low tensor-core utilization it should not yet be treated as a tensor-core-first redesign candidate."
        elif compute_bound:
            reasoning = "Kernel is compute-bound but the evidence does not yet justify a tensor-core-first rewrite; improve launch shape, instruction mix, or register pressure first."
        else:
            reasoning = "Macro evidence is not strongly compute-bound, so tensor-core restructuring is not the first lever to pull."

    shape_regime = str(overrides.get("shape_regime") or _shape_regime(size, effective_m, decode_like_risk, small_m_risk))

    notes: list[str] = []
    if effective_m is not None:
        notes.append(f"effective_m={effective_m}")
    if mma_m_fill_ratio is not None:
        notes.append(f"mma_m_fill_ratio={mma_m_fill_ratio:.2f}")
    if padding_overhead_ratio is not None:
        notes.append(f"padding_overhead_ratio={padding_overhead_ratio:.2f}")
    if tensor_core_pct is not None:
        notes.append(f"ncu_tensor_core_pct={tensor_core_pct:.1f}%")
    if occupancy_pct is not None:
        notes.append(f"occupancy={occupancy_pct:.1f}%")
    if ipc is not None:
        notes.append(f"ipc={ipc:.2f}")
    if top_stall:
        notes.append(f"top_stall={top_stall}")

    return {
        "is_matmul_like": is_matmul_like,
        "supports_tensor_core_candidate": supports_tensor_core_candidate,
        "mma_shape_risk": mma_shape_risk,
        "small_m_risk": small_m_risk,
        "decode_like_risk": decode_like_risk,
        "mma_tile_m": mma_tile_m,
        "effective_m": effective_m,
        "mma_m_fill_ratio": mma_m_fill_ratio,
        "padding_overhead_ratio": padding_overhead_ratio,
        "shape_regime": shape_regime,
        "tensor_core_recommendation": recommendation,
        "tensor_core_reasoning": reasoning,
        "top_stall": top_stall,
        "observations": notes,
    }
