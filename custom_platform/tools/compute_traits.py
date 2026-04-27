"""Platform-neutral workload traits for optimization guidance."""

from __future__ import annotations

from typing import Any


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip().replace(",", "").replace("%", "")
    if not raw:
        return None
    if "=" in raw:
        raw = raw.split("=", 1)[-1]
    try:
        return float(raw)
    except ValueError:
        return None


def _coerce_int(value: Any) -> int | None:
    number = _coerce_float(value)
    return int(number) if number is not None else None


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


def _infer_effective_work_items(size: dict[str, Any], overrides: dict[str, Any]) -> int | None:
    explicit = _coerce_int(overrides.get("effective_work_items"))
    if explicit is not None and explicit > 0:
        return explicit
    total = 1
    seen = False
    for value in size.values():
        number = _coerce_int(value)
        if number is None or number <= 0:
            continue
        total *= number
        seen = True
    return total if seen else None


def _infer_workload_class(kernel_type: str, size: dict[str, Any], overrides: dict[str, Any]) -> str:
    explicit = str(overrides.get("workload_class") or "").strip()
    if explicit:
        return explicit
    lowered = kernel_type.lower()
    keys = set(size)
    if {"M", "N", "K"}.issubset(keys) or any(token in lowered for token in ("matmul", "gemm")):
        return "dense_linear_algebra"
    if any(token in lowered for token in ("norm", "sum", "reduce", "softmax")):
        return "reduction"
    if any(token in lowered for token in ("quant", "elementwise", "activation", "swiglu")):
        return "elementwise"
    if any(key in keys for key in ("seq_len", "seq_len_q", "seq_len_kv")):
        return "sequence"
    return "generic"


def _shape_regime(work_items: int | None, overrides: dict[str, Any]) -> str:
    explicit = str(overrides.get("shape_regime") or "").strip()
    if explicit:
        return explicit
    if work_items is None:
        return "unknown"
    if work_items < 4096:
        return "small"
    if work_items < 1_000_000:
        return "medium"
    return "large"


def compute_kernel_traits(
    kernel_type: str,
    config: dict[str, Any],
    size: dict[str, Any] | None,
    *,
    dtype: Any = None,
    bench_metrics: dict[str, Any] | None = None,
    profile_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    size = dict(size or {})
    overrides = _resolve_trait_overrides(kernel_type, config, size)
    work_items = _infer_effective_work_items(size, overrides)
    workload_class = _infer_workload_class(kernel_type, size, overrides)
    shape_regime = _shape_regime(work_items, overrides)

    bottleneck = str((bench_metrics or {}).get("bottleneck") or "unknown")
    compute_util = _coerce_float((profile_metrics or {}).get("profile_compute_util"))
    memory_util = _coerce_float((profile_metrics or {}).get("profile_memory_util"))
    top_stall = str((profile_metrics or {}).get("profile_top_stall") or "").strip()
    accelerator_candidate = _coerce_bool(overrides.get("accelerator_candidate"))
    if accelerator_candidate is None:
        accelerator_candidate = workload_class in {"dense_linear_algebra", "reduction"}

    memory_dominant = memory_util is not None and (compute_util is None or memory_util >= compute_util)
    if bottleneck == "memory_bound" or memory_dominant:
        recommendation = "memory_path"
        guidance_class = "memory_optimization"
        reasoning = (
            "Bandwidth or memory utilization is the dominant signal; prioritize layout, locality, transfer volume, "
            "and coalescing-style changes."
        )
    elif bottleneck == "compute_bound" or (compute_util is not None and compute_util > 0):
        recommendation = "compute_path"
        guidance_class = "compute_optimization"
        reasoning = (
            "Compute-side utilization is the dominant signal; prioritize algorithmic work reduction, "
            "instruction mix, vectorization, and launch shape."
        )
    elif top_stall:
        recommendation = "profile_driven"
        guidance_class = "profile_driven"
        reasoning = (
            "Profile output identified a stall signal; use the platform profiler details to select the next "
            "experiment."
        )
    else:
        recommendation = "needs_profile_evidence"
        guidance_class = "needs_profile_evidence"
        reasoning = (
            "Benchmark evidence is not specific enough; collect platform profiler evidence before choosing a "
            "structural path."
        )

    next_steps = {
        "memory_path": [
            "Check whether the hot path moves fewer bytes or reuses data more effectively.",
            "Validate that any layout or tiling change remains valid across supported sizes.",
        ],
        "compute_path": [
            "Estimate dynamic instruction or operation coverage before editing.",
            "Prefer changes that affect the steady-state hot path over boundary-only cases.",
        ],
        "profile_driven": [
            "Map the top stall or utilization signal to one focused optimization hypothesis.",
            "Compare the next profile against the current report before keeping the change.",
        ],
        "needs_profile_evidence": [
            "Run the platform profiler and record the dominant stall or utilization signal.",
            "Avoid committing to hardware-specific rewrites without profiler support.",
        ],
    }[recommendation]

    observations: list[str] = []
    if work_items is not None:
        observations.append(f"effective_work_items={work_items}")
    if dtype is not None:
        observations.append(f"dtype={dtype}")
    if top_stall:
        observations.append(f"profile_top_stall={top_stall}")
    if compute_util is not None:
        observations.append(f"profile_compute_util={compute_util}")
    if memory_util is not None:
        observations.append(f"profile_memory_util={memory_util}")

    return {
        "workload_class": workload_class,
        "shape_regime": shape_regime,
        "effective_work_items": work_items,
        "accelerator_candidate": accelerator_candidate,
        "optimization_recommendation": recommendation,
        "optimization_reasoning": reasoning,
        "guidance_class": guidance_class,
        "next_steps": next_steps,
        "top_stall": top_stall,
        "observations": observations,
    }
