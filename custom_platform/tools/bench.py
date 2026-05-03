"""Benchmark harness using the platform adapter abstraction."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from platforms.base import PlatformNotImplementedError
from platforms.registry import get_platform_adapter

try:
    from .compute_traits import compute_kernel_traits, select_primary_size
except ImportError:
    from compute_traits import compute_kernel_traits, select_primary_size


def _import_torch():
    import torch

    return torch


def _compare(output, expected, atol: float, rtol: float) -> dict[str, Any]:
    if hasattr(output, "float") and hasattr(expected, "float"):
        torch = _import_torch()
        out_f = output.float()
        exp_f = expected.float()
        abs_diff = (out_f - exp_f).abs()
        return {
            "match": torch.allclose(out_f, exp_f, atol=atol, rtol=rtol),
            "max_abs_error": abs_diff.max().item(),
        }

    max_abs_error = 0.0

    def _walk(a, b) -> bool:
        nonlocal max_abs_error
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                return False
            return all(_walk(x, y) for x, y in zip(a, b))
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a.keys()) != set(b.keys()):
                return False
            return all(_walk(a[key], b[key]) for key in a)
        try:
            a_f = float(a)
            b_f = float(b)
        except (TypeError, ValueError):
            return a == b
        diff = abs(a_f - b_f)
        max_abs_error = max(max_abs_error, diff)
        limit = atol + rtol * abs(b_f)
        return diff <= limit

    return {"match": _walk(output, expected), "max_abs_error": max_abs_error}


def _write_json_out(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _error_payload(code: str, stage: str, message: str) -> dict[str, str]:
    return {"code": code, "stage": stage, "message": message}


def _has_nan_inf(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_has_nan_inf(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_nan_inf(item) for item in value)
    if hasattr(value, "is_floating_point"):
        torch = _import_torch()
        if value.is_floating_point():
            return bool(torch.isnan(value).any().item() or torch.isinf(value).any().item())
        return False
    if isinstance(value, float):
        return value != value or value in (float("inf"), float("-inf"))
    return False


def _map_floating_leaves(value: Any, transform: Callable[[Any], Any]) -> Any:
    if isinstance(value, dict):
        return {key: _map_floating_leaves(item, transform) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_floating_leaves(item, transform) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_floating_leaves(item, transform) for item in value)
    if hasattr(value, "is_floating_point"):
        return transform(value) if value.is_floating_point() else value
    if isinstance(value, float):
        return transform(value)
    return value


def _default_numerical_stability_cases(
    gen_fn: Callable,
    size: dict[str, Any],
    dtype: Any,
    device: str,
    seed: int = 42,
) -> list[tuple[str, dict[str, Any]]]:
    base_inputs = gen_fn(size, dtype, device, seed=seed)
    return [
        ("baseline", base_inputs),
        ("low_amplitude", _map_floating_leaves(base_inputs, lambda v: v * 0.1)),
        ("high_amplitude", _map_floating_leaves(base_inputs, lambda v: v * 4.0)),
        ("all_zeros", _map_floating_leaves(base_inputs, lambda v: v * 0.0)),
        ("all_same", _map_floating_leaves(base_inputs, lambda v: v * 0.0 + 0.5)),
    ]


def _numerical_stability_cases(
    config: dict[str, Any],
    gen_fn: Callable,
    size: dict[str, Any],
    dtype: Any,
    device: str,
) -> list[tuple[str, dict[str, Any]]]:
    case_builder = config.get("numerical_stability_cases")
    if case_builder is not None:
        return case_builder(size, dtype, device, seed=42)
    return _default_numerical_stability_cases(gen_fn, size, dtype, device, seed=42)


def _prime_kernel(
    kernel_fn: Callable,
    config: dict[str, Any],
    device: str,
    adapter,
) -> dict[str, Any]:
    sizes = config["test_sizes"]
    dtypes = config["test_dtypes"]
    if not sizes:
        raise ValueError("config.test_sizes must not be empty")
    if not dtypes:
        raise ValueError("config.test_dtypes must not be empty")

    label, size = sizes[0]
    dtype = dtypes[0]
    inputs = config["input_generator"](size, dtype, device, seed=42)
    t0 = time.time()
    kernel_fn(**inputs)
    adapter.synchronize()
    return {
        "label": label,
        "dtype": str(dtype),
        "elapsed_ms": (time.time() - t0) * 1000.0,
    }


def run_correctness(kernel_fn: Callable, config: dict[str, Any], device: str) -> dict[str, Any]:
    gen_fn = config["input_generator"]
    ref_fn = config["reference_fn"]
    sizes = config["test_sizes"]
    dtype = config["test_dtypes"][0]
    tol = config["tolerances"][dtype]

    label, size = sizes[0]
    inputs = gen_fn(size, dtype, device, seed=42)
    expected = ref_fn(inputs)
    output = kernel_fn(**inputs)
    cmp = _compare(output, expected, tol["atol"], tol["rtol"])
    smoke_pass = bool(cmp["match"]) and not _has_nan_inf(output)

    stability_pass = True
    stability_max_abs_error = 0.0
    details: list[str] = []
    for case_name, case_inputs in _numerical_stability_cases(config, gen_fn, size, dtype, device):
        expected_case = ref_fn(case_inputs)
        output_case = kernel_fn(**case_inputs)
        if _has_nan_inf(output_case) and not _has_nan_inf(expected_case):
            stability_pass = False
            details.append(f"stability {case_name}: NaN/Inf (reference is clean)")
            continue

        case_cmp = _compare(output_case, expected_case, tol["atol"] * 10.0, tol["rtol"] * 10.0)
        stability_max_abs_error = max(stability_max_abs_error, float(case_cmp["max_abs_error"]))
        if not case_cmp["match"]:
            stability_pass = False
            details.append(
                f"stability {case_name}: max_abs_error={case_cmp['max_abs_error']:.6e}"
            )

    return {
        "checked": True,
        "passed": smoke_pass and stability_pass,
        "label": label,
        "max_abs_error": max(float(cmp["max_abs_error"]), stability_max_abs_error),
        "smoke_test": "PASS" if smoke_pass else "FAIL",
        "numerical_stability": "PASS" if stability_pass else "FAIL",
        "details": details,
    }


def _format_ms_list(values: list[float]) -> str:
    return ",".join(f"{float(v):.4f}" for v in values)


def _summarize_trial_stats(
    trial_ms: list[float],
    *,
    warmup: int,
    rep: int,
    stability_threshold_pct: float,
    timing_source: str,
) -> dict[str, Any]:
    if not trial_ms:
        trial_ms = [0.0]

    values = [float(v) for v in trial_ms]
    sorted_values = sorted(values)
    avg_ms = sum(values) / len(values)
    median_ms = float(statistics.median(sorted_values))
    min_ms = sorted_values[0]
    max_ms = sorted_values[-1]
    std_ms = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    cv_pct = (std_ms / avg_ms * 100.0) if avg_ms > 0 else 0.0
    spread_pct = ((max_ms - min_ms) / median_ms * 100.0) if median_ms > 0 else 0.0
    stable = spread_pct <= stability_threshold_pct

    return {
        "average_ms": avg_ms,
        "median_ms": median_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "std_ms": std_ms,
        "cv_pct": cv_pct,
        "spread_pct": spread_pct,
        "stable": stable,
        "trials_ms": values,
        "trial_count": len(values),
        "warmup": warmup,
        "rep": rep,
        "stability_threshold_pct": stability_threshold_pct,
        "timing_source": timing_source,
    }


def _benchmark_trials(
    adapter,
    fn: Callable[[], Any],
    *,
    warmup: int,
    rep: int,
    trials: int,
    stability_threshold_pct: float,
) -> dict[str, Any]:
    warmup = max(0, int(warmup))
    rep = max(1, int(rep))
    trials = max(1, int(trials))
    values: list[float] = []
    for _ in range(trials):
        adapter.synchronize()
        values.append(float(adapter.benchmark(fn, warmup=warmup, rep=rep)))
        adapter.synchronize()
    return _summarize_trial_stats(
        values,
        warmup=warmup,
        rep=rep,
        stability_threshold_pct=stability_threshold_pct,
        timing_source=f"{getattr(adapter, 'platform_name', 'platform')}.benchmark",
    )


def _result_payload(
    kernel_type: str,
    target_platform: str,
    device_spec,
    correctness: dict[str, Any],
    kernel_stats: dict[str, Any],
    reference_stats: dict[str, Any],
    throughput_tflops: float,
    bandwidth_gb_s: float,
    peak_vram_mb: float,
    bench_time_seconds: float,
    bottleneck: str,
    primary_size_label: str = "",
    primary_size: dict[str, Any] | None = None,
    compute_traits: dict[str, Any] | None = None,
    benchmark_mode: str = "full",
    bench_config: dict[str, Any] | None = None,
    error: dict[str, str] | None = None,
) -> dict[str, Any]:
    peak_compute = float(getattr(device_spec, "peak_tflops_fp16", 0.0) or 0.0)
    peak_memory = float(getattr(device_spec, "peak_bandwidth_gb_s", 0.0) or 0.0)
    pct_peak_compute = (throughput_tflops / peak_compute * 100.0) if peak_compute > 0 else 0.0
    pct_peak_bandwidth = (bandwidth_gb_s / peak_memory * 100.0) if peak_memory > 0 else 0.0
    return {
        "kernel_type": kernel_type,
        "target_platform": target_platform,
        "device_name": getattr(device_spec, "device_name", "unknown"),
        "device_memory_gb": getattr(device_spec, "memory_gb", 0.0),
        "peak_compute_tflops": peak_compute,
        "peak_memory_gbps": peak_memory,
        "correctness": correctness,
        "kernel": kernel_stats,
        "reference": reference_stats,
        "throughput_tflops": throughput_tflops,
        "bandwidth_gb_s": bandwidth_gb_s,
        "peak_vram_mb": peak_vram_mb,
        "achieved_compute_tflops": throughput_tflops,
        "achieved_memory_gbps": bandwidth_gb_s,
        "pct_peak_compute": pct_peak_compute,
        "pct_peak_bandwidth": pct_peak_bandwidth,
        "bottleneck": bottleneck,
        "primary_size_label": primary_size_label,
        "primary_size": primary_size or {},
        "compute_traits": compute_traits or {},
        "benchmark_mode": benchmark_mode,
        "bench_config": bench_config or {},
        "timing_stable": bool(kernel_stats.get("stable", False)) if kernel_stats else False,
        "timing_spread_pct": float(kernel_stats.get("spread_pct", 0.0) or 0.0) if kernel_stats else 0.0,
        "bench_time_seconds": bench_time_seconds,
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a kernel on the target platform")
    parser.add_argument("--platform", default="custom_platform")
    parser.add_argument("--kernel", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--bench-warmup", type=int, default=25)
    parser.add_argument("--bench-rep", type=int, default=100)
    parser.add_argument("--bench-trials", type=int, default=3)
    parser.add_argument("--stability-threshold-pct", type=float, default=1.5)
    args = parser.parse_args()
    bench_warmup = max(0, int(args.bench_warmup))
    bench_rep = max(1, int(args.bench_rep))
    bench_trials = max(1, int(args.bench_trials))
    stability_threshold_pct = max(0.0, float(args.stability_threshold_pct))
    bench_config = {
        "warmup": bench_warmup,
        "rep": bench_rep,
        "trials": bench_trials,
        "stability_threshold_pct": stability_threshold_pct,
    }

    print("=" * 60)
    print("custom_platform Benchmark Harness")
    print("=" * 60)
    print(
        "bench_config: "
        f"warmup={bench_warmup}, rep={bench_rep}, trials={bench_trials}, "
        f"stability_threshold_pct={stability_threshold_pct:.2f}"
    )

    adapter = get_platform_adapter(args.platform)
    payload: dict[str, Any] = {}

    try:
        kernel_module = importlib.import_module("kernel")
        kernel_fn = kernel_module.kernel_fn
        kernel_type = args.kernel or getattr(kernel_module, "KERNEL_TYPE", None)
        target_platform = getattr(kernel_module, "TARGET_PLATFORM", "unknown")
        if kernel_type is None:
            raise RuntimeError("kernel.py has no KERNEL_TYPE")
    except Exception as exc:
        payload = _result_payload(
            kernel_type="unknown",
            target_platform=args.platform,
            device_spec=type("Device", (), {"device_name": "unknown", "memory_gb": 0.0, "peak_tflops_fp16": 0.0, "peak_bandwidth_gb_s": 0.0})(),
            correctness={"checked": False, "passed": False, "label": "", "max_abs_error": 0.0},
            kernel_stats={},
            reference_stats={},
            throughput_tflops=0.0,
            bandwidth_gb_s=0.0,
            peak_vram_mb=0.0,
            bench_time_seconds=0.0,
            bottleneck="unknown",
            error=_error_payload("kernel_import_failed", "import", f"{type(exc).__name__}: {exc}"),
        )
        print("correctness: FAIL")
        print(f"bench_error: failed to import kernel.py ({type(exc).__name__}: {exc})")
        _write_json_out(args.json_out, payload)
        sys.exit(1)

    try:
        from kernel_configs import get_kernel_config

        config = get_kernel_config(kernel_type)
    except KeyError:
        payload = _result_payload(
            kernel_type=kernel_type,
            target_platform=target_platform,
            device_spec=type("Device", (), {"device_name": "unknown", "memory_gb": 0.0, "peak_tflops_fp16": 0.0, "peak_bandwidth_gb_s": 0.0})(),
            correctness={"checked": False, "passed": False, "label": "", "max_abs_error": 0.0},
            kernel_stats={},
            reference_stats={},
            throughput_tflops=0.0,
            bandwidth_gb_s=0.0,
            peak_vram_mb=0.0,
            bench_time_seconds=0.0,
            bottleneck="unknown",
            error=_error_payload("unknown_kernel_type", "setup", f"unknown kernel type '{kernel_type}'"),
        )
        print("correctness: FAIL")
        print(f"bench_error: unknown kernel type '{kernel_type}'")
        _write_json_out(args.json_out, payload)
        sys.exit(1)
    except Exception as exc:
        payload = _result_payload(
            kernel_type=kernel_type,
            target_platform=target_platform,
            device_spec=type("Device", (), {"device_name": "unknown", "memory_gb": 0.0, "peak_tflops_fp16": 0.0, "peak_bandwidth_gb_s": 0.0})(),
            correctness={"checked": False, "passed": False, "label": "", "max_abs_error": 0.0},
            kernel_stats={},
            reference_stats={},
            throughput_tflops=0.0,
            bandwidth_gb_s=0.0,
            peak_vram_mb=0.0,
            bench_time_seconds=0.0,
            bottleneck="unknown",
            error=_error_payload("config_load_failed", "setup", f"{type(exc).__name__}: {exc}"),
        )
        print("correctness: FAIL")
        print(f"bench_error: failed to load kernel config ({type(exc).__name__}: {exc})")
        _write_json_out(args.json_out, payload)
        sys.exit(1)
    print(f"kernel_type: {kernel_type}")
    print(f"target_platform: {target_platform}")

    try:
        device_spec = adapter.detect_device()
        device = adapter.default_device()
        print(f"device_name: {device_spec.device_name}")
        print(f"device_memory_gb: {device_spec.memory_gb}")
        print(f"peak_compute_tflops: {device_spec.peak_tflops_fp16}")
        print(f"peak_memory_gbps: {device_spec.peak_bandwidth_gb_s}")

        print("\n=== KERNEL PRIME ===")
        prime = _prime_kernel(kernel_fn, config, device, adapter)
        print(
            f"kernel_prime: PASS "
            f"(size={prime['label']}, dtype={prime['dtype']}, elapsed_ms={prime['elapsed_ms']:.2f})"
        )

        correctness = run_correctness(kernel_fn, config, device)
        print(f"correctness: {'PASS' if correctness['passed'] else 'FAIL'}")
        print(f"smoke_test: {correctness['smoke_test']}")
        print(f"numerical_stability: {correctness['numerical_stability']}")
        print(f"max_abs_error: {correctness['max_abs_error']:.6e}")
        if not correctness["passed"]:
            payload = _result_payload(
                kernel_type=kernel_type,
                target_platform=target_platform,
                device_spec=device_spec,
                correctness=correctness,
                kernel_stats={},
                reference_stats={},
                throughput_tflops=0.0,
                bandwidth_gb_s=0.0,
                peak_vram_mb=0.0,
                bench_time_seconds=0.0,
                bottleneck="unknown",
                error=_error_payload("correctness_failed", "correctness", "kernel output does not match reference"),
            )
            _write_json_out(args.json_out, payload)
            sys.exit(1)

        label, size = select_primary_size(config)
        dtype = config["test_dtypes"][0]
        inputs = config["input_generator"](size, dtype, device, seed=42)
        ref_fn = config["reference_fn"]
        flops = config["flops_fn"](size)
        nbytes = config["bytes_fn"](size, dtype)

        adapter.reset_peak_memory_stats()
        t0 = time.time()
        kernel_stats = _benchmark_trials(
            adapter,
            lambda: kernel_fn(**inputs),
            warmup=bench_warmup,
            rep=bench_rep,
            trials=bench_trials,
            stability_threshold_pct=stability_threshold_pct,
        )
        wall_s = time.time() - t0
        peak_vram_mb = adapter.get_peak_memory_mb()

        reference_stats = _benchmark_trials(
            adapter,
            lambda: ref_fn(inputs),
            warmup=bench_warmup,
            rep=bench_rep,
            trials=bench_trials,
            stability_threshold_pct=stability_threshold_pct,
        )

        kernel_ms = float(kernel_stats["median_ms"])
        throughput_tflops = flops / (kernel_ms / 1000.0) / 1e12 if kernel_ms > 0 else 0.0
        bandwidth_gb_s = nbytes / (kernel_ms / 1000.0) / 1e9 if kernel_ms > 0 else 0.0
        peak_compute = float(device_spec.peak_tflops_fp16 or 0.0)
        peak_memory = float(device_spec.peak_bandwidth_gb_s or 0.0)
        arithmetic_intensity = flops / nbytes if nbytes > 0 else 0.0
        ridge_point = (peak_compute * 1e12) / (peak_memory * 1e9) if peak_memory > 0 else 0.0
        bottleneck = "memory_bound" if arithmetic_intensity < ridge_point else "compute_bound"
        bench_metrics = {
            "bottleneck": bottleneck,
            "pct_peak_compute": (throughput_tflops / peak_compute * 100.0) if peak_compute > 0 else 0.0,
            "pct_peak_bandwidth": (bandwidth_gb_s / peak_memory * 100.0) if peak_memory > 0 else 0.0,
        }
        benchmark_compute_traits = compute_kernel_traits(
            kernel_type,
            config,
            size,
            dtype=dtype,
            bench_metrics=bench_metrics,
        )

        kernel_stats["label"] = label
        payload = _result_payload(
            kernel_type=kernel_type,
            target_platform=target_platform,
            device_spec=device_spec,
            correctness=correctness,
            kernel_stats=kernel_stats,
            reference_stats=reference_stats,
            throughput_tflops=throughput_tflops,
            bandwidth_gb_s=bandwidth_gb_s,
            peak_vram_mb=peak_vram_mb,
            bench_time_seconds=wall_s,
            bottleneck=bottleneck,
            primary_size_label=label,
            primary_size=size,
            compute_traits=benchmark_compute_traits,
            bench_config=bench_config,
        )

        print(f"latency_ms: {kernel_ms:.4f}")
        print(f"kernel_timing_trials_ms: {_format_ms_list(kernel_stats['trials_ms'])}")
        print(f"kernel_timing_spread_pct: {kernel_stats['spread_pct']:.2f}")
        print(f"kernel_timing_cv_pct: {kernel_stats['cv_pct']:.2f}")
        print(f"kernel_timing_stable: {'yes' if kernel_stats['stable'] else 'no'}")
        print(f"reference_timing_trials_ms: {_format_ms_list(reference_stats['trials_ms'])}")
        print(f"reference_timing_spread_pct: {reference_stats['spread_pct']:.2f}")
        print(f"reference_timing_cv_pct: {reference_stats['cv_pct']:.2f}")
        print(f"reference_timing_stable: {'yes' if reference_stats['stable'] else 'no'}")
        print(f"throughput_tflops: {throughput_tflops:.3f}")
        print(f"bandwidth_gb_s: {bandwidth_gb_s:.1f}")
        print(f"peak_vram_mb: {peak_vram_mb:.1f}")
        print(f"achieved_compute_tflops: {throughput_tflops:.3f}")
        print(f"achieved_memory_gbps: {bandwidth_gb_s:.1f}")
        print(f"peak_compute_tflops: {peak_compute:.3f}")
        print(f"peak_memory_gbps: {peak_memory:.1f}")
        print(f"bottleneck: {bottleneck}")
        print(f"optimization_recommendation: {benchmark_compute_traits['optimization_recommendation']}")
        print(f"optimization_reasoning: {benchmark_compute_traits['optimization_reasoning']}")
        print(f"workload_class: {benchmark_compute_traits['workload_class']}")
        print(f"shape_regime: {benchmark_compute_traits['shape_regime']}")
        print(f"bench_time_seconds: {wall_s:.1f}")
        print(f"benchmark_size: {label}")
        _write_json_out(args.json_out, payload)

    except PlatformNotImplementedError as exc:
        payload = _result_payload(
            kernel_type=kernel_type,
            target_platform=target_platform,
            device_spec=type("Device", (), {"device_name": "unknown", "memory_gb": 0.0, "peak_tflops_fp16": 0.0, "peak_bandwidth_gb_s": 0.0})(),
            correctness={"checked": False, "passed": False, "label": "", "max_abs_error": 0.0},
            kernel_stats={},
            reference_stats={},
            throughput_tflops=0.0,
            bandwidth_gb_s=0.0,
            peak_vram_mb=0.0,
            bench_time_seconds=0.0,
            bottleneck="unknown",
            error=_error_payload("platform_not_implemented", "platform", str(exc)),
        )
        print("correctness: FAIL")
        print(f"bench_error: {exc}")
        _write_json_out(args.json_out, payload)
        sys.exit(1)
    except Exception as exc:
        payload = _result_payload(
            kernel_type=kernel_type,
            target_platform=target_platform,
            device_spec=type("Device", (), {"device_name": "unknown", "memory_gb": 0.0, "peak_tflops_fp16": 0.0, "peak_bandwidth_gb_s": 0.0})(),
            correctness={"checked": False, "passed": False, "label": "", "max_abs_error": 0.0},
            kernel_stats={},
            reference_stats={},
            throughput_tflops=0.0,
            bandwidth_gb_s=0.0,
            peak_vram_mb=0.0,
            bench_time_seconds=0.0,
            bottleneck="unknown",
            error=_error_payload("benchmark_failed", "runtime", f"{type(exc).__name__}: {exc}"),
        )
        print("correctness: FAIL")
        print(f"bench_error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        _write_json_out(args.json_out, payload)
        sys.exit(1)


if __name__ == "__main__":
    main()
