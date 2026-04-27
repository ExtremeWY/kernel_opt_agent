"""Benchmark harness using the platform adapter abstraction."""

from __future__ import annotations

import argparse
import importlib
import json
import os
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
    return {
        "checked": True,
        "passed": bool(cmp["match"]),
        "label": label,
        "max_abs_error": cmp["max_abs_error"],
    }


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
        "bench_time_seconds": bench_time_seconds,
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a kernel on the target platform")
    parser.add_argument("--platform", default="custom_platform")
    parser.add_argument("--kernel", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("custom_platform Benchmark Harness")
    print("=" * 60)

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

        correctness = run_correctness(kernel_fn, config, device)
        print(f"correctness: {'PASS' if correctness['passed'] else 'FAIL'}")
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

        label, size = config["test_sizes"][-1]
        dtype = config["test_dtypes"][0]
        inputs = config["input_generator"](size, dtype, device, seed=42)
        ref_fn = config["reference_fn"]
        flops = config["flops_fn"](size)
        nbytes = config["bytes_fn"](size, dtype)

        adapter.reset_peak_memory_stats()
        t0 = time.time()
        kernel_ms = adapter.benchmark(lambda: kernel_fn(**inputs))
        adapter.synchronize()
        wall_s = time.time() - t0
        peak_vram_mb = adapter.get_peak_memory_mb()

        reference_ms = adapter.benchmark(lambda: ref_fn(inputs))
        adapter.synchronize()

        throughput_tflops = flops / (kernel_ms / 1000.0) / 1e12 if kernel_ms > 0 else 0.0
        bandwidth_gb_s = nbytes / (kernel_ms / 1000.0) / 1e9 if kernel_ms > 0 else 0.0
        peak_compute = float(device_spec.peak_tflops_fp16 or 0.0)
        peak_memory = float(device_spec.peak_bandwidth_gb_s or 0.0)
        arithmetic_intensity = flops / nbytes if nbytes > 0 else 0.0
        ridge_point = (peak_compute * 1e12) / (peak_memory * 1e9) if peak_memory > 0 else 0.0
        bottleneck = "memory_bound" if arithmetic_intensity < ridge_point else "compute_bound"

        kernel_stats = {
            "average_ms": kernel_ms,
            "median_ms": kernel_ms,
            "min_ms": kernel_ms,
            "max_ms": kernel_ms,
            "label": label,
        }
        reference_stats = {
            "average_ms": reference_ms,
            "median_ms": reference_ms,
            "min_ms": reference_ms,
            "max_ms": reference_ms,
        }
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
        )

        print(f"latency_ms: {kernel_ms:.4f}")
        print(f"throughput_tflops: {throughput_tflops:.3f}")
        print(f"bandwidth_gb_s: {bandwidth_gb_s:.1f}")
        print(f"peak_vram_mb: {peak_vram_mb:.1f}")
        print(f"achieved_compute_tflops: {throughput_tflops:.3f}")
        print(f"achieved_memory_gbps: {bandwidth_gb_s:.1f}")
        print(f"peak_compute_tflops: {peak_compute:.3f}")
        print(f"peak_memory_gbps: {peak_memory:.1f}")
        print(f"bottleneck: {bottleneck}")
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
