"""Benchmark harness using the platform adapter abstraction."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback
from typing import Callable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from kernel_configs import KERNEL_CONFIGS
from platforms.base import PlatformNotImplementedError
from platforms.registry import get_platform_adapter


def _compare(output: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> dict:
    out_f = output.float()
    exp_f = expected.float()
    abs_diff = (out_f - exp_f).abs()
    return {
        "match": torch.allclose(out_f, exp_f, atol=atol, rtol=rtol),
        "max_abs_error": abs_diff.max().item(),
    }


def run_correctness(kernel_fn: Callable, config: dict, device: str) -> dict:
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
        "correctness": "PASS" if cmp["match"] else "FAIL",
        "label": label,
        "max_abs_error": cmp["max_abs_error"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a kernel on the target platform")
    parser.add_argument("--platform", default="custom_platform")
    parser.add_argument("--kernel", default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("custom_platform Benchmark Harness")
    print("=" * 60)

    adapter = get_platform_adapter(args.platform)

    try:
        kernel_module = importlib.import_module("kernel")
        kernel_fn = kernel_module.kernel_fn
        kernel_type = args.kernel or getattr(kernel_module, "KERNEL_TYPE", None)
        target_platform = getattr(kernel_module, "TARGET_PLATFORM", "unknown")
        if kernel_type is None:
            raise RuntimeError("kernel.py has no KERNEL_TYPE")
    except Exception as exc:
        print(f"correctness: FAIL")
        print(f"bench_error: failed to import kernel.py ({type(exc).__name__}: {exc})")
        sys.exit(1)

    if kernel_type not in KERNEL_CONFIGS:
        print("correctness: FAIL")
        print(f"bench_error: unknown kernel type '{kernel_type}'")
        sys.exit(1)

    config = KERNEL_CONFIGS[kernel_type]
    print(f"kernel_type: {kernel_type}")
    print(f"target_platform: {target_platform}")

    try:
        device_spec = adapter.detect_device()
        device = adapter.default_device()
    except PlatformNotImplementedError as exc:
        print("correctness: FAIL")
        print(f"bench_error: {exc}")
        sys.exit(1)

    print(f"device_name: {device_spec.device_name}")
    print(f"device_memory_gb: {device_spec.memory_gb}")
    print(f"peak_compute_tflops: {device_spec.peak_tflops_fp16}")
    print(f"peak_memory_gbps: {device_spec.peak_bandwidth_gb_s}")

    try:
        correctness = run_correctness(kernel_fn, config, device)
        print(f"correctness: {correctness['correctness']}")
        print(f"max_abs_error: {correctness['max_abs_error']:.6e}")
        if correctness["correctness"] != "PASS":
            sys.exit(1)

        label, size = config["test_sizes"][-1]
        dtype = config["test_dtypes"][0]
        inputs = config["input_generator"](size, dtype, device, seed=42)
        flops = config["flops_fn"](size)
        nbytes = config["bytes_fn"](size, dtype)

        adapter.reset_peak_memory_stats()
        t0 = time.time()
        kernel_ms = adapter.benchmark(lambda: kernel_fn(**inputs))
        adapter.synchronize()
        wall_s = time.time() - t0
        peak_vram_mb = adapter.get_peak_memory_mb()

        throughput_tflops = flops / (kernel_ms / 1000.0) / 1e12 if kernel_ms > 0 else 0.0
        bandwidth_gb_s = nbytes / (kernel_ms / 1000.0) / 1e9 if kernel_ms > 0 else 0.0
        peak_compute = device_spec.peak_tflops_fp16
        peak_memory = device_spec.peak_bandwidth_gb_s
        arithmetic_intensity = flops / nbytes if nbytes > 0 else 0.0
        ridge_point = (peak_compute * 1e12) / (peak_memory * 1e9) if peak_memory > 0 else 0.0
        bottleneck = "memory_bound" if arithmetic_intensity < ridge_point else "compute_bound"

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

    except PlatformNotImplementedError as exc:
        print("correctness: FAIL")
        print(f"bench_error: {exc}")
        sys.exit(1)
    except Exception as exc:
        print("correctness: FAIL")
        print(f"bench_error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

