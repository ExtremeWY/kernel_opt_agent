"""Mock platform adapter for local end-to-end workflow validation."""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable

from platforms.base import DeviceSpec, PlatformAdapter


class MockPlatformAdapter(PlatformAdapter):
    platform_name = "mock_platform"

    def __init__(self) -> None:
        self._peak_memory_mb = 0.0

    def validate_environment(self) -> list[str]:
        return []

    def detect_device(self) -> DeviceSpec:
        return DeviceSpec(
            platform_name=self.platform_name,
            device_name="Mock Device v1",
            device_id=0,
            memory_gb=8.0,
            peak_tflops_fp16=4.0,
            peak_tflops_bf16=4.0,
            peak_tflops_fp32=2.0,
            peak_bandwidth_gb_s=256.0,
            l2_cache_mb=4.0,
            metadata={"mode": "software_mock"},
        )

    def synchronize(self) -> None:
        return None

    def reset_peak_memory_stats(self) -> None:
        self._peak_memory_mb = 0.0

    def get_peak_memory_mb(self) -> float:
        return self._peak_memory_mb

    def benchmark(self, fn: Callable[[], Any], warmup: int = 5, rep: int = 20) -> float:
        for _ in range(max(0, warmup)):
            fn()
        samples = []
        for _ in range(max(1, rep)):
            t0 = time.perf_counter()
            out = fn()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            samples.append(dt_ms)
            self._peak_memory_mb = max(self._peak_memory_mb, self._estimate_size_mb(out))
        return statistics.median(samples)

    def default_device(self) -> str:
        return "mock_device"

    def profiler_backend_name(self) -> str:
        return "mock_profiler"

    def _estimate_size_mb(self, value: Any) -> float:
        def count_scalars(obj: Any) -> int:
            if isinstance(obj, (int, float, bool)):
                return 1
            if isinstance(obj, list):
                return sum(count_scalars(item) for item in obj)
            if isinstance(obj, tuple):
                return sum(count_scalars(item) for item in obj)
            if isinstance(obj, dict):
                return sum(count_scalars(item) for item in obj.values())
            return 0

        return count_scalars(value) * 8.0 / (1024.0 * 1024.0)
