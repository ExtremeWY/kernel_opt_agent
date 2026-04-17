"""Placeholder adapter for one concrete hardware platform.

Replace this file with a real implementation for your target hardware.
"""

from __future__ import annotations

from platforms.base import DeviceSpec, PlatformAdapter, PlatformNotImplementedError


class PlaceholderCustomPlatformAdapter(PlatformAdapter):
    platform_name = "custom_platform"

    def _todo(self, topic: str) -> PlatformNotImplementedError:
        return PlatformNotImplementedError(
            f"{topic} is not implemented for '{self.platform_name}'. "
            "Fill custom_platform/platforms/custom_platform/adapter.py."
        )

    def validate_environment(self) -> list[str]:
        return [
            "Vendor runtime validation is not implemented.",
            "Vendor compiler detection is not implemented.",
            "Real device availability check is not implemented.",
        ]

    def detect_device(self) -> DeviceSpec:
        raise self._todo("Device detection")

    def synchronize(self) -> None:
        raise self._todo("Device synchronization")

    def reset_peak_memory_stats(self) -> None:
        raise self._todo("Peak memory reset")

    def get_peak_memory_mb(self) -> float:
        raise self._todo("Peak memory query")

    def benchmark(self, fn, warmup: int = 25, rep: int = 100) -> float:
        raise self._todo("Benchmark timing")

    def default_device(self) -> str:
        return "custom_device"

    def profiler_backend_name(self) -> str:
        return "custom_placeholder"

