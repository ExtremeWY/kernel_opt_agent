"""Base abstractions for target hardware integration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


class PlatformNotImplementedError(NotImplementedError):
    """Raised when the placeholder platform backend has not been implemented yet."""


@dataclass
class DeviceSpec:
    platform_name: str
    device_name: str
    device_id: int = 0
    memory_gb: float = 0.0
    peak_tflops_fp16: float = 0.0
    peak_tflops_bf16: float = 0.0
    peak_tflops_fp32: float = 0.0
    peak_bandwidth_gb_s: float = 0.0
    l2_cache_mb: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class PlatformAdapter(ABC):
    """Execution backend abstraction used by prepare, bench, and run_loop."""

    platform_name: str = "unknown"

    @abstractmethod
    def validate_environment(self) -> list[str]:
        """Return human-readable environment issues. Empty means ready."""

    @abstractmethod
    def detect_device(self) -> DeviceSpec:
        """Return the active device specification."""

    @abstractmethod
    def synchronize(self) -> None:
        """Synchronize the target device."""

    @abstractmethod
    def reset_peak_memory_stats(self) -> None:
        """Reset runtime memory accounting before a benchmark run."""

    @abstractmethod
    def get_peak_memory_mb(self) -> float:
        """Return peak memory usage from the last benchmark run."""

    @abstractmethod
    def benchmark(self, fn: Callable[[], Any], warmup: int = 25, rep: int = 100) -> float:
        """Return median runtime in milliseconds."""

    @abstractmethod
    def default_device(self) -> str:
        """Return the device string consumed by input generators."""

    @abstractmethod
    def profiler_backend_name(self) -> str:
        """Return the profiler backend key."""

