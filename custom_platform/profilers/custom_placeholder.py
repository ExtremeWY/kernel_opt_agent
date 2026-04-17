"""Placeholder profiling backend for the target platform."""

from __future__ import annotations

from pathlib import Path

from profilers.base import ProfilerBackend
from platforms.base import PlatformNotImplementedError


class PlaceholderProfilerBackend(ProfilerBackend):
    backend_name = "custom_placeholder"

    def collect(self, kernel_file: str, output_dir: Path) -> Path:
        raise PlatformNotImplementedError(
            "Profiler collection is not implemented. "
            "Fill custom_platform/profilers/custom_placeholder.py."
        )

    def analyze(self, report_path: Path) -> dict[str, str]:
        raise PlatformNotImplementedError(
            "Profiler report parsing is not implemented. "
            "Fill custom_platform/profilers/custom_placeholder.py."
        )

