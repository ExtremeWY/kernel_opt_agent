"""Base profiling abstraction with normalized output fields."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ProfilerBackend(ABC):
    backend_name: str = "unknown"

    @abstractmethod
    def collect(self, kernel_file: str, output_dir: Path) -> Path:
        """Run the platform profiler and return the raw report path."""

    @abstractmethod
    def analyze(self, report_path: Path) -> dict[str, str]:
        """Normalize raw profiler output into platform-neutral metrics."""

