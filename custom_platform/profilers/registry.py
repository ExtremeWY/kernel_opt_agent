"""Registry for profiler backends."""

from __future__ import annotations

from profilers.base import ProfilerBackend
from profilers.custom_placeholder import PlaceholderProfilerBackend
from profilers.mock_profiler import MockProfilerBackend


def get_profiler_backend(backend_name: str) -> ProfilerBackend:
    if backend_name == "mock_profiler":
        return MockProfilerBackend()
    if backend_name == "custom_placeholder":
        return PlaceholderProfilerBackend()
    raise ValueError(f"Unknown profiler backend '{backend_name}'")
