"""Registry for platform adapters."""

from __future__ import annotations

from platforms.base import PlatformAdapter
from platforms.mock_platform.adapter import MockPlatformAdapter
from platforms.custom_platform.adapter import PlaceholderCustomPlatformAdapter


def get_platform_adapter(platform_name: str) -> PlatformAdapter:
    if platform_name == "mock_platform":
        return MockPlatformAdapter()
    if platform_name == "custom_platform":
        return PlaceholderCustomPlatformAdapter()
    raise ValueError(f"Unknown platform '{platform_name}'")
