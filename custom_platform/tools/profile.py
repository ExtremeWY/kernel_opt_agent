"""Profile wrapper that delegates to the platform-specific profiler backend."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platforms.base import PlatformNotImplementedError
from platforms.registry import get_platform_adapter
from profilers.custom_placeholder import PlaceholderProfilerBackend


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a kernel on the target platform")
    parser.add_argument("--platform", default="custom_platform")
    parser.add_argument("--kernel-file", default="kernel.py")
    args = parser.parse_args()

    adapter = get_platform_adapter(args.platform)
    backend_name = adapter.profiler_backend_name()
    if backend_name != "custom_placeholder":
        raise RuntimeError(
            "Only the placeholder backend is wired in this scaffold. "
            "Update tools/profile.py after implementing your profiler backend."
        )

    backend = PlaceholderProfilerBackend()
    kernel_path = ROOT / args.kernel_file
    if not kernel_path.exists():
        print(f"profile_error: kernel file not found: {kernel_path}")
        sys.exit(1)

    try:
        report_path = backend.collect(str(kernel_path), ROOT / "workspace" / "profile_reports")
        results = backend.analyze(report_path)
        print("=== PROFILE ANALYSIS ===")
        for key, val in sorted(results.items()):
            print(f"{key}: {val}")
        print("=== END PROFILE ANALYSIS ===")
    except PlatformNotImplementedError as exc:
        print(f"profile_error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

