"""Profile wrapper that delegates to the platform-specific profiler backend."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platforms.registry import get_platform_adapter
from profilers.registry import get_profiler_backend
from platforms.base import PlatformNotImplementedError


def _write_text(path: Path | None, content: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _render_summary(results: dict[str, str]) -> str:
    lines = ["# Profile Summary", ""]
    for key, val in sorted(results.items()):
        lines.append(f"- {key}: {val}")
    return "\n".join(lines) + "\n"


def _render_details(results: dict[str, str]) -> str:
    lines = ["# Profile Details", ""]
    for key, val in sorted(results.items()):
        lines.append(f"{key}: {val}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a kernel on the target platform")
    parser.add_argument("--platform", default="custom_platform")
    parser.add_argument("--kernel-file", default="kernel.py")
    parser.add_argument("--output-dir", default=str(ROOT / "workspace" / "profile_reports"))
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--details-out", default=None)
    args = parser.parse_args()

    adapter = get_platform_adapter(args.platform)
    backend_name = adapter.profiler_backend_name()
    backend = get_profiler_backend(backend_name)
    kernel_path = ROOT / args.kernel_file
    if not kernel_path.exists():
        print(f"profile_error: kernel file not found: {kernel_path}")
        sys.exit(1)

    try:
        output_dir = Path(args.output_dir)
        report_path = backend.collect(str(kernel_path), output_dir)
        results = backend.analyze(report_path)
        summary_text = _render_summary(results)
        details_text = _render_details(results)
        _write_text(Path(args.summary_out) if args.summary_out else None, summary_text)
        _write_text(Path(args.details_out) if args.details_out else None, details_text)
        print("=== PROFILE ANALYSIS ===")
        print(f"profile_report_path: {report_path}")
        for key, val in sorted(results.items()):
            print(f"{key}: {val}")
        print("=== END PROFILE ANALYSIS ===")
    except PlatformNotImplementedError as exc:
        print(f"profile_error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
