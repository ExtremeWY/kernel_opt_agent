#!/usr/bin/env python3
"""Preflight checks for the custom platform scaffold."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platforms.registry import get_platform_adapter


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def collect_preflight(platform_name: str, kernel_file: Path, allow_placeholder: bool = False) -> dict[str, Any]:
    adapter = get_platform_adapter(platform_name)
    issues = adapter.validate_environment()
    ready = not issues
    if allow_placeholder and issues:
        ready = True

    requirements = [
        {"name": "repo root", "ok": ROOT.exists(), "detail": str(ROOT), "required": True},
        {"name": "kernel file", "ok": kernel_file.exists(), "detail": str(kernel_file), "required": True},
    ]
    for issue in issues:
        requirements.append({"name": "platform validation", "ok": False, "detail": issue, "required": not allow_placeholder})

    return {
        "checked_at": now_iso(),
        "ready": ready,
        "platform": platform_name,
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "host_platform": platform.platform(),
        "allow_placeholder": allow_placeholder,
        "requirements": requirements,
        "warnings": issues if allow_placeholder else [],
        "errors": [] if allow_placeholder else issues,
    }


def render_preflight_markdown(preflight: dict[str, Any]) -> str:
    lines = [
        "# custom_platform Preflight",
        "",
        "## Status",
        f"- ready: {'yes' if preflight.get('ready') else 'no'}",
        f"- checked at: {preflight.get('checked_at', '')}",
        f"- platform: {preflight.get('platform', '')}",
        f"- python: {preflight.get('python_executable', '')}",
        f"- python version: {preflight.get('python_version', '')}",
        f"- allow placeholder: {'yes' if preflight.get('allow_placeholder') else 'no'}",
        "",
        "## Requirements",
        "",
        "| Requirement | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for item in preflight.get("requirements", []):
        lines.append(
            f"| {item.get('name', '')} | {'ok' if item.get('ok') else 'missing'} | {item.get('detail', '')} |"
        )

    warnings = preflight.get("warnings", [])
    if warnings:
        lines.extend(["", "## Warnings"])
        for warning in warnings:
            lines.append(f"- {warning}")

    errors = preflight.get("errors", [])
    if errors:
        lines.extend(["", "## Errors"])
        for error in errors:
            lines.append(f"- {error}")

    return "\n".join(lines) + "\n"


def write_preflight_outputs(preflight: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(preflight, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_preflight_markdown(preflight), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preflight checks for custom_platform")
    parser.add_argument("--platform", default="custom_platform")
    parser.add_argument("--kernel-file", default="kernel.py")
    parser.add_argument("--allow-placeholder", action="store_true")
    parser.add_argument("--json-out", default=str(ROOT / "workspace" / "preflight_check.json"))
    parser.add_argument("--md-out", default=str(ROOT / "workspace" / "preflight_check.md"))
    args = parser.parse_args()

    preflight = collect_preflight(args.platform, ROOT / args.kernel_file, allow_placeholder=args.allow_placeholder)
    write_preflight_outputs(preflight, Path(args.json_out), Path(args.md_out))
    print(render_preflight_markdown(preflight))
    sys.exit(0 if preflight.get("ready") else 1)


if __name__ == "__main__":
    main()
