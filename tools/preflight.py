#!/usr/bin/env python3
"""Environment preflight checks for cuda-evolve."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "workspace"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def trim_output(text: str, max_lines: int = 20) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines] + ["..."])


def run_probe(cmd: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
            cwd=ROOT,
        )
    except OSError as exc:
        return {
            "command": " ".join(cmd),
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "command": " ".join(cmd),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def add_requirement(
    requirements: list[dict[str, Any]],
    errors: list[str],
    name: str,
    ok: bool,
    detail: str,
    *,
    required: bool = True,
) -> None:
    requirements.append(
        {
            "name": name,
            "ok": ok,
            "detail": detail,
            "required": required,
        }
    )
    if required and not ok:
        errors.append(f"{name}: {detail}")


def probe_executable(candidate: str, version_args: list[str]) -> dict[str, Any]:
    resolved = shutil.which(candidate)
    info: dict[str, Any] = {
        "requested": candidate,
        "resolved": resolved or "",
        "exists": bool(resolved),
        "version_command": "",
        "version_returncode": None,
        "version_output": "",
    }
    if not resolved:
        return info

    probe = run_probe([resolved, *version_args])
    output = (probe["stdout"] or probe["stderr"]).strip()
    info["version_command"] = probe["command"]
    info["version_returncode"] = probe["returncode"]
    info["version_output"] = trim_output(output)
    return info


def probe_nvidia_smi() -> dict[str, Any]:
    resolved = shutil.which("nvidia-smi")
    info: dict[str, Any] = {
        "exists": bool(resolved),
        "resolved": resolved or "",
        "query_command": "",
        "returncode": None,
        "query_output": "",
        "gpus": [],
    }
    if not resolved:
        return info

    probe = run_probe([resolved, "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"])
    info["query_command"] = probe["command"]
    info["returncode"] = probe["returncode"]
    info["query_output"] = trim_output((probe["stdout"] or probe["stderr"]).strip())
    if probe["returncode"] == 0:
        for line in probe["stdout"].splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3:
                info["gpus"].append(
                    {
                        "name": parts[0],
                        "compute_capability": parts[1],
                        "driver_version": parts[2],
                    }
                )
    return info


def probe_torch_cuda(gpu_index: int) -> dict[str, Any]:
    info: dict[str, Any] = {
        "importable": False,
        "version": "",
        "cuda_version": "",
        "cuda_available": False,
        "device_count": 0,
        "selected_gpu_index": gpu_index,
        "selected_gpu_name": "",
        "selected_gpu_compute_capability": "",
        "selected_sm": "",
        "error": "",
    }
    try:
        import torch
    except Exception as exc:  # pragma: no cover - import path depends on env
        info["error"] = str(exc)
        return info

    info["importable"] = True
    info["version"] = getattr(torch, "__version__", "")
    info["cuda_version"] = getattr(torch.version, "cuda", "") or ""

    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device_count"] = int(torch.cuda.device_count())
            if 0 <= gpu_index < info["device_count"]:
                info["selected_gpu_name"] = torch.cuda.get_device_name(gpu_index)
                major, minor = torch.cuda.get_device_capability(gpu_index)
                info["selected_gpu_compute_capability"] = f"{major}.{minor}"
                info["selected_sm"] = f"sm_{major}{minor}"
    except Exception as exc:  # pragma: no cover - runtime env dependent
        info["error"] = str(exc)
    return info


def collect_preflight(
    gpu: int = 0,
    kernel_file: Path | None = None,
    *,
    nvcc_bin: str = "nvcc",
    ncu_bin: str = "ncu",
) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    requirements: list[dict[str, Any]] = []

    preflight: dict[str, Any] = {
        "checked_at": now_iso(),
        "ready": False,
        "python_executable": sys.executable,
        "python_version": sys.version.splitlines()[0],
        "selected_gpu_index": gpu,
        "env_vars": {
            "CUDA_PATH": os.environ.get("CUDA_PATH", ""),
            "CUDA_HOME": os.environ.get("CUDA_HOME", ""),
            "CUDA_ROOT": os.environ.get("CUDA_ROOT", ""),
        },
        "requirements": requirements,
        "warnings": warnings,
        "errors": errors,
    }

    add_requirement(requirements, errors, "repo root", ROOT.exists(), str(ROOT))

    if kernel_file is not None:
        add_requirement(requirements, errors, "kernel file", kernel_file.exists(), str(kernel_file))

    torch_info = probe_torch_cuda(gpu)
    preflight["torch"] = torch_info
    add_requirement(
        requirements,
        errors,
        "PyTorch import",
        torch_info["importable"],
        torch_info["version"] if torch_info["importable"] else (torch_info["error"] or "torch import failed"),
    )
    add_requirement(
        requirements,
        errors,
        "CUDA runtime",
        torch_info["cuda_available"],
        f"torch CUDA {torch_info['cuda_version']}" if torch_info["cuda_available"] else (torch_info["error"] or "torch.cuda.is_available() returned false"),
    )
    selected_gpu_ok = torch_info["cuda_available"] and 0 <= gpu < int(torch_info["device_count"])
    add_requirement(
        requirements,
        errors,
        f"GPU index {gpu}",
        selected_gpu_ok,
        f"{torch_info['selected_gpu_name']} ({torch_info['selected_sm']})" if selected_gpu_ok else f"available device count: {torch_info['device_count']}",
    )

    nvidia_smi_info = probe_nvidia_smi()
    preflight["nvidia_smi"] = nvidia_smi_info
    if not nvidia_smi_info["exists"]:
        warnings.append("nvidia-smi not found; GPU model falls back to PyTorch detection.")
    elif nvidia_smi_info.get("returncode") not in (None, 0):
        warnings.append("nvidia-smi is present but GPU query failed.")

    gpu_info: dict[str, Any] = {
        "name": torch_info.get("selected_gpu_name", ""),
        "compute_capability": torch_info.get("selected_gpu_compute_capability", ""),
        "sm": torch_info.get("selected_sm", ""),
        "driver_version": "",
        "source": "torch" if torch_info.get("selected_gpu_name") else "",
    }
    if nvidia_smi_info.get("gpus") and gpu < len(nvidia_smi_info["gpus"]):
        smi_gpu = nvidia_smi_info["gpus"][gpu]
        gpu_info["name"] = smi_gpu.get("name", gpu_info["name"])
        gpu_info["compute_capability"] = smi_gpu.get("compute_capability", gpu_info["compute_capability"])
        gpu_info["driver_version"] = smi_gpu.get("driver_version", "")
        gpu_info["source"] = "nvidia-smi"
    preflight["gpu"] = gpu_info

    nvcc_info = probe_executable(nvcc_bin, ["--version"])
    preflight["nvcc"] = nvcc_info
    add_requirement(
        requirements,
        errors,
        "nvcc executable",
        nvcc_info["exists"],
        nvcc_info["resolved"] or f"cannot resolve {nvcc_bin}",
    )

    ncu_info = probe_executable(ncu_bin, ["--version"])
    preflight["ncu"] = ncu_info
    add_requirement(
        requirements,
        errors,
        "ncu executable",
        ncu_info["exists"],
        ncu_info["resolved"] or f"cannot resolve {ncu_bin}",
    )

    preflight["ready"] = not errors
    return preflight


def render_preflight_markdown(preflight: dict[str, Any]) -> str:
    gpu = preflight.get("gpu") or {}
    torch_info = preflight.get("torch") or {}
    lines = [
        "# cuda-evolve Preflight",
        "",
        "## Status",
        f"- ready: {'yes' if preflight.get('ready') else 'no'}",
        f"- checked at: {preflight.get('checked_at', '')}",
        f"- python: {preflight.get('python_executable', '')}",
        f"- python version: {preflight.get('python_version', '')}",
        f"- selected gpu index: {preflight.get('selected_gpu_index')}",
        "",
        "## Required Environment",
        "",
        "| Requirement | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for item in preflight.get("requirements", []):
        status = "ok" if item.get("ok") else "missing"
        detail = str(item.get("detail", "")).replace("\n", "<br>")
        lines.append(f"| {item.get('name')} | {status} | {detail} |")

    lines.extend(
        [
            "",
            "## GPU",
            f"- model: {gpu.get('name') or 'unknown'}",
            f"- compute capability: {gpu.get('compute_capability') or 'unknown'}",
            f"- sm: {gpu.get('sm') or 'unknown'}",
            f"- driver version: {gpu.get('driver_version') or 'unknown'}",
            f"- source: {gpu.get('source') or 'unknown'}",
            f"- torch version: {torch_info.get('version') or 'unknown'}",
            f"- torch cuda version: {torch_info.get('cuda_version') or 'unknown'}",
            "",
            "## Environment Variables",
            f"- CUDA_PATH: {preflight.get('env_vars', {}).get('CUDA_PATH') or '(unset)'}",
            f"- CUDA_HOME: {preflight.get('env_vars', {}).get('CUDA_HOME') or '(unset)'}",
            f"- CUDA_ROOT: {preflight.get('env_vars', {}).get('CUDA_ROOT') or '(unset)'}",
        ]
    )

    if preflight.get("warnings"):
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {item}" for item in preflight["warnings"])
    if preflight.get("errors"):
        lines.extend(["", "## Errors"])
        lines.extend(f"- {item}" for item in preflight["errors"])
    return "\n".join(lines) + "\n"


def write_preflight_outputs(preflight: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(preflight, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_preflight_markdown(preflight), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cuda-evolve environment preflight checks")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--kernel-file", type=str, default="kernel.py")
    parser.add_argument("--nvcc-bin", type=str, default="nvcc")
    parser.add_argument("--ncu-bin", type=str, default="ncu")
    parser.add_argument("--fail-on-warning", action="store_true")
    parser.add_argument("--json-out", type=str, default=str(WORKSPACE / "preflight_check.json"))
    parser.add_argument("--md-out", type=str, default=str(WORKSPACE / "preflight_check.md"))
    args = parser.parse_args()

    kernel_file = (ROOT / args.kernel_file).resolve() if args.kernel_file else None
    preflight = collect_preflight(args.gpu, kernel_file, nvcc_bin=args.nvcc_bin, ncu_bin=args.ncu_bin)
    json_path = Path(args.json_out)
    md_path = Path(args.md_out)
    write_preflight_outputs(preflight, json_path, md_path)

    print(render_preflight_markdown(preflight))
    if preflight.get("errors"):
        sys.exit(1)
    if args.fail_on_warning and preflight.get("warnings"):
        sys.exit(2)


if __name__ == "__main__":
    main()
