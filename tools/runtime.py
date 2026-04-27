"""Stable runtime selection helpers for tool scripts.

The repo uses ``uv`` to provision ``.venv``, but repeated ``uv run`` invocations
can hang on some hosts before the Python process starts. These helpers pin all
runtime subprocesses to the project interpreter after the environment exists.
"""

from __future__ import annotations

import functools
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "workspace"
RUNTIME_JSON = WORKSPACE / "runtime_env.json"
RUNTIME_MD = WORKSPACE / "runtime_env.md"


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _as_absolute(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else (Path.cwd() / expanded)


def project_venv_python(root: Path = ROOT) -> Path:
    return root / ".venv" / "bin" / "python"


def base_python() -> Path | None:
    base = getattr(sys, "_base_executable", "") or ""
    if not base:
        return None
    candidate = _as_absolute(Path(base))
    return candidate if _is_executable(candidate) else None


def current_python() -> Path:
    return _as_absolute(Path(sys.executable))


@functools.lru_cache(maxsize=None)
def _python_has_torch_cached(python_str: str) -> bool:
    try:
        result = subprocess.run(
            [python_str, "-c", "import torch"],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def python_has_torch(python: Path) -> bool:
    return _python_has_torch_cached(str(python))


def project_venv_ready(root: Path = ROOT) -> bool:
    venv_python = project_venv_python(root)
    return _is_executable(venv_python) and python_has_torch(venv_python)


def preferred_python(root: Path = ROOT) -> Path:
    candidates: list[Path] = []
    override = os.environ.get("CUDA_EVOLVE_PYTHON")
    if override:
        candidate = _as_absolute(Path(override))
        if _is_executable(candidate):
            candidates.append(candidate)

    project_python = project_venv_python(root)
    current = current_python()
    base = base_python()

    for candidate in (project_python, current, base):
        if candidate is not None and _is_executable(candidate) and candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        if python_has_torch(candidate):
            return candidate
    return current


def build_python_cmd(*args: str | Path, python: str | Path | None = None) -> list[str]:
    exe = _as_absolute(Path(python)) if python is not None else preferred_python()
    return [str(exe), *(str(arg) for arg in args)]


def runtime_info(root: Path = ROOT) -> dict[str, Any]:
    preferred = preferred_python(root)
    current = current_python()
    project_python = project_venv_python(root)
    uv_path = shutil.which("uv") or ""
    use_project_python = preferred == project_python if _is_executable(project_python) else False
    python_cmd = shlex.quote(str(preferred))

    return {
        "current_python": str(current),
        "preferred_python": str(preferred),
        "project_venv_python": str(project_python),
        "project_venv_exists": _is_executable(project_python),
        "project_venv_ready": project_venv_ready(root),
        "using_preferred_python": current == preferred,
        "uv_path": uv_path,
        "recommended_runner": "project_python" if use_project_python else "current_python",
        "commands": {
            "prepare": f"{python_cmd} tools/prepare.py",
            "bench": f"{python_cmd} tools/bench.py",
            "ncu_profile": f"{python_cmd} tools/ncu_profile.py",
            "run_loop": f"{python_cmd} tools/run_loop.py",
        },
        "uv_no_sync_commands": {
            "prepare": "uv run --no-sync tools/prepare.py",
            "bench": "uv run --no-sync tools/bench.py",
            "ncu_profile": "uv run --no-sync tools/ncu_profile.py",
            "run_loop": "uv run --no-sync tools/run_loop.py",
        } if uv_path else {},
    }


def render_runtime_markdown(info: dict[str, Any]) -> str:
    commands = info.get("commands", {})
    uv_no_sync = info.get("uv_no_sync_commands", {})
    lines = [
        "# cuda-evolve Runtime",
        "",
        "## Python",
        f"- current python: {info.get('current_python', '')}",
        f"- preferred python: {info.get('preferred_python', '')}",
        f"- project venv python: {info.get('project_venv_python', '')}",
        f"- project venv exists: {'yes' if info.get('project_venv_exists') else 'no'}",
        f"- project venv has runtime deps: {'yes' if info.get('project_venv_ready') else 'no'}",
        f"- using preferred python now: {'yes' if info.get('using_preferred_python') else 'no'}",
        "",
        "## Recommended Commands",
        f"- prepare: `{commands.get('prepare', '')}`",
        f"- bench: `{commands.get('bench', '')}`",
        f"- ncu profile: `{commands.get('ncu_profile', '')}`",
        f"- run loop: `{commands.get('run_loop', '')}`",
    ]
    if uv_no_sync:
        lines.extend(
            [
                "",
                "## uv Fallback",
                "- If you still prefer `uv`, use `uv run --no-sync ...` after `uv sync`/prepare.",
                f"- bench: `{uv_no_sync.get('bench', '')}`",
                f"- ncu profile: `{uv_no_sync.get('ncu_profile', '')}`",
                f"- run loop: `{uv_no_sync.get('run_loop', '')}`",
            ]
        )
    return "\n".join(lines) + "\n"


def write_runtime_outputs(
    json_path: Path = RUNTIME_JSON,
    md_path: Path = RUNTIME_MD,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    info = runtime_info(root)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_runtime_markdown(info), encoding="utf-8")
    return info
