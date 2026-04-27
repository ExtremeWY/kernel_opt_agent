"""Environment preparation and validation for cuda-evolve."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from .preflight import collect_preflight, write_preflight_outputs
    from .runtime import RUNTIME_JSON, RUNTIME_MD, write_runtime_outputs
except ImportError:
    from preflight import collect_preflight, write_preflight_outputs
    from runtime import RUNTIME_JSON, RUNTIME_MD, write_runtime_outputs

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "workspace"
RUNS_DIR = WORKSPACE / "runs"
STRATEGY_MEMORY_FILE = WORKSPACE / "strategy_memory" / "global_strategy_memory.json"
RESULTS_FILE = WORKSPACE / "results.tsv"
MEMORY_FILE = WORKSPACE / "MEMORY.md"
PREFLIGHT_JSON = WORKSPACE / "preflight_check.json"
PREFLIGHT_MD = WORKSPACE / "preflight_check.md"

RESULTS_HEADER = (
    "experiment_id\thypothesis\tcorrectness\ttime_ms\tthroughput\tpeak_vram_mb\tkept"
    "\tpct_peak_compute\tpct_peak_bandwidth\tbottleneck\tgit_sha\tparent_experiment_id"
    "\tncu_top_stall\tncu_occupancy\tncu_l1_hit_rate\tncu_l2_hit_rate"
    "\tstrategy_tags\tstrategy_fingerprint\tstrategy_outcome\tstrategy_reason"
    "\trun_dir\titer_dir\ttargeted_ncu_report\tfull_ncu_report\n"
)


def check_python() -> None:
    v = sys.version_info
    print(f"[✓] Python {v.major}.{v.minor}.{v.micro}")
    if v < (3, 10):
        print("[✗] Python >= 3.10 required")
        sys.exit(1)


def check_triton() -> None:
    try:
        import triton

        print(f"[✓] Triton {triton.__version__}")
    except ImportError:
        print("[!] Triton not installed — Triton kernels will not be available")


def check_git() -> None:
    try:
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT, check=False)
        if result.returncode == 0:
            print("[✓] Git repository OK")
        else:
            print("[!] Git metadata not available — run_loop keep/revert logic will be degraded")
    except FileNotFoundError:
        print("[!] git not found — run_loop keep/revert logic will be degraded")


def init_results() -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(RESULTS_HEADER, encoding="utf-8")
        print(f"[✓] Created {RESULTS_FILE.name}")
        return

    lines = RESULTS_FILE.read_text(encoding="utf-8").strip().split("\n")
    header = lines[0] if lines else ""
    if "strategy_fingerprint" in header and "run_dir" in header:
        print(f"[✓] {RESULTS_FILE.name} exists ({max(0, len(lines) - 1)} experiments recorded)")
        return

    old_rows = lines[1:] if len(lines) > 1 else []
    new_col_count = len(RESULTS_HEADER.strip().split("\t"))
    migrated = [RESULTS_HEADER.strip()]
    for row in old_rows:
        cols = row.split("\t")
        cols.extend([""] * (new_col_count - len(cols)))
        migrated.append("\t".join(cols[:new_col_count]))
    RESULTS_FILE.write_text("\n".join(migrated) + "\n", encoding="utf-8")
    print(f"[✓] Migrated {RESULTS_FILE.name} to artifact-aware schema")


def init_memory() -> None:
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text(
            "# Optimization Log\n\n"
            "This file records the history of optimization experiments.\n\n---\n\n"
            "<!-- New entries should be added below this line, in reverse chronological order. -->\n",
            encoding="utf-8",
        )
        print(f"[✓] Created {MEMORY_FILE.name}")
    else:
        print(f"[✓] {MEMORY_FILE.name} exists")


def init_run_dirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (WORKSPACE / "ncu_reports").mkdir(parents=True, exist_ok=True)
    print(f"[✓] run artifact directory ready: {RUNS_DIR}")


def init_strategy_memory() -> None:
    STRATEGY_MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not STRATEGY_MEMORY_FILE.exists():
        STRATEGY_MEMORY_FILE.write_text('{\n  "version": 1,\n  "updated_at": "",\n  "scopes": {}\n}\n', encoding="utf-8")
        print(f"[✓] Created {STRATEGY_MEMORY_FILE.relative_to(ROOT)}")
    else:
        print(f"[✓] {STRATEGY_MEMORY_FILE.relative_to(ROOT)} exists")


def check_kernel_files() -> None:
    kernel_py = ROOT / "kernel.py"
    references_dir = ROOT / "references"

    if kernel_py.exists():
        print(f"[✓] {kernel_py.name} exists")
    else:
        print(f"[!] {kernel_py.name} not found — create it or copy a kernel from kernels/")

    if references_dir.exists() and (references_dir / "__init__.py").exists():
        print("[✓] references/ package exists")
    else:
        print("[!] references/ package not found — create it before running experiments")


def run_preflight(gpu: int, nvcc_bin: str, ncu_bin: str) -> dict:
    preflight = collect_preflight(gpu, ROOT / "kernel.py", nvcc_bin=nvcc_bin, ncu_bin=ncu_bin)
    write_preflight_outputs(preflight, PREFLIGHT_JSON, PREFLIGHT_MD)
    if preflight.get("ready"):
        print(f"[✓] preflight passed -> {PREFLIGHT_MD.relative_to(ROOT)}")
    else:
        print(f"[!] preflight found blocking issues -> {PREFLIGHT_MD.relative_to(ROOT)}")
    return preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the cuda-evolve workspace")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--nvcc-bin", type=str, default="nvcc")
    parser.add_argument("--ncu-bin", type=str, default="ncu")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  cuda-evolve Environment Check")
    print("=" * 60)
    print()

    check_python()
    check_triton()
    print()
    check_git()
    init_results()
    init_memory()
    init_run_dirs()
    init_strategy_memory()
    runtime = write_runtime_outputs(RUNTIME_JSON, RUNTIME_MD)
    preferred_python = runtime.get("preferred_python", sys.executable)
    if runtime.get("using_preferred_python"):
        print(f"[✓] runtime pinned to {preferred_python}")
    else:
        print(f"[!] current Python differs from preferred runtime: {sys.executable} -> {preferred_python}")
    print()
    check_kernel_files()

    preflight = None
    if not args.skip_preflight:
        print()
        preflight = run_preflight(args.gpu, args.nvcc_bin, args.ncu_bin)

    print()
    print("=" * 60)
    if preflight and not preflight.get("ready"):
        print("  Environment initialized, but preflight is not ready.")
        print(f"  Review {PREFLIGHT_MD.relative_to(ROOT)} before running experiments.")
    else:
        print("  Environment ready. Read program.md to begin.")
    print(f"  Runtime commands: {RUNTIME_MD.relative_to(ROOT)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
