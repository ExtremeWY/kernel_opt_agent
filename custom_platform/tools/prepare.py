"""Environment preparation and workspace initialization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platforms.registry import get_platform_adapter

try:
    from .preflight import collect_preflight, write_preflight_outputs
    from .runtime import RUNTIME_JSON, RUNTIME_MD, write_runtime_outputs
    from .strategy_memory import default_global_strategy_memory
except ImportError:
    from preflight import collect_preflight, write_preflight_outputs
    from runtime import RUNTIME_JSON, RUNTIME_MD, write_runtime_outputs
    from strategy_memory import default_global_strategy_memory

RESULTS_FILE = ROOT / "workspace" / "results.tsv"
MEMORY_FILE = ROOT / "workspace" / "MEMORY.md"
RUNS_DIR = ROOT / "workspace" / "runs"
PREFLIGHT_JSON = ROOT / "workspace" / "preflight_check.json"
PREFLIGHT_MD = ROOT / "workspace" / "preflight_check.md"
STRATEGY_MEMORY_FILE = ROOT / "workspace" / "strategy_memory" / "global_strategy_memory.json"
PROPOSAL_TEMPLATE = ROOT / "workspace" / "optimization_proposal.template.md"

RESULTS_HEADER = (
    "experiment_id\thypothesis\tcorrectness\ttime_ms\tthroughput\tpeak_vram_mb\tkept"
    "\tachieved_compute_tflops\tachieved_memory_gbps\tpeak_compute_tflops\tpeak_memory_gbps"
    "\tbottleneck\tgit_sha\tparent_experiment_id\tprofile_top_stall\tprofile_occupancy"
    "\tprofile_l1_hit_rate\tprofile_l2_hit_rate\tstrategy_tags\tstrategy_fingerprint"
    "\tstrategy_outcome\tstrategy_reason\trun_dir\titer_dir\tprofile_report\n"
)


def init_results() -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(RESULTS_HEADER, encoding="utf-8")


def init_memory() -> None:
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text(
            "# Optimization Log\n\n"
            "This file records cross-kernel optimization history for the target platform.\n",
            encoding="utf-8",
        )


def init_runs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def init_strategy_memory() -> None:
    STRATEGY_MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not STRATEGY_MEMORY_FILE.exists():
        import json

        STRATEGY_MEMORY_FILE.write_text(
            json.dumps(default_global_strategy_memory(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def init_proposal_template() -> None:
    if PROPOSAL_TEMPLATE.exists():
        return
    PROPOSAL_TEMPLATE.write_text(
        "# Optimization Proposal\n\n"
        "## Backend\n- custom_platform\n\n"
        "## Primary references\n"
        "- docs/memory_optimization.md\n"
        "- docs/compute_optimization.md\n"
        "- docs/stall_reasons.md\n"
        "- docs/arch_notes.md\n\n"
        "## Evidence\n"
        "- Fill in the bottleneck diagnosis from benchmark and profiler evidence.\n\n"
        "## Strategy constraints from memory\n"
        "- blocked fingerprints: none\n"
        "- preferred fingerprints: none\n\n"
        "## Strategy tags\n"
        "- baseline\n\n"
        "## This iteration\n"
        "- Describe one focused change.\n"
        "- State why it should improve performance.\n"
        "- State what normalized metric should improve if the hypothesis is correct.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the custom_platform scaffold")
    parser.add_argument("--platform", default="custom_platform")
    parser.add_argument("--kernel-file", default="kernel.py")
    parser.add_argument(
        "--allow-placeholder",
        action="store_true",
        help="Do not fail if the adapter still contains placeholder checks",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Initialize the workspace without validating the platform adapter",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("custom_platform Environment Check")
    print("=" * 60)
    print(f"platform: {args.platform}")

    init_results()
    init_memory()
    init_runs()
    init_strategy_memory()
    init_proposal_template()
    runtime = write_runtime_outputs(RUNTIME_JSON, RUNTIME_MD)
    preferred_python = runtime.get("preferred_python", sys.executable)
    if runtime.get("using_preferred_python"):
        print(f"runtime: using preferred python {preferred_python}")
    else:
        print(f"runtime: current python differs from preferred runtime: {sys.executable} -> {preferred_python}")

    if args.skip_preflight:
        adapter = get_platform_adapter(args.platform)
        issues = adapter.validate_environment()
        print("status: SKIPPED_PREFLIGHT")
        if issues:
            print("environment_issues:")
            for issue in issues:
                print(f"  - {issue}")
    else:
        preflight = collect_preflight(
            platform_name=args.platform,
            kernel_file=ROOT / args.kernel_file,
            allow_placeholder=args.allow_placeholder,
        )
        write_preflight_outputs(preflight, PREFLIGHT_JSON, PREFLIGHT_MD)
        issues = list(preflight.get("errors", [])) + list(preflight.get("warnings", []))
        if issues:
            print("environment_issues:")
            for issue in issues:
                print(f"  - {issue}")
        if not preflight["ready"]:
            print("status: FAIL")
            sys.exit(1)
        print("status: READY")

    print(f"results_file: {RESULTS_FILE}")
    print(f"memory_file: {MEMORY_FILE}")
    print(f"runs_dir: {RUNS_DIR}")
    print(f"strategy_memory_file: {STRATEGY_MEMORY_FILE}")
    print(f"proposal_template: {PROPOSAL_TEMPLATE}")
    print(f"preflight_json: {PREFLIGHT_JSON}")
    print(f"preflight_md: {PREFLIGHT_MD}")
    print(f"runtime_json: {RUNTIME_JSON}")
    print(f"runtime_md: {RUNTIME_MD}")


if __name__ == "__main__":
    main()
