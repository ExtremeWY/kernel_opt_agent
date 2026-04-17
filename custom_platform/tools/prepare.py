"""Environment preparation and workspace initialization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platforms.registry import get_platform_adapter

RESULTS_FILE = ROOT / "workspace" / "results.tsv"
MEMORY_FILE = ROOT / "workspace" / "MEMORY.md"

RESULTS_HEADER = (
    "experiment_id\thypothesis\tcorrectness\ttime_ms\tthroughput\tpeak_vram_mb\tkept"
    "\tachieved_compute_tflops\tachieved_memory_gbps\tpeak_compute_tflops\tpeak_memory_gbps"
    "\tbottleneck\tgit_sha\tparent_experiment_id\tprofile_top_stall\tprofile_occupancy"
    "\tprofile_l1_hit_rate\tprofile_l2_hit_rate\n"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the custom_platform scaffold")
    parser.add_argument("--platform", default="custom_platform")
    parser.add_argument(
        "--allow-placeholder",
        action="store_true",
        help="Do not fail if the adapter still contains placeholder checks",
    )
    args = parser.parse_args()

    adapter = get_platform_adapter(args.platform)
    issues = adapter.validate_environment()

    print("=" * 60)
    print("custom_platform Environment Check")
    print("=" * 60)
    print(f"platform: {args.platform}")

    init_results()
    init_memory()

    if issues:
        print("environment_issues:")
        for issue in issues:
            print(f"  - {issue}")
        if not args.allow_placeholder:
            print("status: FAIL")
            sys.exit(1)

    print(f"results_file: {RESULTS_FILE}")
    print(f"memory_file: {MEMORY_FILE}")
    print("status: READY" if not issues else "status: PLACEHOLDER_READY")


if __name__ == "__main__":
    main()

