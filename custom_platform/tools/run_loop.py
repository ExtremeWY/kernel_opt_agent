"""Minimal experiment loop driver for the scaffold."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "workspace" / "results.tsv"


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)


def _append_result(hypothesis: str, bench_log: str, profile_log: str, kept: bool) -> None:
    metrics: dict[str, str] = {}
    for line in (bench_log + "\n" + profile_log).splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        metrics[key.strip()] = val.strip()

    row = "\t".join(
        [
            metrics.get("experiment_id", ""),
            hypothesis,
            metrics.get("correctness", "UNKNOWN"),
            metrics.get("latency_ms", "0"),
            metrics.get("throughput_tflops", "0"),
            metrics.get("peak_vram_mb", "0"),
            "yes" if kept else "no",
            metrics.get("achieved_compute_tflops", ""),
            metrics.get("achieved_memory_gbps", ""),
            metrics.get("peak_compute_tflops", ""),
            metrics.get("peak_memory_gbps", ""),
            metrics.get("bottleneck", ""),
            metrics.get("git_sha", ""),
            metrics.get("parent_experiment_id", ""),
            metrics.get("profile_top_stall", ""),
            metrics.get("profile_occupancy", ""),
            metrics.get("profile_l1_hit_rate", ""),
            metrics.get("profile_l2_hit_rate", ""),
        ]
    )
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(row + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one scaffold experiment")
    parser.add_argument("--platform", default="custom_platform")
    parser.add_argument("--hypothesis", required=True)
    args = parser.parse_args()

    bench = _run(["python", "tools/bench.py", "--platform", args.platform])
    bench_log = (bench.stdout or "") + ("\n" + bench.stderr if bench.stderr else "")

    profile_log = ""
    if "correctness: PASS" in bench_log:
        profile = _run(["python", "tools/profile.py", "--platform", args.platform])
        profile_log = (profile.stdout or "") + ("\n" + profile.stderr if profile.stderr else "")

    kept = "correctness: PASS" in bench_log and "profile_error:" not in profile_log
    _append_result(args.hypothesis, bench_log, profile_log, kept)

    print("=== RUN LOOP SUMMARY ===")
    print(f"hypothesis: {args.hypothesis}")
    print(f"kept: {'yes' if kept else 'no'}")
    print("=== END RUN LOOP SUMMARY ===")


if __name__ == "__main__":
    main()

