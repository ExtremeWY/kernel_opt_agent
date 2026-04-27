"""Helpers for run manifests and markdown reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def choose_best_iteration(iterations: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for item in iterations:
        bench = item.get("benchmark_result") or {}
        kernel = bench.get("kernel") or {}
        correctness = bench.get("correctness") or {}
        if correctness.get("passed") is False:
            continue
        if not item.get("profile_report_exists", False):
            continue
        median = kernel.get("median_ms")
        avg = kernel.get("average_ms")
        try:
            median_f = float(median)
            avg_f = float(avg)
        except (TypeError, ValueError):
            continue
        candidates.append((median_f, avg_f, item.get("iteration", 0), item))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    return candidates[0][3]


def render_iteration_markdown(record: dict[str, Any]) -> str:
    bench = record.get("benchmark_result") or {}
    kernel = bench.get("kernel") or {}
    correctness = bench.get("correctness") or {}
    strategy = record.get("strategy") or {}
    constraints = strategy.get("constraints") or {}
    profile_metrics = record.get("profile_metrics") or {}
    lines = [
        f"# Iteration v{record.get('iteration')}",
        "",
        "## Overview",
        f"- experiment id: {record.get('experiment_id', '')}",
        f"- hypothesis: {record.get('hypothesis', '')}",
        f"- correctness: {correctness.get('passed')}",
        f"- benchmark rc: {record.get('benchmark_rc')}",
        f"- profile rc: {record.get('profile_rc')}",
        f"- snapshot file: {record.get('snapshot_file')}",
        "",
        "## Strategy Memory",
        f"- tags: {', '.join(strategy.get('tags') or []) or 'none'}",
        f"- fingerprint: {strategy.get('fingerprint') or 'none'}",
        f"- outcome: {strategy.get('outcome') or 'pending'}",
        f"- reason: {strategy.get('reason') or 'not_available'}",
        f"- blocked fingerprints: {', '.join(constraints.get('blocked') or []) or 'none'}",
        f"- preferred fingerprints: {', '.join(constraints.get('preferred') or []) or 'none'}",
        "",
        "## Commands",
        f"- benchmark: `{record.get('benchmark_command', '')}`",
        f"- profile: `{record.get('profile_command', '')}`",
        "",
        "## Benchmark",
        f"- kernel average ms: {kernel.get('average_ms')}",
        f"- kernel median ms: {kernel.get('median_ms')}",
        f"- kernel min ms: {kernel.get('min_ms')}",
        f"- kernel max ms: {kernel.get('max_ms')}",
        f"- throughput: {bench.get('throughput_tflops')}",
        f"- achieved compute: {bench.get('achieved_compute_tflops')}",
        f"- achieved memory: {bench.get('achieved_memory_gbps')}",
        f"- bottleneck: {bench.get('bottleneck')}",
        "",
        "## Profile",
        f"- top stall: {profile_metrics.get('profile_top_stall', 'n/a')}",
        f"- occupancy: {profile_metrics.get('profile_occupancy', 'n/a')}",
        f"- l1 hit rate: {profile_metrics.get('profile_l1_hit_rate', 'n/a')}",
        f"- l2 hit rate: {profile_metrics.get('profile_l2_hit_rate', 'n/a')}",
        "",
        "## Artifacts",
        f"- benchmark json: {record.get('benchmark_json')}",
        f"- profile report: {record.get('profile_report')}",
        f"- profile summary: {record.get('profile_summary_txt')}",
        f"- profile details: {record.get('profile_details_txt')}",
        f"- optimization proposal: {record.get('proposal_path')}",
        "",
    ]
    return "\n".join(lines) + "\n"


def render_final_summary(manifest: dict[str, Any]) -> str:
    preflight = manifest.get("preflight") or {}
    strategy_scope = manifest.get("strategy_memory") or {}
    lines = [
        "# custom_platform Run Summary",
        "",
        "## Run Info",
        f"- run dir: {manifest.get('run_dir', '')}",
        f"- platform: {manifest.get('platform', '')}",
        f"- kernel type: {manifest.get('kernel_type', '')}",
        f"- source kernel: {manifest.get('source_kernel_path', '')}",
        f"- preflight ready: {preflight.get('ready')}",
        f"- preflight report: {preflight.get('markdown_path', '')}",
        "",
        "## Strategy Memory",
        f"- scope key: {strategy_scope.get('scope_key', '')}",
        f"- positive: {len((strategy_scope.get('positive') or {}).keys())}",
        f"- negative: {len((strategy_scope.get('negative') or {}).keys())}",
        f"- rejected: {len((strategy_scope.get('rejected') or {}).keys())}",
        "",
        "## Iterations",
        "",
        "| Iter | Outcome | Correctness | Kernel median ms | Profile report | Snapshot |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for item in manifest.get("iterations", []):
        bench = item.get("benchmark_result") or {}
        correctness = bench.get("correctness") or {}
        kernel = bench.get("kernel") or {}
        strategy = item.get("strategy") or {}
        lines.append(
            f"| v{item.get('iteration')} | {strategy.get('outcome', 'pending')} | {correctness.get('passed')} | {kernel.get('median_ms', '-')} | {'yes' if item.get('profile_report_exists') else 'no'} | {item.get('snapshot_file', '-')} |"
        )
    best = manifest.get("best_iteration")
    lines.extend(["", "## Best"])
    if best is None:
        lines.append("- No eligible best version yet.")
    else:
        lines.append(f"- best iteration: v{best}")
        lines.append(f"- best kernel path: {manifest.get('best_kernel_path', '')}")
    return "\n".join(lines) + "\n"


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def save_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
