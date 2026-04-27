"""Artifact-aware experiment loop driver for the custom_platform scaffold."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from iteration_report import choose_best_iteration, load_manifest, render_final_summary, render_iteration_markdown, save_manifest
from preflight import collect_preflight, write_preflight_outputs
from strategy_memory import (
    build_strategy_fingerprint,
    classify_strategy_outcome,
    ensure_scope,
    extract_strategy_tags,
    load_global_strategy_memory,
    merge_strategy_constraints,
    save_global_strategy_memory,
    sanitize_token,
    update_memory_bucket,
)

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "workspace"
RUNS_ROOT = WORKSPACE / "runs"
RESULTS_FILE = WORKSPACE / "results.tsv"
STRATEGY_MEMORY_FILE = WORKSPACE / "strategy_memory" / "global_strategy_memory.json"
PREFLIGHT_JSON = WORKSPACE / "preflight_check.json"
PREFLIGHT_MD = WORKSPACE / "preflight_check.md"
PROPOSAL_TEMPLATE = WORKSPACE / "optimization_proposal.template.md"
KERNEL_FILE = ROOT / "kernel.py"
RESULTS_HEADER = (
    "experiment_id\thypothesis\tcorrectness\ttime_ms\tthroughput\tpeak_vram_mb\tkept"
    "\tachieved_compute_tflops\tachieved_memory_gbps\tpeak_compute_tflops\tpeak_memory_gbps"
    "\tbottleneck\tgit_sha\tparent_experiment_id\tprofile_top_stall\tprofile_occupancy"
    "\tprofile_l1_hit_rate\tprofile_l2_hit_rate\tstrategy_tags\tstrategy_fingerprint"
    "\tstrategy_outcome\tstrategy_reason\trun_dir\titer_dir\tprofile_report\n"
)


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _run(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_kernel_meta() -> dict[str, str]:
    if not KERNEL_FILE.exists():
        return {"kernel_type": "", "target_platform": ""}
    try:
        spec = importlib.util.spec_from_file_location("_custom_platform_kernel_peek", str(KERNEL_FILE))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return {
            "kernel_type": getattr(module, "KERNEL_TYPE", ""),
            "target_platform": getattr(module, "TARGET_PLATFORM", ""),
        }
    except Exception:
        return {"kernel_type": "", "target_platform": ""}


def _git_sha() -> str:
    result = _run(["git", "rev-parse", "--short", "HEAD"])
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _make_run_dir(run_dir: str | None, resume_from: str | None) -> Path:
    if resume_from:
        return Path(resume_from).resolve()
    if run_dir:
        return Path(run_dir).resolve()
    return (RUNS_ROOT / f"run_{now_stamp()}").resolve()


def _make_iter_dir(run_dir: Path, iteration: int) -> Path:
    return run_dir / f"iter_v{iteration}"


def _copy_snapshot(iter_dir: Path) -> str:
    snapshot_path = iter_dir / "kernel.snapshot.py"
    if KERNEL_FILE.exists():
        shutil.copy2(KERNEL_FILE, snapshot_path)
    return str(snapshot_path.relative_to(ROOT))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _run_bench_to_artifacts(iter_dir: Path, platform: str) -> tuple[int, dict[str, Any], str]:
    benchmark_json = iter_dir / "benchmark_result.json"
    stdout_path = iter_dir / "benchmark.stdout.txt"
    stderr_path = iter_dir / "benchmark.stderr.txt"
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "bench.py"),
        "--platform",
        platform,
        "--json-out",
        str(benchmark_json),
    ]
    result = _run(cmd, timeout=900)
    _write_text(stdout_path, result.stdout or "")
    _write_text(stderr_path, result.stderr or "")
    return result.returncode, _read_json(benchmark_json), " ".join(cmd)


def _run_profile_to_artifacts(iter_dir: Path, platform: str) -> tuple[int, dict[str, str], str]:
    report_path = iter_dir / "profile_report.txt"
    summary_path = iter_dir / "profile_summary.txt"
    details_path = iter_dir / "profile_details.txt"
    stdout_path = iter_dir / "profile.stdout.txt"
    stderr_path = iter_dir / "profile.stderr.txt"
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "profile.py"),
        "--platform",
        platform,
        "--kernel-file",
        "kernel.py",
        "--output-dir",
        str(iter_dir),
        "--summary-out",
        str(summary_path),
        "--details-out",
        str(details_path),
    ]
    result = _run(cmd, timeout=1800)
    _write_text(stdout_path, result.stdout or "")
    _write_text(stderr_path, result.stderr or "")
    metadata = {
        "command": " ".join(cmd),
        "report": str(report_path.relative_to(ROOT)),
        "summary_txt": str(summary_path.relative_to(ROOT)),
        "details_txt": str(details_path.relative_to(ROOT)),
        "stdout_txt": str(stdout_path.relative_to(ROOT)),
        "stderr_txt": str(stderr_path.relative_to(ROOT)),
    }
    return result.returncode, metadata, result.stdout or ""


def _parse_profile_metrics(profile_log: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for line in profile_log.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        metrics[key.strip()] = val.strip()
    return {
        "profile_top_stall": metrics.get("profile_top_stall", ""),
        "profile_occupancy": metrics.get("profile_occupancy", ""),
        "profile_l1_hit_rate": metrics.get("profile_l1_hit_rate", ""),
        "profile_l2_hit_rate": metrics.get("profile_l2_hit_rate", ""),
        "profile_compute_util": metrics.get("profile_compute_util", ""),
        "profile_memory_util": metrics.get("profile_memory_util", ""),
    }


def _extract_profile_report(profile_log: str, default_path: str) -> str:
    for line in profile_log.splitlines():
        if line.startswith("profile_report_path:"):
            value = line.split(":", 1)[1].strip()
            if value:
                try:
                    return str(Path(value).resolve().relative_to(ROOT))
                except Exception:
                    return value
    return default_path


def _ensure_results_file() -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(RESULTS_HEADER, encoding="utf-8")


def _write_proposal_stub(iter_dir: Path, hypothesis: str, kernel_type: str, constraints: dict[str, list[str]]) -> Path:
    proposal_path = iter_dir / "optimization_proposal.md"
    if proposal_path.exists():
        return proposal_path
    if PROPOSAL_TEMPLATE.exists():
        content = PROPOSAL_TEMPLATE.read_text(encoding="utf-8")
    else:
        content = (
            "# Optimization Proposal\n\n"
            "## Backend\n- custom_platform\n\n"
            "## Primary references\n- docs/compute_optimization.md\n\n"
            "## Evidence\n- fill_me\n\n"
            "## Strategy constraints from memory\n- fill_me\n\n"
            "## Strategy tags\n- fill_me_tag\n\n"
            "## This iteration\n- fill_me\n"
        )
    content += (
        "\n\n## Run Metadata\n"
        f"- kernel_type: {kernel_type or 'unknown'}\n"
        f"- hypothesis: {hypothesis}\n"
        f"- blocked: {', '.join(constraints.get('blocked', [])) or 'none'}\n"
        f"- preferred: {', '.join(constraints.get('preferred', [])) or 'none'}\n"
    )
    _write_text(proposal_path, content)
    return proposal_path


def _get_parent_throughput(parent_record: dict[str, Any] | None) -> float:
    if not parent_record:
        return 0.0
    bench = parent_record.get("benchmark_result") or {}
    value = bench.get("throughput_tflops", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _append_result(record: dict[str, Any]) -> None:
    bench = record.get("benchmark_result") or {}
    correctness = bench.get("correctness") or {}
    strategy = record.get("strategy") or {}
    row = [
        record.get("experiment_id", ""),
        record.get("hypothesis", ""),
        "PASS" if correctness.get("passed") else "FAIL",
        str((bench.get("kernel") or {}).get("median_ms", 0.0)),
        str(bench.get("throughput_tflops", 0.0)),
        str(bench.get("peak_vram_mb", 0.0)),
        "yes" if record.get("kept") else "no",
        str(bench.get("achieved_compute_tflops", 0.0)),
        str(bench.get("achieved_memory_gbps", 0.0)),
        str(bench.get("peak_compute_tflops", 0.0)),
        str(bench.get("peak_memory_gbps", 0.0)),
        str(bench.get("bottleneck", "")),
        record.get("git_sha", ""),
        record.get("parent_experiment_id", ""),
        record.get("profile_metrics", {}).get("profile_top_stall", ""),
        record.get("profile_metrics", {}).get("profile_occupancy", ""),
        record.get("profile_metrics", {}).get("profile_l1_hit_rate", ""),
        record.get("profile_metrics", {}).get("profile_l2_hit_rate", ""),
        ",".join(strategy.get("tags", [])),
        strategy.get("fingerprint", ""),
        strategy.get("outcome", ""),
        strategy.get("reason", ""),
        record.get("run_dir", ""),
        record.get("iter_dir", ""),
        record.get("profile_report", ""),
    ]
    with RESULTS_FILE.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(row) + "\n")


def _decide_keep(record: dict[str, Any], previous_record: dict[str, Any] | None) -> tuple[bool, str]:
    bench = record.get("benchmark_result") or {}
    correctness = bench.get("correctness") or {}
    if not correctness.get("passed"):
        return False, "correctness_failed"
    if record.get("profile_expected") and not record.get("profile_report_exists"):
        return False, "profile_missing"
    current_tp = float(bench.get("throughput_tflops", 0.0) or 0.0)
    parent_tp = _get_parent_throughput(previous_record)
    if parent_tp > 0 and current_tp > 0:
        improvement = (current_tp - parent_tp) / parent_tp * 100.0
        if improvement > 1.0:
            return True, f"improved_{improvement:.2f}_percent"
        return False, f"improvement_{improvement:.2f}_below_threshold"
    return True, "baseline_seed"


def run_experiment(
    platform: str,
    hypothesis: str,
    allow_placeholder: bool,
    run_dir_arg: str | None,
    resume_from: str | None,
    proposal_template: str | None,
    preflight_only: bool,
) -> dict[str, Any]:
    _ensure_results_file()
    global PROPOSAL_TEMPLATE
    if proposal_template:
        PROPOSAL_TEMPLATE = Path(proposal_template).resolve()

    run_dir = _make_run_dir(run_dir_arg, resume_from)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    manifest = load_manifest(manifest_path) or {
        "run_dir": str(run_dir.relative_to(ROOT)),
        "platform": platform,
        "kernel_type": "",
        "source_kernel_path": str(KERNEL_FILE.relative_to(ROOT)),
        "preflight": {},
        "iterations": [],
        "best_iteration": None,
        "best_kernel_path": "",
        "strategy_memory": {},
        "updated_at": now_iso(),
    }

    preflight = collect_preflight(platform_name=platform, kernel_file=KERNEL_FILE, allow_placeholder=allow_placeholder)
    write_preflight_outputs(preflight, run_dir / "preflight_check.json", run_dir / "preflight_check.md")
    write_preflight_outputs(preflight, PREFLIGHT_JSON, PREFLIGHT_MD)
    preflight["json_path"] = str((run_dir / "preflight_check.json").relative_to(ROOT))
    preflight["markdown_path"] = str((run_dir / "preflight_check.md").relative_to(ROOT))
    manifest["preflight"] = preflight
    if preflight_only:
        save_manifest(manifest_path, manifest)
        _write_text(run_dir / "final_summary.md", render_final_summary(manifest))
        return {"status": "preflight_only", "run_dir": str(run_dir.relative_to(ROOT))}
    if not preflight["ready"]:
        save_manifest(manifest_path, manifest)
        _write_text(run_dir / "final_summary.md", render_final_summary(manifest))
        return {"status": "error", "reason": "preflight_failed", "run_dir": str(run_dir.relative_to(ROOT))}

    kernel_meta = _load_kernel_meta()
    kernel_type = kernel_meta.get("kernel_type", "")
    manifest["kernel_type"] = kernel_type
    manifest["platform"] = platform

    global_memory = load_global_strategy_memory(STRATEGY_MEMORY_FILE)
    scope_key = sanitize_token(f"{platform}_{kernel_type or 'unknown'}")
    scope = ensure_scope(
        global_memory,
        scope_key,
        {
            "platform": platform,
            "kernel_type": kernel_type or "unknown",
            "source_kernel_path": str(KERNEL_FILE.relative_to(ROOT)),
        },
    )
    constraints = merge_strategy_constraints(scope)

    iteration = len(manifest.get("iterations", [])) + 1
    iter_dir = _make_iter_dir(run_dir, iteration)
    iter_dir.mkdir(parents=True, exist_ok=True)
    snapshot_file = _copy_snapshot(iter_dir)
    proposal_path = _write_proposal_stub(iter_dir, hypothesis, kernel_type, constraints)

    benchmark_rc, benchmark_result, benchmark_command = _run_bench_to_artifacts(iter_dir, platform)
    correctness_passed = bool((benchmark_result.get("correctness") or {}).get("passed"))
    if benchmark_rc != 0 and correctness_passed and not benchmark_result.get("error"):
        benchmark_rc = 0

    profile_rc = None
    profile_metadata = {
        "command": "",
        "report": str((iter_dir / "profile_report.txt").relative_to(ROOT)),
        "summary_txt": str((iter_dir / "profile_summary.txt").relative_to(ROOT)),
        "details_txt": str((iter_dir / "profile_details.txt").relative_to(ROOT)),
        "stdout_txt": str((iter_dir / "profile.stdout.txt").relative_to(ROOT)),
        "stderr_txt": str((iter_dir / "profile.stderr.txt").relative_to(ROOT)),
    }
    profile_metrics: dict[str, str] = {}
    profile_expected = correctness_passed
    if correctness_passed:
        profile_rc, profile_metadata, profile_stdout = _run_profile_to_artifacts(iter_dir, platform)
        profile_metrics = _parse_profile_metrics(profile_stdout)
        profile_metadata["report"] = _extract_profile_report(profile_stdout, profile_metadata["report"])

    profile_report_exists = (ROOT / profile_metadata["report"]).exists()
    previous_record = manifest.get("iterations", [])[-1] if manifest.get("iterations") else None
    strategy_tags = extract_strategy_tags(proposal_path)
    strategy_fingerprint = build_strategy_fingerprint(kernel_type or "unknown", strategy_tags)

    record = {
        "iteration": iteration,
        "experiment_id": f"{kernel_type or 'kernel'}_exp_{iteration:03d}",
        "hypothesis": hypothesis,
        "snapshot_file": snapshot_file,
        "benchmark_command": benchmark_command,
        "benchmark_json": str((iter_dir / "benchmark_result.json").relative_to(ROOT)),
        "benchmark_rc": benchmark_rc,
        "benchmark_result": benchmark_result,
        "profile_command": profile_metadata["command"],
        "profile_rc": profile_rc,
        "profile_report": profile_metadata["report"],
        "profile_summary_txt": profile_metadata["summary_txt"],
        "profile_details_txt": profile_metadata["details_txt"],
        "profile_stdout_txt": profile_metadata["stdout_txt"],
        "profile_stderr_txt": profile_metadata["stderr_txt"],
        "profile_expected": profile_expected,
        "profile_report_exists": profile_report_exists,
        "profile_metrics": profile_metrics,
        "git_sha": _git_sha(),
        "parent_experiment_id": previous_record.get("experiment_id", "") if previous_record else "",
        "run_dir": str(run_dir.relative_to(ROOT)),
        "iter_dir": str(iter_dir.relative_to(ROOT)),
        "proposal_path": str(proposal_path.relative_to(ROOT)),
    }

    outcome, reason = classify_strategy_outcome(record, previous_record)
    record["strategy"] = {
        "tags": strategy_tags,
        "fingerprint": strategy_fingerprint,
        "outcome": outcome,
        "reason": reason,
        "constraints": constraints,
    }
    update_memory_bucket(
        scope[outcome],
        strategy_fingerprint,
        strategy_tags,
        iteration,
        reason,
        outcome,
        record,
        previous_record,
    )
    manifest["strategy_memory"] = {
        "scope_key": scope_key,
        "positive": scope.get("positive", {}),
        "negative": scope.get("negative", {}),
        "rejected": scope.get("rejected", {}),
    }

    kept, keep_reason = _decide_keep(record, previous_record)
    record["kept"] = kept
    record["keep_reason"] = keep_reason

    _write_text(iter_dir / "iteration_summary.md", render_iteration_markdown(record))
    manifest.setdefault("iterations", []).append(record)
    best = choose_best_iteration(manifest["iterations"])
    manifest["best_iteration"] = best.get("iteration") if best else None
    manifest["best_kernel_path"] = best.get("snapshot_file", "") if best else ""
    manifest["updated_at"] = now_iso()
    save_manifest(manifest_path, manifest)
    _write_text(run_dir / "final_summary.md", render_final_summary(manifest))
    save_global_strategy_memory(STRATEGY_MEMORY_FILE, global_memory)
    _append_result(record)
    return {
        "status": "ok",
        "run_dir": str(run_dir.relative_to(ROOT)),
        "iteration": iteration,
        "kept": kept,
        "keep_reason": keep_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one scaffold experiment")
    parser.add_argument("--platform", default="custom_platform")
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--allow-placeholder", action="store_true")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--proposal-template", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    result = run_experiment(
        platform=args.platform,
        hypothesis=args.hypothesis,
        allow_placeholder=args.allow_placeholder,
        run_dir_arg=args.run_dir,
        resume_from=args.resume_from,
        proposal_template=args.proposal_template,
        preflight_only=args.preflight_only,
    )

    print("=== RUN LOOP SUMMARY ===")
    print(f"hypothesis: {args.hypothesis}")
    print(f"status: {result.get('status')}")
    if "iteration" in result:
        print(f"iteration: {result['iteration']}")
    if "kept" in result:
        print(f"kept: {'yes' if result['kept'] else 'no'}")
        print(f"keep_reason: {result.get('keep_reason', '')}")
    print(f"run_dir: {result.get('run_dir', '')}")
    print("=== END RUN LOOP SUMMARY ===")

    if result.get("status") == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
