#!/usr/bin/env python3
"""Artifact-aware experiment runner for cuda-evolve."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .iteration_report import (
        choose_best_iteration,
        load_manifest,
        render_final_summary,
        render_iteration_markdown,
        save_manifest,
    )
    from .preflight import collect_preflight, write_preflight_outputs
    from .runtime import build_python_cmd
    from .strategy_memory import (
        build_strategy_fingerprint,
        ensure_scope,
        extract_strategy_tags,
        classify_strategy_outcome,
        load_global_strategy_memory,
        merge_strategy_constraints,
        save_global_strategy_memory,
        update_memory_bucket,
    )
except ImportError:
    from iteration_report import choose_best_iteration, load_manifest, render_final_summary, render_iteration_markdown, save_manifest
    from preflight import collect_preflight, write_preflight_outputs
    from runtime import build_python_cmd
    from strategy_memory import (
        build_strategy_fingerprint,
        ensure_scope,
        extract_strategy_tags,
        classify_strategy_outcome,
        load_global_strategy_memory,
        merge_strategy_constraints,
        save_global_strategy_memory,
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
KERNEL_CU_FILE = ROOT / "kernel.cu"


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _run(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=timeout,
        check=False,
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _get_kernel_type() -> str:
    if not KERNEL_FILE.exists():
        return ""
    try:
        spec = importlib.util.spec_from_file_location("_kernel_peek", str(KERNEL_FILE))
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return getattr(mod, "KERNEL_TYPE", "")
    except Exception:
        return ""


def _git_sha() -> str:
    result = _run(["git", "rev-parse", "--short", "HEAD"])
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _git_commit(message: str) -> bool:
    if KERNEL_FILE.exists():
        _run(["git", "add", "kernel.py"])
    if KERNEL_CU_FILE.exists():
        _run(["git", "add", "kernel.cu"])
    result = _run(["git", "commit", "-m", message])
    return result.returncode == 0


def _git_revert() -> bool:
    result = _run(["git", "reset", "--hard", "HEAD~1"])
    return result.returncode == 0


def _make_run_dir(run_dir: str | None, resume_from: str | None) -> Path:
    if resume_from:
        return Path(resume_from).resolve()
    if run_dir:
        return Path(run_dir).resolve()
    return (RUNS_ROOT / f"run_{now_stamp()}").resolve()


def _make_iter_dir(run_dir: Path, iteration: int) -> Path:
    return run_dir / f"iter_v{iteration}"


def _copy_snapshot(iter_dir: Path) -> str:
    snapshot_py = iter_dir / "kernel.snapshot.py"
    if KERNEL_FILE.exists():
        shutil.copy2(KERNEL_FILE, snapshot_py)
    if KERNEL_CU_FILE.exists():
        shutil.copy2(KERNEL_CU_FILE, iter_dir / "kernel.snapshot.cu")
    return str(snapshot_py.relative_to(ROOT))


def _read_bench_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _run_bench_to_artifacts(iter_dir: Path, quick: bool, gpu: int) -> tuple[int, dict[str, Any], str]:
    benchmark_json = iter_dir / "benchmark_result.json"
    stdout_path = iter_dir / "benchmark.stdout.txt"
    stderr_path = iter_dir / "benchmark.stderr.txt"
    cmd = build_python_cmd(
        str(ROOT / "tools" / "bench.py"),
        "--gpu",
        str(gpu),
        "--json-out",
        str(benchmark_json),
    )
    if quick:
        cmd.append("--quick")
    result = _run(cmd, timeout=900)
    _write_text(stdout_path, result.stdout or "")
    _write_text(stderr_path, result.stderr or "")
    return result.returncode, _read_bench_json(benchmark_json), " ".join(cmd)


def _run_ncu_mode(
    iter_dir: Path,
    mode: str,
    gpu: int,
    output_name: str,
) -> tuple[int, dict[str, str]]:
    prefix = iter_dir / output_name
    summary_path = iter_dir / f"{output_name}_summary.txt"
    details_path = iter_dir / f"{output_name}_details.txt"
    stdout_path = iter_dir / f"{output_name}_ncu.stdout.txt"
    stderr_path = iter_dir / f"{output_name}_ncu.stderr.txt"
    cmd = build_python_cmd(
        str(ROOT / "tools" / "ncu_profile.py"),
        "--mode",
        mode,
        "--kernel-file",
        str(KERNEL_FILE),
        "--gpu",
        str(gpu),
        "--output-prefix",
        str(prefix),
        "--summary-out",
        str(summary_path),
        "--details-out",
        str(details_path),
        "--stdout-out",
        str(stdout_path),
        "--stderr-out",
        str(stderr_path),
    )
    result = _run(cmd, timeout=1800)
    metadata = {
        "command": " ".join(cmd),
        "report": str(prefix.with_suffix(".ncu-rep").relative_to(ROOT)),
        "summary_txt": str(summary_path.relative_to(ROOT)),
        "details_txt": str(details_path.relative_to(ROOT)),
        "stdout_txt": str(stdout_path.relative_to(ROOT)),
        "stderr_txt": str(stderr_path.relative_to(ROOT)),
    }
    return result.returncode, metadata


def _write_proposal_stub(iter_dir: Path, hypothesis: str, kernel_type: str, constraints: dict[str, list[str]]) -> Path:
    proposal_path = iter_dir / "optimization_proposal.md"
    if proposal_path.exists():
        return proposal_path
    if PROPOSAL_TEMPLATE.exists():
        content = PROPOSAL_TEMPLATE.read_text(encoding="utf-8")
    else:
        content = (
            "# Optimization Proposal\n\n"
            "## Backend\n- fill_me\n\n"
            "## Primary references\n- fill_me\n\n"
            "## Evidence\n- fill_me\n\n"
            "## Strategy constraints from memory\n- fill_me\n\n"
            "## Strategy tags\n- fill_me_tag\n\n"
            "## This iteration\n- fill_me\n"
        )
    content += (
        f"\n\n## Run Metadata\n"
        f"- kernel_type: {kernel_type or 'unknown'}\n"
        f"- hypothesis: {hypothesis}\n"
        f"- blocked: {', '.join(constraints.get('blocked', [])) or 'none'}\n"
        f"- preferred: {', '.join(constraints.get('preferred', [])) or 'none'}\n"
    )
    _write_text(proposal_path, content)
    return proposal_path


def _get_experiment_count() -> int:
    if not RESULTS_FILE.exists():
        return 0
    lines = RESULTS_FILE.read_text(encoding="utf-8").strip().splitlines()
    return max(0, len(lines) - 1)


def _get_last_experiment_id() -> str:
    if not RESULTS_FILE.exists():
        return ""
    lines = RESULTS_FILE.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        return ""
    return lines[-1].split("\t")[0]


def _get_parent_throughput(parent_id: str) -> float:
    if not parent_id or not RESULTS_FILE.exists():
        return 0.0
    lines = RESULTS_FILE.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        return 0.0
    headers = lines[0].split("\t")
    id_idx = headers.index("experiment_id") if "experiment_id" in headers else 0
    tp_idx = headers.index("throughput") if "throughput" in headers else 4
    for line in reversed(lines[1:]):
        cols = line.split("\t")
        if len(cols) <= max(id_idx, tp_idx):
            continue
        if cols[id_idx] == parent_id:
            try:
                return float(cols[tp_idx])
            except ValueError:
                return 0.0
    return 0.0


def _append_result(record: dict[str, Any]) -> None:
    bench = record.get("benchmark_result") or {}
    correctness = bench.get("correctness") or {}
    row = [
        record.get("experiment_id", ""),
        record.get("hypothesis", ""),
        "PASS" if correctness.get("passed") else "FAIL",
        str((bench.get("kernel") or {}).get("median_ms", 0.0)),
        str(bench.get("throughput_tflops", 0.0)),
        str(bench.get("peak_vram_mb", 0.0)),
        "yes" if record.get("kept") else "no",
        str(bench.get("pct_peak_compute", 0.0)),
        str(bench.get("pct_peak_bandwidth", 0.0)),
        str(bench.get("bottleneck", "")),
        record.get("git_sha", ""),
        record.get("parent_experiment_id", ""),
        record.get("ncu_metrics", {}).get("ncu_top_stall", ""),
        record.get("ncu_metrics", {}).get("ncu_occupancy", ""),
        record.get("ncu_metrics", {}).get("ncu_l1_hit_rate", ""),
        record.get("ncu_metrics", {}).get("ncu_l2_hit_rate", ""),
        ",".join((record.get("strategy") or {}).get("tags", [])),
        (record.get("strategy") or {}).get("fingerprint", ""),
        (record.get("strategy") or {}).get("outcome", ""),
        (record.get("strategy") or {}).get("reason", ""),
        record.get("run_dir", ""),
        record.get("iter_dir", ""),
        record.get("targeted_report", ""),
        record.get("full_report", ""),
    ]
    with RESULTS_FILE.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(row) + "\n")


def _decide_keep(record: dict[str, Any], parent_id: str) -> tuple[bool, str]:
    bench = record.get("benchmark_result") or {}
    correctness = bench.get("correctness") or {}
    if not correctness.get("passed"):
        return False, "correctness_failed"
    peak_vram = float(bench.get("peak_vram_mb", 0.0) or 0.0)
    gpu_memory_gb = float(bench.get("gpu_memory_gb", 0.0) or 0.0)
    if gpu_memory_gb > 0 and peak_vram > gpu_memory_gb * 1024 * 0.8:
        return False, "vram_exceeds_80_percent"
    current_tp = float(bench.get("throughput_tflops", 0.0) or 0.0)
    parent_tp = _get_parent_throughput(parent_id)
    if parent_tp > 0 and current_tp > 0:
        improvement = (current_tp - parent_tp) / parent_tp * 100
        if improvement > 1.0:
            return True, f"improved_{improvement:.2f}_percent"
        return False, f"improvement_{improvement:.2f}_below_threshold"
    return True, "baseline_seed"


def _run_preflight(gpu: int, nvcc_bin: str, ncu_bin: str) -> dict[str, Any]:
    preflight = collect_preflight(gpu, KERNEL_FILE, nvcc_bin=nvcc_bin, ncu_bin=ncu_bin)
    write_preflight_outputs(preflight, PREFLIGHT_JSON, PREFLIGHT_MD)
    preflight["json_path"] = str(PREFLIGHT_JSON.relative_to(ROOT))
    preflight["markdown_path"] = str(PREFLIGHT_MD.relative_to(ROOT))
    return preflight


def run_experiment(
    hypothesis: str,
    quick: bool,
    targeted_ncu: bool,
    full_ncu: bool,
    parent_id: str,
    gpu: int,
    nvcc_bin: str,
    ncu_bin: str,
    run_dir_arg: str | None,
    resume_from: str | None,
    preflight_only: bool,
    proposal_template: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    if proposal_template:
        global PROPOSAL_TEMPLATE
        PROPOSAL_TEMPLATE = Path(proposal_template).resolve()

    if not KERNEL_FILE.exists() and not preflight_only:
        return {"status": "error", "reason": "kernel.py not found"}

    run_dir = _make_run_dir(run_dir_arg, resume_from)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    final_summary_path = run_dir / "final_summary.md"
    manifest = load_manifest(manifest_path) or {}

    preflight = _run_preflight(gpu, nvcc_bin, ncu_bin)
    if preflight_only:
        manifest["preflight"] = preflight
        manifest["updated_at"] = now_iso()
        save_manifest(manifest_path, manifest)
        _write_text(final_summary_path, render_final_summary(manifest))
        return {"status": "preflight_only", "run_dir": str(run_dir)}

    kernel_type = _get_kernel_type()
    experiment_index = _get_experiment_count() + 1
    experiment_id = f"{kernel_type}_exp_{experiment_index:03d}" if kernel_type else f"exp_{experiment_index:03d}"
    if not parent_id:
        parent_id = _get_last_experiment_id()

    existing_iterations = manifest.get("iterations", [])
    iteration = len(existing_iterations) + 1
    iter_dir = _make_iter_dir(run_dir, iteration)
    iter_dir.mkdir(parents=True, exist_ok=True)

    strategy_payload = load_global_strategy_memory(STRATEGY_MEMORY_FILE)
    scope_key = kernel_type or "unknown_kernel"
    scope = ensure_scope(strategy_payload, scope_key, {"kernel_type": kernel_type, "gpu": gpu})
    constraints = merge_strategy_constraints(scope)
    proposal_path = _write_proposal_stub(iter_dir, hypothesis, kernel_type, constraints)
    tags = extract_strategy_tags(proposal_path)
    fingerprint = build_strategy_fingerprint(kernel_type or "unknown_kernel", tags)

    snapshot_file = _copy_snapshot(iter_dir)

    if dry_run:
        return {
            "status": "dry_run",
            "experiment_id": experiment_id,
            "run_dir": str(run_dir),
            "iter_dir": str(iter_dir),
            "proposal_path": str(proposal_path),
        }

    _git_commit(f"experiment: {hypothesis}")
    git_sha = _git_sha()

    benchmark_rc, benchmark_result, benchmark_command = _run_bench_to_artifacts(iter_dir, quick=quick, gpu=gpu)
    correctness_pass = (benchmark_result.get("correctness") or {}).get("passed") is True

    targeted_rc = None
    full_rc = None
    targeted_meta: dict[str, str] = {}
    full_meta: dict[str, str] = {}
    ncu_metrics: dict[str, str] = {}

    if correctness_pass and targeted_ncu:
        targeted_rc, targeted_meta = _run_ncu_mode(iter_dir, mode="targeted", gpu=gpu, output_name="targeted")
    if correctness_pass and full_ncu:
        full_rc, full_meta = _run_ncu_mode(iter_dir, mode="full", gpu=gpu, output_name="full")
        full_summary = iter_dir / "full_summary.txt"
        if full_summary.exists():
            for line in full_summary.read_text(encoding="utf-8").splitlines():
                if ":" in line and line.startswith("ncu_"):
                    key, value = line.split(":", 1)
                    ncu_metrics[key.strip()] = value.strip()
    elif correctness_pass and targeted_ncu:
        targeted_summary = iter_dir / "targeted_summary.txt"
        if targeted_summary.exists():
            for line in targeted_summary.read_text(encoding="utf-8").splitlines():
                if ":" in line and line.startswith("ncu_"):
                    key, value = line.split(":", 1)
                    ncu_metrics[key.strip()] = value.strip()

    record: dict[str, Any] = {
        "iteration": iteration,
        "experiment_id": experiment_id,
        "hypothesis": hypothesis,
        "parent_experiment_id": parent_id,
        "git_sha": git_sha,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "iter_dir": str(iter_dir.relative_to(ROOT)),
        "snapshot_file": snapshot_file,
        "proposal_path": str(proposal_path.relative_to(ROOT)),
        "benchmark_command": benchmark_command,
        "benchmark_rc": benchmark_rc,
        "benchmark_json": str((iter_dir / "benchmark_result.json").relative_to(ROOT)),
        "benchmark_result": benchmark_result,
        "targeted_ncu_command": targeted_meta.get("command", ""),
        "targeted_ncu_rc": targeted_rc,
        "targeted_report": targeted_meta.get("report", ""),
        "targeted_summary_txt": targeted_meta.get("summary_txt", ""),
        "targeted_details_txt": targeted_meta.get("details_txt", ""),
        "full_ncu_command": full_meta.get("command", ""),
        "full_ncu_rc": full_rc,
        "full_report": full_meta.get("report", ""),
        "full_summary_txt": full_meta.get("summary_txt", ""),
        "full_details_txt": full_meta.get("details_txt", ""),
        "full_report_exists": bool(full_meta.get("report")) and (ROOT / full_meta["report"]).exists(),
        "ncu_expected": full_ncu,
        "ncu_metrics": ncu_metrics,
        "strategy": {
            "tags": tags,
            "fingerprint": fingerprint,
            "constraints": constraints,
        },
        "updated_at": now_iso(),
    }

    kept, keep_reason = _decide_keep(record, parent_id)
    record["kept"] = kept
    record["decision_reason"] = keep_reason
    if not kept:
        _git_revert()

    previous_record = existing_iterations[-1] if existing_iterations else None
    outcome, reason = classify_strategy_outcome(record, previous_record)
    record["strategy"]["outcome"] = outcome
    record["strategy"]["reason"] = reason

    update_memory_bucket(scope[outcome], fingerprint, tags, iteration, reason, outcome, record, previous_record)
    save_global_strategy_memory(STRATEGY_MEMORY_FILE, strategy_payload)

    iteration_summary = iter_dir / "iteration_summary.md"
    _write_text(iteration_summary, render_iteration_markdown(record))

    manifest.update(
        {
            "run_dir": str(run_dir.relative_to(ROOT)),
            "kernel_type": kernel_type,
            "source_kernel_path": str(KERNEL_FILE.relative_to(ROOT)),
            "preflight": preflight,
            "strategy_memory": {
                "scope_key": scope_key,
                "positive": scope.get("positive", {}),
                "negative": scope.get("negative", {}),
                "rejected": scope.get("rejected", {}),
            },
            "updated_at": now_iso(),
        }
    )
    manifest.setdefault("iterations", []).append(record)
    best = choose_best_iteration(manifest["iterations"])
    manifest["best_iteration"] = best.get("iteration") if best else None
    manifest["best_kernel_path"] = best.get("snapshot_file") if best else ""
    save_manifest(manifest_path, manifest)
    _write_text(final_summary_path, render_final_summary(manifest))

    _append_result(record)
    return {
        "status": "completed",
        "experiment_id": experiment_id,
        "run_dir": str(run_dir),
        "iter_dir": str(iter_dir),
        "kept": kept,
        "decision_reason": keep_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Artifact-aware experiment runner for cuda-evolve")
    parser.add_argument("--hypothesis", type=str, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--targeted-ncu", action="store_true")
    parser.add_argument("--full-ncu", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--nvcc-bin", type=str, default="nvcc")
    parser.add_argument("--ncu-bin", type=str, default="ncu")
    parser.add_argument("--parent-id", type=str, default="")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--proposal-template", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run_experiment(
        hypothesis=args.hypothesis,
        quick=args.quick,
        targeted_ncu=args.targeted_ncu,
        full_ncu=args.full_ncu,
        parent_id=args.parent_id,
        gpu=args.gpu,
        nvcc_bin=args.nvcc_bin,
        ncu_bin=args.ncu_bin,
        run_dir_arg=args.run_dir,
        resume_from=args.resume_from,
        preflight_only=args.preflight_only,
        proposal_template=args.proposal_template,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get("status") == "completed" and not result.get("kept", False):
        sys.exit(1)
    if result.get("status") == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
