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
    from .compute_traits import compute_kernel_traits
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
        build_route_id,
        build_strategy_fingerprint,
        ensure_scope,
        extract_strategy_tags,
        classify_strategy_outcome,
        load_global_strategy_memory,
        merge_strategy_constraints,
        save_global_strategy_memory,
        update_design_boundary_state,
        update_route_state,
        update_memory_bucket,
    )
except ImportError:
    from compute_traits import compute_kernel_traits
    from iteration_report import choose_best_iteration, load_manifest, render_final_summary, render_iteration_markdown, save_manifest
    from preflight import collect_preflight, write_preflight_outputs
    from runtime import build_python_cmd
    from strategy_memory import (
        build_route_id,
        build_strategy_fingerprint,
        ensure_scope,
        extract_strategy_tags,
        classify_strategy_outcome,
        load_global_strategy_memory,
        merge_strategy_constraints,
        save_global_strategy_memory,
        update_design_boundary_state,
        update_route_state,
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
REQUIRED_MARKDOWN_ROUTE_PLAN_TERMS = {
    "prototype_ladder": ("prototype ladder", "current_stage", "current stage"),
    "next_missing_stage": ("next_missing_high_upside_stage", "next missing"),
    "promotion_gate": ("promotion_gate", "promotion gate"),
    "negative_evidence_scope": ("negative_evidence_scope", "negative evidence"),
}


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


def _is_placeholder_text(value: str) -> bool:
    lowered = value.strip().lower()
    if not lowered:
        return True
    placeholders = ("fill_me", "todo", "tbd", "none", "n/a", "unknown")
    return any(token == lowered or token in lowered for token in placeholders)


def _validate_route_metadata(route_metadata: dict[str, Any]) -> list[str]:
    if not route_metadata.get("enabled"):
        return []

    errors: list[str] = []
    required_fields = {
        "route_invariant": str(route_metadata.get("invariant") or ""),
        "route_expected_impact": str(route_metadata.get("expected_impact") or ""),
        "route_stop_condition": str(route_metadata.get("stop_condition") or ""),
    }
    for field, value in required_fields.items():
        if _is_placeholder_text(value):
            errors.append(f"{field}_required")

    budget = int(route_metadata.get("budget") or 0)
    if budget <= 0:
        errors.append("route_budget_must_be_positive")

    if route_metadata.get("iteration_role") == "validation" and route_metadata.get("allow_regression"):
        errors.append("validation_cannot_allow_regression")

    return errors


def _normalize_markdown_field(label: str) -> str:
    return "".join(ch for ch in label.lower() if ch.isalnum())


def _markdown_field_values(text: str, field: str) -> list[str]:
    target = _normalize_markdown_field(field)
    values: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("-") or ":" not in stripped:
            continue
        label, value = stripped.lstrip("-").strip().split(":", 1)
        if _normalize_markdown_field(label) == target:
            values.append(value.strip())
    return values


def _count_filled_markdown_field(text: str, field: str) -> int:
    return sum(1 for value in _markdown_field_values(text, field) if not _is_placeholder_text(value))


def _validate_route_plan(route_plan_path: str | None, min_routes: int = 2) -> tuple[bool, str]:
    if not route_plan_path:
        return False, "route_plan_required"

    path = Path(route_plan_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return False, f"route_plan_not_found:{path}"

    text = path.read_text(encoding="utf-8")
    if "fill_me" in text.lower() or "todo" in text.lower() or "tbd" in text.lower():
        return False, "route_plan_contains_placeholder"

    route_count = 0
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return False, f"route_plan_json_invalid:{exc.msg}"
        if not isinstance(payload, dict):
            return False, "route_plan_json_requires_object_with_prototype_ladder_and_routes"

        ladder = payload.get("prototype_ladder") or {}
        if not isinstance(ladder, dict):
            return False, "prototype_ladder_not_object"
        for field in ("current_stage", "next_missing_high_upside_stage"):
            if _is_placeholder_text(str(ladder.get(field) or "")):
                return False, f"prototype_ladder_{field}_required"

        routes = payload.get("routes")
        if isinstance(routes, list):
            route_count = len(routes)
            for idx, route in enumerate(routes, start=1):
                if not isinstance(route, dict):
                    return False, f"route_{idx}_not_object"
                for field in (
                    "invariant",
                    "expected_impact",
                    "dynamic_coverage",
                    "stop_condition",
                    "ladder_stage_targeted",
                    "promotion_gate",
                    "negative_evidence_scope_if_failed",
                ):
                    if _is_placeholder_text(str(route.get(field) or "")):
                        return False, f"route_{idx}_{field}_required"
        else:
            return False, "route_plan_json_routes_required"
    else:
        lowered = text.lower()
        for name, aliases in REQUIRED_MARKDOWN_ROUTE_PLAN_TERMS.items():
            if not any(alias in lowered for alias in aliases):
                return False, f"route_plan_missing_{name}"
        for field in ("current_stage", "next_missing_high_upside_stage", "evidence_needed_before_local_tuning_resumes"):
            if _count_filled_markdown_field(text, field) < 1:
                return False, f"route_plan_{field}_required"
        for field in (
            "route_id",
            "invariant",
            "ladder_stage_targeted",
            "structural cost removed",
            "dynamic coverage",
            "expected_impact",
            "promotion_gate",
            "negative_evidence_scope_if_failed",
            "stop_condition",
        ):
            if _count_filled_markdown_field(text, field) < min_routes:
                return False, f"route_plan_{field}_needs_at_least_{min_routes}_filled_values"
        route_header_count = 0
        route_id_count = 0
        for line in text.splitlines():
            stripped = line.strip().lower()
            if stripped.startswith("- route_id:"):
                route_id_count += 1
            elif stripped.startswith("## route ") or stripped.startswith("### route "):
                suffix = stripped.split("route", 1)[1].strip()
                if suffix and suffix[0].isdigit():
                    route_header_count += 1
        route_count = max(route_id_count, route_header_count)

    if route_count < min_routes:
        return False, f"route_plan_needs_at_least_{min_routes}_routes_found_{route_count}"

    return True, "route_plan_ok"


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


def _run_bench_to_artifacts(
    iter_dir: Path,
    quick: bool,
    gpu: int,
    artifact_prefix: str = "benchmark",
) -> tuple[int, dict[str, Any], str]:
    benchmark_json = iter_dir / f"{artifact_prefix}_result.json"
    stdout_path = iter_dir / f"{artifact_prefix}.stdout.txt"
    stderr_path = iter_dir / f"{artifact_prefix}.stderr.txt"
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
    *,
    kernel_type: str,
    size: dict[str, Any] | None,
    dtype: Any,
    bench_metrics: dict[str, Any] | None,
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
    if kernel_type:
        cmd.extend(["--kernel-type", kernel_type])
    if dtype is not None:
        cmd.extend(["--dtype", str(dtype)])
    if size:
        cmd.extend(["--size-json", json.dumps(size, ensure_ascii=False, separators=(",", ":"))])
    if bench_metrics:
        cmd.extend(["--bench-metrics-json", json.dumps(bench_metrics, ensure_ascii=False, separators=(",", ":"))])
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


def _write_proposal_stub(
    iter_dir: Path,
    hypothesis: str,
    kernel_type: str,
    constraints: dict[str, list[str]],
    route_metadata: dict[str, Any] | None = None,
) -> Path:
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
        f"- active_routes: {', '.join(constraints.get('active_routes', [])) or 'none'}\n"
        f"- design_boundary_active: {', '.join(constraints.get('design_boundary_active', [])) or 'no'}\n"
        f"- design_boundary_reason: {', '.join(constraints.get('design_boundary_reason', [])) or 'none'}\n"
    )
    if route_metadata and route_metadata.get("enabled"):
        content += (
            "\n## Architecture Route\n"
            f"- route_id: {route_metadata.get('route_id', '')}\n"
            f"- invariant: {route_metadata.get('invariant', '')}\n"
            f"- expected_impact: {route_metadata.get('expected_impact', '')}\n"
            f"- budget: {route_metadata.get('budget', 0)}\n"
            f"- iteration_role: {route_metadata.get('iteration_role', '')}\n"
            f"- allow_regression: {route_metadata.get('allow_regression', False)}\n"
            f"- stop_condition: {route_metadata.get('stop_condition', '')}\n"
            f"- route_plan: {route_metadata.get('route_plan', '') or 'none'}\n"
            "\n## Prototype Ladder\n"
            "- current_stage: fill_me\n"
            "- next_missing_high_upside_stage: fill_me\n"
            "- why_local_micro_tuning_is_premature_or_allowed: fill_me\n"
            "- promotion_gate: fill_me\n"
            "- negative_evidence_scope_if_failed: fill_me\n"
            "\n## No Strong Reference Triage\n"
            "- performance_model_upper_bound: fill_me\n"
            "- affected_dynamic_work_fraction: fill_me\n"
            "- current_self_profile_bottleneck: fill_me\n"
            "- structural_cost_to_remove: fill_me\n"
            "- why_this_is_not_a_local_micro_tweak: fill_me\n"
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


def _quick_benchmark_requires_full_validation(bench: dict[str, Any], parent_id: str) -> tuple[bool, str]:
    if bench.get("benchmark_mode") != "quick":
        return False, ""
    correctness = bench.get("correctness") or {}
    if correctness.get("passed") is not True:
        return False, ""

    kernel = bench.get("kernel") or {}
    if kernel.get("stable") is False:
        spread = float(kernel.get("spread_pct", 0.0) or 0.0)
        return True, f"quick_timing_unstable_spread_{spread:.2f}_percent"

    parent_tp = _get_parent_throughput(parent_id)
    current_tp = float(bench.get("throughput_tflops", 0.0) or 0.0)
    if parent_tp <= 0 or current_tp <= 0:
        return False, ""

    improvement = (current_tp - parent_tp) / parent_tp * 100.0
    bench_config = bench.get("bench_config") or {}
    threshold = float(bench_config.get("stability_threshold_pct", 1.5) or 1.5)
    spread_pct = float(kernel.get("spread_pct", 0.0) or 0.0)
    cv_pct = float(kernel.get("cv_pct", 0.0) or 0.0)
    uncertainty_pct = max(1.0, threshold, spread_pct, cv_pct)

    if -uncertainty_pct <= improvement <= 1.0 + uncertainty_pct:
        return True, f"quick_improvement_{improvement:.2f}_within_uncertainty_{uncertainty_pct:.2f}_percent"
    return False, ""


def _decide_keep(record: dict[str, Any], parent_id: str) -> tuple[bool, str]:
    bench = record.get("benchmark_result") or {}
    correctness = bench.get("correctness") or {}
    route = record.get("architecture_route") or {}
    route_mode = bool(route.get("enabled"))
    route_allow_regression = bool(route.get("allow_regression"))
    route_role = str(route.get("iteration_role") or "")
    if not correctness.get("passed"):
        return False, "correctness_failed"
    peak_vram = float(bench.get("peak_vram_mb", 0.0) or 0.0)
    gpu_memory_gb = float(bench.get("gpu_memory_gb", 0.0) or 0.0)
    if gpu_memory_gb > 0 and peak_vram > gpu_memory_gb * 1024 * 0.8:
        return False, "vram_exceeds_80_percent"
    if (bench.get("kernel") or {}).get("stable") is False:
        return False, "timing_unstable"
    if route_mode and route_allow_regression and route_role != "validation":
        return True, "architecture_route_regression_allowed_for_followup"
    current_tp = float(bench.get("throughput_tflops", 0.0) or 0.0)
    parent_tp = _get_parent_throughput(parent_id)
    if parent_tp > 0 and current_tp > 0:
        improvement = (current_tp - parent_tp) / parent_tp * 100
        if improvement > 1.0:
            return True, f"improved_{improvement:.2f}_percent"
        return False, f"improvement_{improvement:.2f}_below_threshold"
    return True, "baseline_seed"


def _synthesize_guidance(kernel_type: str, benchmark_result: dict[str, Any], ncu_metrics: dict[str, str]) -> dict[str, Any]:
    bench_traits = benchmark_result.get("compute_traits") or {}
    primary_size = benchmark_result.get("primary_size") or {}
    bench_metrics = {
        "bottleneck": benchmark_result.get("bottleneck", ""),
        "pct_peak_compute": benchmark_result.get("pct_peak_compute", 0.0),
        "pct_peak_bandwidth": benchmark_result.get("pct_peak_bandwidth", 0.0),
    }
    merged_traits = dict(bench_traits)
    merged_traits.update(
        compute_kernel_traits(
            kernel_type,
            {},
            primary_size,
            dtype=None,
            bench_metrics=bench_metrics,
            ncu_metrics=ncu_metrics,
        )
    )

    recommendation = merged_traits.get("tensor_core_recommendation", "avoid")
    if recommendation == "recommended":
        guidance_class = "major_redesign_candidate"
        next_steps = [
            "Prioritize an MMA/tensor-core structural experiment.",
            "Preserve MMA-friendly tile fill and launch shape while increasing tensor-core instruction share.",
        ]
    elif recommendation == "compare_first":
        guidance_class = "compare_cuda_vs_tc"
        next_steps = [
            "Run an explicit A/B experiment: CUDA-core path vs tensor-core path with padding or packing.",
            "Use measured throughput, occupancy, and tensor-core pct to decide which path to pursue.",
        ]
    elif recommendation == "needs_ncu_evidence":
        guidance_class = "needs_ncu_evidence"
        next_steps = [
            "Collect targeted NCU evidence before escalating to a tensor-core redesign.",
            "If tensor-core instruction share is low under an MMA-friendly shape, upgrade this path to a high-priority experiment.",
        ]
    else:
        guidance_class = "general_compute_path"
        next_steps = [
            "Do not force a tensor-core rewrite from low tensor_core_pct alone.",
            "Focus on algorithmic simplification, warp-level compute, instruction mix, launch shape, or register pressure.",
        ]

    summary = {
        "guidance_class": guidance_class,
        "shape_regime": merged_traits.get("shape_regime", "generic"),
        "tensor_core_recommendation": recommendation,
        "tensor_core_reasoning": merged_traits.get("tensor_core_reasoning", ""),
        "next_steps": next_steps,
        "kernel_traits": merged_traits,
    }
    analysis = {
        "kernel_type": kernel_type,
        "bench_bottleneck": benchmark_result.get("bottleneck", ""),
        "ncu_bottleneck": ncu_metrics.get("ncu_bottleneck", ""),
        "ncu_top_stall": ncu_metrics.get("ncu_top_stall", ""),
        "summary": summary,
    }
    return {"analysis": analysis, "guidance": summary}


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
    architecture_route: bool,
    route_id: str | None,
    route_invariant: str,
    route_expected_impact: str,
    route_budget: int,
    route_stop_condition: str,
    route_iteration_role: str,
    route_allow_regression: bool,
    route_plan: str | None,
    mark_design_boundary: bool,
    clear_design_boundary: bool,
    design_boundary_reason: str,
    allow_local_after_boundary: bool,
    state_only: bool,
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
    if clear_design_boundary and not dry_run:
        update_design_boundary_state(scope, active=False, reason=design_boundary_reason or "cleared_by_run_loop")
        save_global_strategy_memory(STRATEGY_MEMORY_FILE, strategy_payload)
    if mark_design_boundary and not dry_run:
        reason = design_boundary_reason.strip() or hypothesis
        update_design_boundary_state(scope, active=True, reason=reason)
        save_global_strategy_memory(STRATEGY_MEMORY_FILE, strategy_payload)
    constraints = merge_strategy_constraints(scope)
    route_metadata: dict[str, Any] = {
        "enabled": bool(architecture_route),
        "route_id": "",
        "invariant": route_invariant.strip(),
        "expected_impact": route_expected_impact.strip(),
        "budget": int(route_budget or 0),
        "stop_condition": route_stop_condition.strip(),
        "iteration_role": route_iteration_role,
        "allow_regression": bool(route_allow_regression),
        "route_plan": route_plan or "",
    }
    if architecture_route:
        invariant_for_id = route_metadata["invariant"] or hypothesis
        route_metadata["route_id"] = route_id.strip() if route_id else build_route_id(kernel_type or "unknown_kernel", invariant_for_id)
        if not route_metadata["budget"]:
            route_metadata["budget"] = 8

    design_boundary_active = bool((scope.get("design_boundary") or {}).get("active"))
    if design_boundary_active and not architecture_route and not allow_local_after_boundary and not state_only:
        return {
            "status": "error",
            "reason": "design_boundary_active_requires_architecture_route",
            "design_boundary": scope.get("design_boundary", {}),
            "hint": "Use --architecture-route with route metadata, or pass --allow-local-after-boundary with explicit justification.",
        }

    route_errors = _validate_route_metadata(route_metadata)
    if route_errors:
        return {
            "status": "error",
            "reason": "invalid_architecture_route_metadata",
            "errors": route_errors,
        }

    active_route_ids = set(constraints.get("active_routes") or [])
    is_new_route = architecture_route and route_metadata.get("route_id") not in active_route_ids
    if design_boundary_active and is_new_route:
        route_plan_ok, route_plan_reason = _validate_route_plan(route_plan)
        if not route_plan_ok:
            return {
                "status": "error",
                "reason": route_plan_reason,
                "hint": "Create a route portfolio with at least two route candidates before starting a new architecture route.",
            }
    if state_only:
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
                    "inconclusive": scope.get("inconclusive", {}),
                    "routes": scope.get("routes", {}),
                    "design_boundary": scope.get("design_boundary", {}),
                },
                "updated_at": now_iso(),
            }
        )
        save_manifest(manifest_path, manifest)
        _write_text(final_summary_path, render_final_summary(manifest))
        return {
            "status": "state_only",
            "run_dir": str(run_dir),
            "design_boundary": scope.get("design_boundary", {}),
        }
    proposal_path = _write_proposal_stub(iter_dir, hypothesis, kernel_type, constraints, route_metadata)
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
            "architecture_route": route_metadata,
        }

    _git_commit(f"experiment: {hypothesis}")
    git_sha = _git_sha()

    benchmark_rc, benchmark_result, benchmark_command = _run_bench_to_artifacts(iter_dir, quick=quick, gpu=gpu)
    quick_validation_reason = ""
    quick_benchmark_result: dict[str, Any] | None = None
    quick_benchmark_command = ""
    quick_benchmark_json = ""
    if quick and benchmark_rc == 0:
        needs_full_validation, quick_validation_reason = _quick_benchmark_requires_full_validation(benchmark_result, parent_id)
        if needs_full_validation:
            quick_benchmark_result = benchmark_result
            quick_benchmark_command = benchmark_command
            quick_json_path = iter_dir / "quick_benchmark_result.json"
            quick_stdout_path = iter_dir / "quick_benchmark.stdout.txt"
            quick_stderr_path = iter_dir / "quick_benchmark.stderr.txt"
            for src, dst in (
                (iter_dir / "benchmark_result.json", quick_json_path),
                (iter_dir / "benchmark.stdout.txt", quick_stdout_path),
                (iter_dir / "benchmark.stderr.txt", quick_stderr_path),
            ):
                if src.exists():
                    shutil.copy2(src, dst)
            quick_benchmark_json = str(quick_json_path.relative_to(ROOT))
            benchmark_rc, benchmark_result, benchmark_command = _run_bench_to_artifacts(
                iter_dir,
                quick=False,
                gpu=gpu,
            )
    correctness_pass = (benchmark_result.get("correctness") or {}).get("passed") is True
    primary_size = benchmark_result.get("primary_size") or {}
    primary_dtype = None
    sizes_payload = benchmark_result.get("sizes") or []
    if sizes_payload:
        primary_dtype = sizes_payload[0].get("dtype")

    targeted_rc = None
    full_rc = None
    targeted_meta: dict[str, str] = {}
    full_meta: dict[str, str] = {}
    ncu_metrics: dict[str, str] = {}

    if correctness_pass and targeted_ncu:
        targeted_rc, targeted_meta = _run_ncu_mode(
            iter_dir,
            mode="targeted",
            gpu=gpu,
            output_name="targeted",
            kernel_type=kernel_type,
            size=primary_size,
            dtype=primary_dtype,
            bench_metrics=benchmark_result,
        )
    if correctness_pass and full_ncu:
        full_rc, full_meta = _run_ncu_mode(
            iter_dir,
            mode="full",
            gpu=gpu,
            output_name="full",
            kernel_type=kernel_type,
            size=primary_size,
            dtype=primary_dtype,
            bench_metrics=benchmark_result,
        )
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

    guidance_bundle = _synthesize_guidance(kernel_type, benchmark_result, ncu_metrics)

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
        "quick_validation_reason": quick_validation_reason,
        "quick_benchmark_command": quick_benchmark_command,
        "quick_benchmark_json": quick_benchmark_json,
        "quick_benchmark_result": quick_benchmark_result,
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
        "analysis": guidance_bundle["analysis"],
        "guidance": guidance_bundle["guidance"],
        "architecture_route": route_metadata,
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

    scope.setdefault(outcome, {})
    update_memory_bucket(scope[outcome], fingerprint, tags, iteration, reason, outcome, record, previous_record)
    if route_metadata.get("enabled"):
        update_route_state(scope, route_metadata["route_id"], route_metadata, record)
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
                "inconclusive": scope.get("inconclusive", {}),
                "routes": scope.get("routes", {}),
                "design_boundary": scope.get("design_boundary", {}),
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
        "architecture_route": route_metadata,
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
    parser.add_argument(
        "--architecture-route",
        action="store_true",
        help="Treat this experiment as part of a multi-iteration structural route instead of a local micro-tweak.",
    )
    parser.add_argument("--route-id", type=str, default="", help="Stable id for an architecture route. Auto-generated from invariant if omitted.")
    parser.add_argument("--route-invariant", type=str, default="", help="Structural invariant the route must satisfy.")
    parser.add_argument("--route-expected-impact", type=str, default="", help="Expected dynamic coverage and best-case speedup range.")
    parser.add_argument("--route-budget", type=int, default=0, help="Maximum focused sub-iterations for this route.")
    parser.add_argument("--route-stop-condition", type=str, default="", help="Evidence-based condition for stopping the route.")
    parser.add_argument("--route-plan", type=str, default="", help="Path to a route portfolio plan with at least two candidate architecture routes.")
    parser.add_argument(
        "--route-iteration-role",
        type=str,
        default="prototype",
        choices=["prototype", "repair", "resource_rebalance", "tile_geometry", "validation", "local_tune"],
        help="Role of this sub-iteration inside the route.",
    )
    parser.add_argument(
        "--route-allow-regression",
        action="store_true",
        help="Keep a correctness-passing structural prototype even if it is slower, so later sub-iterations can build on it.",
    )
    parser.add_argument("--mark-design-boundary", action="store_true", help="Mark the current kernel scope as design-boundary limited.")
    parser.add_argument("--clear-design-boundary", action="store_true", help="Clear the design-boundary marker for the current kernel scope.")
    parser.add_argument("--design-boundary-reason", type=str, default="", help="Reason for marking or clearing the design-boundary state.")
    parser.add_argument("--state-only", action="store_true", help="Update strategy-memory state and run metadata without committing or benchmarking.")
    parser.add_argument(
        "--allow-local-after-boundary",
        action="store_true",
        help="Allow a non-route local experiment even while design-boundary mode is active. Requires explicit human-level justification in the hypothesis/proposal.",
    )
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
        architecture_route=args.architecture_route,
        route_id=args.route_id,
        route_invariant=args.route_invariant,
        route_expected_impact=args.route_expected_impact,
        route_budget=args.route_budget,
        route_stop_condition=args.route_stop_condition,
        route_iteration_role=args.route_iteration_role,
        route_allow_regression=args.route_allow_regression,
        route_plan=args.route_plan,
        mark_design_boundary=args.mark_design_boundary,
        clear_design_boundary=args.clear_design_boundary,
        design_boundary_reason=args.design_boundary_reason,
        allow_local_after_boundary=args.allow_local_after_boundary,
        state_only=args.state_only,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get("status") == "completed" and not result.get("kept", False):
        route = result.get("architecture_route") or {}
        if route.get("enabled") and route.get("iteration_role") != "validation":
            sys.exit(0)
        sys.exit(1)
    if result.get("status") == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
