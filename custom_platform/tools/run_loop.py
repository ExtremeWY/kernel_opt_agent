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
        classify_strategy_outcome,
        ensure_scope,
        extract_strategy_tags,
        load_global_strategy_memory,
        merge_strategy_constraints,
        save_global_strategy_memory,
        sanitize_token,
        update_design_boundary_state,
        update_memory_bucket,
        update_route_state,
    )
except ImportError:
    from compute_traits import compute_kernel_traits
    from iteration_report import choose_best_iteration, load_manifest, render_final_summary, render_iteration_markdown, save_manifest
    from preflight import collect_preflight, write_preflight_outputs
    from runtime import build_python_cmd
    from strategy_memory import (
        build_route_id,
        build_strategy_fingerprint,
        classify_strategy_outcome,
        ensure_scope,
        extract_strategy_tags,
        load_global_strategy_memory,
        merge_strategy_constraints,
        save_global_strategy_memory,
        sanitize_token,
        update_design_boundary_state,
        update_memory_bucket,
        update_route_state,
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
REQUIRED_MARKDOWN_ROUTE_PLAN_TERMS = {
    "prototype_ladder": ("prototype ladder", "current_stage", "current stage"),
    "next_missing_stage": ("next_missing_high_upside_stage", "next missing"),
    "promotion_gate": ("promotion_gate", "promotion gate"),
    "negative_evidence_scope": ("negative_evidence_scope", "negative evidence"),
}
RESULTS_HEADER = (
    "experiment_id\thypothesis\tcorrectness\ttime_ms\tthroughput\tpeak_vram_mb\tkept"
    "\tachieved_compute_tflops\tachieved_memory_gbps\tpeak_compute_tflops\tpeak_memory_gbps"
    "\tbottleneck\tgit_sha\tparent_experiment_id\tprofile_top_stall\tprofile_occupancy"
    "\tprofile_l1_hit_rate\tprofile_l2_hit_rate\tstrategy_tags\tstrategy_fingerprint"
    "\tstrategy_outcome\tstrategy_reason\tguidance_class\toptimization_recommendation"
    "\trun_dir\titer_dir\tprofile_report\n"
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


def _record_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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

    if int(route_metadata.get("budget") or 0) <= 0:
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
        if not isinstance(routes, list):
            return False, "route_plan_json_routes_required"
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
    return _record_path(snapshot_path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _run_bench_to_artifacts(iter_dir: Path, platform: str) -> tuple[int, dict[str, Any], str]:
    benchmark_json = iter_dir / "benchmark_result.json"
    stdout_path = iter_dir / "benchmark.stdout.txt"
    stderr_path = iter_dir / "benchmark.stderr.txt"
    cmd = build_python_cmd(
        str(ROOT / "tools" / "bench.py"),
        "--platform",
        platform,
        "--json-out",
        str(benchmark_json),
    )
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
    cmd = build_python_cmd(
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
    )
    result = _run(cmd, timeout=1800)
    _write_text(stdout_path, result.stdout or "")
    _write_text(stderr_path, result.stderr or "")
    metadata = {
        "command": " ".join(cmd),
        "report": _record_path(report_path),
        "summary_txt": _record_path(summary_path),
        "details_txt": _record_path(details_path),
        "stdout_txt": _record_path(stdout_path),
        "stderr_txt": _record_path(stderr_path),
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
                    return _record_path(Path(value))
                except Exception:
                    return value
    return default_path


def _ensure_results_file() -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(RESULTS_HEADER, encoding="utf-8")


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
        (record.get("guidance") or {}).get("guidance_class", ""),
        (record.get("guidance") or {}).get("optimization_recommendation", ""),
        record.get("run_dir", ""),
        record.get("iter_dir", ""),
        record.get("profile_report", ""),
    ]
    with RESULTS_FILE.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(row) + "\n")


def _decide_keep(
    record: dict[str, Any],
    previous_record: dict[str, Any] | None,
    route_metadata: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    bench = record.get("benchmark_result") or {}
    correctness = bench.get("correctness") or {}
    if not correctness.get("passed"):
        return False, "correctness_failed"
    if record.get("profile_expected") and not record.get("profile_report_exists"):
        return False, "profile_missing"
    if (bench.get("kernel") or {}).get("stable") is False:
        return False, "timing_unstable"
    route_metadata = route_metadata or {}
    if (
        route_metadata.get("enabled")
        and route_metadata.get("allow_regression")
        and route_metadata.get("iteration_role") != "validation"
    ):
        return True, "architecture_route_regression_allowed_for_followup"
    current_tp = float(bench.get("throughput_tflops", 0.0) or 0.0)
    parent_tp = _get_parent_throughput(previous_record)
    if parent_tp > 0 and current_tp > 0:
        improvement = (current_tp - parent_tp) / parent_tp * 100.0
        if improvement > 1.0:
            return True, f"improved_{improvement:.2f}_percent"
        return False, f"improvement_{improvement:.2f}_below_threshold"
    return True, "baseline_seed"


def _synthesize_guidance(
    kernel_type: str,
    benchmark_result: dict[str, Any],
    profile_metrics: dict[str, str],
) -> dict[str, Any]:
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
            {"kernel_opt_characteristics": bench_traits},
            primary_size,
            dtype=None,
            bench_metrics=bench_metrics,
            profile_metrics=profile_metrics,
        )
    )

    guidance = {
        "guidance_class": merged_traits.get("guidance_class", "needs_profile_evidence"),
        "shape_regime": merged_traits.get("shape_regime", "unknown"),
        "workload_class": merged_traits.get("workload_class", "generic"),
        "optimization_recommendation": merged_traits.get("optimization_recommendation", "needs_profile_evidence"),
        "optimization_reasoning": merged_traits.get("optimization_reasoning", ""),
        "next_steps": merged_traits.get("next_steps", []),
        "kernel_traits": merged_traits,
    }
    analysis = {
        "kernel_type": kernel_type,
        "bench_bottleneck": benchmark_result.get("bottleneck", ""),
        "profile_top_stall": profile_metrics.get("profile_top_stall", ""),
        "profile_compute_util": profile_metrics.get("profile_compute_util", ""),
        "profile_memory_util": profile_metrics.get("profile_memory_util", ""),
        "summary": guidance,
    }
    return {"analysis": analysis, "guidance": guidance}


def run_experiment(
    platform: str,
    hypothesis: str,
    allow_placeholder: bool,
    run_dir_arg: str | None,
    resume_from: str | None,
    proposal_template: str | None,
    preflight_only: bool,
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
    _ensure_results_file()
    global PROPOSAL_TEMPLATE
    if proposal_template:
        PROPOSAL_TEMPLATE = Path(proposal_template).resolve()

    run_dir = _make_run_dir(run_dir_arg, resume_from)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    manifest = load_manifest(manifest_path) or {
        "run_dir": _record_path(run_dir),
        "platform": platform,
        "kernel_type": "",
        "source_kernel_path": _record_path(KERNEL_FILE),
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
    preflight["json_path"] = _record_path(run_dir / "preflight_check.json")
    preflight["markdown_path"] = _record_path(run_dir / "preflight_check.md")
    manifest["preflight"] = preflight
    if preflight_only:
        save_manifest(manifest_path, manifest)
        _write_text(run_dir / "final_summary.md", render_final_summary(manifest))
        return {"status": "preflight_only", "run_dir": _record_path(run_dir)}
    if not preflight["ready"]:
        save_manifest(manifest_path, manifest)
        _write_text(run_dir / "final_summary.md", render_final_summary(manifest))
        return {"status": "error", "reason": "preflight_failed", "run_dir": _record_path(run_dir)}

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
            "source_kernel_path": _record_path(KERNEL_FILE),
        },
    )
    if clear_design_boundary:
        update_design_boundary_state(scope, active=False, reason=design_boundary_reason or "cleared_by_run_loop")
        save_global_strategy_memory(STRATEGY_MEMORY_FILE, global_memory)
    if mark_design_boundary:
        reason = design_boundary_reason.strip() or hypothesis
        update_design_boundary_state(scope, active=True, reason=reason)
        save_global_strategy_memory(STRATEGY_MEMORY_FILE, global_memory)

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
        route_metadata["route_id"] = route_id.strip() if route_id else build_route_id(
            f"{platform}_{kernel_type or 'unknown'}",
            invariant_for_id,
        )
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
        return {"status": "error", "reason": "invalid_architecture_route_metadata", "errors": route_errors}

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
                "run_dir": _record_path(run_dir),
                "platform": platform,
                "kernel_type": kernel_type,
                "source_kernel_path": _record_path(KERNEL_FILE),
                "preflight": preflight,
                "strategy_memory": {
                    "scope_key": scope_key,
                    "positive": scope.get("positive", {}),
                    "negative": scope.get("negative", {}),
                    "rejected": scope.get("rejected", {}),
                    "inconclusive": scope.get("inconclusive", {}),
                    "routes": scope.get("routes", {}),
                    "design_boundary": scope.get("design_boundary", {}),
                    "guidance_history": scope.get("guidance_history", []),
                },
                "updated_at": now_iso(),
            }
        )
        save_manifest(manifest_path, manifest)
        _write_text(run_dir / "final_summary.md", render_final_summary(manifest))
        return {
            "status": "state_only",
            "run_dir": _record_path(run_dir),
            "design_boundary": scope.get("design_boundary", {}),
        }

    iteration = len(manifest.get("iterations", [])) + 1
    iter_dir = _make_iter_dir(run_dir, iteration)
    iter_dir.mkdir(parents=True, exist_ok=True)
    snapshot_file = _copy_snapshot(iter_dir)
    proposal_path = _write_proposal_stub(iter_dir, hypothesis, kernel_type, constraints, route_metadata)

    benchmark_rc, benchmark_result, benchmark_command = _run_bench_to_artifacts(iter_dir, platform)
    correctness_passed = bool((benchmark_result.get("correctness") or {}).get("passed"))
    if benchmark_rc != 0 and correctness_passed and not benchmark_result.get("error"):
        benchmark_rc = 0

    profile_rc = None
    profile_metadata = {
        "command": "",
        "report": _record_path(iter_dir / "profile_report.txt"),
        "summary_txt": _record_path(iter_dir / "profile_summary.txt"),
        "details_txt": _record_path(iter_dir / "profile_details.txt"),
        "stdout_txt": _record_path(iter_dir / "profile.stdout.txt"),
        "stderr_txt": _record_path(iter_dir / "profile.stderr.txt"),
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
    guidance_bundle = _synthesize_guidance(kernel_type, benchmark_result, profile_metrics)

    record = {
        "iteration": iteration,
        "experiment_id": f"{kernel_type or 'kernel'}_exp_{iteration:03d}",
        "hypothesis": hypothesis,
        "snapshot_file": snapshot_file,
        "benchmark_command": benchmark_command,
        "benchmark_json": _record_path(iter_dir / "benchmark_result.json"),
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
        "analysis": guidance_bundle["analysis"],
        "guidance": guidance_bundle["guidance"],
        "architecture_route": route_metadata,
        "git_sha": _git_sha(),
        "parent_experiment_id": previous_record.get("experiment_id", "") if previous_record else "",
        "run_dir": _record_path(run_dir),
        "iter_dir": _record_path(iter_dir),
        "proposal_path": _record_path(proposal_path),
    }

    kept, keep_reason = _decide_keep(record, previous_record, route_metadata)
    record["kept"] = kept
    record["keep_reason"] = keep_reason

    outcome, reason = classify_strategy_outcome(record, previous_record)
    record["strategy"] = {
        "tags": strategy_tags,
        "fingerprint": strategy_fingerprint,
        "outcome": outcome,
        "reason": reason,
        "constraints": constraints,
    }
    scope.setdefault(outcome, {})
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
    scope.setdefault("guidance_history", []).append(
        {
            "iteration": iteration,
            "experiment_id": record["experiment_id"],
            "guidance_class": record["guidance"].get("guidance_class"),
            "optimization_recommendation": record["guidance"].get("optimization_recommendation"),
            "shape_regime": record["guidance"].get("shape_regime"),
            "workload_class": record["guidance"].get("workload_class"),
        }
    )
    manifest["strategy_memory"] = {
        "scope_key": scope_key,
        "positive": scope.get("positive", {}),
        "negative": scope.get("negative", {}),
            "rejected": scope.get("rejected", {}),
            "inconclusive": scope.get("inconclusive", {}),
            "routes": scope.get("routes", {}),
            "design_boundary": scope.get("design_boundary", {}),
            "guidance_history": scope.get("guidance_history", []),
        }

    if route_metadata.get("enabled"):
        update_route_state(scope, route_metadata["route_id"], route_metadata, record)
        manifest["strategy_memory"]["routes"] = scope.get("routes", {})

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
            "run_dir": _record_path(run_dir),
        "iteration": iteration,
        "kept": kept,
        "keep_reason": keep_reason,
        "architecture_route": route_metadata,
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
    parser.add_argument(
        "--architecture-route",
        action="store_true",
        help="Treat this experiment as part of a multi-iteration structural route instead of a local micro-tweak.",
    )
    parser.add_argument("--route-id", default="", help="Stable id for an architecture route. Auto-generated from invariant if omitted.")
    parser.add_argument("--route-invariant", default="", help="Structural invariant the route must satisfy.")
    parser.add_argument("--route-expected-impact", default="", help="Expected dynamic coverage and best-case speedup range.")
    parser.add_argument("--route-budget", type=int, default=0, help="Maximum focused sub-iterations for this route.")
    parser.add_argument("--route-stop-condition", default="", help="Evidence-based condition for stopping the route.")
    parser.add_argument("--route-plan", default="", help="Path to a route portfolio plan with at least two candidate routes.")
    parser.add_argument(
        "--route-iteration-role",
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
    parser.add_argument("--design-boundary-reason", default="", help="Reason for marking or clearing design-boundary state.")
    parser.add_argument("--state-only", action="store_true", help="Update strategy-memory state and run metadata without benchmarking.")
    parser.add_argument(
        "--allow-local-after-boundary",
        action="store_true",
        help="Allow a non-route local experiment while design-boundary mode is active.",
    )
    args = parser.parse_args()

    result = run_experiment(
        platform=args.platform,
        hypothesis=args.hypothesis,
        allow_placeholder=args.allow_placeholder,
        run_dir_arg=args.run_dir,
        resume_from=args.resume_from,
        proposal_template=args.proposal_template,
        preflight_only=args.preflight_only,
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

    print("=== RUN LOOP SUMMARY ===")
    print(f"hypothesis: {args.hypothesis}")
    print(f"status: {result.get('status')}")
    if result.get("reason"):
        print(f"reason: {result.get('reason')}")
    if result.get("errors"):
        print(f"errors: {', '.join(result.get('errors') or [])}")
    if result.get("hint"):
        print(f"hint: {result.get('hint')}")
    if "iteration" in result:
        print(f"iteration: {result['iteration']}")
    if "kept" in result:
        print(f"kept: {'yes' if result['kept'] else 'no'}")
        print(f"keep_reason: {result.get('keep_reason', '')}")
    if result.get("architecture_route"):
        route = result["architecture_route"]
        print(f"architecture_route: {'yes' if route.get('enabled') else 'no'}")
        if route.get("enabled"):
            print(f"route_id: {route.get('route_id', '')}")
    print(f"run_dir: {result.get('run_dir', '')}")
    print("=== END RUN LOOP SUMMARY ===")

    if result.get("status") == "error":
        sys.exit(1)
    if result.get("status") == "ok" and not result.get("kept", False):
        route = result.get("architecture_route") or {}
        if route.get("enabled") and route.get("iteration_role") != "validation":
            sys.exit(0)
        sys.exit(1)


if __name__ == "__main__":
    main()
