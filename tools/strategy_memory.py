"""Structured strategy memory helpers for optimization experiments."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_SCOPE_TOKEN = "na"
STRATEGY_TAGS_HEADER = "## Strategy tags"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sanitize_token(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip())
    return cleaned.strip("_")[:64] or DEFAULT_SCOPE_TOKEN


def normalize_strategy_tags(tags: list[str]) -> list[str]:
    normalized = []
    blocked_placeholders = {"fill_me_tag", "unlabeled_strategy"}
    for tag in tags:
        clean = re.sub(r"\s+", "_", tag.strip().lower())
        clean = re.sub(r"[^a-z0-9_\-]", "", clean)
        if clean and clean not in blocked_placeholders:
            normalized.append(clean)
    return sorted(set(normalized))


def extract_strategy_tags(proposal_path: Path) -> list[str]:
    if not proposal_path.exists():
        return []
    content = proposal_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    tags: list[str] = []
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower() == STRATEGY_TAGS_HEADER.lower():
            in_section = True
            continue
        if in_section and stripped.startswith("## "):
            break
        if in_section and stripped.startswith("-"):
            tags.append(stripped.lstrip("-").strip())
    return normalize_strategy_tags(tags)


def fingerprint_from_text(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def build_strategy_fingerprint(kernel_type: str, tags: list[str]) -> str:
    canonical = {
        "kernel_type": kernel_type,
        "tags": tags,
    }
    raw = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
    return fingerprint_from_text(raw)


def default_global_strategy_memory() -> dict[str, Any]:
    return {
        "version": 2,
        "updated_at": "",
        "scopes": {},
    }


def load_global_strategy_memory(path: Path) -> dict[str, Any]:
    if not path.exists():
        return default_global_strategy_memory()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("version", 2)
    payload.setdefault("updated_at", "")
    payload.setdefault("scopes", {})
    return payload


def save_global_strategy_memory(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = now_iso()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_scope(global_payload: dict[str, Any], scope_key: str, meta: dict[str, Any]) -> dict[str, Any]:
    scopes = global_payload.setdefault("scopes", {})
    scope = scopes.get(scope_key)
    if scope is None:
        scope = {
            "meta": meta,
            "positive": {},
            "negative": {},
            "rejected": {},
            "inconclusive": {},
            "routes": {},
            "design_boundary": {
                "active": False,
                "reason": "",
                "marked_at": "",
                "cleared_at": "",
            },
            "guidance_history": [],
        }
        scopes[scope_key] = scope
    else:
        scope.setdefault("meta", meta)
        scope.setdefault("positive", {})
        scope.setdefault("negative", {})
        scope.setdefault("rejected", {})
        scope.setdefault("inconclusive", {})
        scope.setdefault("routes", {})
        scope.setdefault(
            "design_boundary",
            {
                "active": False,
                "reason": "",
                "marked_at": "",
                "cleared_at": "",
            },
        )
        scope.setdefault("guidance_history", [])
    return scope


def get_kernel_median_ms(record: dict[str, Any]) -> float | None:
    bench = record.get("benchmark_result") or {}
    kernel = bench.get("kernel") or {}
    median = kernel.get("median_ms")
    if median is None:
        return None
    try:
        return float(median)
    except (TypeError, ValueError):
        return None


def classify_strategy_outcome(record: dict[str, Any], previous_record: dict[str, Any] | None) -> tuple[str, str]:
    bench = record.get("benchmark_result") or {}
    iteration_mode = str(record.get("iteration_mode") or "local_tune")
    if iteration_mode in ("external_probe", "framework_maintenance"):
        return ("inconclusive", f"{iteration_mode}_evidence_only")
    route = record.get("architecture_route") or {}
    route_mode = bool(route.get("enabled"))
    route_role = str(route.get("iteration_role") or "")
    route_milestone = str(route.get("milestone") or "")
    route_validation = route_mode and (route_role == "validation" or route_milestone == "validation")
    route_invariant_satisfied = route.get("invariant_satisfied") is True

    def nonblocking_route(reason: str) -> tuple[str, str]:
        if route_mode and (not route_validation or not route_invariant_satisfied):
            return ("inconclusive", f"architecture_route_subiteration:{reason}")
        return ("rejected", reason)

    if record.get("benchmark_rc") != 0:
        error = bench.get("error") or {}
        code = str(error.get("code") or "").strip()
        return nonblocking_route(f"benchmark_failed:{code or 'unknown'}")
    if bench.get("correctness", {}).get("passed") is False:
        return nonblocking_route("correctness_failed")
    if record.get("targeted_ncu_rc") not in (None, 0):
        return nonblocking_route("targeted_ncu_failed")
    if record.get("full_ncu_rc") not in (None, 0):
        return nonblocking_route("full_ncu_failed")
    if record.get("ncu_expected") and not record.get("full_report_exists"):
        return nonblocking_route("ncu_incomplete")
    if (bench.get("kernel") or {}).get("stable") is False:
        return nonblocking_route("timing_unstable")

    current_median = get_kernel_median_ms(record)
    if previous_record is None:
        return ("positive", "baseline_seed")
    if current_median is None:
        return nonblocking_route("no_current_median")
    previous_median = get_kernel_median_ms(previous_record)
    if previous_median is None:
        return nonblocking_route("no_previous_median")
    if current_median < previous_median:
        return ("positive", "faster_than_previous")
    if route_mode and (not route_validation or not route_invariant_satisfied):
        return ("inconclusive", "architecture_route_subiteration:slower_or_equal_to_previous")
    return ("negative", "slower_or_equal_to_previous")


def build_route_id(kernel_type: str, invariant: str) -> str:
    route_text = invariant.strip() or "unspecified_route_invariant"
    return f"{sanitize_token(kernel_type)}_{fingerprint_from_text(route_text)}"


def update_route_state(
    scope: dict[str, Any],
    route_id: str,
    route_metadata: dict[str, Any],
    record: dict[str, Any],
) -> None:
    routes = scope.setdefault("routes", {})
    route = routes.get(route_id)
    if route is None:
        route = {
            "route_id": route_id,
            "route_type": route_metadata.get("route_type", "architecture_route"),
            "invariant": route_metadata.get("invariant", ""),
            "expected_impact": route_metadata.get("expected_impact", ""),
            "budget": route_metadata.get("budget", 0),
            "stop_condition": route_metadata.get("stop_condition", ""),
            "route_plan": route_metadata.get("route_plan", ""),
            "milestones": {},
            "status": "active",
            "created_at": now_iso(),
            "updated_at": "",
            "subiterations": [],
        }
        routes[route_id] = route

    route["updated_at"] = now_iso()
    if route_metadata.get("invariant"):
        route["invariant"] = route_metadata["invariant"]
    if route_metadata.get("expected_impact"):
        route["expected_impact"] = route_metadata["expected_impact"]
    if route_metadata.get("budget"):
        route["budget"] = route_metadata["budget"]
    if route_metadata.get("stop_condition"):
        route["stop_condition"] = route_metadata["stop_condition"]
    if route_metadata.get("route_plan"):
        route["route_plan"] = route_metadata["route_plan"]
    if route_metadata.get("route_type"):
        route["route_type"] = route_metadata["route_type"]

    strategy = record.get("strategy") or {}
    milestone = str(route_metadata.get("milestone") or "")
    milestone_status = str(route_metadata.get("milestone_status") or "")
    entry = {
        "iteration": record.get("iteration"),
        "experiment_id": record.get("experiment_id", ""),
        "role": route_metadata.get("iteration_role", ""),
        "milestone": milestone,
        "milestone_status": milestone_status,
        "git_sha": record.get("git_sha", ""),
        "kept": bool(record.get("kept")),
        "decision_reason": record.get("decision_reason", ""),
        "strategy_outcome": strategy.get("outcome", ""),
        "strategy_reason": strategy.get("reason", ""),
        "invariant_satisfied": route_metadata.get("invariant_satisfied"),
        "old_bottleneck_removed": route_metadata.get("old_bottleneck_removed"),
        "stage_metrics": route_metadata.get("stage_metrics") or {},
        "failure_class": route_metadata.get("failure_class", ""),
        "negative_evidence_scope": route_metadata.get("negative_evidence_scope", ""),
        "blocks": route_metadata.get("blocks") or [],
        "does_not_block": route_metadata.get("does_not_block") or [],
        "benchmark_json": record.get("benchmark_json", ""),
        "targeted_report": record.get("targeted_report", ""),
        "full_report": record.get("full_report", ""),
    }
    route.setdefault("subiterations", []).append(entry)
    if milestone:
        route.setdefault("milestones", {})[milestone] = {
            "status": milestone_status,
            "iteration": record.get("iteration"),
            "experiment_id": record.get("experiment_id", ""),
            "updated_at": now_iso(),
            "stage_metrics": route_metadata.get("stage_metrics") or {},
            "invariant_satisfied": route_metadata.get("invariant_satisfied"),
            "old_bottleneck_removed": route_metadata.get("old_bottleneck_removed"),
        }
    used_budget = len(route.get("subiterations") or [])
    route["used_budget"] = used_budget

    outcome = strategy.get("outcome", "")
    reason = strategy.get("reason", "")
    if outcome == "positive" and record.get("kept"):
        route["status"] = "active_positive"
    elif (
        (route_metadata.get("iteration_role") == "validation" or route_metadata.get("milestone") == "validation")
        and route_metadata.get("invariant_satisfied") is True
        and outcome == "negative"
    ):
        route["status"] = "negative"
    elif (
        (route_metadata.get("iteration_role") == "validation" or route_metadata.get("milestone") == "validation")
        and route_metadata.get("invariant_satisfied") is True
        and outcome == "positive"
    ):
        route["status"] = "validated_positive"
    elif outcome == "inconclusive":
        route["status"] = "active_repair" if used_budget < int(route.get("budget") or 0) else "budget_exhausted"
    elif "correctness_failed" in reason or "benchmark_failed" in reason:
        route["status"] = "active_repair"


def update_design_boundary_state(scope: dict[str, Any], *, active: bool, reason: str) -> None:
    state = scope.setdefault(
        "design_boundary",
        {
            "active": False,
            "reason": "",
            "marked_at": "",
            "cleared_at": "",
        },
    )
    state["active"] = bool(active)
    if active:
        state["reason"] = reason.strip()
        state["marked_at"] = now_iso()
    else:
        state["cleared_at"] = now_iso()
        if reason.strip():
            state["clear_reason"] = reason.strip()


def update_memory_bucket(
    bucket: dict[str, Any],
    fingerprint: str,
    tags: list[str],
    iteration: int,
    reason: str,
    outcome: str,
    record: dict[str, Any],
    previous_record: dict[str, Any] | None,
) -> None:
    current_median = get_kernel_median_ms(record)
    previous_median = get_kernel_median_ms(previous_record) if previous_record else None
    route = record.get("architecture_route") or {}
    item = bucket.get(fingerprint)
    if item is None:
        item = {
            "tags": tags,
            "first_iteration": iteration,
            "last_iteration": iteration,
            "count": 0,
            "last_outcome": outcome,
            "last_reason": reason,
            "evidence": {},
        }
        bucket[fingerprint] = item
    item["last_iteration"] = iteration
    item["count"] = int(item.get("count", 0)) + 1
    item["last_outcome"] = outcome
    item["last_reason"] = reason
    item["tags"] = tags
    item["evidence"] = {
        "baseline_iteration": previous_record.get("iteration") if previous_record else None,
        "baseline_median_ms": previous_median,
        "current_median_ms": current_median,
        "shape_regime": ((record.get("guidance") or {}).get("shape_regime")),
        "guidance_class": ((record.get("guidance") or {}).get("guidance_class")),
        "kernel_traits": ((record.get("guidance") or {}).get("kernel_traits") or {}),
        "route": {
            "enabled": bool(route.get("enabled")),
            "route_id": route.get("route_id", ""),
            "route_type": route.get("route_type", ""),
            "milestone": route.get("milestone", ""),
            "milestone_status": route.get("milestone_status", ""),
            "invariant_satisfied": route.get("invariant_satisfied"),
            "old_bottleneck_removed": route.get("old_bottleneck_removed"),
            "failure_class": route.get("failure_class", ""),
            "negative_evidence_scope": route.get("negative_evidence_scope", ""),
            "blocks": route.get("blocks") or [],
            "does_not_block": route.get("does_not_block") or [],
        },
    }


def merge_strategy_constraints(scope: dict[str, Any]) -> dict[str, list[str]]:
    blocked = set((scope.get("rejected") or {}).keys())
    preferred = set((scope.get("positive") or {}).keys())
    preferred.difference_update(blocked)
    active_routes = []
    for route_id, route in (scope.get("routes") or {}).items():
        status = str(route.get("status") or "")
        if status.startswith("active") or status == "budget_exhausted":
            active_routes.append(route_id)
    return {
        "blocked": sorted(blocked),
        "preferred": sorted(preferred),
        "active_routes": sorted(active_routes),
        "design_boundary_active": ["yes"] if (scope.get("design_boundary") or {}).get("active") else [],
        "design_boundary_reason": [str((scope.get("design_boundary") or {}).get("reason") or "")],
    }
