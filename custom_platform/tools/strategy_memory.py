"""Structured strategy memory helpers for the custom_platform scaffold."""

from __future__ import annotations

import hashlib
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
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def build_strategy_fingerprint(kernel_type: str, tags: list[str]) -> str:
    raw = json.dumps({"kernel_type": kernel_type, "tags": tags}, sort_keys=True, ensure_ascii=False)
    return fingerprint_from_text(raw)


def default_global_strategy_memory() -> dict[str, Any]:
    return {"version": 1, "updated_at": "", "scopes": {}}


def load_global_strategy_memory(path: Path) -> dict[str, Any]:
    if not path.exists():
        return default_global_strategy_memory()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("version", 1)
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
        scope = {"meta": meta, "positive": {}, "negative": {}, "rejected": {}, "guidance_history": []}
        scopes[scope_key] = scope
    else:
        scope.setdefault("meta", meta)
        scope.setdefault("positive", {})
        scope.setdefault("negative", {})
        scope.setdefault("rejected", {})
        scope.setdefault("guidance_history", [])
    return scope


def get_kernel_median_ms(record: dict[str, Any]) -> float | None:
    bench = record.get("benchmark_result") or {}
    kernel = bench.get("kernel") or {}
    median = kernel.get("median_ms")
    try:
        return float(median) if median is not None else None
    except (TypeError, ValueError):
        return None


def classify_strategy_outcome(record: dict[str, Any], previous_record: dict[str, Any] | None) -> tuple[str, str]:
    bench = record.get("benchmark_result") or {}
    if record.get("benchmark_rc") != 0:
        error = bench.get("error") or {}
        code = str(error.get("code") or "").strip()
        return ("rejected", f"benchmark_failed:{code or 'unknown'}")
    correctness = bench.get("correctness") or {}
    if correctness.get("passed") is False:
        return ("rejected", "correctness_failed")
    if record.get("profile_expected") and record.get("profile_rc") not in (None, 0):
        return ("rejected", "profile_failed")
    if record.get("profile_expected") and not record.get("profile_report_exists"):
        return ("rejected", "profile_incomplete")

    current_median = get_kernel_median_ms(record)
    if previous_record is None:
        return ("positive", "baseline_seed")
    if current_median is None:
        return ("rejected", "no_current_median")
    previous_median = get_kernel_median_ms(previous_record)
    if previous_median is None:
        return ("rejected", "no_previous_median")
    if current_median < previous_median:
        return ("positive", "faster_than_previous")
    return ("negative", "slower_or_equal_to_previous")


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
        "workload_class": ((record.get("guidance") or {}).get("workload_class")),
        "optimization_recommendation": ((record.get("guidance") or {}).get("optimization_recommendation")),
        "kernel_traits": ((record.get("guidance") or {}).get("kernel_traits") or {}),
    }


def merge_strategy_constraints(scope: dict[str, Any]) -> dict[str, list[str]]:
    blocked = set((scope.get("rejected") or {}).keys())
    preferred = set((scope.get("positive") or {}).keys())
    preferred.difference_update(blocked)
    return {"blocked": sorted(blocked), "preferred": sorted(preferred)}
