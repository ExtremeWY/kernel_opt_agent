#!/usr/bin/env python3
"""Lightweight checks for architecture-route framework rules."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from strategy_memory import classify_strategy_outcome, update_route_state
from run_loop import _decide_keep, _validate_route_metadata, _validate_route_plan


def _valid_plan() -> dict:
    return {
        "prototype_ladder": {
            "current_stage": "old single-kernel value-slice ownership",
            "next_missing_high_upside_stage": "chunk precompute plus fused consumer",
            "evidence_needed_before_local_tuning_resumes": "stage benchmark and validation milestone",
        },
        "oracle_free_model": {
            "operator_dependency_graph": "A depends on K,beta,g,chunk,H_v and not value slice",
            "duplicate_work_or_materialization_model": "remove KKT/solve repetition across value slices",
            "stage_cost_model": "precompute_ms + consumer_ms + A read/write bytes",
        },
        "routes": [
            {
                "route_id": "chunk_a_precompute",
                "invariant": "forward consumes A and does not do row solve",
                "ladder_stage_targeted": "dataflow ownership",
                "structural_cost_removed": "duplicated row solve in value-slice forward",
                "dynamic_coverage": "steady-state prefill chunks",
                "expected_impact": "large enough to justify multi-stage route",
                "promotion_gate": "stage benchmark plus full validation",
                "negative_evidence_scope_if_failed": "only this A layout and resource balance",
                "stop_condition": "resource-balanced validation loses beyond uncertainty",
            },
            {
                "route_id": "owner_resident_kkt",
                "invariant": "producer consumes its own KKT without global raw cache",
                "ladder_stage_targeted": "hot-state residency",
                "structural_cost_removed": "global raw matrix cache handoff",
                "dynamic_coverage": "steady-state prefill chunks",
                "expected_impact": "large enough to justify multi-stage route",
                "promotion_gate": "stage benchmark plus full validation",
                "negative_evidence_scope_if_failed": "only this ownership grouping",
                "stop_condition": "resource-balanced validation loses beyond uncertainty",
            },
        ],
    }


def test_route_plan_requires_oracle_free_model() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "route_plan.json"
        payload = _valid_plan()
        path.write_text(json.dumps(payload), encoding="utf-8")
        ok, reason = _validate_route_plan(str(path))
        assert ok, reason

        payload.pop("oracle_free_model")
        path.write_text(json.dumps(payload), encoding="utf-8")
        ok, reason = _validate_route_plan(str(path))
        assert not ok
        assert reason == "oracle_free_model_operator_dependency_graph_required"


def test_route_metadata_requires_structural_fields() -> None:
    errors = _validate_route_metadata(
        {
            "enabled": True,
            "route_type": "architecture_route",
            "invariant": "forward consumes A",
            "expected_impact": "steady-state route",
            "stop_condition": "validation loses after resource rebalance",
            "budget": 8,
            "iteration_role": "prototype",
            "allow_regression": True,
            "milestone": "skeleton",
            "milestone_status": "pending",
        }
    )
    assert errors == []


def test_architecture_route_allow_regression_bypasses_timing_gate() -> None:
    record = {
        "architecture_route": {
            "enabled": True,
            "route_type": "architecture_route",
            "allow_regression": True,
            "iteration_role": "prototype",
            "milestone": "skeleton",
            "milestone_status": "passed",
        },
        "benchmark_result": {
            "correctness": {"passed": True},
            "gpu_memory_gb": 12,
            "peak_vram_mb": 100,
            "kernel": {"stable": False, "spread_pct": 99.0},
            "paired_speedup": {"stable": False, "spread_pct": 99.0},
        },
    }
    keep, reason = _decide_keep(record, parent_id="")
    assert keep
    assert "kept_for_followup" in reason


def test_validation_without_invariant_is_inconclusive() -> None:
    record = {
        "benchmark_rc": 0,
        "architecture_route": {
            "enabled": True,
            "route_type": "architecture_route",
            "iteration_role": "validation",
            "milestone": "validation",
            "invariant_satisfied": False,
        },
        "benchmark_result": {
            "correctness": {"passed": True},
            "kernel": {"stable": True, "median_ms": 10.0},
        },
        "targeted_ncu_rc": None,
        "full_ncu_rc": None,
        "ncu_expected": False,
    }
    previous = {
        "iteration": 1,
        "benchmark_result": {"kernel": {"median_ms": 5.0}},
    }
    outcome, reason = classify_strategy_outcome(record, previous)
    assert outcome == "inconclusive"
    assert "architecture_route_subiteration" in reason


def test_route_state_records_negative_scope() -> None:
    scope = {"routes": {}}
    metadata = {
        "route_type": "architecture_route",
        "route_id": "route_a",
        "invariant": "forward consumes A",
        "expected_impact": "large",
        "budget": 8,
        "stop_condition": "validation loses",
        "iteration_role": "prototype",
        "milestone": "precompute_kernel",
        "milestone_status": "failed",
        "invariant_satisfied": False,
        "old_bottleneck_removed": False,
        "stage_metrics": {"a_ms": 0.4},
        "failure_class": "partial_raw_cache",
        "negative_evidence_scope": "raw_kkt_cache_layout",
        "blocks": ["raw_kkt_global_cache"],
        "does_not_block": ["A_precompute_plus_fused_forward"],
    }
    record = {
        "iteration": 2,
        "strategy": {"outcome": "inconclusive", "reason": "architecture_route_subiteration"},
        "kept": False,
    }
    update_route_state(scope, "route_a", metadata, record)
    entry = scope["routes"]["route_a"]["subiterations"][0]
    assert entry["does_not_block"] == ["A_precompute_plus_fused_forward"]
    assert scope["routes"]["route_a"]["milestones"]["precompute_kernel"]["stage_metrics"]["a_ms"] == 0.4


def test_external_probe_is_inconclusive_evidence_only() -> None:
    record = {
        "iteration_mode": "external_probe",
        "benchmark_rc": 0,
        "benchmark_result": {
            "correctness": {"passed": True},
            "kernel": {"stable": True, "median_ms": 1.0},
        },
    }
    outcome, reason = classify_strategy_outcome(record, previous_record=None)
    assert outcome == "inconclusive"
    assert reason == "external_probe_evidence_only"


def main() -> None:
    for fn in (
        test_route_plan_requires_oracle_free_model,
        test_route_metadata_requires_structural_fields,
        test_architecture_route_allow_regression_bypasses_timing_gate,
        test_validation_without_invariant_is_inconclusive,
        test_route_state_records_negative_scope,
        test_external_probe_is_inconclusive_evidence_only,
    ):
        fn()
    print("route rule checks passed")


if __name__ == "__main__":
    main()
