# Architecture Route Plan

## Prototype Ladder

- current_stage: fill_me_current_ladder_stage
- next_missing_high_upside_stage: fill_me_next_stage
- evidence_needed_before_local_tuning_resumes: fill_me_promotion_evidence

## Oracle-Free Model

- operator_dependency_graph: fill_me_dependencies_by_axis_and_stage
- duplicate_work_or_materialization_model: fill_me_repeated_work_and_added_bytes_model
- stage_cost_model: fill_me_stage_runtime_or_work_budget

## Route 1

- route_id: fill_me_route_a
- invariant: fill_me_structural_invariant
- ladder_stage_targeted: fill_me_ladder_stage
- structural cost removed: fill_me_old_boundary_or_repeated_work
- dynamic coverage: fill_me_steady_state_fraction
- expected_impact: fill_me_best_case_speedup_range
- promotion_gate: fill_me_stage_and_full_benchmark_gate
- negative_evidence_scope_if_failed: fill_me_narrow_scope_and_what_remains_unblocked
- stop_condition: fill_me_finite_stop_condition

## Route 2

- route_id: fill_me_route_b
- invariant: fill_me_second_structural_invariant
- ladder_stage_targeted: fill_me_ladder_stage
- structural cost removed: fill_me_old_boundary_or_repeated_work
- dynamic coverage: fill_me_steady_state_fraction
- expected_impact: fill_me_best_case_speedup_range
- promotion_gate: fill_me_stage_and_full_benchmark_gate
- negative_evidence_scope_if_failed: fill_me_narrow_scope_and_what_remains_unblocked
- stop_condition: fill_me_finite_stop_condition
