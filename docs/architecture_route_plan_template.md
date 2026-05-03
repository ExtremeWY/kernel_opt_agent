# Architecture Route Plan Template

Use this template when a kernel is marked design-boundary limited. Save a copy
under the run directory, for example:

`workspace/runs/run_xxx/architecture_route_plan.md`

The plan must contain at least two structurally distinct route candidates before
starting a new route with `tools/run_loop.py --architecture-route`.

If using JSON instead of Markdown, use a top-level object with
`prototype_ladder` and `routes` keys. A bare JSON array of routes is invalid
because it cannot record the shared ladder state and promotion gate.

## Performance Model

- operator contract:
- runtime-variable properties:
- minimum FLOPs:
- minimum bytes:
- main-loop trip count:
- required synchronization lower bound:
- primitive ceiling:
- current self-profile bottleneck:
- design-boundary reason:

## Prototype Ladder

- current_stage:
- next_missing_high_upside_stage:
- why_local_micro_tuning_is_premature:
- evidence_that_stage_affects_steady_state:
- evidence_needed_before_local_tuning_resumes:

## Route 1

- route_id:
- invariant:
- ladder_stage_targeted:
- structural cost removed:
- dynamic coverage:
- expected_impact:
- required_ownership_or_primitive_change:
- resource_budget:
- promotion_gate:
- negative_evidence_scope_if_failed:
- key risks:
- prototype scope:
- stop_condition:

## Route 2

- route_id:
- invariant:
- ladder_stage_targeted:
- structural cost removed:
- dynamic coverage:
- expected_impact:
- required_ownership_or_primitive_change:
- resource_budget:
- promotion_gate:
- negative_evidence_scope_if_failed:
- key risks:
- prototype scope:
- stop_condition:

## Route Ranking

- preferred first route:
- reason:
- routes rejected before implementation:
- why_rejections_are_not_benchmark_shape_overfitting:
- evidence needed before local micro-tuning resumes:
