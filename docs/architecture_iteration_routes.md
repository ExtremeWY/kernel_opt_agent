# Architecture Iteration Routes

Use separate execution routes for local tuning and structural redesign. A
structural route must not depend on an external high-performance implementation
being available.

## Route Types

| route | use when | keep rule |
| --- | --- | --- |
| `local_tune` | tile sizes, cache hints, unrolls, local shared/global placement | correctness plus stable speedup above uncertainty |
| `architecture_discovery` | the current kernel is design-boundary limited and the next route must be chosen from an operator model | state or prototype evidence; no speedup requirement |
| `architecture_route` | ownership, dataflow, multi-kernel staging, hot-state residency, or primitive route changes | milestone based; non-validation prototypes may be kept for follow-up with `--route-allow-regression` |
| `external_probe` | optional comparison with a separate implementation | evidence only; never a requirement for route selection |
| `framework_maintenance` | benchmark, runner, reporting, or strategy-memory changes | tested as framework behavior, not kernel speedup |

## Oracle-Free Route Selection

Route plans must include:

- `operator_dependency_graph`: which intermediates depend on which axes and
  runtime properties.
- `duplicate_work_or_materialization_model`: repeated work removed by the route
  and new bytes/launches introduced by the route.
- `stage_cost_model`: expected stage cost split, including any precompute,
  consumer, and end-to-end total.

External implementations such as FlashQLA, TileLang, Triton, CUTLASS, or a
vendor library can be recorded as `external_probe` evidence when available, but
they are not prerequisites. A route is valid when the operator model shows a
large structural cost and the route invariant removes that cost.

## Milestones

Use route milestones instead of single-step speed gates:

1. `skeleton`: buildable host/kernel execution graph with stable public API.
2. `precompute_kernel`: standalone producer stage is correct.
3. `consumer_kernel`: consumer/fused forward stage is correct with supplied
   intermediates.
4. `end_to_end_correctness`: full route produces correct outputs.
5. `stage_benchmark`: stage timings and added memory/launch costs are recorded.
6. `resource_rebalance`: shared memory, registers, occupancy, and launch shape
   are brought into a plausible range.
7. `validation`: full benchmark decides whether the route is positive or
   negative.

Only a validation milestone with `invariant_satisfied=true` can mark a broad
route negative. Earlier failures should record scoped negative evidence and what
broader routes they do not block.
