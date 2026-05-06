# Strategy Memory

Structured strategy memory is stored in `workspace/strategy_memory/global_strategy_memory.json`.

The system records:
- `positive`: faster than the previous comparable attempt
- `negative`: valid but slower or equal
- `rejected`: correctness failure, profiling failure, or incomplete evidence
- `inconclusive`: non-validation architecture-route sub-iteration that failed,
  regressed, or produced noisy evidence but should not block the route
- `routes`: active or completed architecture-route histories with invariants,
  budgets, stop conditions, and sub-iteration evidence
- `design_boundary`: per-kernel marker that blocks ordinary local experiments
  until an architecture route or an explicit local override is used

Each strategy is identified by:
- normalized `strategy_tags`
- a stable `strategy_fingerprint`

Each new iteration should:
1. avoid blocked fingerprints from `rejected`
2. prefer fingerprints in `positive`
3. record the current outcome back into the memory store

## Negative-Neighborhood Rule

Do not treat strategy memory as exact-match only. A failed strategy also blocks
nearby variants unless there is new evidence that the bottleneck moved.

Nearby variants include:
- the same hot loop with a different unroll factor, vector width, or load type
- the same shared-memory layout family with a different pitch or tile order
- the same tile-size sweep around a previously rejected tile shape
- the same resource hint family, such as cache preference, carveout, or launch bounds
- a boundary-only specialization of a path whose steady-state version already failed
- a specialization or dispatch branch that only matches fixed benchmark sizes
  instead of a real runtime invariant

To override this rule, the proposal must record:
- the previous negative or rejected fingerprint being revisited
- the new benchmark or NCU evidence that contradicts the old result
- why the expected end-to-end gain now exceeds the keep threshold

## Negative Evidence Scope

A negative experiment should block the tested implementation neighborhood, not
every strategy that shares a broad goal. Record what was still true in the
failed version:

- Did the dominant materialized intermediate or handoff still exist?
- Did the change duplicate expensive work instead of reusing existing results?
- Did it graft a new primitive onto the old ownership model?
- Did it increase register/shared-memory pressure enough to hide the intended
  benefit?

If the answer is yes, the negative result should be scoped narrowly. It blocks
nearby grafts and local variants, but it does not block a full design-boundary
rewrite that removes the old bottleneck and changes ownership/resource balance
together.

Conversely, if an experiment fully removes the intended bottleneck and still
regresses after resource and correctness fixes, then record the broader route
as negative.

## Architecture Route Budget

When NCU/reference comparison, self-profile evidence, source attribution, or a
first-principles performance model shows a design-boundary bottleneck, strategy
memory should track a route, not only isolated edits.

Mark the state before more experiments:

```bash
.venv/bin/python tools/run_loop.py \
  --hypothesis "mark design-boundary limited state" \
  --mark-design-boundary \
  --design-boundary-reason "dominant cost is a dataflow boundary" \
  --state-only
```

While `design_boundary.active` is true, normal local experiments are rejected by
default. A non-route experiment must pass `--allow-local-after-boundary` and
must explain why it is still justified.

For each architecture route, record:
- route type: `architecture_discovery` for route selection or
  `architecture_route` for implementation
- route invariant: the old dataflow or intermediate that must disappear
- expected payoff: dynamic-time coverage and best-case speedup range
- milestone: skeleton, precompute kernel, consumer kernel, end-to-end
  correctness, stage benchmark, resource rebalance, or validation
- milestone status: pending, passed, or failed
- prototype-ladder stage: the current stage, the next missing high-upside
  stage, and why local cleanup is premature or allowed
- promotion gate: the evidence required to move from route exploration back to
  local tuning
- allowed sub-iterations: correctness fixes, resource rebalance, tile geometry,
  synchronization repair, and validation
- stop condition: no plausible fix remains, resource limits make the invariant
  untenable, or a full/stable benchmark proves the route is below threshold
- route plan: a route portfolio containing at least two structurally distinct
  candidates before starting a new route under an active design-boundary marker
- negative evidence scope: what implementation family is blocked, what broader
  route remains unblocked, and whether the old bottleneck was actually removed

Do not abandon a high-upside route after the first failure if that failure is a
race, missing synchronization, register imbalance, or incomplete removal of the
old bottleneck. Fix those within the route budget before marking the route
negative.

Non-validation route sub-iterations use `inconclusive` when they fail
correctness, regress, have incomplete profiling, or remain noisy. This means:

- the code candidate may still be reverted to keep the main worktree stable
- the route remains active in `routes`
- the strategy fingerprint is not added to the blocked `rejected` set
- a later repair can continue the same route if budget remains

Only a route `validation` sub-iteration can mark the broader route negative, and
only if the implementation actually satisfies the route invariant.

## Prototype Ladder State

Use `docs/prototype_ladder.md` to avoid spending many iterations near a
low-ceiling prototype. Strategy memory should make the current ladder position
explicit whenever the reference/model gap is still large.

Record these fields in route plans and proposal notes:

- current stage: ownership, locality, residency, primitive, layout, pipeline,
  grid scheduling, or local cleanup
- next missing stage: the first untested stage with plausible multi-percent
  end-to-end payoff
- stage invariant: what structural fact must be true after the route
- dynamic coverage: how much steady-state work the stage can affect
- promotion criteria: which benchmark/profile conditions allow local tuning to
  resume

Negative evidence must be scoped to the satisfied stage. For example:

- a layout-padding failure blocks that padding neighborhood, not all layout
  swizzles
- a direct swizzle failure under a fixed-layout WMMA consumer blocks that
  incompatible loader/layout pairing, not a route that also changes the consumer
  to manual `ldmatrix`/`mma.sync`, manual fragments, or an explicit descriptor
  layout
- a slower first pipeline blocks that stage-count/resource combination, not all
  overlap routes
- a primitive graft that leaves the old hot-state materialization in place does
  not disprove a route whose invariant is to remove the materialization

When several local variants are positive but each is below threshold and the
kernel remains far from its modeled stage ceiling, keep the positives as local
history but mark or preserve `design_boundary.active=true`. Positive local
evidence does not by itself prove the prototype stage is high-ceiling.

## No-Strong-Reference Mode

A fast reference kernel is helpful but optional. Without one, route state must
record the model-based evidence that justifies structural work:

- operator contract: stable invariants versus runtime-variable properties
- stage model: minimum FLOPs, bytes, loop trips, reductions, and synchronizations
- primitive ceiling: expected upper bound for copy, staging, reduction, scalar
  compute, or MMA-like compute
- source attribution: current hot source/SASS regions responsible for the
  structural cost
- dynamic coverage: what fraction of steady-state work the route can affect

Negative evidence in this mode must be scoped even more carefully. A failed
prototype that still carries the old bottleneck does not disprove the route.

## Impact Gate

Before implementation, estimate whether a strategy can affect enough dynamic
work to matter. The estimate does not need to be exact, but it must be explicit.

Reject or defer a strategy when:
- it only affects tail tiles, diagonal tiles, rare masks, or uncommon shapes and
  the covered work is below the benchmark keep threshold
- it depends on fixed benchmark dimensions, grid sizes, or input properties that
  are expected to vary in real use
- it targets memory layout or cache behavior while roofline and NCU show the
  kernel is compute-bound with low memory-system utilization
- it targets an aggregate metric, such as shared-memory conflicts, without
  source/SASS attribution showing that the edited lines account for a large
  fraction of the metric
- it raises occupancy when achieved occupancy is already high and no stall data
  points to occupancy as the limiter
- it tunes around a materialized intermediate when history says the
  materialization itself is the structural bottleneck

Prefer strategies when:
- they affect the steady-state hot path for the primary benchmark shape
- they remain valid for the operator's intended runtime variability rather than
  hard-coding the current benchmark configuration
- they attack the dominant NCU bottleneck rather than a secondary metric
- when using line-attributed NCU evidence, they directly edit the source/SASS
  lines responsible for the dominant attributed counter
- they replace a known structural cost instead of polishing around it
- their best-case impact is comfortably above the keep/revert threshold
