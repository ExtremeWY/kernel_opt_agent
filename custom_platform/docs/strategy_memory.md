# Strategy Memory

Strategy memory records optimization evidence across iterations so the agent
does not keep retrying the same low-value or already-failed neighborhood.

Outcome buckets:

- `positive`: correctness passed and the comparable stable benchmark improved.
- `negative`: correctness passed but performance was equal or worse.
- `rejected`: correctness, benchmark, profile, or rule validation failed.
- `inconclusive`: non-validation architecture-route sub-iteration that failed,
  regressed, or had noisy evidence but should not block the route.
- `routes`: active or completed architecture-route histories with invariants,
  budgets, stop conditions, and sub-iteration evidence.
- `design_boundary`: per-kernel marker that blocks ordinary local experiments
  until an architecture route or explicit local override is used.

Each strategy is identified by normalized `strategy_tags` from the proposal and
a `strategy_fingerprint` scoped by platform and kernel type.

## Negative-Neighborhood Rule

Do not treat strategy memory as exact-match only. A failed strategy also blocks
nearby variants unless new benchmark/profile evidence shows that the bottleneck
moved.

Nearby variants include:

- the same hot loop with a different unroll factor, vector width, or load type
- the same staged-layout family with a different pitch, padding, or tile order
- the same tile-size sweep around a rejected shape
- the same resource hint family, such as cache, residency, or launch shape
- a boundary-only specialization of a path whose steady-state version failed
- a specialization or dispatch branch that only matches benchmark sizes instead
  of a real API or production invariant

To override this rule, the proposal must record the old negative fingerprint,
the new contradictory evidence, and why the theoretical end-to-end gain now
exceeds the keep threshold.

## Negative Evidence Scope

A negative result blocks the tested implementation neighborhood, not every route
that shares a broad goal. Record what remained true in the failed version:

- Did the dominant materialized intermediate or handoff still exist?
- Did the change duplicate expensive work instead of reusing existing results?
- Did it graft a new primitive onto the old ownership model?
- Did it increase register, temporary-storage, or on-chip-memory pressure enough
  to hide the intended benefit?

If the answer is yes, scope the negative result narrowly. It blocks nearby
grafts and local variants, but it does not block a full design-boundary route
that removes the old bottleneck and changes ownership or resource balance.

Only mark a broader route negative after a correctness-passing, resource-balanced
implementation actually satisfies the route invariant and still fails the
validation gate.

## Design Boundary

When profile comparison, self-profile trends, source attribution, or a
first-principles performance model shows a structural bottleneck, mark the
kernel design-boundary limited before running more local experiments:

```bash
.venv/bin/python tools/run_loop.py \
  --hypothesis "mark design-boundary limited state" \
  --mark-design-boundary \
  --design-boundary-reason "dominant cost is a repeated dataflow handoff" \
  --state-only
```

While `design_boundary.active` is true, normal local experiments are rejected by
default. Use `--architecture-route` with route metadata, or pass
`--allow-local-after-boundary` only when the proposal explains why a local
experiment is still justified.

## Architecture Route Budget

For each architecture route, record:

- route invariant: the old dataflow, ownership, residency, primitive, layout, or
  pipeline boundary that must change
- expected payoff: dynamic-time coverage and best-case speedup range
- prototype-ladder state: current stage, next missing high-upside stage, and why
  local cleanup is premature or allowed
- promotion gate: evidence required to move from route exploration back to local
  tuning
- allowed sub-iterations: correctness repair, resource rebalance, tile geometry,
  synchronization repair, and validation
- stop condition: no plausible fix remains, resource limits make the invariant
  untenable, or stable validation proves the route below threshold
- route plan: a portfolio containing at least two structurally distinct
  candidates before starting a new route under design-boundary mode

Do not abandon a high-upside route after the first failure if that failure is a
race, missing synchronization, resource imbalance, or incomplete removal of the
old bottleneck. Fix those within the route budget.

Non-validation route failures use `inconclusive`. The code candidate may still
be reverted or not kept, but the broader route remains active until validation
or budget exhaustion.

## Prototype Ladder State

Use `docs/prototype_ladder.md` whenever the current kernel is far from its
reference or modeled ceiling. Record these fields in route plans and proposal
notes:

- current stage
- next missing high-upside stage
- stage invariant
- dynamic coverage
- promotion criteria
- negative evidence scope if the route fails

Positive local evidence does not by itself prove a low-ceiling prototype is the
right final route. If the design-boundary gap remains large, keep route mode
active until a higher-ceiling stage is validated or rejected.

## No-Strong-Reference Mode

A fast reference kernel is helpful but optional. Without one, route state must
record model-based evidence:

- operator contract: stable invariants versus runtime-variable properties
- stage model: minimum operations, bytes, loop trips, reductions, and
  synchronization
- primitive ceiling: expected upper bound for copy, staging, reduction,
  vector/matrix compute, or scalar compute
- source attribution: current hot source, IR, or instruction regions
- dynamic coverage: fraction of steady-state work affected by the route

Lack of a strong reference is not a reason to keep micro-tuning a structural
bottleneck.
