# Strategy Memory

Structured strategy memory is stored in `workspace/strategy_memory/global_strategy_memory.json`.

The system records:
- `positive`: faster than the previous comparable attempt
- `negative`: valid but slower or equal
- `rejected`: correctness failure, profiling failure, or incomplete evidence

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
