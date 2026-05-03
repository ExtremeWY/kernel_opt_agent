# Prototype Ladder and Expert Iteration Patterns

This document describes a platform-neutral ladder for moving an operator from a
low-ceiling prototype to a high-ceiling implementation. It is intended for
custom hardware platforms where there may be no fast source-level reference
kernel.

Use this document when:

- the current kernel is far below a performance model, profiler evidence, or a
  primitive microbenchmark ceiling
- local edits keep producing sub-threshold gains
- source or IR attribution shows the same structural cost repeated through the
  steady-state hot path
- the current implementation may be using the wrong ownership, storage level,
  synchronization scope, or hardware primitive

## General Prototype Ladder

Before local tuning, classify the current implementation by its highest
satisfied ladder stage. If the next missing stage has plausible multi-percent
end-to-end payoff and affects steady-state work, prioritize that architecture
route before local cleanup.

| Stage | Question | Typical evidence |
|---|---|---|
| 0. Correct contract baseline | Does the kernel implement the full supported API contract? | Correctness, edge cases, no benchmark-only specialization. |
| 1. Parallel ownership | Is work owned by the right execution unit: lane, vector group, workgroup, persistent workgroup, or multi-workgroup group? | Active lanes/groups, duplicate work, reduction scope, grid/task count. |
| 2. Data locality and tiling | Is dominant reuse captured at the right memory or cache level? | Required bytes vs observed bytes, cache/scratchpad hit rate, transaction efficiency, tile reuse model. |
| 3. Hot-state residency | Are repeatedly consumed intermediates kept in the narrowest valid scope, preferably registers or owner-local state? | Local-memory traffic attribution, spill pressure, bank/port conflicts, synchronization or handoff cost. |
| 4. Hardware primitive route | Is the dominant compute or transfer mapped to the intended platform primitive for this shape regime? | Instruction mix, vector/matrix utilization, copy primitive use, primitive microbenchmark ceiling. |
| 5. Layout matched to primitive | Does the staged layout match the primitive access pattern and storage-bank mapping? | Source/IR/instruction attribution, uncoalesced transactions, bank conflicts, alignment counters. |
| 6. Pipeline and overlap | Are load, compute, reduction, and store stages overlapped with explicit lifetime and wait rules? | Wait/sync stalls, stage depth, in-flight groups, resource budget, replay correctness. |
| 7. Grid scheduling and tile order | Does launch order match reuse, occupancy, load balance, and tail distribution? | Waves per execution unit, cache reuse, tile count, persistent vs nonpersistent comparison. |
| 8. Local cleanup | Are remaining costs small arithmetic, address-generation, branch, tail, cache-hint, or launch-shape details? | Source attribution shows a local site dominates and best-case speedup exceeds the keep threshold. |

## Stage Promotion Gates

Use these gates before moving from architecture exploration to local tuning:

1. **Ceiling gate**: compare current runtime to a reference, first-principles
   model, or primitive microbenchmark. If the gap is large and unavoidable work
   is already near minimum, mark the kernel design-boundary limited.
2. **Primitive gate**: if the dominant work is suitable for a known
   high-throughput primitive, at least one route must test that primitive before
   local variants around a lower-ceiling implementation are allowed.
3. **Residency gate**: if source attribution shows a hot intermediate repeatedly
   moving through shared storage, global storage, or queue handoff, at least one
   route must test an owner-resident design before tuning that handoff.
4. **Layout gate**: bank conflicts, uncoalesced accesses, or inefficient staged
   loads must be tied to specific source, IR, or instruction sites before
   padding or layout variants are accepted as the main route.
5. **Pipeline gate**: a pipeline route must state stage count, buffer lifetime,
   wait/commit discipline, synchronization scope, and resource budget.
6. **Shape-regime gate**: a route must state which operator properties are true
   invariants and which are runtime-variable. Do not specialize on benchmark
   dimensions unless they are part of the API or production contract.
7. **Validation gate**: only a correctness-passing, resource-balanced prototype
   that satisfies the route invariant can validate or reject the route.

## Negative Evidence Scope

Nearby failures often do not disprove the larger route:

- a failed padding variant does not disprove all layout routes
- a slower first pipeline does not disprove pipelining if stage count, lifetime,
  or resource budget was wrong
- a failed abstraction variant does not disprove the lower-level primitive
- a route that still materializes the old hot intermediate does not disprove a
  route whose invariant is to remove that intermediate
- a scheduling variant that loses for small tile counts does not block it for
  large tile-count regimes

Record what was actually removed, what old cost remained, and what resource
limit appeared. Store failed partial implementations as narrow negative
evidence or non-validation route evidence, not as broad route rejection.

## Route Planning Checklist

Every architecture route plan should include:

- current ladder stage and the next missing high-upside stage
- route invariant tied to that stage
- dynamic-work fraction affected by the route
- expected speedup range and why it is above the keep threshold
- required ownership, residency, layout, primitive, pipeline, or scheduling
  change
- resource budget: registers or temporaries, on-chip storage, resident work, and
  synchronization
- correctness risks and repair budget
- stop condition for the route
- evidence required before local micro-tuning resumes
