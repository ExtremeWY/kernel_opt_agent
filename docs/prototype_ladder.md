# Prototype Ladder and Human Kernel Iteration Patterns

This note distills reusable optimization process rules from the local
`manual_cuda_kernel` project. The goal is not to copy a specific kernel, but to
borrow the way an expert kernel engineer moves from a low-ceiling prototype to a
high-ceiling implementation before spending iterations on local tuning.

Use this document when:

- a kernel is far below a performance model, reference profile, or primitive
  ceiling
- local edits keep producing sub-threshold gains
- the current implementation may be using the wrong ownership, storage, or
  hardware primitive
- there is no strong reference implementation, but source attribution shows a
  structural cost repeated through the steady-state hot path

---

## Observed Manual Iteration Patterns

The manual project repeatedly follows the same structure across different
operators.

| Case study | Version progression | Transferable lesson |
|---|---|---|
| SIMT matmul | naive dot -> shared-memory tiling -> thread coarsening -> register tiling -> warp tiling -> vectorized loads / bounds cleanup -> shared layout fix | Establish the hierarchy first: CTA tile, warp tile, thread/register tile. Bounds cleanup and vectorization are late-stage work, not the first route. |
| Tensor-core matmul | block/warp tile -> async copy -> padding -> swizzled shared layout -> better matrix-load primitive / address arithmetic -> multistage pipeline -> tile-order swizzle | Choose the intended compute primitive and data layout as a route. Shared-memory padding is only one member of a broader layout route; a failed padding variant does not disprove swizzling. A swizzled operand layout must be paired with a load primitive or descriptor that understands it; standard WMMA loaders generally require padded normal row/column-major layouts. |
| Reduction | one thread per row -> parallel tree -> thread coarsening -> warp-level reduction -> abstraction comparison -> vectorized loads | First fix algorithmic parallelism and ownership. Then move communication to the narrowest valid scope. Abstractions such as cooperative groups must still be measured. |
| Softmax | naive, online, split-row variants across very different shapes | Shape regimes decide the architecture route. A route that wins for small batch / huge row may lose for large row batches. Do not generalize from one benchmark shape. |
| Attention | high-ceiling first version with tensor-core mainloop, register-resident hot intermediates, warp-owned rows -> shared layout swizzle -> pipeline -> wider matrix-load primitive -> schedule refinement | When the operator is known to be dense-dot dominated, the first serious prototype should already target the high-ceiling primitive and register ownership. Do not spend many iterations on a scalar or materialized-intermediate design. |
| Newer-architecture matmul | older primitive baseline -> new architecture primitive -> transfer granularity fix -> pipeline -> producer/consumer split -> multi-CTA compute -> persistent schedule -> small cache/detail polish | Architecture-specific primitives change the route boundary. Small detail polish appears only after the primitive, pipeline, and scheduling route is already near the ceiling. |
| Row-scaled matmul | non-pipelined variant unexpectedly faster for one path | Pipelining is a hypothesis with resource costs, not a universal upgrade. Validate overlap, register pressure, shared-memory footprint, and power/throughput behavior. |

---

## General Prototype Ladder

Before local tuning, classify the current implementation by its highest
satisfied ladder stage. If the next stage has a plausible multi-percent
end-to-end payoff and has not been tested, prioritize that route over local
micro-tuning.

| Stage | Question | Typical evidence |
|---|---|---|
| 0. Correct reference-shaped baseline | Does the custom kernel implement the contract for all supported runtime-variable properties? | Correctness, edge cases, no benchmark-only specialization. |
| 1. Parallel ownership | Is work owned by the right unit: lane, warp, CTA, persistent CTA, or multi-CTA group? | Source review, active lanes, grid size, reduction scope, duplicate work. |
| 2. Data locality and tiling | Is the dominant reuse captured at the right memory level? | Required bytes vs observed bytes, cache hit rate, sectors/request, tile reuse model. |
| 3. Hot-state residency | Are repeatedly consumed intermediates kept in the narrowest valid scope, preferably registers or warp scope? | Shared/global traffic attribution, LSU instructions, bank conflicts, barriers, spills. |
| 4. Hardware primitive route | Is the dominant compute or transfer mapped to the intended primitive for the shape regime? | Tensor/scalar instruction mix, vector load width, async-copy/TMA usage, warp intrinsics. |
| 5. Layout matched to primitive | Does the staged layout match the primitive's access and bank mapping? | Source-line NCU attribution, shared-memory wavefronts/conflicts, alignment counters. |
| 6. Pipeline and overlap | Are load/compute/store stages overlapped with explicit lifetime and wait rules? | Wait/barrier stalls, stage depth, in-flight groups, resource budget, correctness under replay. |
| 7. Grid scheduling and tile order | Does launch order match reuse, occupancy, and tail distribution? | Waves/SM, L2 reuse, tile count, persistent-vs-nonpersistent comparison. |
| 8. Local cleanup | Are remaining costs small arithmetic, address-generation, branch, tail, cache-hint, or launch-bound details? | Source attribution shows local site dominates and best-case speedup exceeds keep threshold. |

Rules:

- Do not spend normal iterations on Stage 8 while a high-upside lower-numbered
  missing stage remains untested.
- Do not classify a route as failed if the tested version did not actually
  satisfy its stage invariant.
- Once a stage is validated and the kernel is near its modeled ceiling, local
  tuning becomes appropriate again.

---

## Stage Promotion Gates

Use these gates before moving from architectural exploration to local tuning.

1. **Ceiling gate**: compare current runtime to a reference, first-principles
   model, or primitive microbenchmark. If the gap is large and unavoidable bytes
   are already near minimum, mark design-boundary limited.
2. **Primitive gate**: if the dominant work is suitable for a known high-throughput
   primitive, at least one route must test that primitive before local variants
   around a lower-ceiling implementation are allowed.
3. **Residency gate**: if source attribution shows a hot intermediate repeatedly
   moving through shared or global memory, at least one route must test a
   register/warp/owner-resident design before tuning that handoff.
4. **Layout gate**: bank conflicts or uncoalesced staged loads must be tied to
   the specific source/SASS lines and primitive layout. Generic padding, pitch,
   or tile-order sweeps are not enough evidence. The route must also state the
   consumer primitive's layout contract: standard WMMA row/column-major,
   manually addressed `ldmatrix`/`mma.sync`, manually constructed fragments, TMA
   descriptor layout, or another explicit mechanism. Do not swizzle storage
   independently of the load primitive.
5. **Pipeline gate**: a pipeline route must state stage count, buffer lifetime,
   wait/commit discipline, synchronization scope, and resource budget.
6. **Shape-regime gate**: a route must state which operator properties are true
   invariants and which are runtime-variable. Do not specialize on benchmark
   dimensions unless they are part of the contract.
7. **Validation gate**: only a correctness-passing, resource-balanced prototype
   that satisfies the route invariant can validate or reject the route.

---

## Negative Evidence Scope

Manual version histories show that nearby failures often do not disprove the
larger route:

- A failed padding variant does not disprove a swizzled layout route.
- A failed direct swizzle under a fixed-layout WMMA loader does not disprove a
  primitive-matched swizzle route using manual `ldmatrix`/`mma.sync` or manual
  fragment construction.
- A slower first pipeline does not disprove pipelining if the stage count,
  buffer lifetime, or register/shared-memory budget was wrong.
- A failed abstraction variant does not disprove the lower-level primitive.
- A route that still materializes the old hot intermediate does not disprove a
  route whose invariant is to remove that intermediate.
- A tile-order or persistent scheduling variant that loses on small tile counts
  does not block it for large tile-count regimes.

Record what was actually removed, what old cost remained, and what resource
limit appeared. Store failed partial implementations as narrow negative
evidence or non-validation route evidence, not as a broad route rejection.

---

## Experiment Planning Checklist

Every architecture route plan should include:

- current ladder stage and the next missing stage
- route invariant tied to that stage
- dynamic-work fraction affected by the route
- expected speedup range and why it is above the keep threshold
- required primitive or ownership change
- resource budget: registers, shared memory, resident CTAs/warps, and barriers
- correctness risks and repair budget
- stop condition for the route
- evidence needed before local micro-tuning resumes

This checklist is intentionally stricter than a normal one-change experiment.
It prevents the optimizer from staying near a low-ceiling prototype simply
because that prototype is easier to edit.
