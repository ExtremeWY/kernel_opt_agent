# Synchronization Optimization

This note collects synchronization-focused optimization guidance that complements
the generic compute and memory notes in `docs/`.

## Why It Matters

Synchronization overhead rarely appears as the first thing people optimize, but
it often becomes the hidden reason a kernel plateaus:

- `__syncthreads()` can serialize an otherwise healthy tile pipeline.
- poorly placed fences can block issue even when math and memory are ready.
- producer / consumer designs can lose overlap if the barrier protocol is too
  conservative.

When `wait`, `barrier`, or `membar` stalls dominate, the first question is not
"how do I make the barrier faster?" but "do I need this synchronization point at
this scope at all?"

## Main Levers

- Reduce the number of block-wide synchronization points.
- Replace block-wide sync with warp-scoped coordination when the dependency is
  warp-local.
- Replace shared-memory exchange plus sync with register exchange where possible.
- Use explicit producer / consumer staging for async copy pipelines.
- Validate every synchronization change with both correctness and NCU metrics.

## Block vs Warp Scope

### `__syncthreads()`

Use block-wide synchronization only when data produced by one warp is consumed
by another warp in the same block, or when shared-memory reuse truly spans the
full block.

Common anti-patterns:

- adding `__syncthreads()` after every shared-memory write even when each warp
  only reads its own lane group
- leaving barriers inside tight loops after the algorithm has been refactored
- synchronizing "for safety" without proving a cross-warp dependency

### `__syncwarp()`

If communication stays inside a warp, prefer `__syncwarp()` or warp intrinsics.
This reduces the synchronization domain and usually removes idle time from
unrelated warps in the same block.

Important caveat:

- On Volta and newer GPUs, independent thread scheduling means warp-synchronous
  programming should not rely on implicit lockstep alone. Use correct masks and
  explicit warp-level synchronization where required.

## Replace Shared-Memory Exchange with Warp Intrinsics

`__shfl_sync`, `__shfl_down_sync`, `__shfl_xor_sync`, and ballot-style intrinsics
can replace the classic pattern:

1. write partial result to shared memory
2. call `__syncthreads()`
3. read neighbor data from shared memory

Use warp intrinsics for:

- warp reduction
- warp scan / prefix sums
- warp broadcast
- warp-local compaction and voting

Benefits:

- lower latency than shared memory
- no bank conflicts
- fewer synchronization points

## Producer / Consumer Pipelines

Synchronization becomes subtle once the kernel overlaps data movement and
compute.

Typical cases:

- `cp.async` multistage GEMM pipelines
- Hopper / Blackwell TMA pipelines
- warp-specialized kernels with dedicated load and compute warp groups

Guidance:

- make stage ownership explicit: which warp group loads, which computes, which
  waits
- use the narrowest valid synchronization scope
- avoid mixing "old" full-block barriers with async pipeline semantics unless
  there is a clear reason
- record the pipeline depth and barrier scheme in experiment notes

If the pipeline is correct but performance is unstable, inspect:

- `wait` stalls
- `barrier` stalls
- registers per thread
- achieved occupancy

Often the real issue is not the barrier instruction itself but a pipeline that
is too deep for the register / shared-memory budget.

## `cuda::barrier` and `cuda::pipeline`

For modern CUDA C++ kernels, prefer explicit stage semantics when async copy is
part of the algorithm.

Use these abstractions when:

- multiple pipeline stages are active at once
- producer and consumer work must be coordinated precisely
- occasional data corruption appears after "optimizing away" old barriers

They are especially useful when moving from "works most of the time" to
"provably correct under replay, different launch shapes, and compiler changes."

## Cooperative Groups

Cooperative Groups are useful when synchronization scope is neither full block
nor single warp.

Typical use cases:

- tiled reductions
- subgroup-local collaboration inside a larger block
- replacing a block-wide sync with a smaller coordination domain

This is often a cleaner step than hand-rolling masks and lane ownership logic.

## Fences and Memory Ordering

`membar` or `__threadfence*()` stalls are often a sign that ordering guarantees
are stronger than the kernel actually needs.

Questions to ask:

- Is cross-block visibility required, or only block-local ordering?
- Can acquire / release semantics replace a heavier full fence?
- Can the fence move out of an inner loop?
- Can the algorithm avoid write-after-read aliasing that forced the fence?

Prefer the weakest ordering that preserves correctness.

## CUDA Graphs and Launch-Side Synchronization

For chains of small kernels, synchronization overhead may be outside the kernel.
CUDA Graphs can help when:

- the kernel DAG is stable
- launch overhead is significant relative to kernel runtime
- a small-kernel pipeline repeatedly executes with the same structure

This is not a replacement for kernel-level synchronization work, but it is a
useful second check when the device-side stalls look clean and end-to-end
latency is still poor.

## NCU Validation Checklist

Pair synchronization changes with:

- `wait`, `barrier`, and `membar` stall movement
- `Eligible Warps Per Cycle`
- achieved occupancy
- kernel median latency
- correctness under repeated runs

Common failure modes:

- fewer barriers but broken visibility semantics
- lower barrier stall but higher register pressure
- cleaner micro metrics but worse end-to-end latency due to lost overlap

## Cross References

- `wait` / `barrier` / `membar` symptoms: `docs/stall_reasons.md`
- shared-memory and async copy context: `docs/memory_optimization.md`
- warp specialization and register pressure tradeoffs: `docs/compute_optimization.md`
- Triton staging and `num_stages`: `docs/triton_optimization.md`
