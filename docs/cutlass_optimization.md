# CUTLASS Optimization

This note collects CUTLASS-specific optimization guidance that complements the
generic CUDA notes in `docs/`.

## Main Levers

- Tile shape: threadblock, warp, and MMA tile sizes must match the target
  architecture and problem shape.
- Stage count: multistage pipelines help on memory-bound kernels, but too many
  stages can reduce occupancy.
- Epilogue design: fuse bias, activation, normalization, or permutation work
  when global-memory traffic is the bottleneck.
- Scheduling: evaluate split-K, Stream-K, swizzle, and grouped GEMM depending
  on problem shape.
- Architecture features: use collective builders, EVT, TMA, and warp
  specialization where the target GPU supports them.

## Problem Families

CUTLASS optimization choices should start from problem family, not from the most
advanced feature in the template stack.

Typical families:

- regular dense GEMM
- tall-skinny or wide-shallow GEMM
- grouped GEMM
- batched / pointer-array GEMM
- implicit GEMM convolution
- fused GEMM epilogues

Different families want different schedules even on the same GPU.

## Tile Shape and Stage Count

The most important CUTLASS kernel knobs are:

- threadblock tile shape
- warp tile shape
- MMA atom shape
- stage count

Tradeoffs:

- larger tiles raise reuse and arithmetic intensity
- larger tiles also increase shared-memory and register pressure
- deeper multistage pipelines hide memory latency, but can reduce occupancy

When performance regresses after a template change, re-check tile shape and
stages before chasing lower-level issues.

## Scheduling Choices

### Split-K

Use split-K when:

- `K` is large
- `M` and `N` are too small to expose enough CTA-level parallelism

Cost:

- additional reduction or accumulation overhead

### Stream-K

Use Stream-K when:

- work distribution across CTAs is uneven
- standard tile decomposition leaves some SMs underutilized
- irregular shapes make classic scheduling inefficient

### Grouped GEMM

Use grouped scheduling when:

- a batch contains many GEMMs with different shapes
- launch overhead matters
- load balancing across heterogeneous problems matters

## Epilogue Fusion

CUTLASS is often most compelling when epilogue work can be fused into the GEMM.

Common fused patterns:

- bias add
- activation
- residual add
- normalization fragments
- permutation or layout conversion
- softmax-style postprocessing in specialized paths

Benefits:

- less global-memory traffic
- fewer launches
- better reuse of accumulator values

Risks:

- register pressure spikes
- epilogue complexity hides the real bottleneck
- validation becomes harder if too much logic is fused at once

## EVT and Collective Builders

For CUTLASS 3.x and newer APIs:

- use CollectiveBuilder when the default mainloop / epilogue stack is close to
  your target
- use EVT when epilogue fusion complexity exceeds a simple linear-combination
  operator

These abstractions are powerful, but they are not "free". Record the chosen
builder, schedule, and epilogue tree so future iterations can reproduce the
working combination.

## Architecture-Specific Features

### Ampere

- `cp.async`
- multistage pipelines
- TF32 and structured sparsity options

### Hopper

- TMA
- warp specialization
- WGMMA
- grouped and persistent scheduling improvements

### Blackwell

- newer tensor-core paths
- block-scaled / FP4 / mixed-input variants
- more aggressive cluster and pipeline orchestration

Keep architecture-specific advice separate from generic CUTLASS advice in
experiment notes.

## Convolution and Implicit GEMM

For convolution-backed CUTLASS kernels, key dimensions are not just GEMM tile
shape but also:

- iterator design
- data layout
- whether the problem is fprop, dgrad, or wgrad
- whether split-K or grouped scheduling is involved

Implicit GEMM performance problems are often iterator or layout problems before
they are MMA problems.

## Diagnostic Flow

1. Confirm tensor-core path and instruction mix with NCU.
2. Check occupancy, registers, and shared-memory pressure after every tile or
   stage change.
3. Compare schedule variants before deep kernel rewrites.
4. Inspect whether epilogue fusion saved memory traffic or only inflated
   registers.
5. Keep CUTLASS-specific anti-patterns in `CUDA_OPTIMIZATION.md` under a
   dedicated section.

## What To Record Per Iteration

For CUTLASS kernels, experiment notes should capture:

- architecture target
- threadblock / warp / MMA tile shape
- stage count
- schedule type: standard, split-K, Stream-K, grouped, persistent
- epilogue type
- builder or EVT usage

## Practical Integration Points

- Treat CUTLASS kernels as a distinct backend in experiment notes, even if they
  compile through `nvcc`.
- Reuse successful patterns through structured strategy tags instead of prose
  only.
- Cross-reference generic resource issues with `docs/compute_optimization.md`,
  `docs/memory_optimization.md`, and `docs/sync_optimization.md`.
