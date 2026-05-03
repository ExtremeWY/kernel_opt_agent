# CUDA Kernel Optimization Guide

This document is **maintained by the optimization agent**. It serves two roles:

1. a **systematic index** that maps bottlenecks, kernel styles, and hardware
   features to the right reference material
2. a **living record** of optimization patterns that worked or failed on kernels
   in this repository

Use this file as the front door. Use `docs/` for the detailed reference manual.

---

## How To Use This File

When starting a new optimization iteration:

1. identify the macro bottleneck with `tools/bench.py`
2. identify the micro bottleneck with `tools/ncu_profile.py`
3. use the indexes below to jump to the right `docs/` references
4. check the matching strategy tags in the kernel-specific and cross-kernel
   sections
5. only then propose a single focused change

This file should stay compact and navigable. Long-form explanations belong in
`docs/`.

---

## Reference Index

### By Bottleneck

| Bottleneck or symptom | Primary docs | Typical strategy tags |
|---|---|---|
| compute-bound GEMM / tensor-core issues | `docs/compute_optimization.md`, `docs/triton_optimization.md`, `docs/cutlass_optimization.md` | `[tensor-core]`, `[mma-shape]`, `[small-m]`, `[compare-cuda-vs-tc]`, `[tile-size]`, `[data-type]` |
| memory-bound streaming kernel | `docs/memory_optimization.md`, `docs/stall_reasons.md` | `[memory-coalescing]`, `[vectorized-loads]`, `[cache]` |
| low occupancy | `docs/compute_optimization.md`, `docs/triton_optimization.md` | `[occupancy]`, `[register-pressure]`, `[launch-config]` |
| high register count / spills | `docs/compute_optimization.md`, `docs/triton_optimization.md` | `[register-pressure]`, `[tile-size]` |
| warp-count / CTA-geometry mismatch | `docs/compute_optimization.md`, `docs/sync_optimization.md` | `[warp-count]`, `[cta-geometry]`, `[launch-config]`, `[sync]` |
| barrier / wait / fence stalls | `docs/sync_optimization.md`, `docs/stall_reasons.md` | `[sync]`, `[warp-divergence]`, `[algorithmic]` |
| poor L1 / L2 reuse | `docs/memory_optimization.md`, `docs/arch_notes.md` | `[cache]`, `[memory-access]`, `[tile-order]` |
| branch-heavy hot loop | `docs/compute_optimization.md`, `docs/sync_optimization.md` | `[warp-divergence]`, `[algorithmic]` |
| architecture-specific async copy or pipeline issue | `docs/arch_notes.md`, `docs/memory_optimization.md`, `docs/sync_optimization.md` | `[pipeline]`, `[memory-access]`, `[launch-config]` |
| low-ceiling prototype / repeated local polish | `docs/prototype_ladder.md`, `docs/strategy_memory.md` | `[prototype-ladder]`, `[design-boundary]`, `[architecture-route]`, `[register-dataflow]` |

### By Kernel Style

| Kernel style | Primary docs | Typical first checks |
|---|---|---|
| Triton elementwise / reduction | `docs/triton_optimization.md`, `docs/memory_optimization.md` | coalescing, tile width, occupancy |
| Triton GEMM / attention | `docs/triton_optimization.md`, `docs/compute_optimization.md`, `docs/sync_optimization.md` | tensor-core path, tile size, registers, stages |
| CUTLASS GEMM / conv | `docs/cutlass_optimization.md`, `docs/compute_optimization.md`, `docs/memory_optimization.md` | tile shape, stage count, schedule, epilogue |
| CUDA C custom kernel | `docs/compute_optimization.md`, `docs/memory_optimization.md`, `docs/sync_optimization.md` | launch config, warp-count/CTA geometry, shared memory, warp primitives |

### By Architecture Feature

| Feature | Primary docs | Notes |
|---|---|---|
| tensor cores / MMA | `docs/compute_optimization.md` | includes BF16 / FP16 / TF32 guidance |
| `cp.async` / software pipelining | `docs/memory_optimization.md`, `docs/sync_optimization.md` | validate both overlap and correctness |
| TMA / Hopper+ features | `docs/arch_notes.md`, `docs/cutlass_optimization.md`, `docs/triton_optimization.md` | layout and pipeline semantics matter |
| warp specialization | `docs/compute_optimization.md`, `docs/sync_optimization.md` | often limited by registers and barriers |
| swizzle / tile ordering | `docs/memory_optimization.md`, `docs/triton_optimization.md`, `docs/cutlass_optimization.md` | use only when locality justifies it |

---

## Strategy Tag Index

The repository uses short reusable tags in experiment notes and this guide.

| Tag | Meaning | Go read first |
|---|---|---|
| `[tensor-core]` | tensor-core enablement or utilization for matmul-like, MMA-friendly regimes | `docs/compute_optimization.md` |
| `[mma-shape]` | MMA tile fill, padding overhead, and tensor-core shape suitability | `docs/compute_optimization.md`, `docs/arch_notes.md` |
| `[small-m]` | small-M regimes where MMA tile fill is poor | `docs/compute_optimization.md` |
| `[compare-cuda-vs-tc]` | explicit comparison of CUDA-core and tensor-core-with-padding paths | `docs/compute_optimization.md` |
| `[register-dataflow]` | keeping hot intermediates in registers or warp scope instead of materializing them through shared/global memory | `docs/compute_optimization.md`, `docs/sync_optimization.md` |
| `[register-pressure]` | registers per thread, spills, occupancy limit | `docs/compute_optimization.md` |
| `[occupancy]` | active warps / blocks per SM | `docs/compute_optimization.md` |
| `[warp-count]` | warps/block, producer/consumer warp groups, and per-CTA synchronization domain | `docs/compute_optimization.md`, `docs/sync_optimization.md` |
| `[cta-geometry]` | block geometry, tile ownership, and CTA-level work granularity | `docs/compute_optimization.md`, `docs/memory_optimization.md` |
| `[tile-size]` | tile shape, block geometry, MMA decomposition | `docs/triton_optimization.md`, `docs/cutlass_optimization.md` |
| `[launch-config]` | persistent vs non-persistent, block count, grid mapping | `docs/triton_optimization.md`, `docs/compute_optimization.md` |
| `[memory-coalescing]` | load/store access quality | `docs/memory_optimization.md` |
| `[vectorized-loads]` | wide loads / stores and alignment | `docs/memory_optimization.md` |
| `[cache]` | L1/L2 residency, eviction policy, tile reuse | `docs/memory_optimization.md` |
| `[memory-access]` | broader address-generation and staging issues | `docs/memory_optimization.md` |
| `[warp-divergence]` | branch-heavy or mask-heavy hot paths | `docs/compute_optimization.md` |
| `[algorithmic]` | work reduction, loop simplification, mathematical restructuring | `docs/compute_optimization.md`, `docs/triton_optimization.md` |
| `[sync]` | barrier, wait, fence, pipeline protocol | `docs/sync_optimization.md` |
| `[data-type]` | precision path, accumulator type, conversion cost | `docs/compute_optimization.md` |
| `[pipeline]` | async copy depth, producer / consumer overlap | `docs/memory_optimization.md`, `docs/sync_optimization.md` |
| `[prototype-ladder]` | classifying the current implementation stage before choosing local tuning versus structural route work | `docs/prototype_ladder.md` |
| `[design-boundary]` | evidence that the current dataflow or primitive has a hard ceiling and needs a route change | `docs/prototype_ladder.md`, `docs/strategy_memory.md` |
| `[architecture-route]` | multi-sub-iteration structural redesign governed by route invariant, promotion gate, and finite budget | `docs/prototype_ladder.md`, `docs/architecture_route_plan_template.md` |

---

## Investigation Playbooks

### Compute-Bound Playbook

1. confirm whether the kernel is actually matmul / MMA-like
2. check whether the active shape regime is MMA-friendly or small-M / decode-like
3. only then inspect tensor-core path and tensor-core instruction share
4. inspect tile shape and data type
5. inspect warp-count / CTA geometry against the kernel style table in
   `docs/compute_optimization.md`
6. inspect registers per thread and achieved occupancy
7. only then try deeper pipeline, warp specialization, or epilogue fusion
8. estimate dynamic coverage before selecting an experiment: if the change only
   affects boundary tiles, tail predicates, diagonal tiles, or rare shape cases,
   compute the maximum end-to-end speedup and skip it when the estimate is near
   benchmark noise or below the keep threshold
9. if the dominant gap is structural, such as low tensor-core utilization on an
   MMA-friendly compute-bound kernel, prioritize redesigns that move the main
   FLOPs onto the right pipeline before local layout/load/barrier polishing
10. if reference profiling, self-profile trends, source attribution, or a
    first-principles performance model shows much higher total instructions,
    LSU/shared-memory handoff, synchronization, main-loop trip count, or grid
    work than the minimum required work, treat the current kernel as
    design-boundary limited. Stop local polish and choose a route that removes
    the dominant dataflow boundary.
11. scope negative evidence carefully. A failed partial bypass that still keeps
    the old intermediate or duplicates expensive work does not disprove a full
    register/ownership redesign that removes the old boundary.

Read first:

- `docs/compute_optimization.md`
- `docs/triton_optimization.md` or `docs/cutlass_optimization.md`

### Experiment Triage Gate

Before writing code, every candidate experiment must answer:

1. What fraction of the primary benchmark's dynamic work does this change touch?
2. What is the best-case end-to-end speedup if that touched work became free?
3. Has an adjacent strategy already been negative or rejected in
   `memory/<kernel_type>.md` or `workspace/strategy_memory/global_strategy_memory.json`?
4. Does the change attack the primary NCU bottleneck, or only a secondary metric?
5. Is the expected speedup comfortably above the full-benchmark keep threshold?
6. Does the change remain valid for the operator's intended runtime variability,
   or does it only work because the benchmark uses fixed sizes?

Reject the candidate before implementation if the answer is unfavorable. This
prevents burning iterations on changes that are technically plausible but cannot
move the benchmark enough to survive the keep/revert rule.

Do not treat fixed benchmark dimensions as optimization invariants. Compile-time
specializations, dispatch branches, removed checks, or hard-coded constants are
valid only when the specialized property is part of the operator contract or a
documented production invariant. Runtime tile-state checks are acceptable when
they preserve correctness and performance portability across all supported
problem sizes.

### Architecture-Route Triage

Use route-level triage when the evidence says the implementation design, not a
local instruction site, is the limiter.

Start an architecture route when:
- a reference kernel has similar DRAM traffic but far lower instruction count,
  shared-memory handoff, synchronization, or launched work
- no strong reference exists, but the performance model and current-kernel NCU
  show the same kind of structural excess versus the minimum required work or a
  primitive ceiling
- source attribution points to an intermediate or handoff that appears across
  the steady-state hot path
- several local variants around the same handoff have become sub-threshold or
  negative
- the best-case speedup from removing the boundary is several times larger than
  the normal keep threshold

Rules for route execution:
- write a route invariant before editing, such as removing a materialized
  intermediate, changing producer/consumer ownership, or moving the dominant
  operation to the intended hardware pipeline
- allow a finite number of focused sub-iterations for correctness races,
  synchronization repair, resource rebalance, and tile geometry
- do not abandon the route because the first version is wrong or slower if the
  failure still carries the old bottleneck or is a repairable implementation
  issue
- mark the broader route negative only after a version that actually satisfies
  the route invariant has been validated and still cannot beat the old design

Execution support:
- Use `tools/run_loop.py --architecture-route` for route sub-iterations.
- When a kernel is design-boundary limited, mark it with
  `tools/run_loop.py --mark-design-boundary --state-only`. While this marker is
  active, the runner rejects normal local experiments unless
  `--allow-local-after-boundary` is passed with explicit justification.
- New architecture routes under an active design-boundary marker must provide a
  `--route-plan` containing at least two structurally distinct candidate routes.
- Non-validation route failures are recorded as `inconclusive`, not as blocked
  negative evidence. This preserves route budget for correctness, synchronization,
  and resource-rebalance repairs. The runner exits successfully for these
  non-validation inconclusive sub-iterations so route execution can continue.
- Use `--route-allow-regression` only for correctness-passing prototypes that
  need to remain as the base for the next route sub-iteration. Final validation
  must use normal keep/revert criteria.

Anti-pattern:
- repeatedly applying pitch, padding, cache, launch-bound, or branch-shape
  changes to a design whose dominant intermediate is already known to be the
  bottleneck

### No-Strong-Reference Workflow

When there is no high-performance black-box or source reference, replace
reference-gap reasoning with explicit upper-bound reasoning:

1. Record the operator contract: stable dtype/layout/semantic dimensions versus
   runtime-variable sizes and flags.
2. Compute required FLOPs, required bytes, loop trip counts, reductions, and
   synchronization lower bounds for each shape regime.
3. Estimate primitive ceilings from hardware peak, local microbenchmarks, or
   minimal prototypes: copy bandwidth, shared-memory staging, reduction,
   scalar compute, and MMA throughput as applicable.
4. Profile the current kernel and attribute the largest dynamic costs to source
   regions: total instructions, LSU/shared instructions, barriers/waits, spills,
   and memory sectors/request.
5. Generate multiple architecture-route invariants before editing. Each route
   must name the structural cost it removes and the dynamic-work fraction it
   covers.
6. Run minimal prototypes to falsify route invariants. Do not require the first
   prototype to beat the incumbent unless it is the route validation step.
7. Mark a route negative only after a version that actually satisfies the
   invariant is correct, stable, resource-balanced, and still below threshold.

This prevents the optimizer from treating the first correctness-passing kernel
as the only viable neighborhood.

### Prototype-Ladder Workflow

The local `manual_cuda_kernel` project is a useful case study for how expert
CUDA kernels are usually improved: the first large gains come from moving to the
right prototype stage, and the final gains come from local tuning only after the
stage is near its ceiling.

Use `docs/prototype_ladder.md` before proposing a local edit when the current
kernel is still far from a model or reference:

1. Classify the current implementation stage: ownership, data locality,
   hot-state residency, hardware primitive, layout, pipeline, grid scheduling,
   or local cleanup.
2. Name the next missing high-upside stage and estimate its dynamic coverage.
3. If that stage affects the steady-state hot path and has multi-percent
   end-to-end payoff, open an architecture route instead of tuning nearby
   address arithmetic, padding, cache hints, launch bounds, or predicates.
4. Treat the route as unproven until a correctness-passing, resource-balanced
   implementation actually satisfies the route invariant.
5. Scope negative evidence narrowly: a failed padding variant does not disprove
   a layout route, a bad first pipeline does not disprove all pipelines, and a
   partial graft that keeps the old hot intermediate does not disprove a
   residency/ownership redesign.

Manual case-study lessons that should transfer:

- **Matmul progression**: naive/low-level baseline -> CTA/warp/thread tiling ->
  register tiling -> vectorized memory -> layout/pipeline. Do not start by
  polishing bounds checks while the tiling hierarchy is wrong.
- **Tensor-core progression**: choose the intended primitive, then design staged
  layout around that primitive. Swizzle can be a different route from padding,
  not just a small variant of it.
- **Reduction progression**: fix algorithmic parallelism first, then reduce
  communication scope with warp-level primitives, then vectorize.
- **Attention-style dense-dot progression**: the first serious prototype should
  already target high-ceiling compute primitives and keep hot intermediates in
  register or warp ownership where possible; shared-memory materialization of
  steady-state intermediates is a design-boundary candidate, not just a bank
  conflict to tune around.
- **Pipeline progression**: record in-flight stages, buffer lifetime, wait
  discipline, synchronization scope, and resource budget. Pipelining can lose if
  it increases pressure more than it overlaps useful work.
- **Shape-regime progression**: split work, tile order, and persistent scheduling
  routes must state the shape regime they target. A route that wins for one
  regime is not automatically valid for all runtime-variable sizes.

### Memory-Bound Playbook

1. check coalescing and sectors/request
2. inspect vectorization and alignment
3. inspect L1/L2 hit rate and tile order
4. inspect `long_scoreboard` and staging depth

Read first:

- `docs/memory_optimization.md`
- `docs/stall_reasons.md`
- `docs/triton_optimization.md` for Triton kernels

### Synchronization-Limited Playbook

1. check `wait`, `barrier`, and `membar`
2. reduce synchronization scope if valid
3. replace shared-memory exchange with warp intrinsics where possible
4. inspect producer / consumer protocol before changing tile size

Read first:

- `docs/sync_optimization.md`
- `docs/stall_reasons.md`

---

## Kernel-Specific Notes

The sections below capture repository-specific observations. They are more
actionable than generic advice, but less general than the `docs/` references.

When a new experiment succeeds or fails in a transferable way, update:

- the relevant kernel section below
- `Cross-Kernel Optimization Patterns`
- `CUDA_OPTIMIZATION.md` tag usage if a new stable tag emerges

---

## rms_norm (Per-Row RMSNorm)

### Characteristics
- Bottleneck: memory-bound (arithmetic intensity ~3, well below ridge point)
- Data access: streaming reads (input), streaming writes (output), broadcast read (gamma)
- Per-row reduction (sum of squares) followed by element-wise normalization
- Typical sizes: M=2048-4096 rows, N=1024-5120 columns, bf16/fp16

### Effective Optimizations

1. **Maximize occupancy via grid sizing** (40% latency reduction): `[occupancy]` `[launch-config]`
   - Baseline launched 132 blocks (1/SM) with persistent thread loop → 6.25% occupancy
   - Increasing grid to rows (one block per row) → 65% occupancy
   - Key: check NCU "Block Limit Registers" to know max blocks/SM, then size grid accordingly
   - Expected speedup: 1.4-1.8x

2. **Inline helper functions to reduce register pressure** (registers: 96 → 39): `[register-pressure]` `[occupancy]`
   - The `_do_rms_norm` helper function inflated register count due to call overhead
   - Inlining and removing unnecessary type conversions (`hidden.to(gamma.dtype).to(tl.float32)`) cut registers by 60%
   - Lower registers → more blocks/SM → higher theoretical occupancy (31% → 75%)
   - Expected impact: enables other occupancy optimizations

3. **Row-per-block launch instead of persistent threads** (~5% improvement): `[launch-config]` `[occupancy]`
   - For M ≥ 1024, `grid=(M,)` outperforms persistent kernel with `tl.range` loop
   - Eliminates loop overhead, branch resolution stalls, and software pipelining register cost
   - Hardware wave dispatch is efficient for these grid sizes
   - Expected speedup: 1.03-1.05x over persistent with same occupancy

4. **L2 eviction policy hints** (~1.5% improvement): `[cache]` `[memory-access]`
   - `evict_last` for input loads (keep briefly for coalescing across warps)
   - `evict_first` for output stores (streaming write, never re-read)
   - Expected speedup: 1.01-1.02x

5. **Triton autotune for num_warps** (~1% improvement): `[launch-config]`
   - Optimal num_warps varies by column width (N)
   - num_warps=8 optimal for N=4096 on H800
   - Search space: num_warps=[4,8,16,32], num_stages=[1,2]

### Anti-patterns (things that didn't work)

- **num_warps=16 or 32 for N=4096**: Too many threads per block → fewer blocks/SM → lower occupancy
- **Persistent threads with low register inlined kernel**: Loop overhead + software pipelining register cost negated occupancy gains (32 us vs 29 us)
- **2 rows per block**: Branch overhead from bounds checking outweighed dispatch savings
- **evict_first for both load and store**: Input data benefits from brief L2 residency
- **int32 offsets**: Triton already handles offset optimization internally; explicit int32 can generate worse code
- **Division → multiplication by reciprocal**: Compiler already optimizes this for constexpr divisors

---

## qkv_part_rope (QKV Partial Rotary Position Embedding)

### Characteristics
- Bottleneck: memory-bound (arithmetic intensity ~0.34, deeply below ridge point of ~295)
- Data access: streaming read+write of packed QKV tensor, broadcast read of cos/sin tables
- 77% of data is pure copy (nope + V heads), 23% has fp32 rope computation
- CUDA kernel with persistent thread model
- Typical sizes: batch=2, seq=4096, q_heads=10, kv_heads=1, head_dim=256, nope_dim=192

### Effective Optimizations

1. **Increase SeqTile from 2 to 4** (19% latency reduction): `[tile-size]` `[register-pressure]`
   - Doubles work per tile → halves number of tiles (scheduler iterations)
   - Better amortization of per-tile overhead: scheduler coord computation, cos/sin loads
   - ElemPerThread goes from 1 to 2, improving instruction-level parallelism
   - Registers increase from 31 to 60/thread (still fits 1 block of 1024/SM)
   - Expected speedup: 1.15-1.20x
   - NOTE: SeqTile=8 causes register spilling (regression). SeqTile=4 is the sweet spot.

2. **Float4 (16-byte) nope copy** (~1.3% additional improvement): `[memory-coalescing]` `[vectorized-loads]`
   - Widen nope copy from float2 (8B) to float4 (16B) loads/stores
   - Halves instruction count for the dominant 77%-of-traffic nope copy path
   - Small but real gain when combined with SeqTile=4
   - Standalone (without SeqTile increase): negligible improvement

### Anti-patterns (things that didn't work)

- **Doubling grid size (2 blocks/SM)**: With 31 regs/thread, 2 blocks of 1024 fit, giving 64 warps/SM. But performance WORSENED by 7%. 32 warps/SM already provides sufficient memory latency hiding for this access pattern. More warps increase L2 contention.
- **ld.global.lu (last-use) for nope loads**: Streaming hint evicts data that IS reused by adjacent warps in the same block (processing different heads at the same seq position). Caused 7% regression.
- **__launch_bounds__(_, 2) to force register reduction**: Compiler spills to local memory when constrained to fit 2 blocks/SM. Spilling overhead > occupancy benefit (6% regression).
- **SeqTile=8**: Too much register pressure. Registers exceed comfortable range, causing spilling and 20% regression vs SeqTile=4.

---

## swiglu_input_quant (SwiGLU + FP8 Blockwise Quantization)

### Characteristics
- Bottleneck: memory-bound (arithmetic intensity ~0.17, deeply below ridge point ~295)
- Data access: read BF16 input [M, 2N], write BF16 SwiGLU [M, N], write FP8 [M, 2N], write FP32 scales [2N/128, M]
- Multi-output kernel: 3 output tensors with mixed data types (BF16, FP8, FP32)
- Per-block-of-128-columns row-wise absmax scaling for FP8 quantization
- Scale stored in transposed layout (`block_idx_n * m + row_idx`) for downstream matmul
- Typical sizes: M=4096, N=7168, bf16

### Effective Optimizations

1. **Reduce tile size to cut register pressure** (27% total improvement): `[register-pressure]` `[occupancy]` `[tile-size]`
   - Baseline block_size_m=128 → 126 regs/thread → 6.25% occupancy
   - block_size_m=32 with num_warps=8 → 40 regs/thread → 70.9% occupancy
   - This is the single most impactful optimization: 11x occupancy increase
   - Expected speedup: 1.25-1.35x

2. **Non-persistent grid outperforms persistent for high tile count** (8% improvement): `[launch-config]` `[occupancy]`
   - With 7168 tiles across 132 SMs (~54 tiles/SM), hardware wave dispatch is efficient
   - L1 hit rate improved 0% → 32% from better spatial locality
   - Eliminates loop overhead, branch resolution stalls
   - Expected speedup: 1.05-1.10x over persistent with equivalent occupancy

3. **Increase grid to match SM block capacity** (17% improvement): `[occupancy]` `[launch-config]`
   - For persistent kernels: compute max blocks/SM from register + shared mem limits
   - Baseline: 132 blocks (1/SM). Optimal: 3 blocks/SM for this register profile
   - Expected speedup: 1.15-1.20x

4. **L2 eviction hints** (~0.7% improvement): `[cache]` `[memory-access]`
   - `evict_last` on input loads, `evict_first` on all stores
   - Smaller benefit than rms_norm since this kernel has 5 store streams polluting L2
   - Expected speedup: 1.005-1.01x

### Anti-patterns (things that didn't work)

- **num_warps=8 alone (without block_size_m reduction)**: With block_size_m=64, doubling warps from 4→8 didn't change occupancy (same register budget, same blocks/SM). No improvement.
- **Swizzled tile ordering (GROUP_SIZE_M=8)**: Integer arithmetic overhead for the swizzle mapping exceeded L2 locality benefit. 1% regression.
- **num_warps=16**: Too many threads per block reduced blocks/SM, negating the benefit. ~3% regression.
- **block_size_m=128 with any num_warps**: Always register-limited to ≤4 blocks/SM. Fundamental tile size problem.

---

## persistent_matmul (GEMM — C = A @ B)

### Characteristics
- Bottleneck: compute-bound (arithmetic intensity ~341, well above ridge point ~295)
- Data access: tiled reads of A and B matrices, tiled write of C
- Tensor core dominated: 89.7% HMMA utilization at optimal
- Typical sizes: M=2048-8192, N=2048-11008, K=512-8192, bf16

### Effective Optimizations

1. **Remove .to(tl.float32) before tl.dot** (7.7x speedup): `[tensor-core]` `[data-type]`
   - Baseline cast BF16 loads to FP32 before dot product, forcing scalar FP32 FMA
   - Removing cast enables native BF16 tensor cores with FP32 accumulation
   - Expected speedup: 5-10x (single most impactful change for any GEMM kernel)

2. **Expand autotune to proper GEMM tile sizes** (2x speedup on top of #1): `[tile-size]` `[tensor-core]`
   - Baseline had BLOCK_SIZE_N={16,32} — far too narrow for tensor cores
   - Optimal: BLOCK_SIZE_M=128, BLOCK_SIZE_N=128-256, BLOCK_SIZE_K=64-128
   - Expected: 128x128 or 128x256 tiles with 4-8 warps

3. **Use tl.make_block_ptr for TMA loads** (~7% speedup): `[memory-access]` `[register-pressure]`
   - Enables Hopper TMA (Tensor Memory Accelerator) for async memory operations
   - Reduces register pressure from manual pointer arithmetic
   - Better latency hiding via hardware-managed prefetch
   - Expected speedup: 1.05-1.10x

4. **Non-persistent grid for moderate tile counts** (~3% speedup): `[launch-config]`
   - Hardware scheduler outperforms persistent loop when tiles ≤ 4× num_SMs
   - Autotune selects smaller tiles (128x128 vs 128x256) for better load balance
   - Expected speedup: 1.02-1.05x

### Anti-patterns (things that didn't work)

- **Device-side TMA descriptors with B transpose**: `tl.make_tensor_descriptor` requires column-major B input; runtime `.T.contiguous()` copy costs more than TMA saves (2x regression)
- **flatten=True with block pointers on Hopper**: The `flatten` optimization is designed for `tl.make_tensor_descriptor`, not `tl.make_block_ptr` (regression)
- **Epilogue dependency breaking with block pointers**: Adding separate `tile_id_c` for epilogue PID computation adds overhead without benefit for block pointer stores
- **num_stages > 3 with 128x128 tiles**: 4-stage pipeline exceeds shared memory for 2 blocks/SM, forcing single-block occupancy

---

## dsa_forward (Dynamic Sparse Attention)

### Characteristics
- Bottleneck: compute-bound (attention with sparse block indices)
- Data access: Q (per-query-tile), K/V (per-block random access via block_indices), block_indices (streaming)
- Per-token baseline wastes tensor cores: matmul dims [16,128]x[128,64] too small
- GQA-aware: separate code paths for GQA (n_heads_block != n_heads_kv) vs non-GQA
- Typical sizes: batch=4, seq=2048, n_heads=32, n_heads_kv=8, head_dim=128, blk_siz=64

### Effective Optimizations

1. **FlashAttention-style Q-tiling** (8.4x speedup — most impactful): `[tile-size]` `[tensor-core]` `[algorithmic]`
   - Baseline: 1 query token per block → tiny matmuls ([16,128]x[128,64]) waste tensor cores
   - Tiled: BLOCK_Q=64 tokens per block → large matmuls ([64,128]x[128,64]) with K/V reuse
   - Grid changes from (seq_len, n_heads_block) to (num_q_tiles, n_heads)
   - Expected speedup: 5-10x (depending on matmul dimension ratio)

2. **Remove redundant tl.where on attention weights** (12% speedup): `[algorithmic]` `[warp-divergence]`
   - With b_m initialized to `-1e30` (not `-inf`), `exp(masked_score - b_m)` naturally produces ~0
   - `tl.where(mask, b_p, 0)` is mathematically redundant and wastes instructions
   - Insight: exploit numerical properties of online softmax

3. **Simplify causal mask** (3% speedup): `[algorithmic]` `[warp-divergence]`
   - Fold `q_valid` into `sr` (set sr=-1 for invalid queries) so the causal mask doesn't need a separate q_valid branch
   - Skip `sl` check entirely for non-sliding-window attention
   - Fewer mask ops = fewer instructions in the hot inner loop

4. **Pre-computed loop bound** (7% speedup): `[algorithmic]` `[warp-divergence]`
   - Replace `if blk_st <= tile_max_sr` branch inside the K/V block loop with pre-computed `n_valid = min(blk_cnt, max_sr // blk_siz + 1)`
   - Eliminates per-iteration branch and enables better instruction scheduling

### Anti-patterns (things that didn't work)

- **Remove early skip for pipelining**: Unconditionally processing all K/V blocks (including fully-masked ones) for "better pipelining" caused 59% regression. Causal early-exit is essential.
- **GQA-grouped sequential heads**: Grid=(num_q_tiles, n_heads_kv) with inner loop over GQA group to share K/V loads. Reduced parallelism hurts more than K/V reuse helps on H800 with enough heads. 15% regression.
- **V preload + fused rescale**: Loading V right after K to overlap with QK dot. Increased register pressure reduced occupancy. Slight regression.
- **Register pressure is the ceiling**: At ~46% peak compute, further gains require reducing register count per thread — a fundamental Triton compiler limitation for complex attention kernels.

---

## Cross-Kernel Optimization Patterns

Patterns below are indexed by **bottleneck type** rather than kernel. When the agent encounters a specific bottleneck on any kernel, check this section first for transferable techniques. Tags like `[register-pressure]` on per-kernel entries above map to the categories below.

### `[register-pressure]` — Reducing Registers per Thread

| Technique | Observed impact | Source kernel | Applicability |
|-----------|----------------|---------------|---------------|
| Inline helper functions | 60% register reduction (96→39) | rms_norm | Universal: any Triton/CUDA kernel using helper functions |
| Reduce tile size | 69% register reduction (126→40) | swiglu_input_quant | Any kernel where large tiles inflate register usage |
| Use `tl.make_block_ptr` instead of manual pointer arithmetic | ~7% speedup via register savings | persistent_matmul | Hopper+ with TMA support |
| Find the sweet spot tile size (not too small, not too large) | SeqTile=4 optimal, SeqTile=8 spills | qkv_part_rope | Memory-bound kernels with tunable tile dimensions |

**Diagnostic**: NCU `launch__registers_per_thread` > 64 → likely occupancy-limited. Check `launch__occupancy_limit_registers`.

### `[occupancy]` — Increasing Active Warps per SM

| Technique | Observed impact | Source kernel | Applicability |
|-----------|----------------|---------------|---------------|
| Match grid size to SM block capacity | 40% latency reduction | rms_norm | Any persistent kernel with low block count |
| Non-persistent grid for high tile counts | 8% improvement | swiglu_input_quant | When tiles >> SMs (>10x), hardware dispatch wins |
| Reduce tile size to lower registers | 11x occupancy increase | swiglu_input_quant | Register-limited kernels |

**Diagnostic**: NCU `sm__warps_active.avg.pct_of_peak_sustained_active` < 50% → investigate register and shared memory limits.

**Warning**: Increasing occupancy beyond ~50% rarely helps. At >75%, L1 contention can negate benefits (qkv_part_rope: 2 blocks/SM was 7% slower).

### `[memory-coalescing]` / `[vectorized-loads]` — Efficient Memory Access

| Technique | Observed impact | Source kernel | Applicability |
|-----------|----------------|---------------|---------------|
| Float4 (16-byte) vectorized copy | 1.3% improvement | qkv_part_rope | Any kernel with significant data copy paths |
| Vectorized loads via larger BLOCK_SIZE | Implicit in tile sizing | persistent_matmul | All Triton kernels |

**Diagnostic**: NCU coalescing ratio (`memory_l2_theoretical_sectors_global` / `ideal`) > 1.5 → access pattern needs fixing.

### `[cache]` — L1/L2 Cache Optimization

| Technique | Observed impact | Source kernel | Applicability |
|-----------|----------------|---------------|---------------|
| `evict_last` for inputs, `evict_first` for stores | 0.7-1.5% | rms_norm, swiglu | Streaming kernels (read once, write once) |
| Non-persistent grid improves L1 hit rate | 0%→32% L1 hit rate | swiglu_input_quant | When tiles have spatial locality |

**Warning**: `evict_first` on loads can backfire if data is reused by adjacent warps in the same block (qkv_part_rope: 7% regression).

### `[launch-config]` — Persistent vs Non-Persistent Grid

| When to use persistent | When to use non-persistent |
|------------------------|---------------------------|
| Tiles < 4x num_SMs | Tiles > 10x num_SMs |
| Complex scheduling logic | Simple tile-to-block mapping |
| Need cross-tile state | Stateless tiles |

**Empirical rule**: For tile counts > ~500 on H100 (132 SMs), non-persistent tends to win due to lower overhead and better L1 locality.

## flash_attention_2 (CUDA C causal attention)

### Characteristics

- Bottleneck: compute-bound on RTX 4070 Ti SUPER, but still far from peak compute
- Current instruction path: QK score generation and full-prefix PV both use
  partial WMMA paths, but tensor-core share is still low
  (`ncu_tensor_core_pct≈2.0%`)
- Working set behavior: K/V staging plus score/probability/PV handoff are
  shared-memory resident; after run_022 the active WMMA path uses about
  `32.8 KB` dynamic shared memory per CTA
- Main sensitivity observed so far: CTA geometry, shared-memory layout, and staging instruction count all materially affect performance
- Current plateau after `run_015`: `large≈2.86-2.87 ms`, `≈24.0 TFLOPS`,
  `compute_bound`, occupancy already high (`≈82.8%`), and tensor-core
  instruction share still low (`≈2.0%`). The remaining gap is structural:
  QK score generation and full-tile PV are now partially WMMA, but the path
  still materializes scores/probabilities through shared memory and leaves
  boundary/update work outside tensor cores.
- Source/SASS NCU attribution after `run_014` refined the shared-memory issue:
  `~68.8%` of attributed excessive shared wavefronts come from PV WMMA
  matrix-A loads of `prob_tile`, mapping to `kernel.cu:374`; `~22.6%` comes
  from WMMA accumulator shared stores. K/V staging and scalar probability
  writes are not conflict sources. Future bank-conflict work must therefore
  target the PV WMMA-A/probability dataflow or accumulator store/readback path,
  not K/V pitch or generic shared-memory padding.
- `run_015` resolved the dominant PV probability A-load conflict by manually
  constructing the WMMA probability fragment from shared memory. Targeted shared
  bank conflicts fell from the prior `~169.8M` aggregate signal to `62.0M`,
  then to `45.9M` after padding only the PV accumulator scratch pitch to `68`.
- `run_017` tested the next local neighborhoods against source-line evidence:
  worker-warp merging, reduced staging parallelism, fallback-control sinking,
  isolated loop-ending barrier removal, and PV scratch `float2` readback. None
  cleared the keep threshold; the best was only `+0.49%`. The remaining gap is
  therefore not a local readback/sync/layout problem under the current design.
- `run_018` found a remaining CTA-geometry win by reducing owner warps from
  `16` to `8`, with each owner warp split into two 16-lane row groups. Final
  full benchmark reached `large=2.7384 ms`, `25.107 TFLOPS`. More aggressive
  `4`-owner-warp packing regressed, so this dataflow needs at least 16 lanes
  per active query row.
- `run_019` profiled the run_018 best against the PyTorch fused FlashAttention
  reference. Current is still `3.11x` slower on `large`
  (`2.7268 ms` vs `0.8769 ms`). The run_018 change reduced total instructions
  versus run_016 by `23%`, but current remains `7.16x` above reference in total
  instructions, `10.74x` above reference in LSU instructions, and has
  `92.7M` shared store wavefronts versus reference `0.289M`. The next route is
  therefore register/worker-owned PV accumulation and larger CTA dataflow, not
  occupancy tuning or local shared-memory pitch/readback variants.
- `run_020` validated the first part of that route. Full-prefix PV worker
  accumulators are now kept in registers across full-prefix tiles and imported
  into owner accumulators only once before fallback. Final validation reached
  `large=2.6535 ms`, `25.910 TFLOPS`, correctness `PASS`. Targeted NCU moved
  shared load wavefronts `127.7M -> 99.3M`, shared store wavefronts
  `92.7M -> 79.7M`, bank conflicts `62.3M -> 39.9M`, and LSU instructions
  `150.7M -> 142.1M`.
- `run_021` found one remaining high-coverage owner-path duplication: each
  owner lane read the same full-prefix score values once for row max and again
  for probability/sum. Caching those four scores in registers improved final
  validation to `large=2.5966 ms`, `26.478 TFLOPS`, with shared load
  wavefronts `99.3M -> 91.1M` and shared load bank conflicts
  `10.5M -> 6.4M`.
- `run_022` found a larger resource/dataflow win by reusing a single shared
  operand tile for K then V. Dynamic shared memory dropped
  `45888 B -> 28480 B`, shared-memory residency moved from 2 to 3 CTAs/SM, and
  achieved occupancy rose `49.6% -> 74.1%`. A follow-up widened the PV handoff
  scratch enough to import both persistent PV accumulator phases in one round
  while preserving 3 CTAs/SM. Final validation reached `large=2.2611 ms`,
  `30.407 TFLOPS`.
- `run_023` exhausted the adjacent post-run_022 local neighborhoods. Scratch
  aliasing (`+0.38%`) and approximate `ex2` (`+0.11%`) were below threshold;
  direct global K/V WMMA operands, forced launch bounds, first-tile PV direct
  assignment, PV import phase reshaping, and `__ldg` vector staging all
  regressed. Final validation of the unchanged best measured
  `large=2.2567 ms`, `30.466 TFLOPS`, with targeted NCU still compute-bound,
  occupancy `74.1%`, dynamic shared memory `32832 B`, and tensor-core share
  `2.4%`.
- `run_024` confirmed the remaining gap is not solved by direct standard-WMMA
  structural translations. A `64x64 / 4-warp` CTA prototype passed correctness
  but regressed to `large=4.7008 ms` because `2` lanes per row and large
  per-warp PV accumulator state outweighed lower grid/KV-staging counts. A
  probability-materialization bypass that recomputed probability fragments in
  worker warps also passed correctness but regressed to `large=2.7766 ms`.
  Future score/probability/PV bypass work must avoid both shared
  materialization and duplicate exponent/register pressure, likely requiring a
  lower-level MMA/register pipeline rather than another standard-WMMA patch.
- `run_025` continued with CUDA-only structural probes. Key-segment partial PV
  output, reusing score/PV worker warps as owner rows, worker-side probability
  production directly from QK fragments, and simple full-prefix `cp.async`
  staging all passed correctness but regressed. The current standard-WMMA
  design relies on overlap between owner probability generation and worker
  V-staging; moving that work to workers or adding cross-warp partial-output
  reductions lengthens the critical path.
- `run_026` ran 20 CUDA-only iterations and kept no source change. The best
  restored validation was `large=2.2527 ms`. Standard-WMMA M64/M32 larger-query
  CTA branches were improved substantially from the prior M64 prototype but
  still plateaued at `3.0788 ms` / `3.0523 ms`, far slower than the current M16
  dataflow. Manual score stores, `N=128`, 8-worker PV, named barriers,
  padding-zero removal, output-store packing, diagonal masked PV-WMMA,
  owner-warp expansion, worker-side probability materialization, and shared/cache
  hints all failed or stayed below threshold. The remaining credible route is a
  true lower-level CUDA MMA/register mainloop rewrite with explicit fragment
  ownership; do not continue standard-WMMA CTA-size, owner/worker, padding,
  sync, or cache-hint variants without new contradictory NCU evidence.
- `run_027` tested that lower-level CUDA route for 20 iterations. Explicit
  BF16 `mma.sync.m16n8k16` QK and PV primitives were made correct, including
  the required non-transposed B `ldmatrix` direction for the staged K layout.
  However, QK-only explicit MMA still regressed because scores remained
  materialized to shared memory. The best inline-PV/worker-direct-output branch
  reached only quick `large=2.3863 ms` and used `80` registers/thread versus
  baseline `56`, losing the 3-CTA/SM residency that made the current best
  viable. Launch bounds, direct persistent accumulation, unroll limiting,
  worker output packing, 4-owner-warps, probability/shared pitch variants, and
  hybrid standard-WMMA-prefix + inline-diagonal output all regressed. The
  conclusion is to block local inline-MMA grafts onto the current M16 CTA; a
  future attempt must redesign the whole mainloop/ownership model while keeping
  register usage near the baseline budget.
- `run_031` found the first successful post-M16 structural redesign:
  `BM32/BN64/8-warp` row-group mainloop. Four contiguous warps own each
  16-row group, compute four score-column WMMA tiles in parallel, split
  softmax rows, and keep two value-tile PV accumulators per warp. Score and
  probability reuse one shared region with `scorePitch=68` and `probPitch=136`.
  Final full benchmark reached `large=2.0390 ms`, `medium=0.5964 ms`,
  `xlarge=7.6669 ms`, correctness PASS. This is the preferred base for the
  next stage.
- Negative run_031 evidence: direct BM64 row-group is blocked despite lower
  CTA count; best BM64 quick was only `2.7175 ms`. BM32 depends on K/V operand
  pitch `136`; pitch `128` and `144` regressed. Score/prob pitches `76/152` and
  `80/160`, launch bounds, and split-Q pitch `128` also regressed. Do not
  continue blind pitch sweeps; use NCU source attribution before local layout
  work.
- `run_035` corrected the previous route-selection mistake. The successful
  route was not another shared-memory score/probability layout. It kept QK
  score fragments and probability fragments in registers while preserving
  Tensor Core PV. The final active regtc kernel uses `BLOCK_Q=128`,
  `BLOCK_KV=16`, `HEAD_DIM=128`, and `8` warps. Full benchmark reached
  `medium=0.4297 ms`, `large=1.3911 ms`, `xlarge=4.9442 ms`, correctness PASS.
- Transferable run_035 lesson: do not treat failures inside the old
  shared-materialization design as proof that a register-owned redesign is
  invalid. Earlier negative results mostly tested partial bypasses that still
  carried the old intermediate, duplicated exponent work, or grafted lower-level
  MMA onto the old ownership model. The successful route changed the
  dataflow boundary first, then tuned resource balance.
- Run_035 kept changes: shrink the key/value tile to reduce live register
  fragments, fix operand-tile reuse races instead of abandoning the route,
  split K and V shared operand tiles when the saved barrier/instruction cost
  exceeded the lower CTA residency, and only then raise query tile size to
  reduce grid/KV repetition. Rejected follow-ups include shared pitch `128`,
  forced launch bounds, `BLOCK_Q=256`, and padding-zero removal.

### Effective Optimizations

1. **Prune causal work before the dot-product loop** (`[algorithmic]` `[sync]`)
   - Specializing the causal path and stopping tile/inner-loop work before masked future keys cut large-size latency substantially.
   - Why it works: the previous implementation still computed masked QK/PV work and only replaced the score with `-INF` afterward.

2. **Use sequence-length-aware CTA width** (`[occupancy]` `[launch-config]`)
   - `WARPS_PER_BLOCK=8` for `seq_len <= 2048`, `WARPS_PER_BLOCK=4` otherwise.
   - On `large`, occupancy rose from `41.4%` to `82.5%`, and latency improved from `21.11 ms` to `14.26 ms` (~`1.48x`).
   - Why it works: with `16 KB` dynamic shared memory, CTA residency was capped by shared memory rather than registers. Wider CTAs convert the same `5` resident CTAs/SM into many more active warps and let more query warps reuse each staged K/V tile.

3. **Vectorize K/V global-to-shared staging** (`[vectorized-loads]` `[memory-coalescing]`)
   - Replacing scalar pair copies with `uint4` vectorized loads/stores on the K/V staging loop improved `large` from `14.41 ms` to `13.06 ms` and `xlarge` from `83.58 ms` to `65.26 ms`.
   - Why it works: once launch geometry was already tuned, the staging loop became a clearer instruction-count bottleneck. Wider copies reduce per-element address and copy overhead without changing the math path.

4. **Add a short-sequence WMMA score-tile path** (`[tensor-core]` `[mma-shape]` `[tile-size]`)
   - Switching the `seq_len <= 2048` path to a WMMA-assisted score tile while keeping the existing online softmax and V accumulation improved `large` from `12.80 ms` to `11.77 ms`.
   - Why it works: even partial tensor-core use reduces scalar dot-product pressure enough to move the kernel forward when the shape regime is MMA-friendly.

5. **Pad WMMA shared-memory rows** (`[cache]` `[memory-access]` `[tensor-core]`)
   - Padding the WMMA Q/K/V shared-memory pitch improved `large` from `11.77 ms` to `11.45 ms`.
   - Why it works: the original WMMA layout introduced a bad shared-memory bank pattern. Padding preserved the WMMA structure while dramatically improving L1/shared-memory behavior.

6. **Widen the WMMA key tile and use two score warps** (`[launch-config]` `[tensor-core]` `[tile-size]`)
   - Expanding the short-sequence WMMA key tile from `16` to `32` improved `large` from `11.45 ms` to `10.86 ms`.
   - Why it works: a wider score tile amortizes synchronization and staging overhead over more useful QK work, even if occupancy falls somewhat.

7. **Use base-2 online softmax updates** (`[algorithmic]` `[data-type]`)
   - Replacing `expf` with `exp2f` in the online softmax recurrence improved `large` from `10.86 ms` to `10.46 ms`.
   - Why it works: the kernel remains compute-bound after the structural WMMA changes, so cheaper exponential updates still matter in the hot loop.

8. **Vectorize WMMA-path Q/K/V staging** (`[vectorized-loads]` `[memory-coalescing]` `[tensor-core]`)
   - Replacing element-wise WMMA tile staging with `uint4` copies improved `large` from `10.46 ms` to `8.15 ms` and `medium` from `2.74 ms` to `2.09 ms`.
   - Why it works: after the WMMA path exists, staging overhead becomes a major secondary cost. Vectorized copies slash address-generation and copy instruction count without changing the numerical path.

9. **Move full-tile PV accumulation to WMMA and reuse score scratch** (`[tensor-core]` `[pv-path]` `[shared-memory]`)
   - Adding a full-tile PV-WMMA path improved the inherited `large` full benchmark from `5.0569 ms` to `4.3916 ms`, but the naive extra `16x128` float PV scratch cut occupancy to `41.6%`.
   - Reusing the existing score scratch as two-phase `16x64` PV output scratch restored occupancy to `82.8%` and improved the final full benchmark to `large=3.3257 ms`, `20.67 TFLOPS`.
   - Why it works: the steady-state full-tile path covers most dynamic work in the benchmarked shape, and scalar owner-warp `P*V` was the next structural bottleneck after QK score WMMA. The scratch-reuse version keeps the tensor-core PV work without paying a shared-memory residency penalty.

10. **Specialize full-prefix K/V staging** (`[full-tile]` `[staging]` `[control-flow]`)
   - Splitting K/V staging into a branch-free full-prefix path improved `large` from `3.3257 ms` to a best observed `3.2227 ms`; promotion validation measured `3.2473 ms`.
   - Why it works: benchmarked causal shapes spend most tile iterations in full-prefix tiles where all 64 K/V rows are active. Removing inactive-row checks and zeroing from that hot staging loop trims repeated control/copy overhead without changing the WMMA operand layout.

11. **Manually construct the PV probability WMMA A fragment** (`[bank-conflict]` `[pv-path]` `[manual-fragment]` `[tensor-core]`)
   - Replacing standard `wmma::load_matrix_sync` for the PV probability A
     operand with explicit per-lane fragment loads improved `large` from
     `3.2487 ms` to `2.9190 ms` in run_015 (`+11.3%`).
   - Source-line NCU confirmed that the new manual probability load path has
     `0` excessive shared wavefronts, eliminating the previously dominant
     source-attributed conflict from the standard WMMA A-load helper.
   - Why it works: the kernel still uses WMMA for compute, but avoids the
     shared-memory access pattern generated by the standard WMMA A load for the
     materialized probability tile.

12. **Pad only the PV accumulator scratch pitch after the probability fix** (`[bank-conflict]` `[pv-path]` `[scratch-layout]`)
   - Changing the PV scratch leading dimension from `64` to `68` floats after
     the manual probability fragment change improved `large` from `2.9190 ms`
     to `2.8615 ms` and reduced targeted shared bank conflicts from `62.0M` to
     `45.9M`.
   - Why it works: once the probability A-load conflict is removed, the
     accumulator store/readback path becomes the remaining high-coverage shared
     handoff. The pitch change is narrow and preserves the WMMA accumulator
     store mapping.

13. **Reduce owner warps with half-warp row ownership** (`[owner-warps]` `[cta-geometry]` `[dataflow]`)
   - Reducing WMMA owner warps from `16` to `8` and assigning two query rows per
     owner warp with two 16-lane groups improved run_018 `large` from
     `2.8777 ms` to `2.7427 ms`; final full validation measured
     `large=2.7384 ms`, `25.107 TFLOPS`.
   - Why it works: this removes excessive owner-warp/thread overhead while
     preserving the four score/PV worker warps and the existing MMA
     decomposition. It is a structural CTA reduction, not a local readback or
     synchronization tweak.

14. **Keep full-prefix PV accumulators in worker registers** (`[pv-path]` `[dataflow]` `[shared-memory]`)
   - Moving full-prefix PV accumulation into score/PV worker-warp registers and
     importing the accumulated result into owner warps only once before fallback
     improved the run_020 full benchmark to `large=2.6535 ms`,
     `25.910 TFLOPS`.
   - NCU confirmed the mechanism: shared load wavefronts dropped from `127.7M`
     to `99.3M`, shared store wavefronts from `92.7M` to `79.7M`, bank
     conflicts from `62.3M` to `39.9M`, and LSU instructions from `150.7M` to
     `142.1M`.
   - Why it works: it removes repeated per-full-tile
     `PV WMMA -> shared scratch -> owner readback` handoff without introducing
     a larger shared accumulator.

15. **Cache owner score values between full-prefix max and probability passes** (`[score-tile]` `[shared-memory]` `[owner-path]`)
   - The owner row-stat path used to read each full-prefix QK score once during
     the max pass and again during the probability/sum pass. Keeping each
     lane's four score values in registers improved run_021 final validation to
     `large=2.5966 ms`, `26.478 TFLOPS`.
   - NCU confirmed the mechanism: total instructions dropped
     `640.0M -> 632.9M`, shared load wavefronts `99.3M -> 91.1M`, shared load
     bank conflicts `10.5M -> 6.4M`, while registers/thread stayed at `56`.
   - Why it works: it removes a high-coverage duplicate shared read without
     changing the score tile layout or increasing CTA resource residency.

16. **Reuse one shared operand tile for K then V** (`[shared-memory]` `[occupancy]` `[staging]` `[dataflow]`)
   - K and V are not simultaneously live in the current WMMA main loop. Reusing
     the same shared tile for K during QK and then for V during PV reduced
     dynamic shared memory from `45888 B` to `28480 B`.
   - NCU confirmed the resource mechanism: shared-memory residency improved
     from 2 to 3 CTAs/SM and achieved occupancy rose from `49.6%` to `74.1%`.
     Full `large` latency improved from `2.5918 ms` to `2.2853 ms`.
   - Why it works: this preserves the proven shared-memory WMMA operand layout
     but removes unnecessary simultaneous residency. It is a resource/dataflow
     change, not a K/V layout or direct-global operand change.

17. **Spend freed shared-memory budget on one-round PV accumulator import** (`[pv-path]` `[shared-memory]` `[sync]` `[dataflow]`)
   - After single operand tile reuse, widening the PV scratch pitch to `136`
     still allowed 3 CTAs/SM and let both persistent PV accumulator phases be
     stored and imported in one handoff round.
   - NCU after the change showed dynamic shared memory `32832 B`, occupancy
     `74.2%`, instructions `717.2M -> 708.0M`, and shared bank conflicts
     `31.8M -> 30.9M`. Full `large` improved to `2.2566 ms` in the promotion
     run.
   - Why it works: once shared-memory residency has headroom, a wider scratch
     can reduce synchronization and handoff rounds without lowering CTA
     residency.

18. **Remove hot-path intermediate materialization with a register-owned Tensor Core route** (`[register-dataflow]` `[tensor-core]` `[cta-geometry]` `[dataflow]`)
   - Replacing the shared-materialized score/probability route with a register
     fragment route improved the regtc full benchmark from `large=1.9261 ms` to
     `large=1.3911 ms` and `xlarge=6.4678 ms` to `xlarge=4.9442 ms`.
   - The winning configuration keeps score and probability fragments in
     registers, uses Tensor Core for PV, uses `BLOCK_KV=16` to reduce live
     fragment state, splits K and V shared operand tiles to remove an operand
     reuse barrier, and uses `BLOCK_Q=128` to reduce grid/KV repetition.
   - Why it works: the prior design's bottleneck was the dataflow boundary
     itself, not the exact shared-memory layout. Removing the hot intermediate
     reduces instruction and handoff cost enough to beat the occupancy loss from
     higher register/shared-memory use.
   - Execution rule: give this type of route a finite repair budget. In run_035,
     an intermediate `BLOCK_KV=32` version exposed a shared-memory race and
     failed determinism; fixing the synchronization revealed the route was
     strongly positive.

### Anti-patterns

- **Treating old-design partial bypass failures as proof against a full redesign** (`[strategy]` `[dataflow]` `[register-dataflow]`)
  - Several earlier experiments removed only one shared store/read or grafted a
    new primitive onto the old ownership model. They regressed because they
    duplicated expensive work, carried the old intermediate, or increased
    register pressure without deleting the dominant handoff.
  - Reason: negative evidence must be scoped to the dataflow actually tested.
    If the old intermediate remains, the result blocks that local graft, not a
    route that removes the intermediate and rebalances ownership/tile shape
    together.
- **CUDA-only standard-WMMA role/dataflow rewrites after run_024** (`[cuda-only]` `[dataflow]` `[shared-memory]`)
  - Key-segment partial PV output regressed to `large=6.1790 ms`; reusing
    score/PV worker warps as owner rows regressed to `2.3796 ms`; generating
    probabilities directly from QK fragments in worker warps regressed to
    `3.6539 ms`.
  - Reason: these variants attack the right shared-materialization bottleneck
    but move too much row-stat/probability/PV work onto the score/PV worker
    critical path or require large partial-output reductions. The current
    standard-WMMA design depends on owner/worker overlap.
- **Simple `cp.async` conversion for current full-prefix K/V staging** (`[cuda-only]` `[pipeline]` `[staging]`)
  - Replacing hot full-prefix `uint4` staging loops with
    `cp.async.ca.shared.global` regressed to `large=2.4965 ms`.
  - Reason: the current staging bottleneck is not exposed global-memory latency
    that a minimal async-copy conversion can hide. The added commit/wait
    pipeline overhead is worse than the existing vectorized load/store path.
- **Direct `64x64 / 4-warp` standard-WMMA CTA translation** (`[tile-size]` `[launch-config]` `[tensor-core]`)
  - A runtime-general prototype passed correctness but regressed to
    `large=4.7008 ms` versus the current best `2.2567 ms`.
  - Reason: reducing grid count and K/V staging repetition is not sufficient
    when the ownership model leaves only `2` lanes per row and requires eight
    PV accumulator fragments per warp. Larger-M CTA work needs a different
    low-level warp/MMA ownership model, not direct standard-WMMA expansion.
- **Worker-side probability recompute to avoid `prob_tile` stores** (`[dataflow]` `[shared-memory]` `[instruction-count]`)
  - Computing PV probability fragments from `score_tile + row_m` instead of
    storing `prob_tile` passed correctness but regressed to `large=2.7766 ms`.
  - Reason: the removed shared stores were outweighed by duplicate `exp2`
    work and fused-PV register/code pressure. A probability bypass must reuse
    already computed probabilities or keep score fragments live cheaply; simply
    moving exponent work into worker warps is negative.
- **Post-run_023 local variants around the run_022 dataflow** (`[dataflow]` `[staging]` `[pv-path]`)
  - Scratch aliasing and approximate `ex2` were directionally positive but only
    `+0.38%` and `+0.11%`, below threshold. Direct global V and K operands
    regressed to `2.9001 ms` and `3.2990 ms`; forced
    `__launch_bounds__(384,4)` regressed to `3.1151 ms`; first PV tile direct
    assignment regressed to `2.8286 ms`; `__ldg` vector staging regressed to
    `2.7870 ms`.
  - Reason: after single operand tile reuse and one-round PV import, local
    branch/cache/scheduling variants do not change the dominant dataflow. Future
    work should reduce score/probability/PV materialization or change the
    ownership/main-loop structure instead of polishing the same handoff.
- **Post-run_022 single-tile staging micro-variants** (`[shared-memory]` `[staging]` `[sync]`)
  - Removing the duplicate padding store after K/V tile aliasing improved only
    `0.15%`, below threshold. Distributing V restaging across all CTA threads
    regressed because worker-only restaging overlaps better with owner
    probability generation. Removing the final import-readback barrier improved
    `large` by only `0.08%` and was mixed on other sizes.
  - Reason: after the structural single-tile and one-round import wins, the
    nearby padding/restaging/barrier variants no longer remove enough dynamic
    work to survive full-benchmark gating.
- **Standard WMMA probability pitch padding for the PV A operand** (`[bank-conflict]` `[pv-path]` `[shared-memory]`)
  - Padding the probability tile leading dimension from `64` to `72` while
    still using standard `wmma::load_matrix_sync` regressed `large` from
    `3.2487 ms` to `4.1242 ms`.
  - Reason: although it targets the right source-attributed line, the standard
    WMMA load/codegen/layout cost dominates any conflict reduction. Prefer the
    manual probability fragment dataflow instead of nearby standard-WMMA pitch
    sweeps.
- **K/V pitch or generic shared-memory tuning for the current bank-conflict signal** (`[shared-memory]` `[bank-conflict]`)
  - Source attribution showed K/V full-prefix staging, residual staging, scalar `prob_tile` stores, and `v_tile` WMMA-B loads have actual shared wavefronts equal to ideal or zero excessive wavefronts.
  - Reason: the aggregate `~169.8M` bank-conflict signal is dominated by PV WMMA-A loads of `prob_tile` and WMMA accumulator stores. K/V layout work does not attack the source-attributed bottleneck.
- **Direct global WMMA operands for staged K/V in this kernel** (`[memory-access]` `[tensor-core]`)
  - Loading `V` directly from global/L2 in PV-WMMA regressed to `large=3.55 ms`; loading `K` directly from global/L2 in QK-WMMA regressed to `large=3.87 ms`.
  - Reason: despite high L2 hit rate, the current WMMA path depends on the staged/padded shared-memory operand layout. Removing staging loses more than it saves.
- **Late-stage local staging/control variants after full-prefix staging** (`[staging]` `[control-flow]`)
  - Skipping full-prefix pad-zero stores, branch-free full-query Q staging, and duplicating a larger full-prefix fast block did not improve the primary `large` metric beyond the kept staging split.
  - Reason: these affect less dynamic work or add enough code/branch pressure to erase the saved instructions.
- **Adjacent PV-WMMA micro-tweaks after scratch reuse** (`[pv-path]` `[shared-memory]` `[control-flow]`)
  - Compact `V` pitch, `kWmmaBlockN=128`, explicit phase-loop unroll, diagonal-adjacent PV-WMMA coverage, padded `65`-stride PV scratch, and lane-half phase checks all failed to beat the scratch-reuse PV-WMMA best.
  - Reason: after the structural PV-WMMA change, the nearby local layout/control variants were either below the full-benchmark keep threshold or added resource/control cost. Future work should change the dataflow boundary again instead of sweeping neighboring forms of the same implementation.
- **Post-run_015 score pitch and manual accumulator-store variants** (`[bank-conflict]` `[score-tile]` `[pv-path]`)
  - QK score tile pitch `68` improved the primary `large` metric by only
    `0.68%`, below the keep threshold. Manual PV accumulator scratch stores
    improved `large` by only `0.07%` and regressed `medium`/`xlarge`.
  - Reason: after the manual probability fragment and PV scratch-pitch changes,
    the remaining local store/readback conflict is too small for another
    isolated layout/store tweak to reliably move full-benchmark performance.
    Require new source-line evidence before revisiting this neighborhood.
- **Post-run_017 local handoff/sync/worker tweaks** (`[pv-path]` `[sync]` `[worker-warps]`)
  - Merging score/PV worker warps regressed to `large=3.1731 ms`; limiting
    staging work to `256` CTA threads regressed to `large=4.3249 ms`;
    fallback-only control sinking improved only `0.09%`; skipping one
    loop-ending barrier regressed to `large=2.8871 ms`; PV scratch `float2`
    readback improved only `0.49%`.
  - Reason: these are local variants around the same shared-handoff dataflow.
    The current plateau requires reducing CTA/warp count and the
    score/probability/PV handoff together, not shaving isolated instructions
    from a design whose total instruction count is still far above reference.
- **Over-compressing owner rows below 16 lanes per row** (`[owner-warps]` `[cta-geometry]`)
  - Reducing from the kept `8` owner warps to `4` owner warps, with four
    8-lane row groups per owner warp, regressed to `large=3.0249 ms`.
  - Re-testing after run_020 persistent PV accumulation still regressed to
    `large=2.9888 ms` in quick benchmark.
  - Reason: the CTA thread count drops, but each row has too little parallelism.
    Per-lane softmax/readback work and reduced staging participation dominate.
    For this dataflow, 16 lanes per query row is the useful compression point.
- **Naive 32-query-row CTA phase reuse** (`[cta-geometry]` `[staging-reuse]`)
  - Processing two 16-row query phases per staged K/V tile regressed to
    `large=3.7870 ms`.
  - Reason: K/V staging reuse was overwhelmed by doubled owner state,
    serialized query phases, a larger Q shared tile, and extra synchronization.
    Larger query coverage needs a different ownership/dataflow model, not
    serial reuse of the current owner-warp path.
- **Separate score and PV scratch slabs after scratch reuse** (`[shared-memory]` `[dataflow]`)
  - Giving QK score storage and PV accumulator scratch separate shared regions
    regressed to `large=2.9032 ms`.
  - Reason: score/PV scratch aliasing is not the remaining bottleneck; extra
    shared footprint and address-shape cost outweigh any isolation benefit.
- **PV-owned accumulation via a larger shared accumulator** (`[pv-path]` `[shared-memory]` `[dataflow]`)
  - A runtime-general full-prefix shared-accumulator probe passed quick correctness but regressed `large` to `5.0465 ms`.
  - Reason: moving ownership out of owner warps only helps if the accumulator stays low-overhead. Adding a `16x128` shared accumulator increases shared footprint and shared update traffic enough to overwhelm the saved owner readback. Register-level PV ownership would need a different dataflow because standard WMMA fragment layout is not suitable for row-wise online-softmax scaling.
- **Probability-subtile streaming with standard WMMA fragments** (`[score-prob]` `[pv-path]` `[tensor-core]`)
  - A runtime-general `16x16` probability subtile version passed quick correctness but regressed `large` to `5.4388 ms`.
  - Reason: the standard WMMA implementation requires extra full-block barriers between probability-subtile production and PV consumption, and the two-accumulator-fragment PV path increases register/code pressure. The saved `16x64` probability materialization is not enough to offset those costs.
- **Keeping standard WMMA QK accumulator fragments live across row-stat barriers** (`[score-tile]` `[dataflow]` `[tensor-core]`)
  - A run_020 attempt to bypass full-prefix score materialization by preserving
    QK accumulator fragments across multiple CTA-wide barriers exceeded a
    reasonable compile window and was reverted.
  - Reason: in the current monolithic standard-WMMA kernel, this creates too
    much compile/register-allocation pressure. A future score/probability
    bypass should be a smaller lower-level MMA pipeline or separate main-loop
    redesign, not this direct fragment-lifetime extension.
- **BF16 score scatter into probability storage** (`[score-tile]` `[data-type]` `[shared-memory]`)
  - Storing full-prefix QK scores as BF16 in `prob_tile` and overwriting them
    with probabilities passed quick correctness but regressed to
    `large=2.7729 ms`.
  - Reason: manual fragment scatter and BF16 conversion cost exceeded the saved
    shared-memory footprint.
- **Fusing the two full-prefix PV value phases in standard WMMA** (`[pv-path]` `[dataflow]` `[register-pressure]`)
  - Constructing `p_frag` once per `k_step` and feeding both value-column halves
    passed quick correctness but regressed to `large=2.7117 ms`.
  - Reason: simultaneous phase fragments increase register/code pressure more
    than the saved probability-fragment loads help.
- **Post-run_021 local PV/probability load/store formatting** (`[pv-path]` `[probability-store]` `[shared-memory]`)
  - Vectorized manual probability-fragment loads improved only `0.41%`; hoisting
    repeated `row_alpha_tile` loads improved only `0.47%`; vectorized bf16x2
    probability stores regressed `large` by about `8%`.
  - Reason: after persistent PV accumulation and owner score caching, these
    local shared-memory formatting changes do not remove enough dynamic work.
    The probability-store version also introduces shuffle/branch imbalance that
    overwhelms the reduced store count.
- **Score pitch variants after owner score caching** (`[score-tile]` `[scratch-layout]`)
  - Re-testing score tile pitch `68` after run_021's owner score cache tied the
    primary metric and regressed `xlarge`.
  - Reason: score pitch remains below the keep threshold even after the owner
    read pattern changed. Do not continue score-pitch variants without new
    source-level evidence that score-store conflicts have become dominant.
- **Boundary-only score/owner tweaks after `run_009`** (`[causal-mask]` `[score-tile]` `[wmma-hotloop]`)
  - Tail tiles, diagonal-tile cleanups, partial score-warp gating, and similar
    boundary-only changes affect too little of the `seq_len=2048` primary
    benchmark to justify early iterations.
  - Reason: most dynamic work is the steady-state full causal tile path, and the
    keep threshold is `>1%` on full benchmark. These ideas should be deferred
    unless NCU shows the boundary path became dominant.
- **More score-tile layout/load polishing around the materialized score slab** (`[score-tile]` `[shared-memory]`)
  - Compact score subtiles, score-row load packing, fewer score warps,
    column-major score layout, and `kWmmaBlockN=48` retuning have already failed
    or failed to survive full validation in recent runs.
  - Reason: the bottleneck is the materialized score handoff plus scalar PV, not
    the exact local representation of the same handoff. Future experiments
    should reduce/bypass score materialization or move PV to tensor cores.
- **A single fixed wide CTA shape for every sequence length** (`[occupancy]` `[launch-config]`)
  - Fixed `WARPS_PER_BLOCK=8` improved `medium`/`large`, but regressed `xlarge` badly (`195.59 ms` vs ~`85 ms` baseline-scale).
  - Reason: the wider CTA is beneficial where occupancy is the main limiter, but it over-shoots on larger sequence lengths. Launch geometry should depend on sequence length, not stay globally fixed.
- **Shared-memory K/V swizzle on this kernel** (`[memory-access]` `[cache]`)
  - A swizzled K/V tile layout regressed `large` from `14.41 ms` to `16.66 ms`.
  - Reason: the extra index arithmetic outweighed any bank-conflict benefit on this small shared-memory footprint.
- **Bigger short-sequence `BLOCK_N` without evidence of tile-loop overhead** (`[tile-size]` `[launch-config]`)
  - `BLOCK_N=48` and `BLOCK_N=40` both passed correctness but slightly regressed the kept vectorized-staging version.
  - Reason: the kernel is still compute-bound and scalar, so larger tiles did not create enough reuse to offset the extra per-CTA work.
- **CTA widths that violate existing synchronization assumptions** (`[occupancy]` `[launch-config]`)
  - `WARPS_PER_BLOCK=12` produced `NaN/Inf` in the smoke test even though raw latency improved.
  - Reason: nontrivial CTA-shape changes can create partial-CTA tails that return before later `__syncthreads()`. Correctness constraints must be checked before trusting faster timings.

### `[tensor-core]` — Maximizing Tensor Core Utilization

| Technique | Observed impact | Source kernel | Applicability |
|-----------|----------------|---------------|---------------|
| Remove FP32 cast before `tl.dot` | 7.7x speedup | persistent_matmul | CRITICAL: any GEMM kernel |
| Increase matmul dimensions (Q-tiling) | 8.4x speedup | dsa_forward | Attention kernels with small per-token matmuls |
| Tile sizes ≥ 128 for M/N dimensions | 2x speedup | persistent_matmul | All GEMM-dominated kernels |

**Diagnostic**: NCU `sm__inst_executed_pipe_tensor.sum` = 0 → tensor cores not being used at all.

### `[warp-divergence]` / `[algorithmic]` — Branch Elimination

| Technique | Observed impact | Source kernel | Applicability |
|-----------|----------------|---------------|---------------|
| Remove redundant conditional operations | 12% speedup | dsa_forward | Any kernel with `tl.where` in hot loops |
| Pre-compute loop bounds | 7% speedup | dsa_forward | Loops with data-dependent exit conditions |
| Fold validity checks into data (set invalid=-1) | 3% speedup | dsa_forward | Kernels with per-element validity masks |
| Branchless rescaling (AVO technique) | 8.1% speedup | FlashAttention (AVO paper) | Online softmax, any conditional rescale |

**Diagnostic**: NCU `smsp__warps_issue_stalled_membar` high + branches in kernel → branchless conversion may help.

### Anti-patterns That Transfer Across Kernels

These failed consistently across multiple kernel types:

1. **Over-subscribing warps per SM**: Adding more blocks/SM beyond the latency-hiding sweet spot increases L2 contention. Failed on: qkv_part_rope, rms_norm.
2. **`__launch_bounds__` to force more blocks**: Compiler spills registers to local memory. Failed on: qkv_part_rope.
3. **num_warps too high**: Reduces blocks/SM without enough benefit. Failed on: rms_norm (num_warps=16/32), swiglu (num_warps=16).
4. **Swizzled tile ordering on small kernels**: Integer overhead exceeds L2 benefit. Failed on: swiglu_input_quant.
5. **Explicit int32 offsets in Triton**: Compiler already optimizes; manual downcasts can generate worse code. Failed on: rms_norm.
