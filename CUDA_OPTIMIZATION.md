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
| barrier / wait / fence stalls | `docs/sync_optimization.md`, `docs/stall_reasons.md` | `[sync]`, `[warp-divergence]`, `[algorithmic]` |
| poor L1 / L2 reuse | `docs/memory_optimization.md`, `docs/arch_notes.md` | `[cache]`, `[memory-access]`, `[tile-order]` |
| branch-heavy hot loop | `docs/compute_optimization.md`, `docs/sync_optimization.md` | `[warp-divergence]`, `[algorithmic]` |
| architecture-specific async copy or pipeline issue | `docs/arch_notes.md`, `docs/memory_optimization.md`, `docs/sync_optimization.md` | `[pipeline]`, `[memory-access]`, `[launch-config]` |

### By Kernel Style

| Kernel style | Primary docs | Typical first checks |
|---|---|---|
| Triton elementwise / reduction | `docs/triton_optimization.md`, `docs/memory_optimization.md` | coalescing, tile width, occupancy |
| Triton GEMM / attention | `docs/triton_optimization.md`, `docs/compute_optimization.md`, `docs/sync_optimization.md` | tensor-core path, tile size, registers, stages |
| CUTLASS GEMM / conv | `docs/cutlass_optimization.md`, `docs/compute_optimization.md`, `docs/memory_optimization.md` | tile shape, stage count, schedule, epilogue |
| CUDA C custom kernel | `docs/compute_optimization.md`, `docs/memory_optimization.md`, `docs/sync_optimization.md` | launch config, shared memory, warp primitives |

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
| `[register-pressure]` | registers per thread, spills, occupancy limit | `docs/compute_optimization.md` |
| `[occupancy]` | active warps / blocks per SM | `docs/compute_optimization.md` |
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

---

## Investigation Playbooks

### Compute-Bound Playbook

1. confirm whether the kernel is actually matmul / MMA-like
2. check whether the active shape regime is MMA-friendly or small-M / decode-like
3. only then inspect tensor-core path and tensor-core instruction share
4. inspect tile shape and data type
5. inspect registers per thread and achieved occupancy
6. only then try deeper pipeline, warp specialization, or epilogue fusion
7. estimate dynamic coverage before selecting an experiment: if the change only
   affects boundary tiles, tail predicates, diagonal tiles, or rare shape cases,
   compute the maximum end-to-end speedup and skip it when the estimate is near
   benchmark noise or below the keep threshold
8. if the dominant gap is structural, such as low tensor-core utilization on an
   MMA-friendly compute-bound kernel, prioritize redesigns that move the main
   FLOPs onto the right pipeline before local layout/load/barrier polishing

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
- Current instruction path: short-sequence path now uses a partial WMMA score-tile implementation, but tensor-core share is still low (`ncu_tensor_core_pct≈0.4%`)
- Working set behavior: K/V staging is shared-memory resident; dynamic shared memory per CTA is only `16 KB`
- Main sensitivity observed so far: CTA geometry, shared-memory layout, and staging instruction count all materially affect performance
- Current plateau after `run_012`: `large≈3.22-3.25 ms`, `21.2-21.3 TFLOPS`,
  `compute_bound`, occupancy already high (`≈82.8%`), and tensor-core
  instruction share still low (`≈2.1%`). The remaining gap is structural:
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

### Anti-patterns

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
- **PV-owned accumulation via a larger shared accumulator** (`[pv-path]` `[shared-memory]` `[dataflow]`)
  - A runtime-general full-prefix shared-accumulator probe passed quick correctness but regressed `large` to `5.0465 ms`.
  - Reason: moving ownership out of owner warps only helps if the accumulator stays low-overhead. Adding a `16x128` shared accumulator increases shared footprint and shared update traffic enough to overwhelm the saved owner readback. Register-level PV ownership would need a different dataflow because standard WMMA fragment layout is not suitable for row-wise online-softmax scaling.
- **Probability-subtile streaming with standard WMMA fragments** (`[score-prob]` `[pv-path]` `[tensor-core]`)
  - A runtime-general `16x16` probability subtile version passed quick correctness but regressed `large` to `5.4388 ms`.
  - Reason: the standard WMMA implementation requires extra full-block barriers between probability-subtile production and PV consumption, and the two-accumulator-fragment PV path increases register/code pressure. The saved `16x64` probability materialization is not enough to offset those costs.
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
