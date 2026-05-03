# Compute Optimization Reference

Quick-reference for maximizing compute throughput on NVIDIA GPUs.

---

## Tensor Core Utilization

**What**: Tensor cores perform matrix multiply-accumulate (MMA) on small matrix tiles (e.g., 16x16x16) in a single cycle.

**Important gate**: Low tensor-core utilization is **not** by itself a mandate to rewrite for tensor cores. Treat it as actionable only when the kernel is matmul-like and the active problem shape is friendly to MMA tile fill.

**Throughput**: 989.5 TFLOPS FP16 on H100 (with tensor cores) vs ~60 TFLOPS FP32 (without).

**Requirements for tensor core usage**:
- Matrix dimensions must be multiples of 16 (FP16) or 8 (TF32)
- Data must be in FP16, BF16, TF32, FP8, or INT8
- Use `tl.dot` (Triton) or `wmma`/`mma.sync` (CUDA) intrinsics

**Common mistake**: Casting inputs to FP32 before `tl.dot` forces scalar FMA path (16x slower).

**NCU indicator**: `sm__inst_executed_pipe_tensor.sum` as fraction of total instructions.

**Decision flow**:
1. Is the kernel matmul-like or otherwise dominated by MMA-style dense dot products?
2. Is the active shape regime MMA-friendly, or is it small-M / decode-like with poor tile fill?
3. If you only have bench / roofline evidence and the kernel is shape-friendly, treat the state as **needs NCU evidence**, not as an immediate tensor-core mandate.
4. Only after NCU shows low `ncu_tensor_core_pct` should tensor-core work become a high-priority redesign path.
5. The default targeted NCU skill set already includes tensor-core utilization evidence, so a normal targeted profile can promote `needs_ncu_evidence` to `recommended` when the kernel is MMA-friendly and tensor-core instruction share is low.

**Small-M / decode-like warning**:
- For MMA tile `m=16`, an effective M below 16-32 often leaves tensor cores under-filled.
- In decode-like cases such as batch < 16 and short query length, padding and packing overhead can erase tensor-core gains.
- In these regimes, explicitly compare:
  - CUDA-core path
  - tensor-core path with padding / packing
- Useful heuristics to record are `mma_m_fill_ratio = effective_m / padded_m` and `padding_overhead_ratio = padded_m / effective_m - 1`.

**Tile size guidance for GEMM**:

| GPU | Optimal tile | Why |
|-----|-------------|-----|
| H100/H800 | 128x128 or 128x256 | Fills tensor core pipeline, good L2 reuse |
| A100 | 128x128 or 256x64 | Balance between compute and shared memory |
| 4090 | 64x64 or 128x64 | Smaller L2, fewer SMs |

---

## Instruction Mix Optimization

**Principle**: Different instructions use different execution pipelines. Bottleneck is the most-utilized pipeline.

| Pipeline | Operations | Notes |
|----------|-----------|-------|
| FP32 (FMA) | float add, mul, fma | 1 per SM per clock |
| FP16/BF16 | half precision arithmetic | 2x throughput vs FP32 |
| Tensor | MMA operations | 16x+ throughput vs FP32 |
| INT32 | integer arithmetic, address calc | shared with FP32 on some archs |
| SFU | sin, cos, exp, rsqrt, rcp | 4 per cycle per SM, slower than FMA |
| LSU | load/store | limited by memory bandwidth |

**Strength reduction** (reduce expensive ops):
- `x / y` → `x * __frcp_rn(y)` (reciprocal + multiply)
- `exp(x)` → `__expf(x)` (fast math, less accurate)
- `sqrt(x)` → `rsqrt(x) * x` (if rsqrt available)
- `x % power_of_2` → `x & (power_of_2 - 1)` (bitwise mask)

---

## Warp Divergence

**What**: When threads in a warp take different branches, both paths execute sequentially (threads on wrong path are masked off).

**Impact**: Up to 2x slowdown for a single if/else. Worse for nested branches.

**NCU indicator**: `smsp__thread_inst_executed_pred_on.sum` / `smsp__inst_executed.sum` < 32 = divergence.

**Mitigations**:
- **Predicated execution**: Replace branches with arithmetic
  ```c
  // Branch (divergent if condition varies within warp):
  if (cond) x = a; else x = b;
  // Predicated (no divergence):
  x = cond * a + (1 - cond) * b;
  ```
- **Branchless select**: CUDA's `__fsel(cond, a, b)` or ternary on uniform condition
- **Sort data**: Ensure threads in same warp follow same path
- **AVO technique**: Always compute both paths, use multiplication by 0/1 to select
  - Example: rescale factor = `need_rescale ? scale : 1.0` → always compute scale, multiply by 1.0 when not needed
  - Eliminates branches AND enables lighter memory fences

---

## Register Pressure

**Budget**: 65536 registers per SM (Hopper). Divided among all resident threads.

| Regs/thread | Max threads/SM | Max warps/SM | Theoretical occupancy |
|-------------|---------------|-------------|----------------------|
| 32 | 2048 | 64 | 100% |
| 64 | 1024 | 32 | 50% |
| 96 | 682 | 21 | 33% |
| 128 | 512 | 16 | 25% |
| 192 | 341 | 10 | 16% |
| 256 | 256 | 8 | 12.5% |

**NCU indicator**: `launch__registers_per_thread`, `launch__occupancy_limit_registers`

**Reducing register pressure**:
- **Inline helpers**: Function calls reserve registers for the call frame
- **Reduce live variables**: Compute values just before use, not at beginning
- **Split kernel phases**: Compute phase 1 → sync → compute phase 2 (each phase uses fewer registers)
- **Use shared memory for spill**: Explicitly store intermediate values to shared memory
- **Smaller tiles**: Reduce tile size = fewer registers for tile data
- **`__launch_bounds__(maxThreads, minBlocks)`**: Tell compiler the register budget (but may cause spilling if too aggressive)

**Spilling** (registers → local memory): NCU shows `lmem` traffic. 100x+ slower than register access. Avoid at all costs.

---

## Occupancy

**What**: Ratio of active warps to maximum possible warps per SM.

**Limiters** (in priority order):
1. **Registers**: Most common limiter. See table above.
2. **Shared memory**: Each block's shared memory allocation reduces blocks/SM.
3. **Block size**: Warps/SM ≤ 64 (Hopper). Block of 1024 threads = 32 warps.
4. **Blocks per SM**: Hardware limit of 32 blocks/SM (Hopper).

**Diminishing returns**: Going from 25% to 50% occupancy usually helps a lot. Going from 50% to 75% helps less. Above 75% rarely helps and can hurt (more register pressure, more L1 contention).

**When low occupancy is OK**: Compute-bound kernels with high IPC. If tensor cores are >80% utilized *and* the kernel is in an MMA-friendly shape regime, occupancy usually matters less.

---

## Warp Count / CTA Geometry Strategy

Warp count is part of CTA geometry, not an isolated tuning knob. Choose it
together with tile shape, per-thread work, register budget, shared-memory
footprint, and synchronization scope.

Reliable external anchors:

- CUDA Programming Guide and Runtime API: warps are groups of `32` threads;
  resident blocks and warps depend on registers, shared memory, and launch
  configuration; occupancy APIs estimate resident blocks from launch geometry
  and resource use.
- CUDA Best Practices Guide: higher occupancy does not always mean higher
  performance; `128-256` threads/block is a reasonable first experimental range
  for many kernels; several smaller blocks are often better than one large block
  when `__syncthreads()` latency matters.
- Nsight Compute Profiling Guide: use Occupancy, SchedulerStats,
  WarpStateStats, SourceCounters, and stall reasons together. Do not chase stall
  reasons unless schedulers fail to issue regularly.
- Triton tutorials and kernels: `num_warps` is autotuned for matmul and
  attention; persistent softmax derives program count from register pressure and
  divides by `num_warps`; persistent matmul sweeps `4/8` warps and warns that
  persistent variants can fail on small shared-memory devices.

Local reference data:

- See `workspace/runs/run_029/warp_count_reference_bench_summary.md`.
- Hardware: RTX 4070 Ti SUPER, CC 8.9, Triton 3.6.0, PyTorch 2.10.0+cu130.
- Treat these measurements as directional. They validate rule shape, not
  universal constants.

### Evidence Map

| Kernel type | Credible source anchors | Local reference data |
|---|---|---|
| Streaming / elementwise / copy | CUDA Best Practices thread/block heuristics: warp-multiple block sizes, `128-256` threads/block starting range, and smaller CTAs when block-wide synchronization latency matters. Nsight Compute: `MemoryWorkloadAnalysis`, scheduler issue/eligible warps, and sectors/request identify whether memory access quality or latency hiding is the actual bottleneck. | `workspace/runs/run_029/warp_count_reference_bench.py` triad sweep, summarized in `workspace/runs/run_029/warp_count_reference_bench_summary.md`: `1-8` warps changed throughput only by `~1.4%`. |
| Row reduction / scan / row softmax | CUDA Best Practices occupancy guidance: active warps hide latency, but higher occupancy is not automatically better. Triton fused softmax tutorial: distributes wide rows across `num_warps`, then derives persistent program count from register/shared-memory occupancy and `num_warps`. | `run_029` row-sum and persistent-softmax sweeps: `1/4` warps were effectively tied for row sum; persistent softmax was almost insensitive to `1-8` warps at the tested row width. |
| GEMM / MMA / convolution-like dense tiles | Triton matmul tutorial autotunes `num_warps=2/4/8` across tile families; Triton production matmul flags compute `num_warps` from `block_m * block_n`, with a larger floor for persistent kernels. Nsight Compute tensor-pipe and launch-resource sections validate whether a wider tile actually feeds MMA throughput. | `run_029` GEMM sweep: `128x64` was best at `4` warps, while `128x256` required `8` warps to avoid underfeeding the large tile. |
| Attention / producer-consumer | Triton fused attention tutorial uses tile-dependent `num_warps` and has separate producer/consumer-style paths for newer architectures; Nsight Compute `WarpStateStats`, shared-memory wavefront/conflict counters, and barrier/wait stalls determine whether extra warp groups overlap useful work or only enlarge the synchronization domain. | `run_029` attention-forward reference sweep: `BM=128,BN=32` was best at `4` warps; `2` underfed the tile and `8` added overhead on the tested Ada GPU. |
| Persistent kernels | CUDA Programming Guide cluster/persistent-style grid-stride guidance ties fixed grid size to SM count and desired occupancy. Triton fused softmax creates persistent programs from `NUM_SM * occupancy`; Triton persistent matmul launches at most an SM-scaled persistent grid and autotunes `4/8` warps. | `run_029` persistent softmax sweep: `1-8` warps were within `~0.5%`, so grid residency and tile ownership were more important than warp count alone. |

### General Decision Rules

1. Start from kernel style, then adjust with NCU. Do not infer the best warp
   count from occupancy alone.
2. Prefer multiples of one warp. Avoid partial warps unless the API or layout
   forces them.
3. If block-wide barriers are hot, first reduce synchronization scope or split
   work into smaller CTAs before adding more warps to the same CTA.
4. If long-scoreboard or no-eligible-warp stalls dominate and resource limits
   allow more resident warps, increase block count, warp count, or both.
5. If tensor instructions are the intended main pipeline, pick warp count from
   MMA tile shape and tensor-core issue needs, then verify register pressure and
   occupancy.
6. If LSU/shared-memory instructions dominate and tensor instruction count is
   already comparable to a reference, reducing warp groups and shared-memory
   handoff is usually higher leverage than increasing warp count.

### Initial Strategy By Kernel Type

| Kernel type | Initial CTA / warp count | Expand when | Reduce when | Primary NCU checks | Reference evidence |
|---|---|---|---|---|---|
| Streaming / elementwise / copy | Start with `128-256` threads/block in CUDA C, or `1-4` Triton warps for one-dimensional blocks. | `long_scoreboard` or no-eligible-warp stalls are high, occupancy is low, and memory accesses are already coalesced. | Bandwidth is saturated, sectors/request are poor, L2 contention rises, or more warps only change noise-level timing. | `dram__throughput`, global load/store sectors, sectors/request, `sm__warps_active`, scheduler issue/eligible metrics. | Local triad: `1-8` warps changed only `~1.4%` (`593-602 GB/s`), so warp-count tuning was secondary to memory access quality. CUDA Best Practices recommends warp-multiple block sizes and `128-256` threads/block as an initial range. |
| Row reduction / scan / row softmax | Start with `1-4` warps per row/tile. Use more lanes only when row width is large enough to amortize cross-warp reduction. | Wide rows, high per-row reduction latency, low eligible warps, or a single warp has too much serial work. | Register pressure, shared-memory partial reductions, or barrier stalls rise; row width is small enough for warp-local reductions. | `smsp__warps_issue_stalled_short_scoreboard`, `barrier`, `launch__registers_per_thread`, shared wavefront/conflicts, achieved occupancy. | Local row-sum `8192x4096`: `1` and `4` warps were effectively tied; `8` was not better. Triton fused softmax uses `num_warps=8` for wide rows but computes persistent program occupancy as a function of `num_warps`, registers, and shared memory. |
| GEMM / MMA / convolution-like dense tiles | Start with `4` warps for common `64/128`-sized MMA tiles. Use `2` for small-M/small-N tiles and `8` for large `128x256`, `256x128`, FP8, or high-throughput persistent tiles. | Tensor pipe is underfed, MMA tile is large, or autotuned/open-source references use larger warp groups for the same tile family. | Registers/shared memory cut CTA residency too far, `not_selected` or barrier stalls rise, or smaller tiles already saturate tensor throughput. | `sm__inst_executed_pipe_tensor.sum`, tensor throughput, `launch__registers_per_thread`, `launch__shared_mem_per_block`, occupancy limiters, eligible warps. | Triton matmul autotune spans `2/4/8` warps: small tiles include `2`, mainstream `128x64/128x128` use `4`, and large `128x256`/FP8 configs use `8`. Local GEMM reproduced this: `128x64` best at `4`, while `128x256` needed `8`. |
| Attention / producer-consumer | Start with `4` warps/block for forward attention-style CTAs on Ampere/Ada unless there is strong evidence for a wider warp-specialized pipeline. | Distinct producer/consumer roles overlap real work, barrier stalls are low, and larger tiles reduce global/shared handoff without excessive registers. | `barrier`, `wait`, shared-memory wavefronts, or LSU instructions dominate; owner/helper warp groups spend time waiting; tensor instruction count already matches reference. | `smsp__warps_issue_stalled_barrier`, `smsp__warps_issue_stalled_wait`, shared load/store wavefronts, shared bank conflicts, LSU instructions, tensor instructions, block size/grid size. | Triton fused attention autotunes `[4,8]` and keeps a reproducible `4`-warp config. Local attention-forward reference: `BM=128,BN=32` was `1.002 ms` at `4` warps, `1.094 ms` at `8`, and `2.670 ms` at `2`. Counterexample pattern: a much wider CTA can match tensor instruction count yet lose to barrier/shared/LSU overhead. |
| Persistent kernels | Start by choosing persistent grid size from SM count and desired residency, then choose `2-4` warps for reduction/softmax-like persistent kernels or `4-8` for persistent GEMM/TMA tiles. | Each persistent CTA owns a large tile or multiple tiles and has enough independent compute to amortize the persistent scheduler loop. | Too few CTAs leave SMs idle, register/shared-memory budget limits residency, or persistent loop overhead exceeds launch/scheduler savings. | `launch__grid_size`, waves per SM, eligible/active warps, `not_selected`, long scoreboard, register/shared-memory occupancy limits, per-CTA tile count. | Triton fused softmax creates persistent programs from `NUM_SM * occupancy`, where occupancy is divided by `num_warps`. Triton persistent matmul launches at most `NUM_SMS` programs and autotunes `4/8` warps. Local persistent softmax had `1-8` warps within `~0.5%`, so persistent grid/tile sizing mattered more than warp count alone. |

### Anti-Patterns

| Anti-pattern | Why it fails | Better action |
|---|---|---|
| Increasing warps because occupancy is low | Occupancy can be low for good reasons: high register reuse, large tensor tiles, or deliberate persistent scheduling. | Inspect eligible warps, issue slots, register pressure, and tensor/LSU mix first. |
| Using many producer/helper warps without overlap evidence | More warp groups enlarge the synchronization domain and often turn into shared-memory handoff plus barriers. | Prove producer/consumer overlap with NCU; otherwise collapse roles or keep data in registers/warp scope. |
| Retrying warp-count sweeps after a negative result | Adjacent warp-count variants rarely help unless the tile shape or dataflow changed. | Treat warp count and tile/dataflow as one strategy fingerprint in history. |
| Choosing `8/16+` warps for small row reductions | Cross-warp reduction and register pressure can exceed saved serial work. | Keep the reduction warp-local or use `1-4` warps until row width justifies expansion. |
| Treating reference `num_warps` as a copy-paste constant | Reference kernels encode tile shape, stage count, and hardware assumptions. | Match the reference dataflow and tile first; only then compare warp count. |

### Required Evidence Before Changing Warp Count

Record these in the proposal:

- Current and proposed `threads/block`, `warps/block`, grid size, tile shape, and
  per-thread work.
- The resource model: registers/thread, shared memory/block, expected CTAs/SM,
  and resident warps/SM.
- The bottleneck metric that justifies the change: eligible-warps shortage,
  long scoreboard, barrier, MIO throttle, tensor pipe underuse, or LSU/shared
  handoff.
- The expected dynamic coverage. A warp-count change that only affects tails,
  boundary tiles, or rare dispatch paths should be rejected unless it is a
  correctness probe for a larger redesign.
- At least one reference point: official/open-source kernel configuration,
  local benchmark sweep, or NCU comparison against a known-good reference.

### Sources

- CUDA Programming Guide, thread hierarchy / occupancy / launch bounds:
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>
- CUDA Runtime API occupancy helpers:
  <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html>
- CUDA Best Practices Guide, occupancy and thread/block heuristics:
  <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>
- Nsight Compute Profiling Guide, Occupancy, SchedulerStats, WarpStateStats,
  and stall reasons:
  <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>
- Triton fused softmax tutorial:
  <https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html>
- Triton matrix multiplication tutorial:
  <https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html>
- Triton fused attention tutorial:
  <https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html>
- Triton persistent matmul tutorial:
  <https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html>
- Triton production matmul option flags:
  <https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_details/opt_flags_details/opt_flags_nvidia.py>
- Local benchmark script and raw data:
  `workspace/runs/run_029/warp_count_reference_bench.py`,
  `workspace/runs/run_029/warp_count_reference_bench.json`

---

## Warp Specialization (Advanced)

**What**: Different warp groups within a block perform different roles (producer/consumer pattern).

**Example** (FlashAttention-style):
- Warp group 0: Load Q/K tiles (producer)
- Warp group 1: Compute QK^T GEMM (consumer)
- Warp group 2: Compute PV GEMM (consumer)
- Warp group 3: Online softmax correction

**Register rebalancing**: Each warp group may have different register needs. Allocate more registers to compute-heavy groups, fewer to data-movement groups.

**AVO technique**: Adjust per-warp-group register allocation to eliminate spills in bottleneck group:
- Original: 192/80/48 → correction warp spills to local memory
- Optimized: 184/88/56 → no spills, +2.1% performance

---

## Warp-Level Compute Primitives

Warp intrinsics are not just communication helpers. They are compute primitives
that often replace shared-memory staging and synchronization.

Useful tools:

- `__shfl_sync`, `__shfl_down_sync`, `__shfl_xor_sync` for reduction, scan, and
  broadcast
- `__ballot_sync`, `__all_sync`, `__any_sync` for warp-wide condition handling
- `__match_any_sync`, `__match_all_sync` for grouping identical values

Use them when:

- data exchange stays inside a warp
- shared-memory traffic is too high relative to compute
- bank conflicts or barrier overhead show up in NCU

For synchronization details, see `docs/sync_optimization.md`.

---

## Loop and ILP Optimization

Loop structure often decides how much instruction-level parallelism the compiler
can expose.

Useful transformations:

- loop unrolling for small fixed trip counts
- loop fusion when multiple passes touch the same data
- loop fission when one large loop creates too much register pressure
- loop interchange when it improves memory access order
- software pipelining in steady-state load / compute loops

Watch for the tradeoff:

- more unrolling can raise ILP
- too much unrolling can bloat register usage and instruction cache pressure

---

## Reduction and Scan Patterns

Common high-performance reduction structure:

1. reduce inside a warp with shuffle intrinsics
2. combine warp partials through shared memory or a narrow synchronization scope
3. finish block-level reduction
4. reduce across blocks with atomics or a second kernel when needed

Scan / prefix-sum kernels should also be treated as staged algorithms rather
than naive nested loops. The key concern is usually synchronization and memory
traffic, not arithmetic throughput.

---

## Fast Math and Strength Reduction

Use mathematical approximations only when the accuracy budget allows it.

Typical examples:

- reciprocal plus multiply instead of division
- `rsqrt` instead of `1 / sqrt`
- explicit FMA for fused multiply-add patterns
- `exp2`-style fast paths when algebra permits
- bit operations instead of integer divide / modulo by powers of two

Always validate:

- correctness tolerances
- numerical stability on adversarial inputs
- whether the faster instruction mix actually improved kernel time

---

## Compiler Guidance and Verification

Helpful tools and levers:

- `__restrict__` to reduce aliasing pessimism
- `__forceinline__` or `__noinline__` to manage code size and register pressure
- `--use_fast_math` or narrower math flags when appropriate
- PTX / SASS inspection to confirm the intended instruction path

When a change should have produced tensor-core, async-copy, or reduced-spill
codegen but performance does not move, inspect generated code instead of
guessing.

---

## Validation Checklist

Pair compute-oriented changes with:

- tensor-core utilization
- instruction mix
- registers per thread
- local-memory / spill behavior
- achieved occupancy
- kernel latency, not just a single micro metric
