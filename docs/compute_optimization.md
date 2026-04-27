# Compute Optimization Reference

Quick-reference for maximizing compute throughput on NVIDIA GPUs.

---

## Tensor Core Utilization

**What**: Tensor cores perform matrix multiply-accumulate (MMA) on small matrix tiles (e.g., 16x16x16) in a single cycle.

**Throughput**: 989.5 TFLOPS FP16 on H100 (with tensor cores) vs ~60 TFLOPS FP32 (without).

**Requirements for tensor core usage**:
- Matrix dimensions must be multiples of 16 (FP16) or 8 (TF32)
- Data must be in FP16, BF16, TF32, FP8, or INT8
- Use `tl.dot` (Triton) or `wmma`/`mma.sync` (CUDA) intrinsics

**Common mistake**: Casting inputs to FP32 before `tl.dot` forces scalar FMA path (16x slower).

**NCU indicator**: `sm__inst_executed_pipe_tensor.sum` as fraction of total instructions.

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

**When low occupancy is OK**: Compute-bound kernels with high IPC. If tensor cores are >80% utilized, occupancy doesn't matter.

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
