# Memory Subsystem Optimization Reference

Quick-reference for GPU memory hierarchy optimization techniques.

---

## Global Memory Coalescing

**Principle**: When threads in a warp access contiguous memory addresses, the hardware merges requests into fewer transactions.

**Ideal**: 32 threads access 32 consecutive 4-byte addresses = 1 transaction (128 bytes).

**NCU indicators**:
- `memory_l2_theoretical_sectors_global` vs `memory_l2_theoretical_sectors_global_ideal`
- Ratio > 1.5 = significant coalescing waste

**Fixes**:
- Ensure thread i accesses element i (not strided)
- For struct-of-arrays vs array-of-structs: prefer SoA for GPU
- For matrix access: row-major + row-wise thread mapping
- Pad arrays to avoid unaligned accesses on non-power-of-2 sizes

---

## Vectorized Loads/Stores

**Principle**: Single instruction loads more data = fewer instructions for the same bandwidth.

| Type | Bytes per instruction | When to use |
|------|----------------------|-------------|
| `float` / `bf16` | 4 / 2 | Default, suboptimal for bandwidth |
| `float2` / `bf16_4` | 8 | 2x fewer instructions |
| `float4` / `bf16_8` | 16 | 4x fewer instructions, optimal for bandwidth-bound |

**Triton**: Increase `BLOCK_SIZE` (which increases elements per thread) to let the compiler generate wider loads.

**CUDA**: Use `reinterpret_cast<float4*>` for aligned loads.

**Requirements**: Data must be aligned to the vector width (16-byte aligned for float4).

---

## L1 / Shared Memory

**Capacity**: 192 KB per SM on Hopper (configurable split between L1 cache and shared memory).

**Bank conflicts**: Shared memory has 32 banks, 4 bytes wide. Two threads accessing the same bank (but different addresses) cause a conflict, serializing the access.

**Detect**: NCU metric `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`

**Fix bank conflicts**:
- Pad shared memory: `__shared__ float smem[N][N+1]` (add 1 column)
- Restructure access: ensure threads in a warp access different banks
- Swizzle: `smem[threadIdx.x ^ threadIdx.y]`

**Tiling pattern** (for reductions, GEMM):
1. Load tile from global to shared memory
2. `__syncthreads()`
3. Compute from shared memory
4. `__syncthreads()`
5. Repeat for next tile

---

## L2 Cache

**Capacity**: 50 MB (H100/H800), 40 MB (A100)

**Optimization strategies**:
- **Tile ordering**: Process tiles so that neighboring blocks access nearby memory regions
  - Swizzled tile ordering: `GROUP_SIZE_M` parameter in Triton
  - Row-major vs column-major tile walk
- **Eviction hints** (Triton):
  - `eviction_policy='evict_last'`: keep in cache (for data reused across tiles)
  - `eviction_policy='evict_first'`: evict immediately (for streaming writes)
- **Persistence control**: Set `CUDA_DEVICE_MAX_L2_PERSISTENT_LINES` for persistent data

**NCU indicators**:
- `lts__t_sector_hit_rate.pct` < 50% = poor L2 reuse
- Compare `dram__bytes_read.sum` to theoretical minimum bytes

---

## DRAM Bandwidth

**Peak**: 3352 GB/s (H100 SXM HBM3), 2039 GB/s (A100 SXM HBM2e)

**Practical peak**: ~80-90% of theoretical due to overhead.

**Maximizing utilization**:
- Coalesced accesses (see above)
- Vectorized loads (see above)
- Sufficient occupancy to saturate memory pipeline
- Minimize DRAM traffic: compute more per byte loaded

**NCU indicator**: `dram__throughput.avg.pct_of_peak_sustained_elapsed`

---

## Memory Access Pattern Anti-patterns

| Anti-pattern | NCU symptom | Fix |
|---|---|---|
| Strided access | High sectors/request | Transpose data layout or remap threads |
| Unaligned access | Extra transactions | Pad to alignment boundary |
| Random access | Low L1/L2 hit rate | Sort indices, use shared memory staging |
| Redundant loads | Excessive DRAM bytes | Cache in registers or shared memory |
| Write-after-read same address | Memory fence stalls | Restructure to avoid aliasing |

---

## Prefetching and Pipelining

**Software pipelining** (Triton `num_stages`):

```
Stage 0: Load tile N+1 from DRAM    |  Compute on tile N
Stage 1: Load tile N+2 from DRAM    |  Compute on tile N+1
```

Setting `num_stages=2` or `num_stages=3` overlaps loads with compute.

**Trade-off**: Each stage requires shared memory to buffer the loaded tile. Too many stages = excessive shared memory = lower occupancy.

**`cp.async`** (CUDA): Hardware-accelerated async copy from global to shared memory, bypassing register file.

**Rule of thumb**:
- Memory-bound + long_scoreboard stalls: increase `num_stages`
- Already at shared memory limit: reduce `num_stages` or reduce tile size

---

## Read-Only and Specialized Cache Paths

Read-only data can behave differently from general global-memory traffic.

Useful tools:

- `const __restrict__` for alias-free read-only pointers
- `__ldg()` on older or specific read-mostly paths
- texture / surface objects for 2D spatial-locality-heavy access patterns

Do not treat `__ldg()` as an automatic win on every modern GPU. Validate it
against actual cache behavior and kernel latency.

---

## L2 Persistence and Access Policy

If a small hot data region is reused across many blocks, L2 persistence can help.

Relevant mechanisms:

- `cudaAccessPolicyWindow`
- persisting vs streaming access properties

This is most useful when:

- the working set is small relative to L2
- the data is reused across many CTAs
- the normal launch order would otherwise evict it too early

If the kernel is streaming with little temporal reuse, forcing persistence may
hurt rather than help.

---

## Constant Memory

Constant memory is effective when:

- the data is small
- many threads read the same location at the same time

It is a poor choice when each lane accesses different constant addresses, since
that can serialize the access.

Typical use cases:

- small lookup tables
- scalar coefficient blocks
- compact parameter sets shared across the whole kernel

---

## Data Layout Choices

Data layout often dominates memory efficiency before any lower-level tuning.

Common rules:

- prefer structure-of-arrays over array-of-structures for warp-parallel field
  access
- pad rows or leading dimensions to preserve alignment
- reorder irregular problems when locality is poor

Good examples:

- CSR row bucketing for sparse workloads
- padding matrix leading dimensions for vectorized loads
- swizzled layout or block reordering for better cache reuse

---

## Host-to-Device and Runtime Memory Paths

These do not change kernel micro-architecture directly, but they matter for
end-to-end throughput.

Useful tools:

- pinned host memory for faster async transfers
- `cudaMallocAsync` / `cudaFreeAsync`
- CUDA memory pools
- `cudaMemPrefetchAsync` and `cudaMemAdvise` for Unified Memory workloads
- zero-copy only for very small or one-touch datasets

Keep these separate from kernel-only conclusions in experiment notes.

---

## Validation Checklist

Pair memory-oriented changes with:

- sectors per request / coalescing quality
- L1 and L2 hit rates
- DRAM throughput
- shared-memory bank conflict counters
- kernel latency and effective throughput

If a local metric improves but total runtime gets worse, the bottleneck probably
moved rather than disappeared.
