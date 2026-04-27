# GPU Architecture Quick Reference

Key specifications and quirks for optimization, organized by architecture.

---

## Hopper (SM 9.0) — H100, H800, H200

**Compute**:
- 132 SMs
- FP16/BF16 tensor cores: 989.5 TFLOPS (H100 SXM)
- FP32: ~60 TFLOPS (CUDA cores only)
- TMA (Tensor Memory Accelerator): hardware-managed async copy with address generation

**Memory**:
- HBM3: 80 GB, 3352 GB/s (H100 SXM) / 4800 GB/s (H200)
- L2 cache: 50 MB
- L1/shared memory: 256 KB per SM (configurable split)
- Register file: 65536 32-bit registers per SM

**Key features**:
- **TMA**: `tl.make_block_ptr` (Triton) or `cp.async.bulk` (CUDA). Handles multi-dimensional addressing and boundary checks in hardware. Reduces register pressure from pointer arithmetic.
- **Thread Block Clusters**: group blocks across SMs for distributed shared memory
- **Warp specialization**: different warp groups can be assigned different roles with independent register budgets
- **Asynchronous barriers**: `cp.async` with `arrive`/`wait` pattern for producer-consumer overlap

**Quirks**:
- `tl.make_tensor_descriptor` requires specific memory layouts (e.g., column-major B for GEMM)
- `flatten=True` only works with `tl.make_tensor_descriptor`, NOT with `tl.make_block_ptr`
- L2 partition camping can occur with certain grid launch patterns — use swizzled tile ordering
- Persistent kernels benefit from TMA but need careful barrier management
- Hopper tensor cores are most attractive when the kernel is truly matmul-like and the active shape fills MMA tiles well; small-M or decode-like regimes can lose the throughput advantage to padding overhead and launch inefficiency

---

## Ada Lovelace (SM 8.9) — RTX 4090, RTX 4070 Ti Super, L40S

**Compute**:
- 128 SMs (4090), 142 SMs (L40S)
- FP16 tensor cores: 330 TFLOPS (4090)
- Third-gen RT cores

**Memory**:
- GDDR6X: 24 GB, 1008 GB/s (4090) / 48 GB, 864 GB/s (L40S)
- L2 cache: 72 MB (4090) / 48 MB (L40S)

**Quirks**:
- Large L2 makes tile ordering less critical than on HBM GPUs
- GDDR6X has different latency characteristics than HBM
- Consumer cards: no NVLink, no MIG

### RTX 4070 Ti Super Quick Specs

| Component | RTX 4070 Ti Super | Notes |
|-----------|-------------------|-------|
| Compute Capability | 8.9 (`sm_89`) | Ada Lovelace, TSMC 4N 5nm |
| SMs | 66 | AD103-275 chip, roughly 84% of full AD103 |
| CUDA Cores | 8,448 | 128 per SM |
| Tensor Cores | 264 | 4th gen, FP8 support |
| RT Cores | 66 | 3rd gen |
| FP16 Tensor Core Dense | 176.5 TFLOPS | Dense tensor-core peak |
| L2 Cache | 48 MB | Larger than A100's 40 MB |
| Shared Memory | 128 KB/SM | Max 100 KB user-configurable |
| Registers | 64K 32-bit/SM | 255 per thread max |
| Memory | 16 GB GDDR6X | 256-bit interface |
| Memory Bandwidth | 672 GB/s | 21 Gbps effective |
| TDP | 285 W | Consumer board power limit |
| Max Threads/SM | 2,048 | 64 warps |
| Max Threads/Block | 1,024 | 32 warps |
| Warp Size | 32 | Unchanged from prior NVIDIA architectures |
| PCIe | 4.0 x16 | About 31.5 GB/s host link bandwidth |

**Optimization notes**:
- Relative to H100/A100, the much lower memory bandwidth means bandwidth-bound kernels hit the ridge point sooner.
- The 48 MB L2 is large for a consumer card and often helps persistent or cache-friendly tiling strategies.
- Since this is a consumer Ada card, assume no NVLink and no MIG when planning multi-GPU or partitioning workflows.
- For matmul-like kernels in small-M or decode-like regimes, compare CUDA-core and tensor-core-with-padding paths explicitly instead of assuming tensor cores win.

---

## Ampere (SM 8.0) — A100

**Compute**:
- 108 SMs
- FP16/BF16 tensor cores: 312 TFLOPS (A100 SXM)
- TF32 tensor cores: 156 TFLOPS

**Memory**:
- HBM2e: 80 GB, 2039 GB/s (A100 SXM)
- L2 cache: 40 MB
- L1/shared memory: 192 KB per SM (configurable)

**Key differences from Hopper**:
- No TMA — must use `cp.async` manually
- No Thread Block Clusters
- `cp.async` supports up to 16 bytes per thread (float4)
- Async copy is SM-initiated, not hardware-addressed like TMA

**Quirks**:
- MIG (Multi-Instance GPU): can partition into 7 instances
- `__shfl_sync` works on full warp only (no sub-warp shuffle)
- A100-80GB PCIe has lower bandwidth (1935 GB/s) than SXM variant

---

## Blackwell (SM 10.0) — B200, B100

**Compute**:
- 2250 TFLOPS FP16 (B200)
- Fifth-gen tensor cores with FP4 support
- 2048 warp registers per SM (fixed allocation across warp groups)

**Memory**:
- HBM3e: 192 GB, 8000 GB/s (B200)
- L2 cache: 64 MB

**Key features**:
- **Fixed warp register budget**: 2048 registers per SM divided among warp groups. Must be explicitly balanced (e.g., 184/88/56 across 3 groups).
- **FP4 tensor cores**: 2x throughput over FP8
- **Second-gen TMA**: enhanced async copy capabilities

**Quirks**:
- Register spilling is catastrophic: local memory latency is very high relative to the fast tensor cores
- Register rebalancing between warp groups is a key optimization lever (AVO: +2.1%)
- The 2048-register budget is NOT per-warp — it's shared across all warp groups in the SM

---

## Common Specifications Lookup

| GPU | Arch | SMs | FP16 TFLOPS | BW (GB/s) | VRAM | L2 (MB) | Ridge Point (FLOPs/Byte) |
|-----|------|-----|-------------|-----------|------|---------|--------------------------|
| B200 | Blackwell | - | 2250 | 8000 | 192 GB | 64 | ~281 |
| H200 | Hopper | 132 | 989.5 | 4800 | 141 GB | 50 | ~206 |
| H100 SXM | Hopper | 132 | 989.5 | 3352 | 80 GB | 50 | ~295 |
| H800 | Hopper | 132 | 989.5 | 3352 | 80 GB | 50 | ~295 |
| A100 SXM | Ampere | 108 | 312 | 2039 | 80 GB | 40 | ~153 |
| 4090 | Ada | 128 | 330 | 1008 | 24 GB | 72 | ~327 |
| 4070 Ti Super | Ada | 66 | 176.5 | 672 | 16 GB | 48 | ~263 |
| L40S | Ada | 142 | 362 | 864 | 48 GB | 48 | ~419 |
| L4 | Ada | 58 | 121 | 300 | 24 GB | 48 | ~403 |

Ridge point = FP16 TFLOPS * 1e12 / (BW * 1e9) = arithmetic intensity threshold.
Below ridge point: memory-bound. Above ridge point: compute-bound.
