# Custom Platform Implementation Guide

This document describes what must be added or modified to turn this scaffold into a working AI-agent-driven kernel auto-optimization framework for one concrete hardware platform.

## 1. Mandatory Code Files

### `platforms/custom_platform/adapter.py`

This is the primary hardware integration point. You must implement:

- environment validation
- device enumeration
- hardware property discovery
- synchronization
- peak memory statistics
- kernel timing
- roofline peak specification lookup
- optional kernel build/loading helpers

Expected outputs:

- a real `DeviceSpec`
- a benchmark timing path that measures kernel runtime on the target platform
- accurate peak compute and memory bandwidth values for roofline analysis

Typical platform-specific dependencies:

- vendor runtime API
- vendor compiler or JIT
- device management CLI
- framework bridge such as PyTorch custom backend, SYCL, HIP, CANN, or a vendor SDK

### `profilers/custom_placeholder.py`

This file must be replaced with a real profiling backend. You must implement:

- profile collection
- raw report parsing
- normalized metric generation
- high-level findings and actions

The framework expects normalized metrics such as:

- `profile_compute_util`
- `profile_memory_util`
- `profile_l1_hit_rate`
- `profile_l2_hit_rate`
- `profile_occupancy`
- `profile_register_pressure`
- `profile_spill_pressure`
- `profile_top_stall`
- `profile_vectorization_efficiency`
- `profile_coalescing_efficiency`

### `kernels/*.py`

Replace the sample placeholder kernel with real target-platform kernels. Each kernel module should export:

- `KERNEL_TYPE`
- `TARGET_PLATFORM`
- `kernel_fn(**inputs)`

If the target platform requires compilation or module loading, add helper functions inside the kernel module or centralize them in the platform adapter.

## 2. Configuration Files To Fill

### `kernel_configs/*.toml`

For each kernel type, define:

- benchmark sizes
- dtypes
- correctness tolerances
- edge-case sizes
- metadata such as multi-output kernels

### `kernel_configs/*.py`

For each kernel type, implement:

- `input_generator`
- `reference_fn`
- `flops_fn`
- `bytes_fn`

These files are platform-neutral. They define the optimization problem and correctness contract.

## 3. Documentation To Add

### `docs/arch_notes.md`

Add platform architecture notes such as:

- compute units, cores, or tiles
- execution group model such as warp, wavefront, subgroup, or thread block analog
- matrix acceleration units
- cache and scratchpad hierarchy
- register limits
- occupancy or residency limits

### `docs/memory_optimization.md`

Map memory tuning concepts to the target platform:

- global memory transaction rules
- alignment rules
- vectorized access width
- scratchpad or shared-memory equivalent
- cache hint mechanisms
- async copy or prefetch capabilities

### `docs/compute_optimization.md`

Describe:

- matrix instruction set
- supported low-precision data types
- reduction patterns
- instruction throughput bottlenecks
- divergence model

### `docs/stall_reasons.md`

Translate profiler stall metrics into human-usable meanings:

- what the stall means
- what causes it
- how to mitigate it

## 4. Knowledge Base To Maintain

### `knowledge/custom_platform/OPTIMIZATION.md`

This file is the platform-specific optimization memory for the agent. Populate it with:

- successful patterns
- failed patterns
- cross-kernel heuristics
- architecture-specific dos and donts

Keep the format concise and greppable so the agent can use it during hypothesis generation.

## 5. Tooling To Wire In

### `tools/prepare.py`

You may need to extend it to check:

- vendor compiler
- profiler CLI
- runtime libraries
- backend framework installation

### `tools/bench.py`

Only modify this file if the target platform needs extra benchmark outputs beyond the normalized schema already supported by `PlatformAdapter`.

### `tools/profile.py`

Only modify this file if the normalized schema needs new generic fields. Prefer putting platform-specific parsing inside the profiler backend instead.

### `tools/run_loop.py`

Modify only if your keep/revert policy depends on platform-specific constraints not already represented by the generic metrics.

## 6. Hardware Documentation You Need

Before implementation, collect these documents from the target hardware vendor:

- programming guide
- runtime API reference
- compiler or kernel language reference
- profiler user guide
- profiler metric reference
- architecture tuning guide
- device specification sheet with peak compute, bandwidth, cache sizes, and memory size
- matrix engine or tensor engine instruction reference if available

## 7. Minimum Path To First Working Port

1. Implement `platforms/custom_platform/adapter.py`.
2. Implement `profilers/custom_placeholder.py`.
3. Replace `kernels/example_op.py` with one real kernel.
4. Update `kernel_configs/example_op.*` to match the real kernel.
5. Fill `docs/*.md` with platform terminology and profiler metric mapping.
6. Run `python custom_platform/tools/prepare.py --allow-placeholder` first, then without the flag after implementation.
7. Copy a real kernel to `custom_platform/kernel.py` and run the benchmark and profiler flow.

## 8. Files You Usually Do Not Need To Change

These should remain mostly generic:

- `platforms/base.py`
- `profilers/base.py`
- `kernel_configs/__init__.py`
- `references/__init__.py`
- `workspace/MEMORY.md`
- `workspace/results.tsv`

If you need to edit them, the abstraction boundary is probably still too weak.

