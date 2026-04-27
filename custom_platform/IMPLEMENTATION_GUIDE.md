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

## 2. Generic Workflow Files You Now Need To Maintain

### `tools/preflight.py`

Fill or extend:

- adapter readiness checks
- kernel file checks
- vendor CLI presence checks
- platform-specific warnings that should appear in `workspace/preflight_check.{json,md}`

### `tools/bench.py`

Keep the normalized output schema intact. Only change this file if the platform needs extra generic fields.

The benchmark JSON should continue to provide:

- correctness
- kernel timing
- reference timing
- achieved compute and memory throughput
- roofline peaks
- bottleneck classification
- structured `error`

### `tools/profile.py`

Keep this wrapper generic. Put vendor-specific collection and parsing inside the profiler backend whenever possible.

This tool should continue to write:

- `profile_report.txt` or the platform equivalent report path
- `profile_summary.txt`
- `profile_details.txt`
- normalized metrics on stdout

### `tools/run_loop.py`

This file now owns the artifact-aware experiment loop. Update it only if your platform requires extra keep/revert constraints.

It is responsible for:

- `workspace/runs/run_<timestamp>/`
- `iter_vN/`
- `run_manifest.json`
- `final_summary.md`
- `iteration_summary.md`
- proposal template materialization
- strategy memory updates

### `tools/strategy_memory.py`

Keep the schema stable unless the generic experiment semantics truly need to change.

The framework expects:

- `positive`
- `negative`
- `rejected`
- `strategy_tags`
- `strategy_fingerprint`
- blocked and preferred constraints

### `tools/iteration_report.py`

This file renders markdown from the structured artifacts. Update it if you add new generic metrics or new report sections.

## 3. Configuration Files To Fill

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

## 4. Documentation To Add

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

### `docs/experiment_artifacts.md`

Keep this file aligned with the artifact directory layout produced by `tools/run_loop.py`.

### `docs/strategy_memory.md`

Keep this file aligned with:

- `workspace/strategy_memory/global_strategy_memory.json`
- `tools/strategy_memory.py`
- the proposal tagging workflow

## 5. Knowledge Base To Maintain

### `knowledge/custom_platform/OPTIMIZATION.md`

This file is the platform-specific optimization memory for the agent. Populate it with:

- successful patterns
- failed patterns
- cross-kernel heuristics
- architecture-specific dos and donts

Keep the format concise and greppable so the agent can use it during hypothesis generation.

## 6. Workspace Files And Templates

These generic files are now part of the scaffold and usually should not be removed:

- `workspace/results.tsv`
- `workspace/preflight_check.json`
- `workspace/preflight_check.md`
- `workspace/runs/.gitkeep`
- `workspace/optimization_proposal.template.md`
- `workspace/strategy_memory/global_strategy_memory.json`

You may change the template text, but keep the purpose of each file unchanged.

## 7. Hardware Documentation You Need

Before implementation, collect these documents from the target hardware vendor:

- programming guide
- runtime API reference
- compiler or kernel language reference
- profiler user guide
- profiler metric reference
- architecture tuning guide
- device specification sheet with peak compute, bandwidth, cache sizes, and memory size
- matrix engine or tensor engine instruction reference if available

## 8. Minimum Path To First Working Port

1. Implement `platforms/custom_platform/adapter.py`.
2. Implement `profilers/custom_placeholder.py`.
3. Replace `kernels/example_op.py` with one real kernel.
4. Update `kernel_configs/example_op.*` to match the real kernel.
5. Fill `docs/*.md` with platform terminology and profiler metric mapping.
6. Run `python custom_platform/tools/prepare.py --allow-placeholder` first, then without the flag after implementation.
7. Copy a real kernel to `custom_platform/kernel.py` and run the benchmark/profile loop.
8. Verify `workspace/runs/run_<timestamp>/` contains complete artifacts and strategy memory updates.

The scaffold also ships with `mock_platform`, which is useful for validating:

- manifest generation
- artifact layout
- benchmark JSON output
- profiler summary generation
- strategy memory updates

Do not use the mock backend as a performance baseline for the real platform.

## 9. Files You Usually Do Not Need To Change

These should remain mostly generic:

- `platforms/base.py`
- `profilers/base.py`
- `kernel_configs/__init__.py`
- `references/__init__.py`
- `workspace/MEMORY.md`

If you need to edit them, the abstraction boundary is probably still too weak.
