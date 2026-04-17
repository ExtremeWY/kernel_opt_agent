# custom_platform

This directory is a platform-neutral kernel auto-optimization scaffold derived from the main `cuda-evolve` workflow.

It keeps the same experiment loop shape:

1. prepare environment
2. benchmark kernel correctness and performance
3. profile micro-architectural behavior
4. form a hypothesis
5. modify the kernel
6. keep or revert based on measured results
7. record experiment history

Unlike the CUDA-specific root project, the hardware-specific logic here is intentionally left as placeholders. You are expected to implement one concrete platform adapter, one concrete profiler backend, and the corresponding documentation.

## Layout

```text
custom_platform/
├── README.md
├── IMPLEMENTATION_GUIDE.md
├── program.md
├── pyproject.toml
├── docs/
├── kernel_configs/
├── kernels/
├── kernels_optimized/
├── knowledge/
├── memory/
├── platforms/
├── profilers/
├── references/
├── tools/
└── workspace/
```

## Core Design

- `platforms/` defines the execution and device abstraction.
- `profilers/` defines the profiling abstraction and normalized metric schema.
- `tools/bench.py` performs correctness checks, timing, and macro roofline classification through the platform adapter.
- `tools/profile.py` collects hardware counters through the profiler backend.
- `tools/run_loop.py` drives the experiment loop and writes `workspace/results.tsv`.
- `kernel_configs/` declares kernel shapes, dtypes, tolerances, and reference functions.
- `references/` contains correctness specifications independent of the target hardware.

## Current State

- The framework is complete enough to show how the pieces fit together.
- The custom hardware implementation is intentionally missing.
- Any method that depends on the real platform raises a descriptive placeholder error until you implement it.

## First Step

Read [IMPLEMENTATION_GUIDE.md](/home/et/cuda/cuda-kernel-agent/cuda-evolve-oss/custom_platform/IMPLEMENTATION_GUIDE.md) and then fill:

- `platforms/custom_platform/adapter.py`
- `profilers/custom_placeholder.py`
- `docs/*.md`
- `knowledge/custom_platform/OPTIMIZATION.md`

