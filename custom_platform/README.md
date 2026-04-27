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
- `tools/preflight.py` writes `workspace/preflight_check.{json,md}` and provides a platform-neutral readiness report.
- `tools/bench.py` performs correctness checks, timing, and macro roofline classification through the platform adapter.
- `tools/profile.py` collects normalized profiler outputs through the profiler backend.
- `tools/strategy_memory.py` maintains structured `positive/negative/rejected` strategy history.
- `tools/iteration_report.py` renders iteration and run summaries from saved artifacts.
- `tools/run_loop.py` drives the experiment loop and writes per-run artifacts under `workspace/runs/`.
- `kernel_configs/` declares kernel shapes, dtypes, tolerances, and reference functions.
- `references/` contains correctness specifications independent of the target hardware.

## Artifact-Aware Workflow

Each run is stored under `workspace/runs/run_<timestamp>/`.

Per run:
- `run_manifest.json`
- `final_summary.md`
- `preflight_check.json`
- `preflight_check.md`

Per iteration:
- `benchmark_result.json`
- `benchmark.stdout.txt`
- `benchmark.stderr.txt`
- `profile_report.txt`
- `profile_summary.txt`
- `profile_details.txt`
- `profile.stdout.txt`
- `profile.stderr.txt`
- `optimization_proposal.md`
- `iteration_summary.md`

## Strategy Memory

Structured strategy memory lives in `workspace/strategy_memory/global_strategy_memory.json`.

- `positive`: faster than the previous comparable attempt
- `negative`: valid but slower or equal
- `rejected`: correctness failure, profiling failure, or incomplete evidence

The experiment loop writes:
- normalized `strategy_tags`
- stable `strategy_fingerprint`
- blocked and preferred fingerprints for the next iteration

## Optimization References

- [docs/arch_notes.md](/home/et/cuda/cuda-kernel-agent/cuda-evolve-oss/custom_platform/docs/arch_notes.md)
- [docs/compute_optimization.md](/home/et/cuda/cuda-kernel-agent/cuda-evolve-oss/custom_platform/docs/compute_optimization.md)
- [docs/memory_optimization.md](/home/et/cuda/cuda-kernel-agent/cuda-evolve-oss/custom_platform/docs/memory_optimization.md)
- [docs/stall_reasons.md](/home/et/cuda/cuda-kernel-agent/cuda-evolve-oss/custom_platform/docs/stall_reasons.md)
- [docs/experiment_artifacts.md](/home/et/cuda/cuda-kernel-agent/cuda-evolve-oss/custom_platform/docs/experiment_artifacts.md)
- [docs/strategy_memory.md](/home/et/cuda/cuda-kernel-agent/cuda-evolve-oss/custom_platform/docs/strategy_memory.md)
- [knowledge/custom_platform/OPTIMIZATION.md](/home/et/cuda/cuda-kernel-agent/cuda-evolve-oss/custom_platform/knowledge/custom_platform/OPTIMIZATION.md)

## Current State

- The framework now includes preflight checks, per-run artifact directories, structured strategy memory, and markdown summaries.
- The custom hardware implementation is still intentionally missing.
- A software-only `mock_platform` is included so the full workflow can be rehearsed without vendor hardware.
- Any method that depends on the real platform raises a descriptive placeholder error until you implement it.

## Local Dry Run

Use the mock path when you want to validate the workflow itself:

- `python custom_platform/tools/prepare.py --platform mock_platform`
- `python custom_platform/tools/run_loop.py --platform mock_platform --hypothesis "mock dry run"`

## First Step

Read [IMPLEMENTATION_GUIDE.md](/home/et/cuda/cuda-kernel-agent/cuda-evolve-oss/custom_platform/IMPLEMENTATION_GUIDE.md) and then fill:

- `platforms/custom_platform/adapter.py`
- `profilers/custom_placeholder.py`
- `docs/*.md`
- `knowledge/custom_platform/OPTIMIZATION.md`
