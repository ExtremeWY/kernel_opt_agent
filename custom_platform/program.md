# custom_platform Program

You are an autonomous kernel optimization agent working on a non-CUDA hardware platform.

Follow this protocol strictly.

## Available Kernels

The `kernels/` directory contains baseline kernels. Do not modify them.

Optimized kernels are written to `kernels_optimized/`.

Each kernel module must export:

- `KERNEL_TYPE: str`
- `TARGET_PLATFORM: str`
- `kernel_fn(**inputs)`

## Setup Phase

1. Run `python tools/prepare.py`.
2. Read `knowledge/custom_platform/OPTIMIZATION.md`.
3. Read `workspace/MEMORY.md`.
4. Copy one baseline kernel to `kernel.py`.
5. Read the matching `kernel_configs/` files.
6. Read the matching `references/` implementation.
7. Read the platform docs under `docs/`.

## Experiment Loop

1. Benchmark with `python tools/bench.py`.
2. Profile with `python tools/profile.py`.
3. Form one focused hypothesis.
4. Modify `kernel.py`.
5. Benchmark again.
6. Keep only changes with correctness pass and meaningful improvement.
7. Record the result in `workspace/results.tsv`, `memory/<kernel_type>.md`, and `workspace/MEMORY.md`.
8. If a reusable pattern is found, update `knowledge/custom_platform/OPTIMIZATION.md`.

## Rules

- One hypothesis per experiment.
- Always ground the hypothesis in measured data.
- Use normalized profiler outputs, not raw vendor-specific counter names, when writing your reasoning.
- Preserve baseline kernels unchanged.

