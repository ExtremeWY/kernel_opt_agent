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

1. Run `.venv/bin/python tools/prepare.py`.
   For a local dry run without real hardware, use `--platform mock_platform`.
   If `workspace/runtime_env.md` selects a different interpreter on this machine, use those commands for all subsequent tool invocations.
2. Read `knowledge/custom_platform/OPTIMIZATION.md`.
3. Read `workspace/MEMORY.md`.
4. Read `workspace/strategy_memory/global_strategy_memory.json`.
5. Copy one baseline kernel to `kernel.py`.
6. Read the matching `kernel_configs/` files.
7. Read the matching `references/` implementation.
8. Read the platform docs under `docs/`, especially `docs/experiment_artifacts.md` and `docs/strategy_memory.md`.

## Experiment Loop

1. Run `.venv/bin/python tools/run_loop.py --hypothesis "<one focused change>"`.
   For a full mock success path, use `--platform mock_platform`.
2. Inspect `workspace/runs/run_<timestamp>/preflight_check.md`.
3. Read `iter_vN/benchmark_result.json` and `iter_vN/profile_summary.txt`.
4. Create or update `iter_vN/optimization_proposal.md`.
5. Keep only one focused hypothesis per iteration.
6. Modify `kernel.py`.
7. Re-run benchmark and profile through the loop.
8. Keep only changes with correctness pass, complete profiling evidence, and meaningful improvement.
9. Record the result in `workspace/results.tsv`, `memory/<kernel_type>.md`, and `workspace/MEMORY.md`.
10. If a reusable pattern is found, update `knowledge/custom_platform/OPTIMIZATION.md`.

## Required Artifacts

Each run should produce:
- `run_manifest.json`
- `final_summary.md`
- `preflight_check.json`
- `preflight_check.md`

Each iteration should produce:
- `benchmark_result.json`
- `benchmark.stdout.txt`
- `benchmark.stderr.txt`
- `profile_report.txt`
- `profile_summary.txt`
- `profile_details.txt`
- `optimization_proposal.md`
- `iteration_summary.md`

## Strategy Memory Rules

- Every proposal must include `## Strategy tags`.
- Use normalized profiler outputs, not raw vendor-specific counter names, when writing your reasoning.
- Avoid fingerprints listed in the current scope as blocked.
- Prefer fingerprints listed in the current scope as preferred.
- Record the outcome as `positive`, `negative`, or `rejected`.

## Rules

- One hypothesis per experiment.
- Always ground the hypothesis in measured data.
- Preserve baseline kernels unchanged.
- Do not treat placeholder platform errors as performance findings. Replace the adapter and profiler backend first.
- The `mock_platform` path is for workflow rehearsal only. Do not treat its metrics as hardware truth.
