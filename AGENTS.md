# Repository Guidelines

## Project Structure & Module Organization

This repository is `cuda-evolve`, an agent-driven CUDA/Triton kernel optimization harness. The active editable kernel lives at `kernel.py`; `kernel.cu` is used when a CUDA source variant is present. Baseline kernels are in `kernels/`, optimized outputs in `kernels_optimized/`, reference implementations in `references/`, and benchmark/input metadata in `kernel_configs/`. Automation lives in `tools/`, with `tools/bench.py` for correctness/performance and `tools/run_loop.py` for artifact-aware iterations. `docs/` and `CUDA_OPTIMIZATION.md` contain optimization guidance. `workspace/` stores generated run artifacts, logs, results, and profiler reports.

## Build, Test, and Development Commands

- `uv sync --extra dev`: install runtime and development dependencies.
- `.venv/bin/python tools/prepare.py`: validate CUDA, Python, and profiler setup; check `workspace/runtime_env.md` for machine-specific commands.
- `.venv/bin/python tools/bench.py`: run correctness checks and benchmark the active `kernel.py`.
- `.venv/bin/python tools/bench.py --quick`: run a faster benchmark pass for iteration.
- `.venv/bin/python tools/run_loop.py --hypothesis "increase tile size" --targeted-ncu`: run one experiment iteration and archive artifacts.
- `.venv/bin/ruff check .`: lint Python imports and style.

## Coding Style & Naming Conventions

Use Python 3.10+ syntax and keep lines at or below 120 characters, matching `pyproject.toml`. Ruff enforces `E`, `F`, `W`, and import ordering (`I`). Use snake_case for Python files, functions, and variables. Kernel modules should export `KERNEL_TYPE`, `kernel_fn`, `get_inputs`, `get_flops`, and `get_bytes` when following the standard harness contract. Keep changes focused on `kernel.py`, `kernel.cu`, configs, or docs unless framework behavior is intentionally changing.

## Testing Guidelines

Benchmark validation is the primary test path. Run `tools/bench.py` before and after kernel changes, and use `--quick` only for early iteration. For new kernels, add a PyTorch reference in `references/` and matching config in `kernel_configs/` so correctness tolerances and shape sweeps are explicit. If adding unit tests, place them under `tests/` and run with `.venv/bin/pytest`.

## Commit & Pull Request Guidelines

Recent history uses short imperative subjects, sometimes with Conventional Commit prefixes such as `fix:` and `docs:`. Prefer `type: concise summary` when possible, for example `fix: preserve benchmark artifact output`. Pull requests should describe the kernel or tool change, include benchmark commands and key results, note CUDA/GPU environment details, and link relevant `workspace/runs/...` artifacts when performance claims depend on them.

## Agent-Specific Instructions

Treat `workspace/runs/`, `workspace/results.tsv`, and profiler exports as generated experiment evidence. Do not delete prior artifacts unless explicitly requested. For optimization work, read `program.md`, inspect current run history, make one focused hypothesis per iteration, and record outcomes in the workspace logs.
