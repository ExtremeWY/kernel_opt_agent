# cuda-evolve

Autonomous GPU kernel optimization system driven by AI agents.

## Overview

cuda-evolve lets AI agents (Claude, Codex, etc.) autonomously profile, analyze, and optimize GPU kernels through iterative experimentation. Given a kernel, the agent:

1. **Checks** the environment with preflight and records blocking issues
2. **Profiles** the kernel to understand performance characteristics (compute-bound vs memory-bound)
3. **Modifies** the kernel code
4. **Benchmarks** the modified kernel against the reference implementation with structured JSON output
5. **Profiles** the kernel with targeted/full Nsight Compute artifact capture
6. **Decides** whether to keep or revert the change
7. **Logs** the result to `workspace/MEMORY.md`, `workspace/results.tsv`, and `workspace/runs/`
8. **Repeats** until satisfactory performance is achieved

## Project Structure

```
cuda-evolve/
├── program.md              # Agent workflow protocol
├── CUDA_OPTIMIZATION.md    # Agent-maintained optimization knowledge base
├── workspace/              # Runtime outputs and shared logs
│   ├── MEMORY.md           # Global optimization log (shared across sessions)
│   ├── preflight_check.*   # Environment readiness report
│   ├── results.tsv         # Experiment results tracking
│   ├── runs/               # Per-run / per-iteration experiment artifacts
│   ├── strategy_memory/    # Structured strategy fingerprints and outcomes
│   └── ncu_reports/        # Shared NCU profiling exports
├── tools/
│   ├── bench.py            # Benchmark harness & correctness checking
│   ├── merge_results.py    # Merge benchmark / result files
│   ├── ncu_profile.py      # Nsight Compute profiling
│   ├── prepare.py          # Environment preparation & validation
│   ├── preflight.py        # Detailed environment and tool checks
│   ├── run_loop.py         # Agent / optimization loop driver
│   ├── iteration_report.py # Iteration / run summary rendering
│   └── strategy_memory.py  # Strategy fingerprint memory helpers
├── kernel.py               # The kernel being optimized (editable by agent)
├── references/             # Reference implementations (per-kernel modules)
├── kernels/                # Baseline kernels (READ-ONLY, bring your own)
├── kernels_optimized/      # Agent-optimized kernels (output)
├── docs/                   # Optimization references and artifact docs
├── memory/                 # Per-kernel experiment logs
└── pyproject.toml
```

## Quick Start

```bash
# Install dependencies
uv sync

# Prepare the environment
uv run tools/prepare.py

# Add your kernel to kernels/ (see "Adding Your Own Kernels" below)
# Then select it for optimization:
cp kernels/your_kernel.py kernel.py

# Run a benchmark
uv run tools/bench.py

# Run an artifact-aware experiment iteration
uv run tools/run_loop.py --hypothesis "increase tile size" --targeted-ncu --full-ncu

# Or kick off the agent loop (via your AI agent):
# "Read program.md and start optimizing the kernel."
```

## How It Works

The agent reads `program.md` which defines the experimental protocol. Each iteration:

1. Runs `tools/preflight.py` or `tools/prepare.py` to validate the environment
2. Examines profiling data to understand the bottleneck
3. Consults `CUDA_OPTIMIZATION.md` and `docs/` for optimization strategies
4. Makes a focused change to `kernel.py`
5. Runs `tools/run_loop.py` to benchmark, profile, and archive artifacts
6. Records the outcome in `workspace/MEMORY.md`, `workspace/results.tsv`, and `workspace/runs/run_xxx/iter_vN/`

## Optimization References

The `docs/` directory is the curated reference set for agents and manual runs.
The most commonly used references are:

- `docs/compute_optimization.md`
- `docs/memory_optimization.md`
- `docs/sync_optimization.md`
- `docs/stall_reasons.md`
- `docs/triton_optimization.md`
- `docs/cutlass_optimization.md`
- `docs/arch_notes.md`

## Adding Your Own Kernels

To add a kernel for the agent to optimize:

1. **Create the kernel module** at `kernels/your_kernel.py` exporting:
   - `KERNEL_TYPE: str` -- identifier (e.g. `"rms_norm"`)
   - `kernel_fn(**inputs) -> torch.Tensor` -- the kernel to optimize
   - `get_inputs() -> dict` -- generates sample inputs
   - `get_flops() -> int` -- total FLOPs for roofline analysis
   - `get_bytes() -> int` -- total bytes accessed for roofline analysis

2. **Add a reference implementation** under `references/` (pure PyTorch, used for correctness checking)

3. **Add a benchmark config** under `kernel_configs/` with test sizes, tolerances, input generator, and reference function

4. **Copy to `kernel.py`** and start optimizing:
   ```bash
   cp kernels/your_kernel.py kernel.py
   uv run tools/bench.py
   ```

## Requirements

- Python >= 3.10
- CUDA-capable GPU
- CUDA Toolkit
- [uv](https://github.com/astral-sh/uv) package manager

## Experiment Artifacts

Each `tools/run_loop.py` iteration writes a dedicated artifact directory:

- `workspace/runs/run_<timestamp>/iter_vN/benchmark_result.json`
- `workspace/runs/run_<timestamp>/iter_vN/benchmark.stdout.txt`
- `workspace/runs/run_<timestamp>/iter_vN/benchmark.stderr.txt`
- `workspace/runs/run_<timestamp>/iter_vN/targeted.ncu-rep`
- `workspace/runs/run_<timestamp>/iter_vN/full.ncu-rep`
- `workspace/runs/run_<timestamp>/iter_vN/optimization_proposal.md`
- `workspace/runs/run_<timestamp>/iter_vN/iteration_summary.md`

See `docs/experiment_artifacts.md` and `docs/strategy_memory.md` for the artifact and structured strategy memory schema.

## License

MIT
