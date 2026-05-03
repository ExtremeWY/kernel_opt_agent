# cuda-evolve Program

You are an autonomous GPU kernel optimization agent. Follow this protocol strictly.

## Available Kernels

The `kernels/` directory contains **baseline (read-only)** kernels. **Never modify files in `kernels/`** — they serve as the unmodified reference.

Optimized kernels are saved to `kernels_optimized/`, which mirrors the structure of `kernels/`.

**Directory layout:**
```
kernels/                        # Baseline (READ-ONLY, never modify)
├── <your_kernel>.py            # Python module (Triton or wrapper)
└── <your_kernel>.cu            # Optional: CUDA C source (paired with .py wrapper)

kernels_optimized/              # Optimized versions (agent writes here)
├── <your_kernel>.py
└── <your_kernel>.cu            # If the kernel uses CUDA C
```

Each kernel module (`.py`) must export:
- `KERNEL_TYPE: str` -- identifier matching a key in `kernel_configs/` (i.e. a `<name>.toml` + `<name>.py` pair)
- `kernel_fn(**inputs) -> torch.Tensor` (or tuple)
- `get_inputs() -> dict`
- `get_flops() -> int` (for roofline analysis)
- `get_bytes() -> int` (for roofline analysis)

## Non-Negotiable Constraint

All kernel optimization work must optimize the custom kernel implementation itself.

- Do **not** replace the kernel with library calls such as PyTorch SDPA, cuDNN attention, cuBLAS, CUTLASS wrappers, Triton reference kernels, or any other prebuilt fused operator.
- Do **not** add runtime dispatch that routes “fast” cases to library functions and leaves only fallback cases on the custom kernel.
- Benchmark results only count if the measured path executes the optimized `kernel.py` / `kernel.cu` implementation directly.
- Library code may be used only for **reference correctness** and **performance comparison**, never as the optimized solution.

**Kernel types:**
- **Triton kernels**: Single `.py` file containing Triton code directly
- **CUDA C kernels**: A `.py` wrapper that compiles and loads a companion `.cu` file via `torch.utils.cpp_extension.load_inline()`

To add a new kernel, create `kernel_configs/<name>.toml` (sizes, dtypes, tolerances) and `kernel_configs/<name>.py` (input_generator, reference_fn, flops_fn, bytes_fn). The registry auto-discovers them at import time.

## Setup Phase

1. Run `.venv/bin/python tools/prepare.py` to validate the environment (CUDA, GPU, dependencies).
   If `workspace/runtime_env.md` selects a different interpreter on this machine, use those commands for all subsequent tool invocations.
2. Review `workspace/preflight_check.md`. If preflight is not ready, fix blocking issues before starting experiments.
3. Read `CUDA_OPTIMIZATION.md` to review optimization strategies discovered in previous runs. (This file is maintained by you — the agent — and may be empty on the first run.)
4. Read `workspace/MEMORY.md` for the global optimization summary across all kernels.
4. **Select a kernel** to optimize. Copy from the baseline `kernels/` directory (or from `kernels_optimized/` if a previous optimized version exists):
   ```bash
   # First run: start from baseline
   cp kernels/<your_kernel>.py kernel.py

   # Resuming: start from last optimized version (if it exists)
   cp kernels_optimized/<your_kernel>.py kernel.py

   # For CUDA C kernels, also copy the .cu file:
   cp kernels/<your_kernel>.cu kernel.cu    # if it exists
   ```
5. Read the per-kernel log in `memory/<kernel_type>.md` if it exists, to review past experiments for this specific kernel.
6. Read the relevant optimization references in `docs/`, especially
   `docs/prototype_ladder.md`, `docs/triton_optimization.md`,
   `docs/cutlass_optimization.md`, `docs/sync_optimization.md`,
   `docs/compute_optimization.md`, `docs/memory_optimization.md`, and
   `docs/stall_reasons.md`.
7. Read `docs/strategy_memory.md` and respect the `blocked` / `preferred` fingerprints already recorded in `workspace/strategy_memory/global_strategy_memory.json`.
8. Read `kernel.py` to understand the current kernel implementation.
9. Read the relevant module(s) in `references/` (per-kernel reference implementations) to understand the correctness specification.

## Experiment Loop

Repeat the following cycle:

### Step 1: Benchmark (baseline or after change)

Run the benchmark harness. It auto-detects `KERNEL_TYPE` from `kernel.py` and should emit structured JSON:

```bash
.venv/bin/python tools/bench.py --json-out workspace/last_bench.json > run.log 2>&1
```

For quick iteration (skip numerical stability, determinism, edge cases):

```bash
.venv/bin/python tools/bench.py --quick --json-out workspace/last_bench.json > run.log 2>&1
```

Quick mode is directional evidence only. It benchmarks fewer cases but still emits multi-trial timing stability fields. If timing is unstable, or if the measured improvement/regression is close to the benchmark noise band, run the full benchmark before making a keep/revert decision.

Read `run.log` and extract the key metrics:

```bash
grep "correctness\|throughput_tflops\|speedup_vs_pytorch\|pct_peak_compute\|pct_peak_bandwidth\|bottleneck\|kernel_timing_spread_pct\|kernel_timing_stable\|peak_vram_mb" run.log
```

The benchmark reports:
- **correctness**: 5-stage verification (smoke, shape sweep, numerical stability, determinism, edge cases)
- **throughput_tflops**: Achieved throughput
- **bandwidth_gb_s**: Achieved memory bandwidth
- **pct_peak_compute**: % of GPU's theoretical compute peak
- **pct_peak_bandwidth**: % of GPU's theoretical bandwidth peak
- **bottleneck**: `compute_bound` or `memory_bound` (from roofline analysis)
- **speedup_vs_pytorch**: Speedup vs PyTorch reference implementation
- **kernel_timing_spread_pct / kernel_timing_stable**: Multi-trial timing noise guard. Treat unstable results as inconclusive, not as optimization evidence.

### Step 2: Macro Performance Analysis

Analyze the benchmark results to understand the kernel's **macro-level** performance characteristics:

1. **Compute throughput**: How close is `pct_peak_compute` to the GPU's theoretical peak?
2. **Memory bandwidth**: How close is `pct_peak_bandwidth` to the GPU's theoretical bandwidth?
3. **Bottleneck classification**: Is the kernel `compute_bound` or `memory_bound`?
4. **Roofline position**: Where does the kernel sit on the roofline? How far from the ridge point?

This gives you the **direction** of optimization (memory vs. compute), but not the **specific** cause.

### Step 2b: Performance Model When No Strong Reference Exists

A fast black-box reference profile is optional, not a requirement. If there is
no trusted high-performance implementation to compare against, build a
first-principles performance model before choosing experiments:

1. **Operator contract**: record which dimensions, dtypes, layouts, and semantic
   flags are true API or production invariants, and which are runtime-variable.
2. **Minimum work**: estimate required FLOPs, required bytes, reductions,
   synchronization points, and main-loop trip counts for each supported shape
   regime.
3. **Primitive ceilings**: estimate feasible upper bounds from hardware peak,
   local microbenchmarks, or small isolated prototypes for copy bandwidth,
   shared-memory staging, reductions, and MMA or scalar compute primitives.
4. **Stage budget**: split the kernel into load/stage, core compute,
   reduction/normalization, and store/epilogue stages. Estimate the dynamic work
   fraction and best-case speedup if each stage became free.
5. **Route candidates**: generate at least two structurally different route
   invariants when the current kernel is far below the modeled ceiling. Do not
   start with local pitch, padding, cache, branch, or launch-hint tweaks unless
   the model shows those sites dominate end-to-end time.
6. **Prototype ladder**: classify the current implementation by the ladder in
   `docs/prototype_ladder.md`: parallel ownership, data locality, hot-state
   residency, hardware primitive, layout, pipeline, grid scheduling, then local
   cleanup. If a higher-upside earlier stage is missing, test that architecture
   route before spending iterations on later-stage micro-tuning.

The model is allowed to be approximate, but it must be explicit enough to reject
low-coverage ideas before implementation.

When the model indicates a design-boundary bottleneck, mark that state in
strategy memory before running more experiments:

```bash
.venv/bin/python tools/run_loop.py \
  --hypothesis "mark design-boundary limited state" \
  --mark-design-boundary \
  --design-boundary-reason "mainloop instruction/sync cost exceeds modeled ceiling" \
  --state-only
```

Once marked, `tools/run_loop.py` rejects normal local experiments by default.
Use `--architecture-route` with a route plan, or explicitly pass
`--allow-local-after-boundary` only when the proposal explains why a local
experiment is still justified.

### Step 3: NCU Deep Analysis

After understanding the macro picture, use NCU + ncu-cli to identify the **specific** bottleneck:

```bash
.venv/bin/python tools/ncu_profile.py --mode targeted --output-prefix workspace/ncu_reports/manual_targeted > ncu.log 2>&1
```

Extract the key findings:

```bash
grep "ncu_bottleneck\|ncu_top_stall\|ncu_finding\|ncu_action\|ncu_occupancy\|ncu_l1_hit_rate\|ncu_l2_hit_rate" ncu.log
```

For targeted analysis (e.g., memory access patterns, warp stalls):

```bash
.venv/bin/python tools/ncu_profile.py --mode targeted --skills roofline,memory,warp_stall --output-prefix workspace/ncu_reports/manual_targeted > ncu.log 2>&1
```

Note: `--skills` now extends the default targeted skill set. If you truly want to replace the default targeted skills, use `--replace-skills`, but that can drop required evidence such as `tensor_core` and `occupancy`.

To compare before/after an optimization:

```bash
.venv/bin/python tools/ncu_profile.py --diff before.csv after.csv > ncu_diff.log 2>&1
```

**NCU analysis tells you the *specific* cause:**

- **Memory-bound kernels**: Which cache level is the bottleneck? Are loads coalesced? What's the L1/L2 hit rate? How many DRAM bytes are transferred?
- **Compute-bound kernels**: First ask whether the kernel is matmul/MMA-like, then whether the active shape regime is MMA-friendly, then inspect tensor core utilization, instruction mix, and warp behavior.

### Step 4: Hypothesize

Combine the macro analysis (Step 2) and NCU deep analysis (Step 3) to formulate a **single, focused** hypothesis. Create an `optimization_proposal.md` for the iteration and make sure it contains a `## Strategy tags` section.

> Hypothesis: [What you plan to change and why you expect it to improve performance]
> Macro evidence: [Which `tools/bench.py` metric(s) indicate the bottleneck direction]
> NCU evidence: [Which ncu-cli finding(s) pinpoint the specific cause]
> Expected impact: [Estimated dynamic-time coverage and expected end-to-end speedup]

**Hypothesis workflow:**
1. **Macro**: `tools/bench.py` roofline → is it compute-bound or memory-bound? How far from peak?
2. **Kernel traits**: before assuming tensor cores matter, classify whether the kernel is matmul/MMA-like and what shape regime it is in (`full_m`, `small_m`, `small_m_decode_like`, etc.).
3. **Micro**: `ncu-cli analyze` → what is the *specific* bottleneck? (stall type, cache miss, uncoalesced access, etc.)
4. **Tensor-core gate**: only treat low `tensor_core_pct` as actionable when the kernel is matmul-like *and* the active shape regime is MMA-friendly. For small-M / decode-like regimes, compare CUDA-core and tensor-core-with-padding paths first. For bench-only evidence on an MMA-friendly kernel, classify it as needing NCU confirmation rather than as an automatic tensor-core recommendation.
5. **Knowledge**: Check `CUDA_OPTIMIZATION.md` → does a known optimization address this? The "Cross-Kernel Optimization Patterns" section at the bottom organizes techniques by bottleneck type (e.g., `[register-pressure]`, `[occupancy]`, `[tensor-core]`) for easy lookup regardless of which kernel you're optimizing.
6. **Docs**: Read relevant files in `docs/` for the specific bottleneck. The `docs/` directory contains curated references on stall reasons, synchronization, memory optimization, compute optimization, framework-specific tuning, and architecture-specific notes.
7. **History**: Check `memory/<kernel_type>.md` → has this been tried before for this kernel?
8. **Impact gate**: estimate the maximum possible end-to-end gain before editing code. Use benchmark shapes, loop trip counts, tile counts, predicate/branch coverage, NCU instruction/stall shares, and prior timing deltas. If the affected dynamic work is a small boundary case or the theoretical end-to-end gain is below the keep threshold, reject the idea before implementation.
9. **Generality gate**: reject benchmark-shape overfitting before editing code. A proposed optimization must be valid for the operator's intended runtime variability, not just for the current `kernel_configs/` sizes. Do not specialize on runtime-varying dimensions, batch counts, grid sizes, or other benchmark constants unless they are explicitly part of the operator contract or production invariant. Prefer optimizations based on stable facts such as dtype, hardware architecture, fixed layout contracts, fixed semantic dimensions, or runtime tile-state checks that work for arbitrary supported problem sizes.
10. **History-neighborhood gate**: if a proposed change is an adjacent variant of a negative or rejected strategy (same hot path, same data layout family, same tile sweep, same load/store trick), skip it unless there is new contradictory NCU evidence showing that the bottleneck moved.
11. **Priority gate**: rank all candidate hypotheses by expected end-to-end impact divided by implementation/validation risk. Spend early iterations on the dominant bottleneck only; leave cleanup and noise-floor micro-tuning for after a structural improvement is working.
12. **Design-boundary gate**: if a fast reference profile, self-profile trend, source attribution, or the Step 2b performance model shows a large gap in total instructions, LSU/shared-memory handoff, synchronization, main-loop trip count, or launched work while unavoidable DRAM traffic is already near the minimum, classify the current implementation as design-boundary limited. In that state, do not keep spending iterations on local layout, padding, cache hints, branch cleanup, or small launch-shape variants. The next proposals must remove or replace the dominant dataflow boundary.
13. **Negative-evidence scope gate**: do not over-generalize a failed experiment. A negative result only blocks the dataflow and implementation scope that was actually tested. If a failed change still kept the dominant old intermediate, duplicated expensive work, or grafted a new primitive onto the old ownership model, it does not disprove a full redesign that removes the old boundary.
14. **Architecture-route budget gate**: when a high-upside design-boundary route is selected, define a route-level invariant and budget before editing. The route may take multiple sub-iterations to become correct and fast. Correctness failures, races, register pressure, or first-version regressions should trigger focused fixes inside the same route until the route budget is exhausted, not immediate abandonment of the route. The budget must still be finite and evidence-driven.
15. **No-reference route gate**: when there is no high-performance reference, route selection must be justified by the performance model and current-kernel NCU/source attribution. The proposal must name the structural cost to remove, the affected dynamic-work fraction, the primitive ceiling being targeted, and why the route is not a benchmark-shape specialization.
16. **Prototype-ladder gate**: before proposing local cleanup, record the current prototype-ladder stage and the next missing high-upside stage from `docs/prototype_ladder.md`. If the next missing stage targets the dominant steady-state work and has plausible multi-percent payoff, it must become an architecture route instead of a local experiment.
17. **Stage-promotion gate**: a route can only be considered rejected after a correctness-passing, resource-balanced implementation actually satisfies its route invariant. Failed partial grafts, implementations that still carry the old hot intermediate, or first versions with repairable synchronization/resource issues are narrow negative evidence or route sub-iterations, not proof that the broader stage is invalid.

**Rules:**
- One change per experiment. Do not combine unrelated optimizations.
- If you've tried this before (check per-kernel log), try something different.
- Read the current scope's `blocked` / `preferred` fingerprints before writing the proposal. Do not repeat blocked strategies unless you have new contradictory evidence.
- Always ground hypotheses in NCU evidence, not guesswork.
- Do not spend an iteration on a change whose estimated best-case end-to-end speedup is below the keep threshold, unless the experiment is a minimal correctness probe for a larger structural redesign.
- Do not design optimizations that only win because the benchmark uses fixed sizes. In particular, do not add compile-time specializations, dispatch branches, hard-coded constants, or removed checks for dimensions/properties that are expected to vary in real use. If a specialization is proposed, the proposal must state why the specialized property is a true API/production invariant rather than a benchmark artifact.
- For boundary-only changes (tail cases, partial tiles, uncommon predicates, rare dtype/shape paths), compute what fraction of the benchmarked dynamic work they cover. If that fraction is small, deprioritize them behind changes that affect the steady-state hot path.
- Runtime checks for common tile states are allowed when they preserve correctness and performance portability across all supported sizes. Benchmark-only dispatch is not allowed.
- Do not turn a structural bottleneck into a series of local layout/load/barrier tweaks. If NCU and history indicate the design itself is wrong, the next experiments must change the design boundary.
- When pursuing a design-boundary route, keep the route invariant explicit. Examples of route invariants include “remove a materialized intermediate from the hot path,” “change the ownership model so the producer consumes its own result,” or “move the dominant work to the intended hardware pipeline.” Sub-iterations may tune tile shape, resource balance, or synchronization only if they preserve that invariant.
- Do not classify an architecture route as failed because one intermediate implementation regressed while it still carried the old bottleneck. Record whether the tested version truly removed the dominant intermediate or only partially bypassed it.
- If a local family has produced several sub-threshold wins while the reference gap remains large, stop the family even if each individual change is plausible. A chain of small wins can still lead to a hard design ceiling.
- Do not keep optimizing a low-ceiling prototype just because it is the nearest
  editable code. If the operator model points to a different ownership,
  residency, primitive, or pipeline stage with much larger expected payoff,
  switch to an architecture route and budget multiple repair iterations.
- If there is no strong reference gap to quote, replace “reference gap” with
  “gap to modeled primitive or stage ceiling.” Lack of a reference is not a
  reason to continue local micro-tuning after the model and NCU identify a
  structural boundary.
- Never satisfy an optimization task by swapping in a library implementation. Improve the custom kernel code itself.

### Step 5: Modify

Edit `kernel.py` (and `kernel.cu` for CUDA C kernels) to implement your hypothesis.

### Step 6: Commit

```bash
# Triton kernels:
git add kernel.py
git commit -m "experiment: <brief description of change>"

# CUDA C kernels (when kernel.cu exists):
git add kernel.py kernel.cu
git commit -m "experiment: <brief description of change>"
```

### Step 7: Benchmark

```bash
.venv/bin/python tools/bench.py > run.log 2>&1
```

**IMPORTANT**: Always redirect to `run.log`. Do NOT let output flood your context window.

### Step 8: Decide

| Condition | Action |
|-----------|--------|
| correctness = FAIL | **REVERT** immediately: `git reset --hard HEAD~1` (reverts both `kernel.py` and `kernel.cu`) |
| correctness = PASS, timing unstable | **RUN FULL/STABLE BENCHMARK** before deciding; do not keep/revert from an unstable quick result |
| correctness = PASS, full/stable benchmark throughput improved (>1%) | **KEEP** |
| correctness = PASS, full/stable benchmark throughput same or worse | **REVERT**: `git reset --hard HEAD~1` |

If `--quick` and full benchmark disagree, use the full benchmark result. If the delta is within the configured timing stability threshold, treat the experiment as inconclusive and prefer another hypothesis instead of tuning around noise.

Architecture-route exception:

- A structural route must still revert broken code unless explicitly kept as a
  correctness-passing prototype, but a failed or slower non-validation
  sub-iteration is recorded as `inconclusive`, not as a route-blocking negative.
  `tools/run_loop.py` exits successfully for such non-validation route
  sub-iterations so a route executor can continue until the route budget or stop
  condition is reached.
- Use route mode for multi-sub-iteration structural work:
  ```bash
  .venv/bin/python tools/run_loop.py \
    --hypothesis "prototype new ownership route" \
    --architecture-route \
    --route-invariant "remove the dominant dataflow boundary" \
    --route-expected-impact "affects steady-state mainloop; target >5% end-to-end" \
    --route-budget 8 \
    --route-stop-condition "validated implementation below threshold or budget exhausted" \
    --route-plan workspace/runs/run_xxx/architecture_route_plan.md \
    --route-iteration-role prototype \
    --quick
  ```
- When design-boundary mode is active and the route id is new, the route plan is
  mandatory. It must contain at least two structurally distinct route candidates
  and must not contain placeholder fields such as `fill_me`, `todo`, or `tbd`.
  It must also record the current prototype-ladder stage, the next missing
  high-upside stage, route promotion criteria, and the scope of negative
  evidence.
- Use `--route-allow-regression` only when a correctness-passing prototype must
  remain in the worktree for follow-up sub-iterations. Do not use it for final
  validation. The route must eventually pass normal full/stable keep criteria in
  a `--route-iteration-role validation` run.

### Step 9: Record

**9a. Append to `workspace/results.tsv`:**

```
experiment_id	hypothesis	correctness	time_ms	throughput	peak_vram_mb	kept	pct_peak_compute	pct_peak_bandwidth	bottleneck	git_sha	parent_experiment_id	ncu_top_stall	ncu_occupancy	ncu_l1_hit_rate	ncu_l2_hit_rate	strategy_tags	strategy_fingerprint	strategy_outcome	strategy_reason	run_dir	iter_dir	targeted_ncu_report	full_ncu_report
```

The extended columns capture micro-architectural context for lineage tracking:
- `pct_peak_compute`, `pct_peak_bandwidth`, `bottleneck`: from `tools/bench.py` roofline output
- `git_sha`: `git rev-parse --short HEAD` for exact reproducibility
- `parent_experiment_id`: which experiment this was derived from
- `ncu_top_stall`, `ncu_occupancy`, `ncu_l1_hit_rate`, `ncu_l2_hit_rate`: from `tools/ncu_profile.py` output
- `strategy_tags`, `strategy_fingerprint`, `strategy_outcome`, `strategy_reason`: from `optimization_proposal.md` + `workspace/strategy_memory/global_strategy_memory.json`
- `run_dir`, `iter_dir`, `targeted_ncu_report`, `full_ncu_report`: artifact lineage
- architecture-route metadata is stored in each run manifest and in
  `workspace/strategy_memory/global_strategy_memory.json` under `routes`.
  Non-validation route failures are `inconclusive` so they do not poison the
  blocked fingerprint set.

**9b. Archive per-iteration artifacts under `workspace/runs/run_xxx/iter_vN/`:**

- `benchmark_result.json`
- `benchmark.stdout.txt`
- `benchmark.stderr.txt`
- `targeted.ncu-rep` and/or `full.ncu-rep`
- `targeted_summary.txt`, `targeted_details.txt`
- `full_summary.txt`, `full_details.txt`
- `optimization_proposal.md`
- `iteration_summary.md`

**9c. Update per-kernel log (`memory/<kernel_type>.md`):**

Record the detailed experiment for this specific kernel:
- Experiment ID and hypothesis
- Macro analysis (`tools/bench.py` roofline metrics)
- NCU analysis (specific bottleneck, stall types, cache hit rates)
- Result (kept / reverted) and key observations
- What you learned that could inform the next experiment

**9d. Update `workspace/MEMORY.md` (global summary):**

Keep a concise cross-kernel summary:
- Which kernel was optimized and current best speedup
- High-level insights that transfer across kernels

**9e. Update `CUDA_OPTIMIZATION.md` (if a new optimization pattern was discovered):**

When an optimization **succeeds**, add it to `CUDA_OPTIMIZATION.md` under the appropriate kernel type section. Include:
- What the optimization is
- Why it works for this kernel type
- Expected speedup range
- When an optimization **fails**, add it to the "Anti-patterns" section for that kernel type.

### Step 10: Repeat

Return to Step 1. Continue until:
- Performance gains have plateaued (< 1% improvement over 3 consecutive experiments)
- You have exhausted all known optimizations in `CUDA_OPTIMIZATION.md` and cannot generate new hypotheses from NCU data

If `workspace/strategy_memory/global_strategy_memory.json` marks a strategy fingerprint as rejected, avoid repeating it without new evidence.

## Switching Kernels

When you finish optimizing one kernel, save the optimized version to `kernels_optimized/` and move to the next:

```bash
# Save optimized kernel
cp kernel.py kernels_optimized/<kernel_name>.py
cp kernel.cu kernels_optimized/<kernel_name>.cu    # if CUDA C kernel

# Switch to next kernel -- copy from baseline (or from kernels_optimized/ if resuming)
cp kernels/<next_kernel>.py kernel.py
cp kernels/<next_kernel>.cu kernel.cu              # if CUDA C kernel

# Per-kernel logs are in memory/<kernel_type>.md -- they persist across sessions
# workspace/MEMORY.md has the global summary -- cross-kernel insights are valuable
```

**Important:** Never modify files in `kernels/`. The baseline must remain intact for comparison and reproducibility.

Before starting the new kernel, review `memory/<kernel_type>.md` for any past experiments on it, and check `CUDA_OPTIMIZATION.md` for transferable optimization patterns.

## Memory-Bound Kernel Optimization Priority

Most kernels in this repo are memory-bound. The optimization priority for memory-bound kernels is:

1. **Coalescing** -- NCU tells you if loads/stores are uncoalesced (sectors/request > 4). Fix memory layout or access pattern.
2. **Vectorized loads** -- Use `float4`/`bf16_8` loads to maximize bandwidth per instruction.
3. **L2 cache locality** -- Reorder tile indices so neighboring blocks access nearby memory. NCU shows L2 hit rate.
4. **Prefetching / pipelining** -- `num_stages` in Triton, `cp.async` in CUDA. NCU shows Long Scoreboard stalls.
5. **Reduce memory traffic** -- Fuse operations, avoid redundant reads/writes. NCU shows total DRAM bytes.
6. **Shared memory tiling** -- For reduction patterns, load to shared memory first. NCU shows bank conflicts.

**Yes, you can and should modify the kernel code for memory-bound kernels.** The optimization is about *how* data moves, not *what* is computed. Typical changes:

- Adjust `block_size` and `num_stages` (Triton) or thread/block config (CUDA)
- Change memory access patterns for better coalescing
- Add prefetching / software pipelining
- Use vectorized loads (`tl.load` with larger block sizes, or `float4` in CUDA)
- Reorder loop dimensions for better cache behavior

## Memory & Knowledge Structure

```
cuda-evolve/
├── kernels/                    # Baseline kernels (READ-ONLY)
│   ├── <name>.py               # Python module (Triton or CUDA C wrapper)
│   └── <name>.cu               # Optional: CUDA C source (paired with .py)
├── kernels_optimized/          # Optimized kernels (agent saves here)
│   ├── <name>.py
│   └── <name>.cu               # If CUDA C kernel
├── kernel_configs/             # Per-kernel benchmark configs (TOML data + Python callables)
│   ├── <name>.toml             # Sizes, dtypes, tolerances, edge_sizes
│   └── <name>.py               # input_generator, reference_fn, flops_fn, bytes_fn
├── tools/                      # CLI scripts (bench, NCU, run_loop, prepare, merge)
├── references/                 # Per-kernel reference implementations (correctness spec; READ-ONLY)
├── workspace/                  # Working artifacts (results, memory summary, NCU exports)
│   ├── MEMORY.md               # Global summary across all kernels
│   ├── results.tsv             # Raw experiment results (extended schema with NCU metrics, lineage)
│   └── ncu_reports/            # NCU report outputs
├── kernel.py                   # Current working kernel module
├── kernel.cu                   # Current working CUDA C source (when applicable)
├── CUDA_OPTIMIZATION.md        # Agent-maintained: optimization patterns by kernel type + cross-kernel patterns
├── memory/
│   └── <kernel_type>.md        # Detailed experiment log per kernel
└── docs/                       # Reference documentation (read during kernel optimization)
    ├── stall_reasons.md        # NCU warp stall type reference
    ├── memory_optimization.md  # Memory subsystem optimization guide
    ├── compute_optimization.md # Compute optimization guide
    ├── prototype_ladder.md     # Structural route ladder and promotion gates
    ├── architecture_route_plan_template.md # Structural route plan template
    └── arch_notes.md           # GPU architecture specifications
```

- **`kernels/`**: Baseline kernels. **Never modify.** These are the starting point and comparison reference.
- **`kernels_optimized/`**: Mirrors `kernels/` structure. The agent saves the best optimized version of each kernel here after finishing optimization.
- **`kernel_configs/`**: Per-kernel benchmark configurations. Each kernel has a `.toml` file (declarative data: sizes, dtypes, tolerances) and a companion `.py` file (callables: input generator, reference wrapper, flops/bytes functions). Auto-discovered by `tools/bench.py` at import time. To add a new kernel, create `<name>.toml` + `<name>.py` here.
- **`tools/`**: Runnable harnesses and helpers. Invoke with `.venv/bin/python tools/<script>.py` after `uv sync`.
- **`references/`**: Per-kernel PyTorch reference code for correctness. **Never modify.**
- **`CUDA_OPTIMIZATION.md`**: Grows over time as the agent discovers what works. Organized by kernel type with tagged entries (e.g., `[register-pressure]`, `[occupancy]`). Includes a "Cross-Kernel Optimization Patterns" section for transferable techniques.
- **`memory/<kernel_type>.md`**: Detailed per-kernel experiment log with full NCU analysis, hypotheses, and outcomes. This is the primary record for each kernel.
- **`workspace/MEMORY.md`**: High-level cross-kernel summary. Kept concise — just the current best results and transferable insights.
- **`docs/`**: Curated reference documentation on GPU optimization. During kernel optimization, treat these files as read-only. Framework-maintenance tasks may update them, but must keep the changes general and documented.
- **`workspace/results.tsv`**: Extended schema with NCU micro-metrics, git SHA, parent experiment lineage, and bottleneck classification.
- **`workspace/ncu_reports/`**: Directory for NCU profiling exports and related artifacts.

## Available Tools

| Tool | Purpose | Usage |
|------|---------|-------|
| `tools/bench.py` | Correctness + performance benchmark | `.venv/bin/python tools/bench.py > run.log 2>&1` |
| `tools/ncu_profile.py` | NCU micro-architecture profiling | `.venv/bin/python tools/ncu_profile.py > ncu.log 2>&1` |
| `tools/run_loop.py` | Automated experiment cycle | `.venv/bin/python tools/run_loop.py --hypothesis "..."` |
| `tools/prepare.py` | Environment validation | `.venv/bin/python tools/prepare.py` |
| `tools/merge_results.py` | Merge multi-agent results | `.venv/bin/python tools/merge_results.py ../worktree` |

## Multi-Agent Parallel Optimization

When multiple agents need to optimize **different kernels** simultaneously, use **git worktree** to give each agent an isolated working directory. This avoids conflicts on `kernel.py`, logs, git state, and GPU resources.

### Setup

From the main repository, create one worktree per kernel/agent:

```bash
# Ensure main is clean
git checkout main

# Create isolated worktrees (one per kernel)
git worktree add ../cuda-evolve-matmul   -b agent/matmul
git worktree add ../cuda-evolve-rms-norm -b agent/rms-norm
git worktree add ../cuda-evolve-swiglu   -b agent/swiglu
```

Each worktree is an independent directory with its own `kernel.py`, `workspace/results.tsv`, `workspace/MEMORY.md`, `memory/`, `traces/`, and git working state. All worktrees share the same `.git` repository, so commit history is unified and branches can be merged.

### Branch Naming Convention

Use `agent/<kernel_name>` branches (e.g. `agent/matmul`, `agent/rms-norm`). Each agent commits only to its own branch.

### GPU Isolation

Bind each agent to a separate GPU via `CUDA_VISIBLE_DEVICES`:

```bash
# Agent A (matmul) — GPU 0
cd ../cuda-evolve-matmul
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/bench.py > run.log 2>&1

# Agent B (rms_norm) — GPU 1
cd ../cuda-evolve-rms-norm
CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/bench.py > run.log 2>&1
```

If only **one GPU** is available, agents can edit code in parallel but must **serialize benchmark execution** to avoid VRAM contention and timing interference.

### Per-Agent Workflow

Each agent follows the standard Experiment Loop (above) inside its own worktree. No changes to the loop itself — the isolation is at the directory/branch level.

### Merging Results Back to Main

After each agent completes optimization, merge its branch into `main`:

```bash
cd /path/to/main-repo
git merge agent/matmul   --no-ff -m "merge: matmul optimization results"
git merge agent/rms-norm --no-ff -m "merge: rms-norm optimization results"
```

**Conflict expectations by file:**

| File | Conflict risk | Resolution |
|------|--------------|------------|
| `kernels_optimized/<name>.py` | None — different files | Auto-merge |
| `memory/<kernel_type>.md` | None — different files | Auto-merge |
| `workspace/results.tsv` | Low — append-only | Concatenate rows (keep header once) |
| `workspace/MEMORY.md` | Low — different sections | Merge by section |
| `CUDA_OPTIMIZATION.md` | Low — different kernel type sections | Merge by section |

You can use `tools/merge_results.py` to assist with `workspace/results.tsv` merging (see below).

### Cleanup

```bash
git worktree remove ../cuda-evolve-matmul
git worktree remove ../cuda-evolve-rms-norm
```

## Automated Experiment Runner

For faster iteration, use `tools/run_loop.py` to automate Steps 6-9 (commit, benchmark, decide, record):

```bash
# Edit kernel.py with your change, then:
.venv/bin/python tools/run_loop.py --hypothesis "increase tile size from 64 to 128"

# With NCU profiling:
.venv/bin/python tools/run_loop.py --hypothesis "vectorize loads" --targeted-ncu

# Quick mode (skip correctness stages 3-5):
.venv/bin/python tools/run_loop.py --hypothesis "try num_warps=8" --quick

# Dry run (show what would happen):
.venv/bin/python tools/run_loop.py --hypothesis "test change" --dry-run

# Structural route mode:
.venv/bin/python tools/run_loop.py \
  --hypothesis "prototype new mainloop ownership" \
  --architecture-route \
  --route-invariant "remove the dominant steady-state handoff" \
  --route-expected-impact "mainloop coverage high; target >5% end-to-end" \
  --route-budget 8 \
  --route-iteration-role prototype \
  --targeted-ncu
```

The runner automatically:
- Commits `kernel.py` (and `kernel.cu` if present) before benchmarking
- Runs `tools/bench.py` and parses metrics
- Optionally runs `tools/ncu_profile.py`
- Applies keep/revert decision (>1% improvement threshold for normal
  experiments; route-aware inconclusive handling for architecture routes)
- Appends full metadata to `workspace/results.tsv` (including git_sha, NCU metrics)
- Outputs a compact summary

This reduces token usage and eliminates the risk of forgetting to commit or revert.

## Important Rules

1. **Never break correctness.** Every change must pass all 5 correctness stages.
2. **During kernel optimization tasks, never modify files in `tools/` (harness scripts), `references/`, or `kernels/`.** These are fixed baselines and evaluation harnesses. Save optimized kernels to `kernels_optimized/`. Framework-maintenance tasks are the explicit exception and must keep harness behavior documented and tested.
3. **One change at a time.** Isolate variables to understand causality.
4. **Always commit before benchmarking.** Commit both `kernel.py` and `kernel.cu` (if present). This enables clean reverts.
5. **Read per-kernel log before each experiment.** Check `memory/<kernel_type>.md` to learn from past attempts on this kernel.
6. **Always run NCU analysis.** Every experiment should include both macro (`tools/bench.py`) and micro (ncu-cli) analysis. Don't hypothesize without evidence.
7. **Use roofline data and NCU findings together.** Macro tells you the direction, NCU tells you the specific cause.
8. **VRAM must not exceed 80% of GPU memory.** Treat as regression and revert.
9. **Maintain the knowledge base.** Update `CUDA_OPTIMIZATION.md` when you discover new optimization patterns or anti-patterns. Future runs depend on this.
