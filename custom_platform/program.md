# custom_platform Program

You are an autonomous kernel optimization agent working on a custom hardware
platform. Follow this protocol strictly.

## Available Kernels

The `kernels/` directory contains baseline kernels. Do not modify them.
Optimized kernels are written to `kernels_optimized/`.

Each active kernel module must export:

- `KERNEL_TYPE: str`
- `TARGET_PLATFORM: str`
- `kernel_fn(**inputs)`

## Non-Negotiable Constraint

All optimization work must improve the custom kernel implementation itself.

- Do not replace the kernel with a prebuilt library implementation.
- Do not add runtime dispatch that sends fast cases to a library and leaves only
  fallback cases on the custom kernel.
- Reference code may be used for correctness and performance comparison, never
  as the optimized implementation.

## Setup Phase

1. Run `.venv/bin/python tools/prepare.py`.
   For local workflow rehearsal without real hardware, use
   `--platform mock_platform`.
2. If `workspace/runtime_env.md` selects a different interpreter, use that
   interpreter for all subsequent tool invocations.
3. Read `knowledge/custom_platform/OPTIMIZATION.md`.
4. Read `workspace/MEMORY.md` and `memory/<kernel_type>.md` if present.
5. Read `workspace/strategy_memory/global_strategy_memory.json`.
6. Copy one baseline or last optimized kernel to `kernel.py`.
7. Read the matching `kernel_configs/` and `references/` modules.
8. Read relevant files in `docs/`, especially
   `docs/prototype_ladder.md`, `docs/architecture_route_plan_template.md`,
   `docs/experiment_artifacts.md`, `docs/strategy_memory.md`,
   `docs/compute_optimization.md`, `docs/memory_optimization.md`,
   `docs/stall_reasons.md`, and `docs/arch_notes.md`.

## Experiment Loop

Use the run loop unless explicitly debugging a tool:

```bash
.venv/bin/python tools/run_loop.py --hypothesis "<one focused change>"
```

For a local mock success path:

```bash
.venv/bin/python tools/run_loop.py --platform mock_platform --hypothesis "<one focused change>"
```

Each iteration must:

1. Run preflight and inspect `workspace/runs/run_<timestamp>/preflight_check.md`.
2. Run benchmark and profile through `tools/run_loop.py`.
3. Read `iter_vN/benchmark_result.json`, `profile_summary.txt`, and
   `profile_details.txt`.
4. Fill or update `iter_vN/optimization_proposal.md` before editing code.
5. Keep exactly one focused hypothesis per iteration.
6. Modify `kernel.py` and only necessary same-kernel source dependencies.
7. Re-run the loop and decide from correctness, stable timing, and complete
   profile evidence.
8. Record the result in `workspace/results.tsv`, `memory/<kernel_type>.md`, and
   `workspace/MEMORY.md`.
9. If a reusable platform pattern is found, update
   `knowledge/custom_platform/OPTIMIZATION.md`.

## Performance Model When No Strong Reference Exists

A fast black-box reference is optional. If there is no trusted high-performance
implementation, build a first-principles performance model before choosing
experiments:

1. Record the operator contract: stable dimensions, dtypes, layouts, semantic
   flags, and runtime-variable properties.
2. Estimate minimum operations, bytes, reductions, synchronization points, and
   main-loop trip counts for each supported shape regime.
3. Estimate primitive ceilings from hardware peak, local microbenchmarks, or
   small isolated prototypes for copy, staging, reduction, vector/matrix
   compute, scalar compute, and synchronization.
4. Split the kernel into load/stage, core compute, reduction/normalization, and
   store/epilogue stages. Estimate dynamic coverage and best-case speedup if
   each stage became free.
5. Generate at least two structurally different route candidates when the
   current kernel is far below the modeled ceiling.
6. Classify the current implementation with `docs/prototype_ladder.md`. If a
   higher-upside earlier stage is missing, test that architecture route before
   spending iterations on later-stage micro-tuning.

## Hypothesis Gates

Every proposal must pass these gates before code edits:

- **Evidence gate**: ground the hypothesis in benchmark plus normalized profile
  evidence, or in an explicit no-reference performance model.
- **Impact gate**: estimate maximum possible end-to-end gain. Reject ideas whose
  dynamic coverage is small or theoretical gain is below the keep threshold.
- **Generality gate**: reject benchmark-shape overfitting. Do not specialize on
  runtime-varying dimensions, counts, layout states, or platform conditions
  unless they are true API or production invariants.
- **History-neighborhood gate**: skip adjacent variants of negative/rejected
  strategies unless new evidence shows that the bottleneck moved.
- **Priority gate**: spend early iterations on the dominant bottleneck. Leave
  cleanup and noise-floor tuning for after a structural improvement is working.
- **Design-boundary gate**: if profile/model evidence shows a large gap in total
  work, handoff cost, synchronization, main-loop trip count, or launched work,
  mark the design-boundary state and switch to an architecture route.
- **Negative-evidence scope gate**: a failed partial graft that still carries
  the old bottleneck does not disprove a full redesign that removes it.
- **Architecture-route budget gate**: high-upside structural routes get a finite
  repair budget. First-version correctness failures or regressions should be
  fixed inside the route until validation or budget exhaustion.
- **Prototype-ladder gate**: before local cleanup, record current ladder stage,
  next missing high-upside stage, promotion gate, and negative evidence scope.

Rules:

- One change per experiment.
- Do not tune a low-ceiling prototype just because it is the nearest editable
  code.
- Runtime checks for common states are allowed when they preserve correctness
  and performance portability across all supported sizes. Benchmark-only
  dispatch is not allowed.
- Boundary-only changes must state their dynamic-work coverage and should be
  deprioritized behind steady-state changes when coverage is small.
- Mock platform metrics are workflow evidence only, not hardware truth.

## Design-Boundary And Architecture Route Mode

Mark design-boundary state when the current implementation is structurally
limited:

```bash
.venv/bin/python tools/run_loop.py \
  --hypothesis "mark design-boundary limited state" \
  --mark-design-boundary \
  --design-boundary-reason "dominant cost is a repeated dataflow handoff" \
  --state-only
```

Run a structural route with explicit metadata:

```bash
.venv/bin/python tools/run_loop.py \
  --hypothesis "prototype new ownership route" \
  --architecture-route \
  --route-invariant "remove the dominant dataflow boundary" \
  --route-expected-impact "affects steady-state mainloop; target >5% end-to-end" \
  --route-budget 8 \
  --route-stop-condition "validated implementation below threshold or budget exhausted" \
  --route-plan workspace/runs/run_xxx/architecture_route_plan.md \
  --route-iteration-role prototype
```

When design-boundary mode is active and the route id is new, the route plan is
mandatory. It must contain at least two structurally distinct route candidates
and must not contain placeholders such as `fill_me`, `todo`, or `tbd`.

Use `--route-allow-regression` only when a correctness-passing structural
prototype must remain as the base for follow-up sub-iterations. Do not use it
for final validation.

## Decide

| Condition | Action |
|---|---|
| correctness failed | Reject the experiment. |
| profile required but missing | Reject the experiment. |
| timing unstable | Treat as inconclusive; run stable/full validation before keep/reject. |
| stable full/comparable benchmark improved by threshold | Keep and record positive evidence. |
| stable full/comparable benchmark same or worse | Record negative evidence. |
| non-validation route failed or regressed | Record `inconclusive`, not broad route rejection. |

Only a `validation` route sub-iteration can mark the broader route negative, and
only when the implementation satisfies the route invariant.

## Required Artifacts

Each run should produce:

- `run_manifest.json`
- `final_summary.md`
- `preflight_check.json`
- `preflight_check.md`

Each iteration should produce:

- `kernel.snapshot.py`
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

## Strategy Memory Rules

- Every proposal must include `## Strategy tags`.
- Use normalized profiler outputs, not raw vendor-specific counter names.
- Avoid blocked fingerprints unless new evidence is recorded.
- Prefer proven fingerprints only when the current shape regime and bottleneck
  match the recorded evidence.
- Record outcomes as `positive`, `negative`, `rejected`, or `inconclusive`.
- Update route state and design-boundary state whenever structural evidence
  changes.
