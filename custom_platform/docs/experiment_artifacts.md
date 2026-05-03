# Experiment Artifacts

Each run is stored under `workspace/runs/run_<timestamp>/`.

Per run:
- `run_manifest.json`
- `final_summary.md`
- `preflight_check.json`
- `preflight_check.md`

Per iteration:
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

Architecture-route metadata:
- `run_manifest.json` records `architecture_route` for each sub-iteration.
- `workspace/strategy_memory/global_strategy_memory.json` records route state
  under `scopes.<platform_kernel>.routes`.
- `workspace/strategy_memory/global_strategy_memory.json` also records
  `scopes.<platform_kernel>.design_boundary`. When active, normal local
  experiments are rejected unless explicitly overridden.
- A route plan can be stored as `architecture_route_plan.md` or JSON in the run
  directory. Under an active design-boundary marker, new architecture routes
  must point to a plan with at least two route candidates.
- JSON route plans must be top-level objects with `prototype_ladder` and
  `routes` keys. Bare route arrays are rejected because they cannot represent
  shared ladder state.
- Route plans must include prototype-ladder state: current stage, next missing
  high-upside stage, promotion gate, and negative-evidence scope.
- Non-validation route failures are recorded as `inconclusive`; review them as
  repair evidence, not as blocked strategy fingerprints.
- If `--route-allow-regression` is used, the kept kernel is a route prototype,
  not a validated best kernel. A later `validation` sub-iteration must pass the
  normal full/stable keep criteria.
