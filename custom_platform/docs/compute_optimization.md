# Compute Optimization Placeholder

Replace this document with compute tuning guidance for the target platform.

Recommended sections:

- matrix acceleration instructions
- reduction patterns
- divergence model
- register pressure model
- instruction throughput limits
- mixed precision guidance

## Execution Group / Workgroup Geometry Strategy

Choose the number of execution groups per workgroup together with tile shape,
per-lane work, register or temporary-storage budget, on-chip-memory footprint,
and synchronization scope. Do not tune group count as an isolated knob.

Initial platform-neutral rules:

- Start from kernel style: streaming, reduction, dense matrix/vector compute,
  producer-consumer, or persistent scheduling.
- Increase groups only when profile evidence shows latency hiding or primitive
  feed rate is the limiting factor and resource limits still allow useful
  residency.
- Reduce groups when synchronization, waiting, local-storage handoff, register
  pressure, or occupancy loss dominates.
- For dense matrix/vector primitives, choose geometry from the primitive tile
  and issue requirements first, then verify register pressure and residency.
- For producer-consumer kernels, add helper groups only if profiler evidence
  proves overlap. Otherwise they often enlarge the synchronization domain.
- For persistent kernels, choose grid residency and tile ownership first; group
  count is secondary to keeping all execution units fed.

Required evidence before changing geometry:

- current and proposed workgroup size, execution-group count, tile shape, and
  per-lane work
- resource model: registers/temporaries, on-chip storage, expected resident
  workgroups, and active execution groups
- bottleneck metric: eligible-work shortage, memory dependency, synchronization,
  primitive underuse, local-storage contention, or instruction issue pressure
- dynamic coverage: reject geometry changes that only affect tails or rare paths
  unless they are a probe for a larger route

Anti-patterns:

- increasing group count only because occupancy looks low
- copying a reference group count without matching its dataflow and tile shape
- retrying adjacent group-count sweeps after a negative result without a changed
  bottleneck
- using many producer/helper groups without measured overlap
