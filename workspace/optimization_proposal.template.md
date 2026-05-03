# Optimization proposal

## Backend
- cuda

## Primary references
- docs/memory_optimization.md
- docs/compute_optimization.md
- docs/stall_reasons.md
- docs/arch_notes.md

## Evidence
- Fill in the bottleneck diagnosis from benchmark and NCU evidence.

## Impact gate
- Benchmark shape(s) affected:
- Hot-path coverage estimate:
- Theoretical best-case end-to-end speedup:
- Keep threshold comparison:
- Decision: proceed / reject before implementation

## Generality gate
- Runtime-varying dimensions/properties touched:
- Stable invariant used, if any:
- Does this rely on fixed benchmark sizes or benchmark-only dispatch? yes / no
- If specialized, why is the specialized property a real operator/API/production invariant rather than a benchmark artifact?
- Decision: proceed / reject before implementation

## Strategy constraints from memory
- blocked fingerprints: none
- preferred fingerprints: none
- adjacent negative/rejected strategies:
- new contradictory evidence if repeating a nearby strategy:
- negative evidence scope: exact implementation / local neighborhood / broader route
- old bottleneck still present in prior negative? yes / no / unknown

## Strategy tags
- baseline

## Design-boundary gate
- Is the current kernel design-boundary limited by reference/NCU evidence? yes / no
- Dominant old boundary or materialized intermediate:
- Route invariant this proposal preserves:
- Does this proposal fully remove or replace that boundary, or only patch around it?
- If this is a high-upside route, route budget and stop condition:
- Why failures in adjacent old-design experiments do or do not apply:

## This iteration
- Describe one focused change.
- State why it should improve performance.
- State what metric or stall should improve if the hypothesis is correct.
- State why this is higher priority than smaller boundary/tail/layout tweaks.
