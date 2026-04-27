# Strategy Memory

Structured strategy memory is stored in `workspace/strategy_memory/global_strategy_memory.json`.

The system records:
- `positive`: faster than the previous comparable attempt
- `negative`: valid but slower or equal
- `rejected`: correctness failure, profiling failure, or incomplete evidence

Each strategy is identified by:
- normalized `strategy_tags`
- a stable `strategy_fingerprint`

Each new iteration should:
1. avoid blocked fingerprints from `rejected`
2. prefer fingerprints in `positive`
3. record the current outcome back into the memory store
