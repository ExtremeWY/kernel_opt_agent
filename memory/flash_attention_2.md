# flash_attention_2 Optimization Log

## 2026-05-02 - flash_attention_2_20260502_run057_nonfast_iter

- Analyzed the manual-v4 fast path design: `BQ64/BK32/4warps`, swizzled shared
  memory, K/V double buffering, `ldmatrix.x4`, explicit `mma.sync`, and
  register-resident score/probability. The original no-tail v4 path is correct
  only for full Q blocks and full K/V tiles, which is why non-64-multiple
  sequence lengths previously fell back to the generic `BQ64/BK16/4warps`
  regtc kernel.
- Ran 10 non-fast-path experiments on `seq_len=1031,2057,4099,8191`,
  `B=8,H=8,D=128,bf16`, both causal and non-causal. The only high-coverage
  route that consistently moved performance was a tail-safe v4 dataflow with
  guarded Q/K/V staging, invalid-score masking, guarded output stores, and
  `__launch_bounds__(128,3)`.
- Best raw route (`v10_tail_v4_launchbounds3`) achieved two-run median
  non-fast speedups of about `1.034x` overall, `1.051x` causal, and `1.017x`
  non-causal versus the previous fallback.
- Did not integrate raw v10 for all paths because it introduced avoidable
  non-causal divisible fast-path risk. Integrated a conservative variant:
  keep the existing no-tail v4 full fast path for
  `!causal && seq_len % 64 == 0`, add a new tail-safe v4 kernel with
  `launch_bounds(128,3)`, and route all `causal && seq_len > 0` through the
  tail-safe v4 kernel. Non-causal non-divisible lengths continue to use the
  existing BK16 regtc fallback until non-causal gains are more stable.
- Post-integration validation passed full correctness. Representative causal
  full benchmark results: `medium=0.2605 ms`, `large=0.9051 ms`,
  `xlarge=3.4045 ms` in the stable targeted rerun. Edge correctness passed for
  causal and non-causal `seq_len=127,128,129,1031,2057,4099,8191,8192`.

## 2026-05-02 - flash_attention_2_20260502_run054_manual_borrow

- Analyzed and experimented with techniques from
  `manual_cuda_kernel/07_attention/attention_v1.cu` through `attention_v5.cu`.
  The useful manual progression on this GPU was v2 shared-memory swizzle,
  v3 K/V cp.async pipeline, and v4 `ldmatrix.x4`; manual v5's `BK64 + V single
  buffer` was not fastest for the current square non-causal sweep.
- Rejected a non-causal full-tile predicate-free branch inside the current
  `BK16` kernel. Correctness passed, but same-run A/B non-causal sweep regressed
  by about `4%~6%`, so the small predicate savings did not justify the branch
  and codegen cost.
- Rejected a generic `BK32` version of the current register-score kernel.
  Correctness passed for causal and non-causal cases, but same-run A/B
  non-causal sweep regressed by about `2%~5%`. The larger score/prob/PV live
  state and shared-memory footprint outweighed fewer mainloop iterations.
- Kept an isolated manual-v4-style non-causal fast path for
  `seq_len % 64 == 0`: `BQ64/BK32/4warps`, swizzled shared memory, K/V
  double-buffering, `ldmatrix.x4`, and explicit `mma.sync`. Causal and
  non-64-multiple non-causal shapes continue to use the original `BK16`
  register-score/probability kernel.
- Kept-path A/B non-causal sweep (`B=8,H=8,D=128,bf16,causal=false`) improved
  all tested sequence lengths versus the pre-run baseline:
  `1024 +3.46%`, `2048 +3.54%`, `3072 +3.43%`, `4096 +4.17%`,
  `5120 +3.21%`, `6144 +5.17%`, `7168 +4.54%`, `8192 +4.05%`.
- Correctness passed for the kept path and fallback path, including causal,
  non-causal, and non-multiple `seq_len=1031`. Default causal full benchmark
  after the change also passed: `medium=0.2716 ms`, `large=0.9465 ms`,
  `xlarge=3.5273 ms`.

## 2026-05-02 - flash_attention_2_20260502_run052_cleanup_no_scalar

- Cleaned the active CUDA source around the current best
  `flash_attention_2_forward_regtc_kernel<64,16,128,4,causal>` path. Removed
  the obsolete scalar long-sequence fallback, historical WMMA/BM32 kernels,
  dead launch wrappers, and unused helper/constants tied to those paths.
- The entrypoint now always dispatches the register-score/probability explicit
  Tensor Core kernel for both `seq_len <= 4096` and `seq_len > 4096`; there is
  no `launch_scalar` threshold path left. The implementation remains CUDA-only
  and uses explicit `ldmatrix + mma.sync`; score/probability stay in registers.
- Replaced the remaining `wmma::fragment` accumulator storage with plain
  register arrays and removed the `<mma.h>` dependency. Added an explicit
  `BLOCK_KV == 16` static assertion because this kernel computes one 16-column
  KV tile per mainloop iteration.
- Full benchmark after cleanup passed correctness:
  `medium=0.2685 ms`, `large=0.9448 ms`, `xlarge=3.5536 ms`. This is within
  the same performance band as the previous kept source, with full-large timing
  still marked unstable (`spread=3.24%`).
- Explicit `seq_len > 4096` correctness passed against PyTorch SDPA reference:
  causal `4097`, `4608`, and `8192` all passed with `max_err=3.90625e-03`;
  non-causal `4097`, `4608`, and `8192` all passed with
  `max_err=4.882812e-04`.
- Synced the cleaned source to `kernels_optimized/flash_attention_2.cu` and
  saved the before/after artifacts under
  `workspace/runs/run_052_cleanup_no_scalar/`.

## 2026-04-30 - flash_attention_2_20260430_run051

- Continued from the run_050/run_051 register-score Tensor Core path:
  `flash_attention_2_forward_regtc_kernel<64,16,128,4,causal>`. The optimized
  implementation remains custom CUDA C only, with score/probability in
  registers and QK/PV using explicit `ldmatrix + mma.sync`.
- Kept `v8_full_active_rows_staging_fastpath`: a runtime
  `active_rows == BLOCK_KV` fast path for the active async K/V staging loops.
  Full/tail tiles still use the guarded fallback, so this is not compile-time
  specialization on runtime-varying `seq_len`.
- Default full benchmark reached the requested all-size `<4%` gap target:
  `medium=0.2640 ms (+2.47% vs PyTorch)`,
  `large=0.9169 ms (+3.74%)`,
  `xlarge=3.4228 ms (+2.58%)`, correctness PASS.
- 9-trial full validation also stayed below 4%:
  `medium=0.2648 ms (+3.78%)`,
  `large=0.9166 ms (+3.58%)`,
  `xlarge=3.4425 ms (+2.90%)`; medium timing was marked unstable, but default
  full was stable on all kernel sizes.
- NCU confirms the mechanism: v8 preserved tensor instructions
  (`17,301,504`), registers/thread (`129`), dynamic shared memory (`17,408B`),
  and zero shared bank conflicts, while reducing total instructions
  `196.8M -> 189.8M` and LSU instructions `22.7M -> 20.0M`.
- Rejected `v7_cp_async_l2_128b`: correctness passed, but full/9-trial did not
  clear the keep threshold (`medium +4.15%`, `large +4.42%`, `xlarge +3.38%`).
  Do not continue cache-hint prefetch-size sweeps without new NCU evidence.

## 2026-04-30 - flash_attention_2_20260430_run050

- Continued from the run_048/run_049 restored best
  `flash_attention_2_forward_regtc_kernel<64,16,128,4,causal>`, preserving
  CUDA-only implementation, register-resident score/probability, and explicit
  `ldmatrix + mma.sync` Tensor Core use.
- Kept one launch-path cleanup: removed the active regtc
  `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize)` calls.
  The regtc path uses only `17408B` dynamic shared memory, so per-launch opt-in
  is unnecessary. Synced this kept change to `kernels_optimized/flash_attention_2.cu`.
- Current stable full validation:
  `medium=0.2744 ms (+6.95% vs PyTorch)`,
  `large=0.9483 ms (+6.02%)`,
  `xlarge=3.5415 ms (+5.45%)`, correctness PASS. The requested all-size
  `<4%` target was not reached.
- Rejected this round:
  warp-local full-prefix score/mask fastpath, removing post-launch
  `cudaGetLastError`, dropping redundant C++ validation, direct output
  allocation, lazy softmax scaling to avoid accumulator alpha, branching around
  accumulator alpha when max is unchanged, skipping Python validation, global
  base pointer hoist, and a `BQ128/4warps` dual-panel structure.
- Important negative evidence:
  the accumulator-alpha hotspot cannot be fixed by lazy scaling or runtime
  branch insertion in the current ownership model; both regress to about
  `1.04 ms` quick-large. The `BQ128/4warps` dual-panel route is correctness
  viable but regresses to about `4.23 ms`, indicating register pressure/spills
  overwhelm K/V reuse. Do not continue adjacent BQ128 dual-ownership or
  accumulator-branch variants without a new design that first reduces live
  accumulator state.
- Next route:
  the current `BQ64/BK16/4w` register-score dataflow is design-boundary
  limited. Further work should start only from a new ownership redesign with a
  credible register-pressure plan and NCU keep metrics; skip additional
  predicate, address, launch, allocator, final-store, BK32/BK64 parameter, and
  BQ128 dual-panel variants.

## 2026-04-30 - flash_attention_2_20260430_run048

- Continued from run_047 register-only coalesced final writeback. Restored the
  final reciprocal hoist regression so the active path is again
  `flash_attention_2_forward_regtc_kernel<64,16,128,4,causal>` with
  score/probability in registers and explicit `ldmatrix + mma.sync`.
- Synced the current best root kernel into
  `kernels_optimized/flash_attention_2.cu` and `.py`.
- Full validation after five experiments:
  `medium=0.2751 ms` (`+5.05%` vs PyTorch), `large=0.9606 ms`
  (`+5.53%`), `xlarge=3.5482 ms` (`+3.60%`). Correctness passed; xlarge is
  within the requested 4% target, medium/large are not.
- NCU for the restored v1 candidate: `1.0868 ms`, `204.4M` total instructions,
  `17.3M` tensor instructions, `23.2M` LSU instructions, `129 regs/thread`,
  `17408B` dynamic shared, global store sectors `~1.05M` equal to ideal,
  shared bank conflicts `0`.
- Rejected this round:
  hand-unrolled accumulator alpha mapping (SASS-equivalent), `cp.async`
  `ignore-src` staging, pre-scaling accumulators before PV MMA, manual shared
  base-address arithmetic, and row-level `valid_cols` predicate compression.
  These were correctness-pass experiments but either produced identical NCU
  instruction counts or clear quick-large regressions.
- Next route:
  do not spend more iterations on staging branch variants, predicate reshaping,
  shared address hoisting, final-store micro-tuning, launch hints, or resource
  threshold tricks. The remaining <4% target requires a new ownership/mainloop
  design that reduces the `BK16` instruction/update overhead without repeating
  the already failed naive `BK32/BK64` neighborhoods.

## 2026-04-30 - flash_attention_2_20260430_run046

- Continued from run_045 final `BQ64/BK16/4w` register-score/prob explicit
  Tensor Core path. The optimized implementation remains custom CUDA C only.
- Kept a structural shared-memory alias change: after Q fragments are cached in
  registers, K/V double-buffer operand tiles reuse the Q shared region. This
  reduced dynamic shared memory from `34816B` to `17408B`, raised active warps
  to about `24.3%`, and removed the remaining shared bank conflicts in targeted
  NCU.
- Kept a double-buffer boundary cleanup: only the first tile uses the load-ready
  barrier at the start of the loop, avoiding a back-to-back CTA barrier between
  consecutive prefetched tiles.
- Best full result:
  `medium=0.2755 ms (1.0540x PyTorch)`,
  `large=0.9443 ms (1.0487x)`,
  `xlarge=3.5537 ms (1.0500x)`. The target “all full benchmark sizes within
  5% of PyTorch” was not stably reached.
- v2 targeted NCU:
  `gpu__time_duration.sum=1.0998 ms`, `sm=40.75%`, `dram=18.27%`,
  `204.8M` total instructions, `17.3M` tensor instructions,
  `129 regs/thread`, `17408B` dynamic shared, shared bank conflicts `0`, and
  global store sectors still about `2.10M`.
- Rejected:
  first-tile preload outside mainloop, forced `__launch_bounds__(128,4)`,
  first-tile owner-warp staging, half-warp final store, full-visible valid-key
  flag, deleting the unused regtc `batch_size` parameter, skipping the final
  empty `cp.async.wait_group`, `BQ128/8w` after aliasing, disabling
  double-buffering, and `launch_bounds + shared carveout`.
- Next useful route:
  the remaining high-upside target is register-only coalesced final writeback
  or a deeper accumulator ownership change that reduces global store sectors
  without returning to shared scratch. Do not continue cache hints, padding
  cleanup, maxrregcount/launch_bounds, BQ128/BQ96/BQ80 sweeps, or full-visible
  predicate branches without new contradictory source-line evidence.

## 2026-04-29 - flash_attention_2_20260429_run042_profile_compare

- Profiled the current CUDA kernel against the local Triton reference and
  PyTorch fused SDPA reference on `B=8,H=8,S=2048,D=128,bf16,causal`, after
  full benchmark across medium/large/xlarge.
- Benchmark large: CUDA `1.0394 ms`, Triton best `1.0279 ms`
  (`BM=128, BN=32, 4warps, stages=2`), PyTorch `0.9267 ms`. CUDA is now close
  to Triton in event timing but still about `12%` slower than PyTorch.
- NCU large: CUDA total instructions `228.4M` vs Triton `108.1M` and PyTorch
  `89.4M`; tensor instructions remain matched (`17.3M` vs PyTorch `17.3M`).
  Global store sectors are still about `2x` reference, shared store wavefronts
  are `2.75x` Triton and `5.69x` PyTorch, and `short_scoreboard/wait/barrier`
  stalls remain far above reference.
- Next route: source/SASS line attribution before more local changes; then only
  pursue high-coverage routes that reduce mainloop instruction/LSU/stall count,
  especially a true register-score/prob `BK64` ownership redesign. Do not repeat
  active pitch128, pad-zero removal, Q `cp.async`, final scratch writeback,
  `kv_end` loop-bound, final `inv_l` hoist, or split K/V `cp.async` neighbors
  without new contradictory profile evidence.

## 2026-04-29 - flash_attention_2_20260429_run041

- Continued from the run_040 explicit-MMA/register-score path. The active CUDA
  implementation remains `flash_attention_2_forward_regtc_kernel<64,16,128,4,causal>`.
  Score/probability remain register-resident and QK/PV use lower-level
  `ldmatrix + mma.sync`; no Triton/library dispatch is used as the optimized
  implementation.
- Final full validation reached the requested target:
  `medium=0.2996 ms`, `large=0.9986 ms`, `xlarge=3.6720 ms`, correctness PASS,
  large timing stable with spread `0.98%`. Full-large validation before the
  all-size run measured `large=0.9902 ms`.
- Kept changes:
  `cp.async.ca` K/V staging with packed direct bf16 pair writeback from run_040,
  skipping the final per-KV-loop CTA barrier when there is no next K/V tile,
  and warp-private Q staging with `__syncwarp()` instead of CTA-wide
  cooperative Q staging. A post-revert full-large validation measured
  `large=0.9894 ms`.
- NCU on the final kept version showed `compute_bound (sm=38.7%, dram=16.7%)`,
  `registers/thread=136`, dynamic shared `26.1 KB`, total instructions
  `228.4M`, tensor instructions `17.3M`, global store sectors `2.10M`, shared
  store wavefronts `1.63M`, and low L1/high L2 hit pattern. The remaining gap
  is still instruction/scheduling overhead around the explicit-MMA mainloop
  rather than missing tensor-core work.
- Rejected in this round:
  split K/V `cp.async` staging, final shared-scratch vector writeback,
  active regtc shared pitch128, host-side shared allocation shrink to pitch128,
  regtc pad-zero-store removal, `kv_end` loop-bound rewrite, final `inv_l`
  hoist, Q tile `cp.async`, and warp-private Q pad-zero ownership. These either
  failed correctness, regressed full-large timing, or were unstable.
- Lesson:
  The successful gains came from removing synchronization/data-movement
  boundaries that NCU and code ownership supported: unnecessary final CTA
  barrier and unnecessary CTA-wide Q-staging synchronization. Adjacent cache
  hints, loop-bound rewrites, and final writeback scratch routes should not be
  retried without new contradictory profile evidence.

## 2026-04-29 - flash_attention_2_20260429_run038

- Continued from run_037 best `BQ64/BK16/4w` register-score Tensor Core kernel.
- Final kept active path uses explicit lower-level `ldmatrix + mma.sync` for both
  QK and PV in `flash_attention_2_forward_regtc_kernel<64,16,128,4,causal>`.
  Score/probability remain register-resident; no Triton/CUTLASS/cuDNN/cuBLAS or
  PyTorch SDPA implementation is used.
- Final full validation: `medium=0.3744 ms`, `large=1.1719 ms`,
  `xlarge=3.9850 ms`, correctness PASS, large timing spread `0.39%`.
  The requested `large < 1.0 ms` target was not reached within the 20
  sub-iteration budget.
- Kept structural changes:
  explicit QK/PV MMA route (`1.1972 ms` full large), PV `ldmatrix.x4.trans`
  with corrected high-half lane addresses (`1.1851 ms`), and QK K operand
  `ldmatrix.x4` with lanes 16-31 addressing high key columns (`1.1796 ms`,
  final rerun `1.1719 ms`).
- NCU for the explicit-MMA route vs the run_038 WMMA start: total instructions
  `278.5M -> 225.3M`, LSU instructions `67.0M -> 32.7M`, registers/thread
  `165 -> 132`, tensor instructions unchanged. This confirms the main win is
  removal of WMMA fragment/LSU wrapper overhead, not a change in mathematical
  work or global traffic.
- Rejected: explicit-MMA `BK32` dual operand (`1.4180 ms` quick), `BK32`
  single operand (`2.0161 ms` quick), `BK16` single operand (`1.5871 ms`
  quick), full-prefix branch hotpath (`1.4296 ms` quick), XOR 4-lane
  reduction (`1.3742 ms` quick), replacing residual accumulator fragment with
  `float[8]` (`1.2037 ms` full), and direct probability pair packing
  (`1.1928 ms` quick). Do not continue these neighborhoods without new NCU
  evidence.
- Next useful route: a new explicit-MMA mainloop/ownership design that further
  reduces shared operand movement and softmax/PV register live range. Avoid
  returning to WMMA API, BK32/BK64, single-operand staging, branch cleanup, or
  local probability packing variants.

## 2026-04-29 - flash_attention_2_20260429_run037

- Ran 21 CUDA-only experiments for the requested 20-round deep optimization
  cycle. Iteration 21 was a correction experiment to invalidate an earlier
  noisy pitch128 conclusion that did not affect the active regtc path.
- Final kept active kernel:
  `flash_attention_2_forward_regtc_kernel<64,16,128,4,causal>` for the
  wide-CTA path. The implementation is synced in `kernel.cu` and
  `kernels_optimized/flash_attention_2.cu`.
- Kept structural changes:
  moved from `BQ128/BK16/8warps` to `BQ64/BK16/4warps`, and cached the full
  per-warp Q WMMA fragment set in registers before the KV loop. Score and
  probability remain register-resident, and QK/PV still use Tensor Core.
- Final full benchmark:
  `large=1.3142 ms`, `52.316 TFLOPS`, correctness PASS, timing spread
  `0.77%`. Versus the trusted run_037 rerun baseline `large=1.3796 ms`, this
  is about `+4.7%`.
- Supplementary full validation after syncing artifacts:
  `large=1.3200 ms`, correctness PASS, large timing spread `0.31%`; this is
  consistent with the 5-trial final rerun.
- Compare benchmark:
  CUDA `1.3159 ms`, Triton best `0.9795 ms`, PyTorch `0.8911 ms`. The current
  CUDA kernel is still about `1.34x` slower than Triton and `1.49x` slower than
  PyTorch on the large case.
- NCU after the kept structure showed the expected mechanism:
  LSU instructions `0.800x`, shared load wavefronts `0.654x`, and NCU time
  `0.953x` versus the old regtc profile. Total instructions only moved to
  `0.981x`, and shared store wavefronts increased to `1.84x`, so the remaining
  bottleneck is still WMMA fragment/operand movement and mainloop organization.
- Rejected or blocked:
  standard-WMMA `BK32/BK64` variants, `BQ96/6w`, `BQ80/5w`, direct global Q/V,
  half Q-cache, launch bounds, full-warp fastpath, output address/inv_l
  micro-tuning, active pitch128, and `__exp2f` intrinsic. Do not repeat these
  adjacent variants without new contradictory NCU evidence.
- Next useful route:
  lower-level `mma.sync`/`ldmatrix` register-layout redesign on the validated
  `BQ64/BK16/4w` dataflow, then possibly a reference-like `BN32/BN64`
  producer-consumer mainloop. Do not continue WMMA API micro-tuning in the
  current neighborhood.

## 2026-04-29 - flash_attention_2_20260429_run035

- Continued the register-score/probability Tensor Core route. Final active
  kernel is `flash_attention_2_forward_regtc_kernel<128,16,128,8,causal>` for
  `seq_len <= 4096`.
- Hard constraint satisfied: QK score fragments and softmax probability
  fragments remain in registers; no `score_tile` / `prob_tile` shared-memory
  materialization is used in the active regtc path. PV uses Tensor Core via
  `wmma::mma_sync`.
- Final full benchmark:
  `medium=0.4297 ms`, `large=1.3911 ms`, `xlarge=4.9442 ms`, correctness PASS.
  Versus regtc v1 (`large=1.9261 ms`, `xlarge=6.4678 ms`), this is about
  `+27.8%` on large and `+23.6%` on xlarge.
- Kept structural changes: reduced `BLOCK_KV` from `64` to `16`, fixed the
  operand-tile reuse race with a barrier during the intermediate BKV32 probe,
  split K and V into separate shared operand tiles, then raised query tile to
  `BLOCK_Q=128` with `8` warps after the split-K/V dataflow made it profitable.
- Final targeted NCU: `128 regs/thread`, `44.5 KB` shared/block, `2 CTA/SM`,
  `32.4%` active warps, `283.9M` total instructions, `17.8M` tensor
  instructions. Grid for large is `1024` CTAs.
- Rejected: shared pitch `128`, forced `__launch_bounds__(128,5)`,
  `BLOCK_Q=256/16 warps`, and removing regtc padding zero stores. Do not repeat
  these adjacent variants without new contradictory NCU evidence.
- Next useful route: keep the same register score/prob + Tensor Core dataflow,
  but replace standard WMMA with explicit lower-level `mma.sync` only if it
  reduces WMMA fragment/control overhead without raising register pressure.

## 2026-04-29 - flash_attention_2_20260429_run031

- Ran 20 CUDA-only lower-level design iterations. Final kept version is a new
  `BM32/BN64/8-warp` row-group CUDA mainloop, not the failed BM64 standard-WMMA
  route and not a Triton/library replacement.
- Final full benchmark: `medium=0.5964 ms`, `large=2.0390 ms`,
  `xlarge=7.6669 ms`, correctness PASS. Versus run_031 baseline
  `large=2.3135 ms`, this is `+11.9%` on large and `+12.9%` on xlarge.
- Key kept structure: two 16-row groups per CTA; four contiguous warps per row
  group; four score-column WMMA tiles computed in parallel; softmax rows split
  across the row-group warps; each warp owns two value-tile PV accumulators.
  Score and probability share one shared-memory region with `scorePitch=68` and
  `probPitch=136`; K/V operand pitch remains `136`.
- Final targeted NCU: compute-bound, SM `68.2%`, DRAM `8.4%`, occupancy
  `33.0%`, registers/thread `64`, tensor-core pct `4.5%`, shared bank
  conflicts `16.43M`.
- Negative evidence: BM64 row-group variants remained too slow even after 10
  attempts; best BM64 was quick `2.7175 ms`. BM32 operand pitch `128` and
  `144`, score/prob pitches `76/152` and `80/160`, launch bounds, and split-Q
  pitch `128` all regressed.
- Next useful route: continue from BM32 final with NCU source attribution and
  target shared footprint/synchronization reduction. Do not return to BM64
  row-group or blind pitch sweeps without new source-line evidence.

## 2026-04-28 - flash_attention_2_20260428_run027

- Ran 20 CUDA-only iterations focused on the previously identified high-upside
  lower-level `mma.sync` / register-dataflow route. No Triton, CUTLASS, cuBLAS,
  cuDNN, PyTorch SDPA, or library dispatch was used.
- No source change was kept. Final restored full benchmark:
  `medium=0.6675 ms`, `large=2.3435 ms`, `xlarge=9.0584 ms`, correctness PASS.
  The `large < 1.8 ms` target was not reached.
- Validated explicit BF16 `mma.sync.m16n8k16` operand mapping:
  QK scalar packing was correct but regressed to `3.0568 ms`; `ldmatrix`
  required non-transposed B loads for the staged K layout. Even after A-fragment
  reuse, QK-only explicit MMA remained much slower (`~2.80-2.90 ms`) because it
  still wrote scores to shared memory and increased instruction/register cost.
- Explored inline PV `mma.sync + ldmatrix` and worker direct-output dataflow.
  The best branch reached quick `large=2.3863 ms`, but it remained slower than
  the restored baseline and used `80` registers/thread versus baseline `56`,
  losing the 3-CTA/SM residency budget. Launch bounds, direct persistent
  accumulation, reduced unroll, split phases, output packing, and scratch/pitch
  variants all regressed.
- Tried preserving baseline standard-WMMA full-prefix PV while moving only the
  diagonal/output path to worker-side direct output. This also regressed
  (`2.96-3.01 ms`), indicating the extra code/register pressure outweighs the
  removed scratch import in the current monolithic kernel.
- Learned:
  explicit-MMA grafts onto the current M16 standard-WMMA CTA are blocked by
  register pressure and codegen. Further progress needs a new mainloop
  architecture that reduces K/V reload and score/probability materialization
  together while staying near the baseline register budget. Do not continue
  QK-only inline MMA, inline-PV grafts, direct-output variants, owner-warp
  reductions, pitch 128/64 sweeps, launch-bound forcing, or output packing in
  this dataflow.

## 2026-04-28 - flash_attention_2_20260428_run026

- Ran 20 CUDA-only optimization iterations from the current run_023/run_025
  best. No Triton, PyTorch SDPA, CUTLASS, cuDNN, cuBLAS, or library dispatch was
  used.
- No source change was kept. Final restored validation:
  `medium=0.6319 ms`, `large=2.2527 ms`, `xlarge=8.5937 ms`, correctness PASS.
  The `large < 1.8 ms` target was not reached.
- Reverted structural negatives:
  manual QK score stores (`2.8115 ms`), `N=128` key tiles (`4.1051 ms` after
  correctness fix), 8-worker single-phase PV (`2.4858 ms`), named barriers
  (`2.2835 ms`), padding-zero removal (`2.8122 ms`), vectorized output stores
  (`2.8359 ms`), diagonal masked PV-WMMA (correctness FAIL, `2.7278 ms`),
  16-owner-warp expansion (`4.6609 ms`), and worker-side probability
  materialization (`3.1004 ms`).
- Explored the high-upside larger-query CTA route for 8 iterations. M64 improved
  from the prior `4.7008 ms` neighborhood to `3.0788 ms` with 16 warps and
  role-split QK/softmax/PV, and M32 partial-softmax reached `3.0523 ms`; both
  remained far slower than the current M16 best. Standard-WMMA M32/M64 is now
  blocked for this kernel.
- Resource hints (`PreferredSharedMemoryCarveout=100`,
  `cudaFuncCachePreferShared`) produced only noise-band results around
  `2.256 ms`, below the keep threshold.
- Learned:
  the current standard-WMMA dataflow is effectively exhausted. Further progress
  toward `large < 1.8 ms` requires a true lower-level MMA/register mainloop
  rewrite with explicit fragment ownership, not another standard-WMMA CTA-size,
  owner/worker, padding, sync, or cache-hint variant.

## 2026-04-28 - flash_attention_2_20260428_run025

- Continued CUDA-only optimization from the run_024 conclusion. No Triton,
  PyTorch, CUTLASS, or library dispatch was used.
- Reverted iter_v1: key-segment PV partial-output dataflow passed correctness
  but regressed to `large=6.1790 ms`. The large `4 x 16 x 128` float partial
  scratch, eight PV accumulator fragments per worker warp, reductions, and
  synchronization overwhelmed the removed shared score/probability traffic.
- Reverted iter_v2: reusing the four score/PV worker warps as owner rows while
  preserving `16` lanes per row passed correctness but regressed to
  `large=2.3796 ms`. Reducing CTA threads by one third was not enough to pay
  for added worker-warp responsibilities and register/scheduling pressure.
- Reverted iter_v3: producing probabilities directly from QK fragments in
  score/PV worker warps passed correctness but regressed to `large=3.6539 ms`.
  It removed score-tile materialization but moved probability generation onto
  the worker critical path and lost the current overlap between owner
  probability generation and worker V staging.
- Reverted iter_v4: CUDA `cp.async` for full-prefix K/V staging passed
  correctness but regressed to `large=2.4965 ms`. The simple async-copy
  conversion added pipeline overhead and waits without hiding enough latency.
- Learned:
  the current standard-WMMA CUDA dataflow is tightly balanced around overlap
  between owner probability generation and worker V staging. CUDA-only changes
  that move probability work to score/PV workers, reuse score workers as owners,
  or add partial-output reductions lengthen the critical path. Further progress
  toward `large < 1.8 ms` likely requires a full lower-level MMA/register
  rewrite with explicit fragment ownership, not another local standard-WMMA
  variant.

## 2026-04-28 - flash_attention_2_20260428_run024

- Compared the current run_023 best against the PyTorch fused FlashAttention
  reference with fresh NCU. Current remains `2.56x` slower on `large`
  (`2.2567 ms` vs `0.8809 ms`) despite nearly identical DRAM traffic and nearly
  identical tensor instruction count. The gap is total/LSU instruction overhead,
  shared-memory handoff, and repeated K/V staging from the `16x64` CTA shape.
- Source attribution on the current best showed the top remaining sources:
  WMMA accumulator shared stores (`12.98M` excessive wavefronts),
  `score_tile` owner reads (`4.06M`), `prob_tile` stores (`4.06M`), and full
  prefix K/V staging (`65.01M` L2 theoretical sectors each).
- Reverted iter_v1: an independent `64x64 / 4-warp` standard-WMMA CTA
  prototype passed correctness but regressed to `large=4.7008 ms`. The direct
  translation reduced grid count but used only `2` lanes per row and carried
  too much register/shared-memory cost.
- Reverted iter_v2: direct worker-side probability fragment construction from
  `score_tile + row_m` passed correctness but regressed to `large=2.7766 ms`.
  Removing `prob_tile` stores was outweighed by duplicate `exp2` work and
  fused-PV register/code pressure.
- Learned:
  the remaining high-value gap cannot be closed by standard-WMMA local
  rewrites. Block `standard_wmma_m64_four_warp_two_lanes_per_row` and
  `standard_wmma_prob_bypass_worker_recompute_exp2`. A future attempt needs
  lower-level MMA/register control that avoids both shared score/probability
  materialization and duplicate probability computation, or it should stop
  rather than spend iterations on adjacent pitch/cache/barrier variants.

## 2026-04-28 - flash_attention_2_20260428_run023

- Ran 10 impact-gated iterations from the run_022 best. Starting stable
  baseline rerun: `medium=0.6369 ms`, `large=2.2663 ms`,
  `xlarge=8.6350 ms`, `large=30.337 TFLOPS`, correctness `PASS`.
- No new source change was kept. Final validation of the unchanged current best:
  `medium=0.6319 ms`, `large=2.2567 ms`, `xlarge=8.5886 ms`,
  `large=30.466 TFLOPS`, correctness `PASS`, timing stable.
- Final targeted NCU: compute-bound, SM `65.9%`, DRAM `7.3%`, occupancy
  `74.1%`, registers/thread `56`, dynamic shared memory `32832 B`,
  tensor-core share `2.4%`, shared bank conflicts `30.9M`.
- Reverted / rejected:
  scratch aliasing was only `+0.38%`; direct global V regressed to
  `2.9001 ms`; persistent PV `alpha == 1` branch tied/regressed;
  `__launch_bounds__(384,4)` regressed to `3.1151 ms`; approximate inline
  `ex2` was only `+0.11%`; direct global K regressed to `3.2990 ms`;
  first PV tile direct assignment regressed to `2.8286 ms`; PV import phase gap
  regressed slightly; `__ldg` vector staging regressed to `2.7870 ms`.
- Learned:
  the neighborhoods adjacent to run_022's single operand tile and one-round PV
  import are exhausted. Direct-global WMMA operands, launch-bound forcing,
  exponent instruction micro-tuning, cache-load formatting, and local PV branch
  scheduling do not have enough upside in the current dataflow. Future work
  should cross a design boundary by reducing score/probability/PV
  materialization or changing ownership/main-loop structure while remaining
  runtime-general for variable sequence lengths.

## 2026-04-28 - flash_attention_2_20260428_run022

- Ran 5 impact-gated iterations from the run_021 best. Starting full benchmark:
  `medium=0.7148 ms`, `large=2.5918 ms`, `xlarge=9.8184 ms`,
  `large=26.527 TFLOPS`, correctness `PASS`, stable timing.
- Kept iter_v1: reused a single shared operand tile for K then V. K is staged
  and consumed for QK, then V is restaged into the same shared tile before PV or
  scalar fallback. Full benchmark improved to `large=2.2853 ms`.
- NCU confirmed the mechanism: dynamic shared memory dropped
  `45888 B -> 28480 B`, shared-memory occupancy limit moved `2 -> 3` CTAs/SM,
  and achieved occupancy rose `49.6% -> 74.1%`.
- Kept iter_v4: enlarged PV scratch pitch to `136` and imported both persistent
  PV accumulator phases in one handoff round. This preserved 3 CTAs/SM
  (`32832 B` dynamic shared), reduced instructions `717.2M -> 708.0M`, and
  improved the promotion full benchmark to `large=2.2566 ms`.
- Final validation after reverting sub-threshold variants:
  `medium=0.6314 ms`, `large=2.2611 ms`, `xlarge=8.5860 ms`,
  `large=30.407 TFLOPS`, correctness `PASS`. Versus run_022 baseline this is
  `+11.7%` on `medium`, `+12.8%` on `large`, and `+12.6%` on `xlarge`.
- Reverted / rejected:
  duplicate padding-store cleanup was only `+0.15%`; all-thread V restaging
  regressed; removing the final import-readback barrier was only `+0.08%` on
  `large` and mixed across sizes.
- Learned:
  the high-value post-run_021 direction was not another score/probability
  layout tweak. It was changing shared-memory residency and handoff granularity
  while preserving the proven shared WMMA operand path. The local neighborhoods
  around padding cleanup, V restaging scheduling, and import barrier shaving are
  now exhausted.

## 2026-04-28 - flash_attention_2_20260428_run021

- Ran 5 impact-gated iterations from the run_020 best. Starting full benchmark:
  `medium=0.7345 ms`, `large=2.6632 ms`, `xlarge=10.1643 ms`,
  correctness `PASS`, stable timing.
- Kept iter_v3: cached each owner lane's four full-prefix QK scores in
  registers between the row-max pass and the probability/sum pass. This removes
  the second shared-memory read of the same `score_tile` values without changing
  tile shape, sequence-length behavior, or score storage layout.
- Final validation:
  `medium=0.7126 ms`, `large=2.5966 ms`, `xlarge=9.8181 ms`,
  `large=26.478 TFLOPS`, correctness `PASS`. Versus run_021 baseline this is
  `+2.98%` on `medium`, `+2.50%` on `large`, and `+3.41%` on `xlarge`.
- Targeted NCU on iter_v3 confirmed the mechanism:
  total instructions `640.0M -> 632.9M`, shared load wavefronts
  `99.3M -> 91.1M`, shared load bank conflicts `10.5M -> 6.4M`,
  tensor instructions unchanged, registers/thread unchanged at `56`, and
  occupancy stayed `~49.6%`.
- Reverted / rejected:
  bf16x2 manual probability-fragment loads were only `+0.41%`; row-alpha load
  hoisting was only `+0.47%`; combining them was rejected by the impact gate;
  bf16x2 probability stores regressed `large` by `~8%`; score pitch `68` after
  the kept score cache tied/regressed versus iter_v3.
- Learned:
  there was still one real high-coverage duplicate shared-read pattern in the
  owner score/probability path. After removing it, the remaining local
  probability-load, alpha-load, probability-store, and score-pitch neighborhoods
  are below threshold or negative. The next high-value route should return to a
  structural score/probability handoff reduction or a reference-like CTA
  main-loop redesign, not more local shared-memory formatting.

## 2026-04-28 - flash_attention_2_20260428_run020

- Ran 5 iterations from the run_019 route. Starting full benchmark:
  `medium=0.7496 ms`, `large=2.7287 ms`, `xlarge=10.4572 ms`,
  correctness `PASS`.
- Kept iter_v1: full-prefix PV worker persistent accumulators. PV WMMA
  accumulators now remain in worker-warp registers across full-prefix tiles and
  are imported into owner accumulators only once before the diagonal/fallback
  tile. Full benchmark: `medium=0.7290 ms`, `large=2.6410 ms`,
  `xlarge=10.1760 ms`, correctness `PASS`.
- Final validation after reverting later experiments:
  `medium=0.7287 ms`, `large=2.6535 ms`, `xlarge=10.1332 ms`,
  `large=25.910 TFLOPS`, correctness `PASS`. Versus the run_020 baseline this
  is `+2.83%` on `large`, `+2.88%` on `medium`, and `+3.20%` on `xlarge`.
- NCU confirmed the intended movement:
  shared load wavefronts `127.7M -> 99.3M`, shared store wavefronts
  `92.7M -> 79.7M`, shared bank conflicts `62.3M -> 39.9M`, and LSU
  instructions `150.7M -> 142.1M`.
- Reverted negatives:
  `4` owner warps after persistent PV still regressed to `large=2.9888 ms`;
  live QK accumulator fragment score-bypass was interrupted after excessive
  compile time; BF16 score-in-prob-tile regressed to `2.7729 ms`; fused
  two-phase PV regressed to `2.7117 ms`.
- Learned:
  Register-owned full-prefix PV accumulation is a real structural improvement,
  but the current standard-WMMA monolithic kernel is sensitive to register/code
  pressure. Do not continue local variants around `4` owner warps, BF16 score
  scatter, fused PV phases, or keeping standard WMMA QK fragments live across
  barriers. Next useful work needs either a smaller lower-level MMA
  score/probability pipeline or a reference-like CTA main-loop redesign.

## 2026-04-28 - flash_attention_2_20260428_run019_profile_compare

- Profiled the run_018 best against the PyTorch fused FlashAttention reference
  on the primary `large` shape. Current benchmark: `large=2.7268 ms`,
  `25.214 TFLOPS`; reference: `0.8769 ms`, `78.407 TFLOPS`. Current remains
  `3.11x` slower.
- NCU confirmed the run_018 owner-warp reduction helped but did not change the
  main bottleneck. Versus run_016 current, block size fell `640 -> 384`,
  launched threads fell `5.24M -> 3.15M`, and total instructions fell
  `833.9M -> 640.4M`; tensor instructions stayed flat at `16.78M`.
- Compared with reference, current still launches `12x` more threads, executes
  `7.16x` more total instructions and `10.74x` more LSU instructions, with
  shared store wavefronts `92.7M` versus reference `0.289M`.
- Key conclusion:
  the next useful route is not more owner packing, pitch/layout sweeps,
  readback packing, or occupancy tuning. The high-value target is a structural
  dataflow change that removes `PV WMMA -> shared scratch -> owner readback`
  and then enables a larger query-row CTA without serially reusing the current
  owner path.
- Next plan:
  first prototype PV-worker persistent output accumulation on the current
  `16x64` path; if it works, reduce owner responsibilities and then design a
  true `32/64` query-row CTA. If this cannot be done without reintroducing a
  large shared accumulator, stop the local variants and move to a reference-like
  main-loop rewrite.

## 2026-04-28 - flash_attention_2_20260428_run018

- Ran 5 structure-focused iterations after run_017 proved local sync/readback
  tweaks were below threshold. Starting baseline:
  `large=2.8777 ms`, `23.8919 TFLOPS`, correctness `PASS`.
- Kept iter_v4: reduced WMMA owner warps from `16` to `8`, with each owner
  warp split into two 16-lane half-warps that each own one query row. Score/PV
  worker warps stayed unchanged. This cut CTA threads from `640` to `384` while
  preserving the existing MMA worker decomposition.
- Final full validation:
  `medium=0.7456 ms`, `large=2.7384 ms`, `xlarge=10.4531 ms`,
  `large=25.107 TFLOPS`, correctness `PASS`. Versus the run_018 baseline,
  the large-shape improvement is about `4.84%`.
- Reverted structural negatives:
  `32`-query-row two-phase CTA regressed to `large=3.7870 ms`; separate
  score/PV scratch regressed to `2.9032 ms`; reducing to `4` owner warps
  regressed to `3.0249 ms`.
- Reverted neutral:
  pre-scaling owner accumulators before PV-WMMA improved only `0.84%`, below
  the keep threshold.
- Learned:
  Owner-warp count was a real structural overhead, but only down to the
  half-warp-per-row point. More aggressive 8-lane row ownership loses too much
  row-local parallelism and staging participation. Future CTA-geometry work
  should preserve at least 16 lanes per active query row or change the MMA
  ownership model more deeply; do not revisit simple 4-owner-warp packing.

## 2026-04-27 - flash_attention_2_20260427_run017

- Ran source-line instruction/shared attribution and 5 impact-gated
  experiments from the run_015 best. Starting baseline:
  `large=2.8550 ms`, `24.0817 TFLOPS`, correctness `PASS`.
- No source change was kept. The best directional candidate was vectorized
  `float2` PV scratch readback (`large=2.8409 ms`, `24.2008 TFLOPS`), but the
  gain was only `+0.49%`, below the `>1%` keep threshold.
- Reverted negative experiments:
  worker-warp merge regressed to `large=3.1731 ms`; limiting staging/padding to
  `256` CTA threads regressed to `large=4.3249 ms`; skipping the full-prefix
  loop-ending barrier regressed to `large=2.8871 ms`.
- Reverted neutral experiments:
  fallback-only control sinking improved only `+0.09%`; PV scratch `float2`
  readback improved only `+0.49%`.
- Source attribution for the current best showed the remaining gap is
  structural: large instruction sites are standard WMMA QK operand loads,
  owner/PV handoff, and PV scratch readback; shared excessive wavefronts remain
  concentrated in accumulator store/readback. K/V staging dominates L2 sectors
  because the current `16x64` query-key CTA reloads K/V more often than a
  reference-like larger query tile.
- Learned:
  The local neighborhoods around worker count, staging parallelism,
  fallback-control sinking, isolated barrier removal, and readback packing are
  exhausted under the current dataflow. The next credible route must change the
  design boundary: reduce CTA/warp count and shared handoff together, or move
  toward a reference-like larger query tile / register-owned fused
  score-softmax-PV path. Do not spend more iterations on sub-threshold local
  readback/layout/sync variants without new contradictory NCU evidence.

## 2026-04-27 - flash_attention_2_20260427_run015

- Ran 5 iterations from the run_014 source-attribution route. Starting point:
  `medium=0.8790 ms`, `large=3.2487 ms`, `xlarge=12.5089 ms`,
  `large=21.163 TFLOPS`, correctness `PASS`.
- Kept iter_v2: manually constructed the PV probability WMMA A fragment from
  shared memory instead of using standard `wmma::load_matrix_sync` on
  `prob_tile`. Full benchmark improved to `large=2.9190 ms`,
  `23.553 TFLOPS` (`+11.3%`). Targeted NCU reported
  `ncu_smem_bank_conflicts=62.0M`, occupancy `82.8%`, registers/thread `40`.
- Source-line attribution on iter_v2 confirmed the intended effect: the new
  `load_pv_prob_fragment_elem` path showed `0` excessive shared wavefronts,
  removing the dominant `prob_tile` WMMA-A load conflict source identified in
  run_014.
- Kept iter_v3: padded only the PV accumulator scratch pitch from `64` to `68`
  floats. Full benchmark improved further to `large=2.8615 ms`,
  `24.027 TFLOPS`; targeted shared bank conflicts dropped to `45.9M`.
- Reverted iter_v1, iter_v4, and iter_v5:
  standard WMMA probability pitch `72` regressed to `large=4.1242 ms`; QK score
  pitch `68` was only `+0.68%` and below the keep threshold; manual PV
  accumulator stores were only `+0.07%` on `large` and regressed non-primary
  sizes.
- Final validation: `medium=0.7742 ms`, `large=2.8686 ms`,
  `xlarge=11.0733 ms`, `large=23.968 TFLOPS`, correctness `PASS`. Versus the
  run_015 baseline this is `+13.25%` on `large`, `+13.54%` on `medium`, and
  `+12.96%` on `xlarge`.
- Learned:
  For this kernel's current plateau, the high-value bank-conflict fix was not
  a standard-WMMA pitch sweep. It required bypassing the WMMA A shared-load
  helper for the PV probability operand while preserving the same MMA compute
  path. The remaining accumulator-store/readback conflicts are smaller, and
  direct local variants are now at or below the noise floor unless a larger
  dataflow change removes the handoff.

## 2026-04-27 - flash_attention_2_20260427_run014_smem_attribution

- Ran Nsight Compute source/SASS shared-memory conflict attribution on the current run_012 best using a temporary `-lineinfo` rebuild of the same `kernel.cu`.
- Source-attributed excessive shared wavefronts total `165.35M`, close to the previous targeted raw bank-conflict count of `~169.8M`.
- Dominant source is not K/V staging. The largest source is WMMA matrix-A shared loads for the PV path: `113.77M` excessive wavefronts (`68.8%`) from inlined `mma.hpp:163`, mapping to `kernel.cu:374`, `wmma::load_matrix_sync(p_frag, prob_tile + k_step, kWmmaBlockN)`.
- Second source is WMMA accumulator shared stores: `37.36M` (`22.6%`) from inlined `mma.hpp:474`, split between PV scratch and QK score store.
- K/V full-prefix staging, K/V residual staging, scalar `prob_tile` writes, and `v_tile` WMMA-B loads all showed actual shared wavefronts equal to ideal or zero excessive wavefronts.
- Route update:
  Do not spend iterations on K/V pitch, K/V swizzle, direct K/V global operands, or scalar scratch pitch micro-tuning for this bottleneck. The next credible route must directly change the `prob_tile -> WMMA-A shared load -> PV MMA -> shared scratch store -> owner readback` dataflow, likely with lower-level MMA/register-level control rather than more standard-WMMA shared-memory layout variants.

## 2026-04-24 - flash_attention_2_20260424_run013_iter1

- Tested the first post-run_012 structural plan: move full-prefix PV accumulation away from owner warps by adding a PV-warp-updated shared accumulator and importing it once before the diagonal/fallback path.
- Generality gate passed: the change used a runtime full-prefix tile-state check and did not depend on fixed benchmark sequence lengths.
- Correctness passed in quick mode, but performance regressed badly: `large=5.0465 ms`, `13.624 TFLOPS`, versus the run_012 best range of `large≈3.22-3.25 ms`.
- Decision: reverted. The shared accumulator preserved correctness but added enough shared footprint and shared update traffic to erase the intended handoff reduction, effectively returning performance toward the pre-PV-WMMA-scratch-reuse plateau.
- Learned:
  A viable PV-owned accumulator cannot be implemented as a larger shared-memory accumulator in this kernel. It would need a lower-overhead/register-level dataflow, but standard WMMA fragment layout does not safely expose row mapping for row-wise online-softmax alpha scaling. Future work should avoid this shared-accumulator variant.

## 2026-04-24 - flash_attention_2_20260424_run013_iter2

- Tested the second post-run_012 structural plan: consume probabilities as `16x16` subtiles and compute both value halves with two PV accumulator fragments, instead of materializing a full `16x64` probability tile and running two PV phases.
- Generality gate passed: the change used runtime full-prefix/subtile behavior and did not depend on fixed benchmark sequence lengths.
- Correctness passed in quick mode, but performance regressed badly: `large=5.4388 ms`, `12.641 TFLOPS`.
- Decision: reverted. The straightforward standard-WMMA implementation needs extra barriers between probability-subtile production and PV consumption, and keeping two accumulator fragments increases register/code pressure. These costs are larger than the saved full-probability materialization and phase merge.
- Learned:
  Score/prob subtile streaming is not viable as a simple shared-memory WMMA rewrite in this kernel. A future attempt would need a different warp-specialized/lower-level MMA pipeline, not all-thread fragment declarations and per-subtile barriers.

## 2026-04-24 - flash_attention_2_20260424_run012

- Ran 10 impact-gated experiments starting from the `run_011` PV-WMMA scratch-reuse best (`medium=0.8903 ms`, `large=3.3257 ms`, `xlarge=12.9403 ms`).
- The only kept improvement was full-prefix K/V staging specialization. Full-prefix tiles dominate the benchmarked shapes, so the K/V copy loop now has a branch-free path when all 64 rows are active. Best full result from the round: `medium=0.8777 ms`, `large=3.2227 ms`, `xlarge=12.5980 ms`, `large=21.334 TFLOPS`. Promotion validation measured `large=3.2473 ms`, still about `2.4%` faster than run_011.
- Negative results:
  Direct global/L2 operands for `V` and `K` both regressed (`large=3.55 ms` and `3.87 ms` quick), confirming the staged shared-memory operand layout is still important for WMMA.
- More negative results:
  Single-phase 8-warp PV was correct but below threshold, padded PV scratch failed correctness, full-prefix control duplication tied the best on `large`, skipping full-prefix pad-zero regressed, and branch-free Q staging regressed.
- Post-change NCU on the kept candidate:
  `compute_bound (sm=55.9%, dram=5.1%)`, `ncu_occupancy=82.8%`, `ncu_l1_hit_rate=1.9%`, `ncu_l2_hit_rate=97.6%`, `ncu_tensor_core_pct=2.1%`, `ncu_smem_bank_conflicts≈169.8M`.
- Learned for the next iteration:
  Local staging/control-flow cleanup is close to exhausted. The next high-probability work needs a deeper dataflow change that raises tensor-core-covered work or removes score/probability/shared handoff, not more direct-global operand swaps, padding variants, or branch duplication.

## 2026-04-24 - flash_attention_2_20260424_run011

- Ran 10 impact-gated iterations starting from the `run_009` best kernel (`medium=1.32 ms`, `large=5.0569 ms`, `xlarge=19.94 ms`, `large=13.596 TFLOPS`).
- The round explicitly prioritized changes with high dynamic coverage over neighboring score-layout, boundary, and codegen micro-tweaks. The central hypothesis was that the remaining gap was scalar PV accumulation after the existing partial-WMMA QK score path.
- Trial 1:
  Tested a compact shared-memory `V` layout for the scalar PV path. Correctness passed and quick `large` moved to `5.0207 ms`, but the gain was only about `0.7%`, below the keep threshold, so it was reverted.
- Trial 2:
  Added a full-tile PV-WMMA path: owner warps produce BF16 probabilities and score warps compute `P*V` with WMMA. After adding `cudaFuncSetAttribute` for the larger dynamic shared-memory requirement, full benchmark passed at `large=4.3916 ms`. NCU showed the cost of the naive design: occupancy fell to `41.6%`.
- Trial 3:
  Reused the existing score scratch as two-phase `16x64` PV output scratch instead of allocating a separate `16x128` float scratch. This restored occupancy and became the round best. Full benchmark: `medium=0.8952 ms`, `large=3.3236 ms`, `xlarge=12.9145 ms`, `large=20.686 TFLOPS`.
- Trials 4-9:
  Tested follow-up variants around the new structure: compact `V` pitch, `kWmmaBlockN=128`, explicit phase-loop unroll, diagonal-adjacent PV-WMMA coverage, `65`-stride PV scratch, and lane-half phase checks. These either regressed, failed correctness, or stayed below the `>1%` keep threshold, so they were reverted.
- Trial 10:
  Promoted and revalidated the best scratch-reuse PV-WMMA kernel. Full benchmark: `medium=0.8903 ms`, `large=3.3257 ms`, `xlarge=12.9403 ms`, with `large=20.673 TFLOPS`.
- Post-change targeted NCU on the final kept version:
  `compute_bound (sm=57.0%, dram=4.9%)`, `ncu_occupancy=82.8%`, `ncu_l1_hit_rate=1.8%`, `ncu_l2_hit_rate=97.7%`, `ncu_tensor_core_pct=2.1%`, `ncu_smem_bank_conflicts≈169.9M`, `registers/thread=40`.
- Learned for the next iteration:
  The high-value step was structural PV-WMMA plus resource-aware scratch reuse. Neighboring shared-memory pitch, tile-width, branch-shape, and boundary-coverage tweaks did not survive measurement. Future work should change the design boundary again: reduce score/probability materialization or raise tensor-core-covered work further, rather than iterating on low-coverage tails or local layout polish.

## 2026-04-23 - flash_attention_2_20260423_run009

- Ran 5 deeper iterations starting from the `run_007/iter_v6` best kernel (`medium=1.36 ms`, `large=5.24 ms`, `xlarge=20.85 ms`).
- This round focused on the remaining scalar work around the score-tile handoff rather than on score-tile layout itself. The main question was whether moving score scaling out of the owner path would create a new, real full-benchmark win.
- Trial 1:
  Moved `sm_scale_log2e` from the owner update loop into the WMMA score warp by scaling `c_frag` before `wmma::store_matrix_sync`. This immediately improved quick benchmark to `large=5.04 ms` with full correctness, making it the strongest branch of the round.
- Trial 2:
  Combined score pre-scaling with the compact row-major score-subtile layout from the previous round. Correctness passed, but quick benchmark slipped to `5.06 ms`, so the layout branch still did not beat the simpler score-scaled baseline.
- Trial 3:
  Kept score pre-scaling and changed owner score reads to `float2` pair loads. Correctness passed, but quick benchmark regressed to `5.21 ms`.
- Trial 4:
  Re-tested `kWmmaBlockN=48` on top of score pre-scaling in case the score/owner work balance had shifted. Correctness passed, but quick benchmark regressed to `5.20 ms`.
- Trial 5:
  Restored the score pre-scaled branch and ran full validation plus targeted NCU. This was kept as the final best of the round. Full benchmark improved to `medium=1.32 ms`, `large=5.06 ms`, `xlarge=19.94 ms`, with `large=13.596 TFLOPS`.
- Post-change targeted NCU on the kept version (`iter_v5`) for `large`:
  `compute_bound (sm=75.6%, dram=4.4%)`, `ncu_occupancy=82.8%`, `ncu_l1_hit_rate=0.9%`, `ncu_l2_hit_rate=93.9%`, `ncu_tensor_core_pct=0.3%`, `ncu_smem_bank_conflicts=24513814`, `registers/thread=40`.
- Learned for the next iteration:
  There was still one meaningful scalar cost left in the score-tile handoff, and moving score scaling into the score warp captured it. After that win, further tuning around the same materialized score-tile structure still failed to help. The next serious step is still a deeper rewrite that avoids full score-tile materialization instead of continuing to polish around it.

## 2026-04-23 - flash_attention_2_20260423_run008

- Ran 10 deeper iterations starting from the `run_007/iter_v6` best kernel (`medium=1.36 ms`, `large=5.24 ms`, `xlarge=20.85 ms`).
- This round followed the reference-vs-current NCU analysis directly. The focus was on reducing score-scratch shared-memory cost, reducing total score/owner warp overhead, and testing whether launch-time resource partitioning hints could help the current WMMA path.
- Trial 1:
  Reduced owner warps by mapping `2` queries to each owner warp. Correctness passed, but quick benchmark regressed badly to `large=7.12 ms`. The current owner/update path does not tolerate this register/control-flow aggregation well.
- Trials 2-4:
  Reworked the score scratch from one `16x64` slab into compact `16x16` score subtiles with row-major layout and pitch sweeps. The best quick result (`iter_v3`) reached `large=5.2371 ms`, but full validation in `iter_v10` landed at `medium=1.37 ms`, `large=5.25 ms`, `xlarge=20.87 ms`, which is slightly slower than the inherited best. This branch was not kept.
- Trials 5-6:
  Tested fewer score warps and a column-major score-subtile layout. Both regressed clearly (`large=5.85 ms` and `5.74 ms` respectively), indicating that the current kernel is not bottlenecked simply by the count of score warps or by the row-major store pattern alone.
- Trials 7-9:
  Tested launch/runtime resource-partitioning hints: `__launch_bounds__(640, 1)`, explicit shared carveout, and explicit L1 preference. `launch_bounds` and `PreferL1` regressed badly (`large=7.79 ms` and `7.57 ms`), while shared carveout was effectively noise (`5.26 ms`).
- Result:
  No change was kept. The round ended by restoring the inherited `run_007/iter_v6` best snapshot.
- Learned for the next iteration:
  This kernel is now past the point where score-tile layout tuning, warp-count reshaping, or launch-time cache/shared-memory hints are likely to produce real gains. The remaining structural gap to the reference almost certainly sits in the existence of the score-tile materialization itself. The next serious attempt should target a deeper rewrite that consumes score data closer to where MMA produces it instead of writing and rereading full score tiles through shared memory.

## 2026-04-23 - flash_attention_2_20260423_run007

- Starting point was the kept `run_006/iter_v2` kernel:
  `medium=1.54 ms`, `large=5.86 ms`, `xlarge=22.96 ms`, with `large=11.743 TFLOPS`.
- This round focused on what remained after causal owner-path specialization: score-warp cleanup and staging-side fixed costs.
- Trials 1-5:
  Tested score scaling in the score warp, missing diagonal-offset specialization, partial-tile score-warp gating, inactive-row zero-fill removal, and a combined partial-tile path. All of these were either flat or regressed in quick benchmarking.
- Trial 6:
  Rewrote the K/V shared-memory padding loop to touch only the trailing `8` pad elements per row instead of scanning all `136` columns. This was kept and became the round best. Full benchmark improved to `medium=1.36 ms`, `large=5.24 ms`, `xlarge=20.85 ms`, with `large=13.118 TFLOPS`.
- Trials 7-10:
  Built on the new best with more staging/epilogue tweaks: removing inactive-row zero fill, full-query-block Q staging, compact Q pad-only staging, and `__frcp_rn` in the epilogue. None of them beat the new baseline.
- Post-change NCU on the final kept version (`iter_v6`) for `large`:
  `compute_bound (sm=75.3%, dram=3.9%)`, `ncu_occupancy=82.8%`, `ncu_l1_hit_rate=0.9%`, `ncu_l2_hit_rate=97.8%`, `ncu_tensor_core_pct=0.3%`, `ncu_smem_bank_conflicts=24575475`, `registers/thread=40`.
- Learned for the next iteration:
  Once the causal owner path was specialized, the next large win came from trimming scalar staging overhead, not from more score-warp or epilogue tweaks. The next serious step probably needs a deeper restructuring of the WMMA dataflow rather than another branch-level cleanup.

## 2026-04-23 - flash_attention_2_20260423_run006

- Starting point was the kept `run_004/iter_v14` kernel:
  `medium=1.76 ms`, `large=6.80 ms`, `xlarge=26.67 ms`, with `large=10.116 TFLOPS`.
- This round focused on the remaining scalar/control overhead inside the causal WMMA owner path rather than on tile geometry or dispatch policy.
- Trial 1:
  Added an `alpha==1` fast path for scores that do not exceed the running max. This was kept. Full benchmark improved to `medium=1.68 ms`, `large=6.52 ms`, `xlarge=25.60 ms`. Targeted NCU on `large` rose to `compute_bound (sm=82.5%, dram=2.5%)` with `ncu_occupancy=82.9%`.
- Trial 2:
  Added a causal full-tile fast path so tiles entirely left of the diagonal use a fixed 64-key owner loop. This was kept and became the final best of the round. Full benchmark improved to `medium=1.54 ms`, `large=5.86 ms`, `xlarge=22.96 ms`, with `large=11.743 TFLOPS`.
- Trial 3:
  Specialized the diagonal tile using aligned common prefixes. Quick benchmark was only marginally better than `iter_v2` and did not justify a full run.
- Trial 4:
  Packed each lane's two `V` shared-memory loads into one 64-bit load on top of the new owner-path specializations. Correctness passed, but quick benchmark regressed to `large=6.19 ms`.
- Trial 5:
  Revisited the packed epilogue writeback on top of the new owner-path specializations. Correctness passed, but quick benchmark regressed slightly to `large=5.90 ms`.
- Post-change NCU on the final kept version (`iter_v2`) for `large`:
  `compute_bound (sm=78.2%, dram=2.8%)`, `ncu_occupancy=82.8%`, `ncu_l1_hit_rate=0.8%`, `ncu_l2_hit_rate=97.7%`, `ncu_tensor_core_pct=0.3%`, `ncu_smem_bank_conflicts=23621297`, `registers/thread=40`.
- Learned for the next iteration:
  There was still significant value in specializing the causal owner loop, and this round extracted it. The next obvious control-flow cases are much smaller than the full-tile case, so future gains are less likely to come from another branch-only tweak and more likely to require a deeper rewrite of how score/update work is organized around MMA.

## 2026-04-23 - flash_attention_2_20260423_run005

- Starting point was the kept `run_004/iter_v14` kernel:
  `medium=1.76 ms`, `large=6.80 ms`, `xlarge=26.67 ms`, with `large=10.116 TFLOPS`.
- This round intentionally avoided more WMMA tile sweeps and instead targeted the scalar work still left inside the WMMA path.
- Trial 1:
  Folded `sm_scale_log2e` into BF16 `q_tile` staging. This failed smoke correctness with `max_abs_error` above tolerance, so it was reverted immediately. The precision loss from pre-scaling into BF16 was too large.
- Trial 2:
  Packed the two per-lane `V` shared-memory reads into one 64-bit load. Correctness passed, but quick benchmark only moved `large` to `6.7920 ms`, which is too small to treat as real.
- Trial 3:
  Rewrote the owner loop as a fixed-trip 64-iteration unrolled loop. Correctness passed, but `large` regressed badly to `7.4942 ms`, indicating higher register/code-size cost outweighed any loop-control savings.
- Trial 4:
  Tried explicit `__exp2f` in the WMMA owner loop. This did not compile under the current CUDA 13 toolchain in this build configuration, so it was dropped.
- Trial 5:
  Packed the epilogue writeback into one 64-bit store per lane and replaced the final division with `__frcp_rn`. Correctness passed, but quick benchmark regressed slightly to `6.8073 ms`.
- Result:
  No change was kept. The round ended by restoring the inherited `run_004/iter_v14` best snapshot.
- Learned for the next iteration:
  The current WMMA design appears to be past the point where small scalar micro-optimizations matter. The next serious gain likely requires moving a larger share of the score/update computation into a different tensor-core-friendly structure rather than polishing the existing owner-warp loop.

## 2026-04-23 - flash_attention_2_20260423_run004

- Starting point was the kept `run_003/iter_v5` kernel with the short-sequence WMMA path and the scalar long-sequence fallback.
- Baseline at the start of this round:
  `medium=2.09 ms`, `large=8.15 ms`, `xlarge=64.54 ms`, with `large=8.436 TFLOPS`.
- Trial 1:
  Increased the WMMA query tile height from `8` to `16`. This was kept. Full benchmark improved to `medium=1.85 ms`, `large=7.17 ms`, `xlarge=63.90 ms`.
- Trial 2:
  Increased the WMMA key tile width from `32` to `48`. This was kept. Full benchmark improved to `medium=1.86 ms`, `large=6.94 ms`, `xlarge=63.84 ms`.
- Trial 3:
  Increased the WMMA key tile width from `48` to `64`. This was kept. Full benchmark improved to `medium=1.76 ms`, `large=6.78 ms`, `xlarge=63.85 ms`.
- Trials 4-11:
  Explored more WMMA padding and tile variants plus a few long-sequence dispatch thresholds. These either regressed or failed to launch/correctness-check and were reverted.
- Trial 12:
  Increased the long-sequence scalar fallback CTA width to `8` warps/block. This was kept. Full benchmark improved to `medium=1.76 ms`, `large=6.76 ms`, `xlarge=51.89 ms`, making the long-sequence fallback materially less costly.
- Trial 13:
  Increased long-sequence scalar `BLOCK_N` from `32` to `48`. It slightly improved `xlarge`, but regressed `large`, so it was not kept as the round best.
- Trial 14:
  Raised the WMMA dispatch threshold from `2048` to `4096` so `xlarge` also uses the WMMA path. This was kept as the final best of the round. Full benchmark reached `medium=1.76 ms`, `large=6.80 ms`, `xlarge=26.67 ms`, with `large=10.116 TFLOPS`.
- Trials 15-20:
  Swept larger and smaller WMMA tiles (`N=80`, `N=48`), shared-memory carveout, extra/less padding, and `M=8`. All of them regressed; `M=8` and `N=80` regressed badly.
- Post-change NCU on the final kept version (`iter_v14`) for `large`:
  `compute_bound (sm=80.7%, dram=2.5%)`, `ncu_occupancy=82.8%`, `ncu_l1_hit_rate=0.8%`, `ncu_l2_hit_rate=101.7%`, `ncu_tensor_core_pct=0.2%`, `ncu_smem_bank_conflicts=23681896`, `registers/thread=40`.
- Learned for the next iteration:
  The decisive win in this round came from dispatching `4096` into the existing WMMA path, not from improving the inner WMMA micro-kernel itself. The kept kernel is still compute-bound with very low tensor-core instruction share, so the next structural step should be a deeper MMA rewrite of the score/update path rather than more tile-shape sweeps.

## 2026-04-22 - flash_attention_2_20260422_run003

- Starting point was the previously kept vectorized-staging scalar kernel.
- Baseline current kernel:
  `medium=3.25 ms`, `large=12.80 ms`, `xlarge=64.97 ms`, `large=5.370 TFLOPS`, `0.069x` vs PyTorch.
- Baseline NCU on `large`:
  `compute_bound (sm=94.5%, dram=1.7%)`, `ncu_occupancy=99.1%`, `ncu_l1_hit_rate=1.8%`, `ncu_l2_hit_rate=95.6%`, `ncu_tensor_core_pct=0.0%`, `ncu_smem_bank_conflicts=0`, `registers/thread=40`.
- Trial 1:
  Added a short-sequence WMMA score-tile path with a dedicated score warp and kept the existing online softmax/V accumulation structure. This was kept. Full benchmark improved to `medium=3.04 ms`, `large=11.77 ms`, `xlarge=64.55 ms`. NCU showed `ncu_tensor_core_pct=0.3%`, confirming the MMA path was active but still only a small slice of total instructions.
- Trial 2:
  Padded the WMMA shared-memory pitch for Q/K/V tiles. This was kept. Full benchmark improved to `medium=2.96 ms`, `large=11.45 ms`, `xlarge=64.77 ms`. NCU showed `ncu_l1_hit_rate=20.9%` and `ncu_smem_bank_conflicts=14775395`, a major improvement from the unpadded WMMA variant.
- Trial 3:
  Increased the WMMA key tile width to `32` and used two score warps. This was kept. Full benchmark improved to `medium=2.83 ms`, `large=10.86 ms`, `xlarge=64.79 ms`. Occupancy dropped to `82.5%`, but the wider score tile amortized synchronization and staging enough to outweigh the lower residency.
- Trial 4:
  Converted the online softmax recurrence to base-2 form with `exp2f`. This was kept. Full benchmark improved to `medium=2.74 ms`, `large=10.46 ms`, `xlarge=63.90 ms`.
- Trial 5:
  Vectorized WMMA-path Q/K/V global-to-shared staging with `uint4` copies. This was the largest win of the round and was kept. Final full benchmark improved to `medium=2.09 ms`, `large=8.15 ms`, `xlarge=64.54 ms`, with `large=8.436 TFLOPS`, `0.109x` vs PyTorch.
- Post-change NCU on the kept version:
  `compute_bound (sm=78.6%, dram=2.0%)`, `ncu_occupancy=82.6%`, `ncu_l1_hit_rate=0.5%`, `ncu_l2_hit_rate=98.8%`, `ncu_tensor_core_pct=0.4%`, `ncu_smem_bank_conflicts=42295409`, `registers/thread=40`.
- Learned for the next iteration:
  The kernel can gain materially from partial WMMA adoption even before tensor-core utilization becomes dominant, but the current design still leaves most of the hot path on scalar work. The next major step should target a more complete MMA decomposition, especially for the long-sequence path and the PV/update side of the attention loop.

## 2026-04-21 - flash_attention_2_20260421_run002

- Starting point was the kept size-aware launch version from `run_001`.
- Baseline current kernel:
  `medium=3.59 ms`, `large=14.41 ms`, `xlarge=83.58 ms`, `large=4.771 TFLOPS`, `0.438x` vs PyTorch.
- Baseline NCU on `large`:
  `compute_bound (sm=94.5%, dram=1.2%)`, `ncu_occupancy=82.5%`, `ncu_l1_hit_rate=1.5%`, `ncu_l2_hit_rate=100.8%`, `ncu_tensor_core_pct=0.0%`, `ncu_smem_bank_conflicts=6768170`, `registers/thread=40`.
- Trial 1:
  Swizzled the shared-memory K/V tile layout to reduce bank conflicts. It regressed all sizes (`large=16.66 ms`, `medium=4.22 ms`, `xlarge=86.16 ms`) and was reverted. The extra address arithmetic cost more than the conflict reduction helped.
- Trial 2:
  Vectorized K/V staging with `uint4` global-to-shared copies. This was kept. Full benchmark improved to `medium=3.34 ms`, `large=13.06 ms`, `xlarge=65.26 ms`, with `large=5.265 TFLOPS`.
- Post-change NCU on the kept version:
  `compute_bound (sm=93.2%, dram=1.5%)`, `ncu_occupancy=82.5%`, `ncu_l1_hit_rate=1.0%`, `ncu_l2_hit_rate=97.4%`, `ncu_tensor_core_pct=0.0%`, `ncu_smem_bank_conflicts=15013778`, `registers/thread=37`.
- Trial 3:
  Increased short-sequence `BLOCK_N` from `32` to `48`. Correctness passed, but `large` regressed slightly to `13.32 ms`; reverted.
- Trial 4:
  Increased short-sequence CTA width from `8` to `12` warps/block. Raw timing looked best (`large=12.93 ms`), but smoke correctness failed with `NaN/Inf in output`. Most likely cause is a partial-CTA path returning before later `__syncthreads()` when the wider CTA shape no longer divides the sequence-tail work safely. Reverted immediately.
- Trial 5:
  Tuned short-sequence `BLOCK_N` from `32` to `40`. Correctness passed, but `large` regressed slightly to `13.37 ms`; reverted.
- Learned for the next iteration:
  After launch-shape tuning, further gains came from cutting staging instruction count, not from wider CTAs or larger `BLOCK_N`. The kernel remains compute-bound with `ncu_tensor_core_pct=0.0%`, so the next substantial step probably requires a real MMA/tensor-core path rather than more scalar tile-shape tweaking.

## 2026-04-21 - flash_attention_2_20260421_iter1

- Baseline current kernel:
  `large=21.11 ms`, `3.257 TFLOPS`, `0.301x` vs PyTorch.
- Baseline NCU:
  `compute_bound`, `ncu_occupancy=41.4%`, `ncu_l1_hit_rate=7.0%`, `ncu_l2_hit_rate=99.2%`, `ncu_tensor_core_pct=0.0%`, `launch__shared_mem_per_block_dynamic=16384`.
- Trial 1:
  Fixed `WARPS_PER_BLOCK=8` improved `large` to `14.19 ms`, but full benchmark regressed `xlarge` to `195.59 ms`. This was not kept as a global launch shape.
- Kept change:
  Size-aware launch dispatch in [kernel.cu](/home/et/cuda/cuda-kernel-agent/cuda-evolve-oss/kernel.cu:165), using `8` warps/block for `seq_len <= 2048` and `4` warps/block otherwise.
- Final result:
  `medium=3.61 ms`, `large=14.26 ms`, `xlarge=83.21 ms`.
  Compared with the pre-round baseline, `large` improved by about `1.48x`.
- Post-change NCU on `large`:
  `compute_bound (sm=74.4%, dram=15.7%)`, `ncu_occupancy=82.5%`, `ncu_l1_hit_rate=1.5%`, `ncu_l2_hit_rate=92.5%`, `ncu_tensor_core_pct=0.0%`.
- What changed micro-architecturally:
  Occupancy on `large` roughly doubled (`41.4% -> 82.5%`) with the same register count (`40`) and the same dynamic shared memory footprint (`16 KB`), because the wider CTA shape converts the shared-memory residency cap into far more active warps per SM.
- Learned for the next iteration:
  The next major performance jump will likely need a tensor-core path. Launch-shape tuning still helps materially, but the kernel remains far from peak compute and still executes with `ncu_tensor_core_pct=0.0%`.
