# qwen35moe_gdn_prefill

## 2026-05-06 CUDA-only Stop-150 Closure

- Continued CUDA-only optimization from the promoted K^T manual-swizzle source
  hash `58f89eeaf6c8b098379dc4bc2795fb063a271b473bc632041f4481df82d67434`.
  No Triton/TileLang/CUTLASS/library replacement path was introduced.
- Fixed the iteration harness so duplicate or transform-inapplicable candidates
  are recorded as `skip` and do not advance the no-improvement stop counter.
  This prevented stale pattern errors from satisfying the user's stop condition.
- Kept `cand_0098` (`kt_padded,m48`): `K^T @ U` now uses manual A-fragment
  construction directly from the normal padded `khm` shared layout through
  `load_ktrans_a_manual_from_padded`, and `M_LD` is 48. Same-process paired
  confirmation over six shapes reported geomean current-best/candidate
  `1.016117`, minimum per-shape ratio `1.013688`.
- Continued the CUDA-only stream until the stop condition was genuinely met:
  150 consecutive valid no-improvement candidates after `cand_0098`. Final
  iteration summary is
  `workspace/qwen35_gdn_current_iter/run_stop150_fixed_1778031416/summary.json`.
- Final full benchmark `workspace/current_iter_stop150_fixed_final_full.json`
  correctness PASS. Medians:
  `2048=0.544275`, `4096=1.045169`, `8192=2.021053`,
  `2113=0.561139`, `4097=1.049865`, `8191=2.031025` ms.
  Large-shape throughput was `8.2187` TFLOPS, speedup vs PyTorch reference was
  `4.9819x`, and peak VRAM was `678 MB`.
- Final targeted NCU for the main launch:
  `workspace/ncu_reports/current_stop150_final_main_targeted.ncu-rep`.
  Summary: compute_bound (`sm=49.8%`, `dram=37.7%`), occupancy `64.2%`,
  `64` registers/thread, L1/L2 hit `42.8%/85.5%`, shared bank conflicts
  `9,954,142`, tensor-core instruction share `2.0%`.
- Final synchronized CUDA source hash:
  `7168f15a1c49d627c586272eacf9d72cc972b1eee6bca86cb02707295b4bae43`.

## 2026-05-05 K^T @ U Manual Swizzle Promotion

- Promoted `workspace/kernel_kt_manual_swz_candidate.cu` to the active CUDA
  sources after same-process paired A/B benchmarking against the previous
  current best. `kernel.cu` and `kernels_optimized/qwen35moe_gdn_prefill.cu`
  now match candidate SHA256
  `0f2abd912539a6cb2a1e7676b1ae5a08a491fca293a011527c0f9263bd27657f`.
- Paired benchmark artifact:
  `workspace/main_kt_manual_swz_sameprocess_pair_fixed31.json`. Fixed 31 trials,
  warmup 25, rep 100, CUDA event timing, same process, shared inputs per shape,
  alternating candidate/current-best order. Candidate was faster on all six
  shapes; geomean current-best/candidate ratio was `1.00984`.
- Active quick benchmark after promotion:
  `workspace/active_kt_manual_swz_quick.json`; correctness PASS with worst shape
  sweep error `8.20e-4`. Timing remains noisy under the strict 1.5% spread gate,
  so future sub-percent changes still need same-process paired A/B evidence.

## 2026-05-05 Main Kernel Remaining Swizzle Consumer Sweep

- Continued CUDA-only experiments on the main kernel's remaining plausible
  swizzle + `ldmatrix`/`mma.sync` consumers. No Triton/TileLang/CUTLASS/library
  replacement path was introduced, and no new source change was kept.
- `Q @ S0` swizzled-Q route: reused `upd_s` after `K @ S0` as a temporary
  swizzled Q shared tile and consumed it with the existing manual MMA helper.
  Quick correctness PASS and large quick improved (`1.097770 -> 1.085725` ms),
  but full benchmark was mixed: `2048=0.566999`, `4096=1.092101`,
  `8192=2.125569`, `2113=0.583179`, `4097=1.097493`, `8191=2.118763` ms.
  It improved 4096/8192/2113 but regressed or tied 2048/4097/8191 versus the
  current best, so it was not kept.
- `A @ W` route: directly swizzling padded `PH_LD=40` storage failed
  correctness (`max_abs_error=1.79e-2`), confirming that this padded layout is
  not a valid primitive-matched swizzle shape. A repaired contiguous `32x32`
  swizzled-A shadow plus shared `W` staging passed quick correctness but slowed
  large quick to `1.103166` ms, so it was rejected.
- `P @ U` route: used contiguous `32x32` swizzled-P storage, kept base output in
  `m_s`, staged `U` into the upper part of `upd_s`, and wrote P@U into `upd_s`
  with manual MMA. Quick correctness PASS but large quick was `1.102543` ms,
  slower than current best, so it was rejected.
- `K^T @ U` route: regenerated a swizzled K shadow in `m_s`, staged scaled `U`
  in `ph_s`, and tried three `ldmatrix` lane mappings for the transposed K
  operand. All three failed smoke correctness with NaN/Inf. Treat this as an
  unresolved fragment-mapping implementation issue, not performance evidence;
  any future retry should start with an isolated `K^T @ U` fragment unit test
  rather than another full-kernel blind mapping.
- Restored active `kernel.cu` to the current best synchronized hash
  `fcb55a5e969003c3406cedb00b4f8ccc03e7ddd8dfc80888cedfa2663e4bc3d9`.

## 2026-05-05 Isolated K^T Fragment Mapping Test

- Added `workspace/kt_fragment_mapping_test.py`, an isolated CUDA extension test
  for one `K[32,128]`, `U[32,32]` tile. It compares candidate fragment mappings
  for `K^T @ U` against a Torch reference before touching the full kernel.
- Result: direct `ldmatrix` A-operand mappings all produced large errors
  (`max_abs` about `23.6-31.2`). The correct mapping is manual A-fragment
  construction from swizzled K using the PTX `m16n8k16` A-fragment row/column
  formula, paired with `ldmatrix.x4.trans` for shared `U` and
  `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`. Isolated error was
  `max_abs=1.91e-6`.
- Retried the main `K^T @ U` stage using that verified mapping. Important
  implementation fixes: `ldmatrix` B must read `U` from shared memory, not the
  global `uh` workspace; and temporary swizzled K must use enough storage
  (`upd_s` is large enough, `m_s` is not). The helper updated `state_s` directly
  from MMA accumulators instead of materializing the full update matrix.
- Full benchmark `workspace/main_kt_manual_swz_sharedb_tmp_full.json`
  correctness PASS, numerical stability PASS, determinism PASS. Medians:
  `2048=0.566951`, `4096=1.083217`, `8192=2.121527`,
  `2113=0.588197`, `4097=1.104272`, `8191=2.118157` ms.
- Compared with current best
  (`0.566682/1.103893/2.130160/0.584291/1.097218/2.109167` ms), this improves
  4096 and 8192 but regresses the tail cases and is still timing-unstable, so it
  was not kept. Candidate source saved at
  `workspace/kernel_kt_manual_swz_candidate.cu` with SHA256
  `0f2abd912539a6cb2a1e7676b1ae5a08a491fca293a011527c0f9263bd27657f`.
  Active source restored to best hash
  `fcb55a5e969003c3406cedb00b4f8ccc03e7ddd8dfc80888cedfa2663e4bc3d9`.

## 2026-05-05 Main Kernel Swizzled-K Route

- CUDA-only change; no Triton/TileLang/CUTLASS/library replacement path was
  introduced. `kernel.cu` and `kernels_optimized/qwen35moe_gdn_prefill.cu` are
  synchronized at SHA256
  `fcb55a5e969003c3406cedb00b4f8ccc03e7ddd8dfc80888cedfa2663e4bc3d9`.
- Applied the primitive-matched swizzle approach to the main kernel's `K @ S0`
  stage. `state_s` stays in its normal padded layout (`STATE_LD=40`) for
  standard WMMA and scalar consumers; `khm` stays normal padded (`KH_LD=152`) for
  `K^T @ U`; a lifetime-aliased swizzled K copy (`khm_swz` in `upd_s`) feeds
  only the manual `ldmatrix`/`mma.sync` `K @ S0` helper.
- The shared swizzle address helper must swizzle offsets relative to the
  swizzled buffer base. Swizzling the absolute shared address was incorrect once
  `khm_swz` started at `upd_s` instead of shared-memory offset zero.
- Full benchmark `workspace/main_khm_swz_tmp_clean_full.json`: correctness PASS,
  numerical stability PASS, determinism PASS. Medians:
  `2048=0.566682`, `4096=1.103893`, `8192=2.130160`,
  `2113=0.584291`, `4097=1.097218`, `8191=2.109167` ms.
- Compared with the precompute-only swizzle full benchmark
  (`0.579999/1.115785/2.201993/0.606924/1.124678/2.171115` ms), the main
  swizzled-K route improved all six lengths by about `1-4%`, and all six medians
  are faster than the saved TileLang `flashqla_tilelang_ms_median` target.
- NCU `workspace/ncu_reports/main_khm_swz_tmp.*`: aggregate shared bank
  conflicts were `3,515,485`, higher than the earlier main-kernel profile around
  `3.15M`. The keep decision is therefore based on full timing plus correctness,
  not on a lower aggregate bank-conflict counter.
- Rejected nearby variants: swizzling `state_s` for both `K @ S0` and `Q @ S0`
  reduced bank conflicts much more but slowed the full kernel; temporary
  swizzled `state_s` copies also passed correctness after the relative-address
  fix but raised conflicts and did not beat the swizzled-K route.

## 2026-05-05 Precompute Shared Swizzle / LDMATRIX

- CUDA-only change; no Triton/TileLang/CUTLASS/library replacement path was added.
- Applied FlashAttention-style XOR swizzle to the precompute kernel's BF16 shared `kh_s` / `qh_pre_s` tiles, and replaced the two precompute `wmma::load_matrix_sync` `K@K^T` / `Q@K^T` calls with hand-written `ldmatrix.sync.aligned.m8n8` + `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` so the swizzled layout is consumed correctly.
- Scope is intentionally limited to precompute Q/K tiles. Main-kernel WMMA shared matrices are still normal padded layouts (`STATE_LD=40`, `KH_LD=152`, `PH_LD=40`, `M_LD=40`, `UPD_LD=40`), because directly swizzling WMMA-consumed storage without manual `ldmatrix` would break the layout contract.
- NCU precompute result: shared bank conflicts dropped from the e332 reprofile `7,768,002` to `430,809`; registers rose from 93 to 95/thread; occupancy stayed about 25%.
- NCU main result stayed effectively unchanged: shared bank conflicts `3,135,701` vs e332 `3,146,237`, registers 64/thread, occupancy about 64%.
- Full correctness PASS. Full benchmark medians were mixed/noisy but comparable to e332: `2048=0.579999`, `4096=1.115785`, `8192=2.201993`, `2113=0.606924`, `4097=1.124678`, `8191=2.171115` ms. Same-process CUDA paired comparison against e332 gave positive median ratios for all six sizes, geo baseline/candidate `1.02287`, but spreads remained high, so treat the end-to-end timing gain as directional while the bank-conflict reduction is confirmed.
- Synchronized source hash: `87770580a787ef0afba5ab355014517765e8bf04a944872deb48f38c95ec2d7d`.

## 2026-05-05 Framework Rule Update: Primitive-Matched Swizzle

- Added a framework rule for shared-memory bank-conflict work: swizzle is a
  layout contract with the consumer primitive, not a standalone store-index
  tweak.
- For standard WMMA consumers, keep normal row/column-major shared layout and
  use compatible fixes such as padding or leading-dimension changes. For
  FlashAttention-style XOR-swizzled shared operands, also change the consumer to
  manual `ldmatrix`/`mma.sync`, manual fragment construction, or an explicit
  descriptor layout.
- Evidence source: `qwen35moe_gdn_prefill` precompute Q/K swizzle reduced NCU
  shared bank conflicts from `7,768,002` to `430,809` only after the WMMA
  precompute GEMMs were replaced by matching hand-written `ldmatrix`/`mma.sync`.

## 2026-05-05 CUDA-only Faster-than-TileLang Closure

- Resumed from CUDA source hash `6cb3dab43533...`; `kernel.cu` and `kernels_optimized/qwen35moe_gdn_prefill.cu` were kept synchronized, and no Triton/TileLang/CUTLASS/library replacement path was introduced. TileLang was used only through `workspace/tilelang_flashqla_profile_saved.json` as the comparison target.
- Re-measured the BF16 `qk_cache` full benchmark first: correctness PASS, but current CUDA was still slower than TileLang on all six sizes in that run (`2048=0.660070`, `4096=1.287967`, `8192=2.504750`, `2113=0.703755`, `4097=1.308225`, `8191=2.501244` ms).
- Kept main shared-K staging: moved main-kernel `khm` from global workspace into shared memory and aliased `base_out` with `upd_s` so `UPD_LD=40` could be retained without exceeding the default shared-memory limit. This improved all six full-benchmark sizes.
- Kept padded shared-K leading dimension route: added WMMA helpers with explicit A/B leading dimensions and set `KH_LD=152`. This removed the major shared-load bank-conflict regression from the first shared-K version and produced the first all-faster full benchmark (`workspace/current_cuda_kshared_ld152_full.json`), though a repeat showed the remaining 2048/2113 margins were still within timing noise.
- Kept `actual == 1` scalar tail path in the main CUDA kernel. For one-token tail chunks it computes the single-token `K@S`, `Q@S`, `QK`, output, and state outer-product update directly with warp reductions instead of running the full 32-token WMMA chunk path.
- Final full benchmark `workspace/current_cuda_tail1_scalar_full.json`: correctness PASS and stop condition met against `flashqla_tilelang_ms_median`:
  `2048=0.582348 < 0.585162`,
  `4096=1.126116 < 1.197815`,
  `8192=2.192968 < 2.446402`,
  `2113=0.601638 < 0.610942`,
  `4097=1.126337 < 1.214904`,
  `8191=2.184340 < 2.475015` ms.
- Timing note: the final full run reports `timing_reliable: no` because the strict spread gate is still above 1.5% on some sizes, but the user-defined stop condition is median faster-than-TileLang and is met by the latest full benchmark.
- Rejected in this pass: Q tile in shared with compressed state/update strides (4.8-7.5% slower), BF16 `A` solve intermediates (2-3% slower), `KH_LD=144` (mixed/worse than 136), `KH_LD=168` (ptxas shared-memory overflow), `STATE_LD=32` (tail and long regression), lower-triangular-only qk_cache stores (about 7% slower), and `launch_bounds(THREADS,1)` (severe occupancy/register regression).
- Final synchronized CUDA source hash: `e3320addf4a74e247c690bb40f0fa0464fb0eed6a206b88653d30d2dbc785425`.

## 2026-05-05 CUDA-only TileLang-gap Closure

- Continued from CUDA source hash `091166f8949d...` and kept all implementation work inside `kernel.cu` / `kernels_optimized/qwen35moe_gdn_prefill.cu`; TileLang was used only as the stop-condition timing target.
- Re-measured BF16 `qk_cache`: correctness PASS, max TileLang ratio improved slightly (`1.15970 -> 1.15946`), so it was retained as a small objective improvement.
- Kept `precompute_qk_float2`: vectorized precompute Q/K global loads with `float2` and shared writes with `bfloat162`. Full benchmark improved every tested length by about `3.4-4.5%`.
- Kept BF16 `qk_cache` vector store: stores precomputed QK cache as `bfloat162`; this mainly improved 4096/tail cases and slightly reduced the max-ratio objective.
- Kept main `P` construction BF16x2 path: loads BF16 `qk_cache` with `bfloat162`, converts to `float2`, and stores `ph_s` as `bfloat162` while preserving the runtime lower-triangular mask.
- Final full benchmark, correctness PASS, stop condition met against `workspace/tilelang_flashqla_profile_saved.json` (`flashqla_tilelang_ms_median`, all CUDA medians within 10%):
  `2048=0.638907 ms / ratio 1.0918`,
  `4096=1.248321 ms / ratio 1.0422`,
  `8192=2.428995 ms / ratio 0.9929`,
  `2113=0.671670 ms / ratio 1.0994`,
  `4097=1.254848 ms / ratio 1.0329`,
  `8191=2.433561 ms / ratio 0.9833`.
- Final synchronized CUDA source hash: `6cb3dab4353330d22298f04936302c79696f6c32ea62861ca4434a0b4132475e`.

## 2026-05-04 FlashQLA-Logic Triton Three-Stage Prototype

- Added `kernels_optimized/qwen35moe_gdn_prefill_triton_flashqla.py`, a Triton implementation that follows FlashQLA forward kernel logic rather than the earlier single-program direct translation.
- The forward path is split into the same conceptual stages as FlashQLA:
  - chunk-local gate cumsum kernel writes `g_cum`.
  - KKT/solve kernel computes `A = inv(I + StrictLower(beta_i * K_i @ K_j))` per `(batch, H_v, chunk)`.
  - fused forward kernel iterates chunks for one `(batch, H_v, value-slice)`, computes `W = V - exp(g) * K@S`, `Vd = (A * exp(g_i-g_j) * beta_j) @ W`, output `scale*exp(g)*Q@S + scale*Lower(exp(g_i-g_j)*QK)@Vd`, and updates state with `exp(g_last)*S + K^T@(exp(g_last-g)*Vd)`.
- `tools/bench_qwen35_gdn_triton.py` now supports `--variant flashqla`, `--chunk-size`, and `--num-warps`.
- Quick configuration search:
  - `chunk_size=32, BLOCK_DV=32, num_warps=4` was best.
  - `chunk_size=64` is closer to upstream FlashQLA's Hopper default but slower on this Ada/Triton setup.
  - `BLOCK_DV=64` and forward `num_warps=8` were slower than `BLOCK_DV=32,num_warps=4`.
- Strict paired benchmark against current CUDA best, warmup 25, rep 100, min 9 trials, max 31, target CI95 <= 1%, spread threshold 1.5%, correctness PASS vs CUDA best. Worst max abs error was <= `2.30e-4`.
- Results, reported as current CUDA best over FlashQLA-style Triton:
  - `2048`: Triton `0.6315 ms`, CUDA `1.8834 ms`, CUDA/Triton `2.9695x`.
  - `4096`: Triton `1.2172 ms`, CUDA `3.7435 ms`, CUDA/Triton `3.0712x`.
  - `8192`: Triton `2.3985 ms`, CUDA `7.3859 ms`, CUDA/Triton `3.0702x`.
  - `2113`: Triton `0.7015 ms`, CUDA `1.9620 ms`, CUDA/Triton `2.7936x`.
  - `4097`: Triton `1.3270 ms`, CUDA `3.7638 ms`, CUDA/Triton `2.8205x`.
  - `8191`: Triton `2.6743 ms`, CUDA `7.4598 ms`, CUDA/Triton `2.7952x`.
- Timing note: paired CI95 was below 1% for all cases, but spread remained above the strict 1.5% threshold (`3.65-6.96%`). The speedup margin is far larger than the spread, so the conclusion is directionally strong, but sub-percent follow-up tuning should still use the strict gating rules.
- Main mechanism: separating KKT/triangular solve from the recurrent/output pass avoids the earlier Triton single-kernel live-state blowup. It also changes the solve from per-value-column row recurrence to FlashQLA's precomputed chunk `A` reused across value slices, which removes duplicated triangular-solve work inside the value-slice CTAs.

## 2026-05-04 Triton FlashQLA-Style Prototype

- Added `kernels_optimized/qwen35moe_gdn_prefill_triton.py`, an experimental single-kernel Triton implementation with the same external FP32 llama.cpp-style interface as the CUDA kernel.
- FlashQLA-inspired structure used: one Triton program owns `(batch, H_v, value-slice)`, keeps recurrent state across chunks, computes chunk-local `K@S`, `K@K^T`, `Q@S`, `Q@K^T`, `P@U`, and `K_restored^T@U` with BF16 `tl.dot` and FP32 accumulation, while keeping gate prefix, triangular solve, and state accumulation in FP32.
- Added `tools/bench_qwen35_gdn_triton.py` to benchmark the Triton prototype against the current CUDA best with adjacent paired timing.
- Configuration search:
  - `BLOCK_DV=16` was slower than `32`.
  - `BLOCK_DV=64` was slower than `32`.
  - `num_warps=4` was much faster than `8` or `16`, indicating the Triton route is dominated by register/live-state pressure rather than raw Tensor Core issue width.
- Final strict comparison used `BLOCK_DV=32`, `num_warps=4`, warmup 25, rep 100, min 9 trials, max 31, target paired CI95 <= 1%, spread threshold 1.5%. Correctness vs current CUDA best passed for all tested lengths, worst max abs error <= `8.11e-5`.
- Results, reported as current CUDA best over Triton prototype:
  - `2048`: Triton `4.552 ms`, CUDA `1.882 ms`, CUDA/Triton `0.413x`.
  - `4096`: Triton `9.236 ms`, CUDA `3.735 ms`, CUDA/Triton `0.403x`.
  - `8192`: Triton `18.510 ms`, CUDA `7.396 ms`, CUDA/Triton `0.401x`.
  - `2113`: Triton `3.763 ms`, CUDA `1.954 ms`, CUDA/Triton `0.518x`.
  - `4097`: Triton `7.218 ms`, CUDA `3.752 ms`, CUDA/Triton `0.517x`.
  - `8191`: Triton `15.724 ms`, CUDA `7.437 ms`, CUDA/Triton `0.471x`, but paired spread hit `23.33%` after 31 trials, so treat this one as direction-only.
- Conclusion: a plain Triton single-program port can express the FlashQLA-style chunk math and uses Tensor Core `tl.dot`, but it is 2-2.5x slower than the current hand-written CUDA kernel. The likely cause is that Triton keeps large `state_s`, `KKT`, and `U` block tensors live in registers/spills, while the CUDA kernel explicitly manages shared memory and block-wide WMMA work. A competitive Triton route would need a different dataflow, likely split kernels or a lower-level Triton/Gluon implementation with explicit shared-memory ownership.

## 2026-05-04 Fused Value-Slice Ownership Route 100-Valid Iteration

- Added `tools/qwen35_gdn_pairv_iter.py` for the next structural route after rejecting global QK/KKT materialization. The tested invariant was to keep `Q @ K^T` and `K @ K^T` local by assigning one CTA to both value slices (`v0=0` and `v0=64`) for a single `(batch, H_v, chunk)`.
- Candidate dimensions: `WARPS=16/20/12/8`, launch-bound min blocks `1/2`, plain vs `__ldg` loads for QK/gate/V/state, and `krest` from global vs shared. A template bug in the shared `krest` path was fixed before the strict run; build/correctness errors were not counted as valid optimization evidence.
- Completed 100 strict keep/reject records with the same paired adaptive gate: warmup 25, rep 100, min 9 trials, max 31 trials, target paired CI95 <= 1%, spread threshold 1.5%, and keep requiring correctness plus improvement beyond the uncertainty band.
- Result: 0 keep, 100 reject. No candidate showed a positive median improvement. Median improvement range was about `-46.09%` to `-43.49%`, with median `-44.76%` versus the current best.
- Conclusion: fusing both value slices inside the same CTA avoids global QK/KKT cache traffic, but the saved recomputation is overwhelmed by halved CTA parallelism, larger dynamic shared memory/live state, and more work per CTA. The current optimized kernel remains unchanged.
- Next structural route should not merge value slices inside one CTA. If continuing, test ownership across the two `H_v` heads that share one `H_k` while keeping each value slice separate, or use a cooperative CTA/cluster handoff only if it preserves enough CTA parallelism and avoids global QK/KKT materialization.

## 2026-05-04 Strict Adaptive 100-Valid Iteration

- Benchmark framework changes for this pass: paired adjacent kernel/reference timing, alternating order, adaptive trial extension, spread/CV/CI95 stability fields, and keep gating that requires correctness plus an improvement larger than the timing uncertainty band. Full defaults are warmup 25, rep 100, min 9 trials, max 31 trials, target paired CI95 <= 1%, spread threshold 1.5%.
- `tools/run_loop.py` now treats unstable kernel/reference/paired-speedup timing as inconclusive, and quick results near the uncertainty band require full validation. `tools/qwen35_gdn_auto_iter.py` was extended to run non-destructively from a chosen best source, skip inapplicable repeated transforms, and stop by valid keep/reject count.
- Re-ran 13 previously excluded directions under the strict gate: `ph_global`, `mat_global`, both global, `m_shared`, `u_shared`, `uh_shared`, `krest_shared`, `launch_bounds_2`, `workspace_compact`, `ldg_all`, `parallel_first`, `parallel_second`, and `parallel_both`. None was kept. Some showed positive medians, but all were below or inside instability/uncertainty; no standalone previous exclusion was proven wrong.
- Completed 100 strict attempts first, then supplemented because 27 early candidates were generation/build errors. Final valid performance evidence count is 100 keep/reject records: 1 keep and 99 rejects. Additional non-evidence records: 27 original errors and 9 skipped inapplicable repeated transforms after the harness fix.
- Kept change: `g_scaled + u_shared`.
  - Cache `scale * g_prefix[row]` once in shared memory as `g_scaled[row]`.
  - Move chunk-local `u[CHUNK, BLOCK_DV]` from global workspace to shared memory.
  - Strict paired 4096 comparison against the prior best: candidate `3.641 ms`, prior best `3.746 ms`, improvement `2.91%`, uncertainty `1.49%`, 9 actual trials.
- No further local `warps`, `BLOCK_DV`, `ldg`, shared-staging, launch-bound, workspace-compact, or triangular-loop variant produced a stable keep signal. A few `warps20/static_j` style variants had positive medians but failed the spread gate, so they remain inconclusive rather than accepted.
- Final selected source is `kernels_optimized/qwen35moe_gdn_prefill.cu`, matching `workspace/qwen35_gdn_auto_iter/run_100_strict/best.cu` after the kept patch.
- Final strict full benchmark against the original FP32 scalar CUDA reference, RTX 4070 Ti SUPER, warmup 25 / rep 100 / min 9 / max 31, correctness PASS, worst max abs error `1.90e-4`:
  - `2048`: optimized `1.837 ms`, reference `2.603 ms`, paired speedup `1.417x`.
  - `4096`: optimized `3.650 ms`, reference `5.099 ms`, paired speedup `1.397x`.
  - `8192`: optimized `7.251 ms`, reference `10.337 ms`, paired speedup `1.426x`.
  - `2113`: optimized `1.917 ms`, reference `2.622 ms`, paired speedup `1.371x`.
  - `4097`: optimized `3.658 ms`, reference `5.116 ms`, paired speedup `1.396x`.
  - `8191`: optimized `7.261 ms`, reference `10.380 ms`, paired speedup `1.427x`.
- Timing note: kernel and reference medians were mostly individually stable, and paired CI95 was below 1%, but paired spread stayed around `1.8-2.5%`, so `timing_reliable` is still `no` under the strict 1.5% spread rule. Use the speedups as the best strict directional estimate, not as a hard stable keep/revert boundary for smaller follow-up deltas.
- The 2x target was not reached. The remaining gap is structural: the route still materializes multiple chunk intermediates through workspace/shared handoff and keeps the triangular/state update as block-wide phased work. The next credible route should change ownership/dataflow, not continue local cache/launch-shape tuning.

## 2026-05-04 FlashQLA-Style Ownership Route 100-Valid Iteration

- Added `tools/qwen35_gdn_struct_iter.py` for non-destructive architecture-route experiments. The route tested a FlashQLA-inspired ownership split: matrices that depend only on `(batch, H_k, chunk)` are computed by a precompute kernel, then reused by main CTAs that are still owned by `(batch, H_v, value-slice)`.
- Structural invariant tested: move ownership of `K @ K^T` and/or `Q @ K^T` from every main CTA to a separate `(H_k, chunk)` producer. This should reduce recomputation across the two `H_v` heads sharing one `H_k` and the two value slices, at the cost of an extra kernel launch and global matrix cache traffic.
- Candidate dimensions:
  - precompute mode: `pre_both`, `pre_kkt`, `pre_qk`.
  - precompute warps: `4`, then a small subset of `8`.
  - main warps: `16`, `12`, `20`, `8`.
  - value split: `BLOCK_DV=64` and `32`.
  - cache loads: plain global load vs `__ldg`.
  - main launch bounds: min blocks `1` vs `2`.
- `BLOCK_DV=128` was removed from the valid candidate pool after ptxas reported shared data `0x15a00` bytes vs `0xc000` max. These build errors were not counted as valid optimization evidence.
- Completed 100 valid keep/reject records: 12 from the first run before skipping `BLOCK_DV=128`, plus 88 from the supplement. Result: 0 keep, 100 reject. There were 8 build errors from the invalid `BLOCK_DV=128` subfamily.
- Best positive medians were all rejected by the spread gate:
  - `pre_kkt, pre_warps4, warps16, block32, plain, lb2`: candidate `3.362 ms` vs best `3.633 ms`, median improvement `8.19%`, but uncertainty/spread `3.45%`.
  - `pre_both, pre_warps4, warps16, block32, ldg, lb2`: candidate `3.443 ms` vs best `3.709 ms`, median improvement `8.17%`, but uncertainty `6.30%`.
  - `pre_both, pre_warps4, warps20, block32, ldg, lb2`: candidate `3.451 ms` vs best `3.640 ms`, median improvement `5.28%`, but uncertainty `2.65%`.
- Stable negative evidence:
  - `pre_qk, pre_warps4, warps8, block64, ldg, lb1` was about `-20.7%`.
  - `pre_both, pre_warps4, warps8, block32, plain, lb2` was about `-6.0%`.
- Conclusion: global precompute/reuse of QK/KKT is not a keepable FlashQLA-style route in this implementation. The recomputation saved by moving QK/KKT ownership is eaten by extra global matrix cache writes/reads, cache-copy phases, and launch overhead/noise. The next structural attempt should not materialize QK/KKT in global memory; it should fuse ownership inside the same CTA/CTA group, for example by computing one `H_k` chunk for both paired `H_v` heads or both value slices in one cooperative block/cluster so QK/KKT remain in shared/register fragments.

## 2026-05-03 BF16 Tensor Core Chunk Prefill

- Baseline/reference: original llama.cpp-style FP32 scalar GDN recurrence, ported as a CUDA extension in `kernels/qwen35moe_gdn_prefill.cu`.
- Optimized route: chunk size 64 prefill path in `kernels_optimized/qwen35moe_gdn_prefill.cu`.
- External tensors remain FP32 to preserve the llama.cpp GDN call semantics.
- Q/K/state/chunk intermediates are converted to BF16 for WMMA GEMMs with FP32 accumulators.
- Gate prefix, triangular solve, and recurrent state update accumulation remain FP32.
- Benchmark shapes: Qwen3.5 MoE `D=128`, `H_k=16`, `H_v=32`, `KDA=false`, batch 1, seqlen 2048/4096/8192 plus 2113/4097/8191 tail cases.

## 2026-05-03 FlashQLA-Informed Iteration 1-20

- FlashQLA prefill design reference: chunk size 64, value-dimension splitting to raise CTA count, BF16/FP16 Tensor Core matmuls with FP32 accumulators, FP32 gate/state path, and chunk-local fused dataflow that keeps recurrent state hot instead of treating every chunk as an independent global-memory round trip.
- Iterations 1-13 before this pass: kept `BLOCK_DV=64`, `WARPS=16`, shared-memory parallel gate prefix, KKT lower-triangle pre-scaling, and `exp2f(prefix * log2e)`; rejected `BLOCK_DV=32`, 4/12-warps, serial prefix, global reciprocal cache, triangular unroll variants, repeated-prefix redo, and full-chunk q/k load split.
- Iteration 14 kept: copied only each z-CTA's `128x64` state slice instead of redundantly copying the full `128x128` state matrix from both z CTAs.
- Iteration 15 kept: moved each CTA's FP32 recurrent state slice to shared memory across chunks, converting it to BF16 staging for WMMA and writing final state once at the end. This was the biggest gain, reducing 4096 quick median to about `5.60 ms`.
- Iteration 16 kept: changed output epilogue from global store plus read-modify-write to workspace base-output staging plus a single global `out` store after `P@U`.
- Iteration 17 rejected: `BLOCK_DV=32` on the shared-state route regressed to about `8.3-9.0 ms`; the extra duplicated Q/K/KKT/solve work outweighed higher CTA count.
- Iteration 18 rejected: `WARPS=8` regressed to about `6.0-6.7 ms`; the current WMMA tile schedule still needs 16 warps.
- Iteration 19 kept: removed three chunk-local `__syncthreads()` calls with no true data dependency (`uh` conversion, base-output staging, final `out` write).
- Iteration 20 kept: replaced the 64-element gate prefix Hillis-Steele block scan with a two-warp `__shfl_up_sync` scan, reducing gate scan barriers from 12 to 2.
- Final full benchmark, warmup 3 / rep 10 / trials 1, correctness PASS, worst max abs error `2.03e-4`:
  - `2048`: optimized `2.892 ms`, FP32 scalar reference `2.952 ms`, `1.021x`.
  - `4096`: optimized `5.983 ms`, reference `4.943 ms`, `0.826x`.
  - `8192`: optimized `9.757 ms`, reference `10.299 ms`, `1.056x`.
  - `2113`: optimized `2.639 ms`, reference `2.543 ms`, `0.963x`.
  - `4097`: optimized `5.041 ms`, reference `5.145 ms`, `1.021x`.
  - `8191`: optimized `10.874 ms`, reference `10.526 ms`, `0.968x`.
- SASS check on the final extension confirmed `HMMA.16816.F32.BF16`, so the active path is still BF16 Tensor Core MMA with FP32 accumulators.
- Remaining bottleneck: this route is still far below Tensor Core peak because Q/K/K-restored/U/P intermediates are staged through global workspace, and the FP32 triangular solve remains serial by row inside each chunk. The next useful route should move more chunk-local BF16 staging into shared memory or redesign the solve/state update dataflow; local launch-shape tweaks are mostly exhausted.

## 2026-05-03 FlashQLA Storage-Hierarchy Iteration 21-40

- FlashQLA storage model used for this pass:
  - Global memory is limited to model inputs, output, final state, and optional chunk state.
  - Producer warps stream Q/K/V/A/g/b into double-buffered shared memory.
  - Consumer roles keep hot recurrent state, output, U/V/P/A/G intermediates in fragments/local storage and use shared memory only for cross-role handoff.
  - Gate derived values are precomputed in FP32 shared/local form (`g`, `exp(g)`, `g_last/g`) to avoid repeated transcendental/divide work.
  - Value-dimension splitting is a scheduling knob: smaller `block_DV` raises CTA count but duplicates Q/K and chunk-matrix work.
- Current CUDA route cannot directly reproduce FlashQLA's Hopper-style register role split on Ada WMMA, so the experiments targeted the largest local mismatches: global workspace staging, redundant transpose, row-level barriers, chunk size, and value split.
- Kept changes:
  - Removed the explicit `kh_t` BF16 transpose workspace. KKT and QK now read `khm` as a WMMA column-major B operand via `wmma_gemm_bf16_bf16_f32_rm_b_col`.
  - Shrunk workspace layout by aliasing the old `kh_t` storage with `sh`.
  - Removed 31/63 per-row `__syncthreads()` calls from the triangular solve; each value column is an independent recurrence, so a single block barrier after the solve is sufficient.
  - Kept shared `g_inv` and rewrote gate ratios as multiplies.
  - Changed `CHUNK` from 64 to 32. This reduced chunk-local `KKT`, `QK`, `P@U`, and triangular-solve work enough to offset the doubled chunk count.
  - Simplified the gate prefix scan for `CHUNK=32` to one full warp and one block barrier.
- Rejected or reverted:
  - Moving `state/q/k/kt/sh` together to dynamic shared (~96KB) regressed.
  - Moving only `sh` to dynamic shared regressed or was neutral.
  - Moving Q/K BF16 staging to dynamic shared regressed for both chunk 64 and chunk 32.
  - Packed `bfloat162` Q/K staging did not beat compiler-generated scalar/pack code.
  - `BLOCK_DV=128` lost too much parallelism; `BLOCK_DV=32` still duplicated too much Q/K/KKT/P work.
  - `WARPS=8` and `WARPS=12` were slower than `WARPS=16`.
  - `CHUNK=16` and initial `CHUNK=48` exposed the full-mask warp-scan assumption; after fixing with `activemask`, `CHUNK=48` was correct but slower than 32. Final fixed `CHUNK=32` uses the faster full-warp scan.
- Final selected constants: `CHUNK=32`, `BLOCK_DV=64`, `WARPS=16`.
- Final 3-trial full benchmark, warmup 5 / rep 20, correctness PASS, worst max abs error `1.90e-4`:
  - `2048`: optimized `1.933 ms`, FP32 scalar reference `2.573 ms`, `1.331x`.
  - `4096`: optimized `3.959 ms`, reference `5.126 ms`, `1.295x`.
  - `8192`: optimized `7.809 ms`, reference `10.518 ms`, `1.347x`.
  - `2113`: optimized `2.070 ms`, reference `2.647 ms`, `1.279x`.
  - `4097`: optimized `3.913 ms`, reference `5.286 ms`, `1.351x`.
  - `8191`: optimized `7.672 ms`, reference `10.576 ms`, `1.379x`.
- The 2x speedup target was not reached in 20 iterations. Versus the previous selected full benchmark at 4096 (`5.983 ms`), this pass improved to `3.959 ms` (`1.51x` over the prior optimized kernel), but speedup over the FP32 scalar reference is `1.30x`.
- SASS check on the final extension hash `a09bf9e0765d` confirmed `HMMA.16816.F32.BF16`.
- Remaining design boundary: the current WMMA implementation still materializes Q/K/SH/U/P/K-restored/M/UPD through global workspace and uses block-wide phases. A credible path to 2x likely needs a FlashQLA-like role-split rewrite that keeps `h/o/u/p` in registers/fragments and uses shared memory as handoff, or a multi-kernel/precompute design that avoids recomputing QK/KKT across the two value slices and the two H_v heads sharing each H_k head.
