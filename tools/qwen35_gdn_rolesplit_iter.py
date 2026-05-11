#!/usr/bin/env python3
"""Run FlashQLA-style role-split iterations for qwen35moe_gdn_prefill.

The route keeps the stable log-prefix decay ratio as an FP32 CTA-local handoff
and sweeps broader ownership/resource choices around it: Q residency, U/Vd
handoff location, ratio placement, leading dimensions, and launch bounds.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
KERNEL_CU = ROOT / "kernel.cu"
WORKSPACE = ROOT / "workspace"
DEFAULT_PYTHON = "/home/et/miniconda3/bin/python3.13"


@dataclass(frozen=True)
class Candidate:
    q_mode: str
    ratio_place: str
    uh_mode: str
    warps: int
    pre_warps: int
    state_ld: int
    kh_ld: int
    ph_ld: int
    m_ld: int
    upd_ld: int
    launch_min_blocks: int


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def replace_one(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"pattern not found: {old!r}")
    return text.replace(old, new, 1)


def regex_replace_one(text: str, pattern: str, repl: str) -> str:
    text2, n = re.subn(pattern, repl, text, count=1)
    if n != 1:
        raise ValueError(f"regex pattern matched {n} times: {pattern}")
    return text2


def apply_candidate(base: str, cand: Candidate) -> str:
    text = base
    text = regex_replace_one(text, r"constexpr int WARPS = \d+;", f"constexpr int WARPS = {cand.warps};")
    text = regex_replace_one(text, r"constexpr int PRE_WARPS = \d+;", f"constexpr int PRE_WARPS = {cand.pre_warps};")
    text = regex_replace_one(
        text,
        r"__global__ __launch_bounds__\(THREADS, \d+\)",
        f"__global__ __launch_bounds__(THREADS, {cand.launch_min_blocks})",
    )
    for name, value in (
        ("STATE_LD", cand.state_ld),
        ("KH_LD", cand.kh_ld),
        ("PH_LD", cand.ph_ld),
        ("M_LD", cand.m_ld),
        ("UPD_LD", cand.upd_ld),
    ):
        text = regex_replace_one(text, rf"constexpr int {name} = \d+;", f"constexpr int {name} = {value};")

    if cand.q_mode == "shared":
        text = replace_one(
            text,
            "  bf16_t* qh      = reinterpret_cast<bf16_t*>(ws + TC_QH_OFF);\n",
            "",
        )
        text = replace_one(
            text,
            "  __shared__ __align__(16) bf16_t kh_s_main[CHUNK * KH_LD];\n",
            "  __shared__ __align__(16) bf16_t kh_s_main[CHUNK * KH_LD];\n"
            "  __shared__ __align__(16) bf16_t qh_s_main[CHUNK * D];\n",
        )
        text = replace_one(
            text,
            "  bf16_t* ph = ph_s;\n",
            "  bf16_t* qh = qh_s_main;\n"
            "  bf16_t* ph = ph_s;\n",
        )
    elif cand.q_mode == "reload_swz":
        text = replace_one(
            text,
            "        *reinterpret_cast<__nv_bfloat162*>(qh + row * D + col2) = q_bf16;\n",
            "        (void)q_bf16;\n",
        )
        text = replace_one(
            text,
            "      *reinterpret_cast<__nv_bfloat162*>(qh + row * D + col2) = q_bf16;\n",
            "      (void)q_bf16;\n",
        )
        old = (
            "    // Base output: scale * G_t * (Q @ S0).\n"
            "    wmma_gemm_bf16_bf16_f32_rm_b_ld_c_ld<CHUNK, BLOCK_DV, D, STATE_LD, M_LD>(qh, sh, m, warp_id, num_warps);\n"
            "    __syncthreads();\n"
        )
        new = (
            "    // Base output: scale * G_t * (Q @ S0). Reload Q into the\n"
            "    // role-local swizzled tile instead of using the global QH workspace.\n"
            "    for (int idx = tid; idx < CHUNK * (D / 2); idx += blockDim.x) {\n"
            "      const int row = idx / (D / 2);\n"
            "      const int col2 = (idx - row * (D / 2)) * 2;\n"
            "      float q0 = 0.0f;\n"
            "      float q1 = 0.0f;\n"
            "      if (row < actual) {\n"
            "        const int t = chunk0 + row;\n"
            "        const int64_t base_off = ((static_cast<int64_t>(seq) * tokens + t) * H_K + kh) * D + col2;\n"
            "        const float2 qv = *reinterpret_cast<const float2*>(q + base_off);\n"
            "        q0 = qv.x;\n"
            "        q1 = qv.y;\n"
            "      }\n"
            "      *reinterpret_cast<__nv_bfloat162*>(khm_swz + swizzled_bf16_index<D>(row, col2)) =\n"
            "          __floats2bfloat162_rn(q0, q1);\n"
            "    }\n"
            "    __syncthreads();\n"
            "    mma_gemm_bf16_bf16_f32_rm_smem_a_swz_ld_b_ld_c_ld<CHUNK, BLOCK_DV, D, D, STATE_LD, M_LD>(khm_swz, sh, m, warp_id, num_warps);\n"
            "    __syncthreads();\n"
        )
        text = replace_one(text, old, new)
    elif cand.q_mode != "global":
        raise ValueError(f"unknown q mode: {cand.q_mode}")

    if cand.uh_mode == "shared":
        if cand.q_mode == "shared":
            text = replace_one(
                text,
                "  bf16_t* uh      = reinterpret_cast<bf16_t*>(ws + TC_UH_OFF);\n",
                "",
            )
        else:
            text = replace_one(
                text,
                "  bf16_t* qh      = reinterpret_cast<bf16_t*>(ws + TC_QH_OFF);\n"
                "  bf16_t* uh      = reinterpret_cast<bf16_t*>(ws + TC_UH_OFF);\n",
                "  bf16_t* qh      = reinterpret_cast<bf16_t*>(ws + TC_QH_OFF);\n",
            )
        text = replace_one(
            text,
            "  __shared__ __align__(16) bf16_t ph_s[CHUNK * PH_LD];\n",
            "  __shared__ __align__(16) bf16_t ph_s[CHUNK * PH_LD];\n"
            "  __shared__ __align__(16) bf16_t uh_s[CHUNK * BLOCK_DV];\n",
        )
        text = replace_one(
            text,
            "  bf16_t* ph = ph_s;\n"
            "  float* m = m_s;\n",
            "  bf16_t* ph = ph_s;\n"
            "  bf16_t* uh = uh_s;\n"
            "  float* m = m_s;\n",
        )
    elif cand.uh_mode != "global":
        raise ValueError(f"unknown uh mode: {cand.uh_mode}")

    if cand.ratio_place == "m_tail":
        text = replace_one(
            text,
            "  float* ratio_s = upd_s + CHUNK * BLOCK_DV;\n",
            "  // ratio_s is kept in the tail columns of m_s for this candidate.\n",
        )
        text = replace_one(
            text,
            "      ratio_s[row * CHUNK + col] = ratio;\n",
            "      m_s[row * M_LD + BLOCK_DV + col] = ratio;\n",
        )
        text = replace_one(
            text,
            "        ratio = ratio_s[row * CHUNK + col];\n",
            "        ratio = m_s[row * M_LD + BLOCK_DV + col];\n",
        )
    elif cand.ratio_place != "upd":
        raise ValueError(f"unknown ratio place: {cand.ratio_place}")

    return text


def shared_bytes(c: Candidate) -> int:
    d = 128
    chunk = 32
    block_dv = 32
    return (
        d * c.state_ld * 2
        + chunk * c.kh_ld * 2
        + 3 * chunk * 4
        + 2 * 4
        + chunk * c.ph_ld * 2
        + chunk * c.m_ld * 4
        + d * c.upd_ld * 4
        + (chunk * d * 2 if c.q_mode == "shared" else 0)
        + (chunk * block_dv * 2 if c.uh_mode == "shared" else 0)
    )


def generate_candidates(limit: int) -> list[Candidate]:
    current = Candidate("global", "upd", "global", 16, 6, 40, 136, 40, 48, 40, 2)
    candidates: list[Candidate] = [current]
    seen = {current}

    q_modes = ["global", "shared", "reload_swz"]
    ratio_places = ["upd", "m_tail"]
    uh_modes = ["global", "shared"]
    warps_opts = [16, 12, 20, 8]
    pre_warps_opts = [6, 4, 8]
    state_lds = [40, 32, 48]
    kh_lds = [136, 128, 144, 152]
    ph_lds = [40, 32, 48]
    m_lds = [48, 64, 40, 56, 32]
    upd_lds = [40, 36, 32, 44]
    lbs = [2, 1]

    def score(c: Candidate) -> tuple[int, int, int, int, int, int, int, int]:
        route_rank = {
            ("global", "upd", "global"): 0,
            ("global", "upd", "shared"): 1,
            ("shared", "upd", "global"): 2,
            ("shared", "upd", "shared"): 3,
            ("global", "m_tail", "global"): 4,
            ("global", "m_tail", "shared"): 5,
            ("reload_swz", "m_tail", "global"): 6,
            ("reload_swz", "m_tail", "shared"): 7,
        }.get((c.q_mode, c.ratio_place, c.uh_mode), 9)
        return (
            route_rank,
            abs(c.warps - 16),
            abs(c.pre_warps - 6),
            abs(c.state_ld - 40),
            abs(c.kh_ld - 136),
            abs(c.ph_ld - 40),
            abs(c.m_ld - 48),
            abs(c.upd_ld - 40),
            0 if c.launch_min_blocks == 2 else 1,
        )

    pools: dict[tuple[str, str, str], list[Candidate]] = {}
    for vals in itertools.product(q_modes, ratio_places, uh_modes, warps_opts, pre_warps_opts, state_lds, kh_lds, ph_lds, m_lds, upd_lds, lbs):
        c = Candidate(*vals)
        if c in seen:
            continue
        if c.ratio_place == "m_tail" and c.m_ld < 64:
            continue
        if c.q_mode == "reload_swz" and c.ratio_place != "m_tail":
            continue
        if c.q_mode == "shared" and c.ratio_place == "m_tail":
            # Shared Q already pressures static shared heavily; keep this family
            # focused unless a smaller resource variant admits it naturally.
            if c.state_ld > 32 or c.upd_ld > 32:
                continue
        if shared_bytes(c) > 49152:
            continue
        # base_out and FP32 ratio_s reuse require at least 2048 floats.
        if c.ratio_place == "upd" and c.upd_ld * 128 < 2048:
            continue
        if c.q_mode == "reload_swz" and c.upd_ld * 128 < 1024:
            continue
        pools.setdefault((c.q_mode, c.ratio_place, c.uh_mode), []).append(c)
    for items in pools.values():
        items.sort(key=score)

    route_order = [
        ("global", "upd", "global"),
        ("global", "upd", "shared"),
        ("shared", "upd", "global"),
        ("shared", "upd", "shared"),
        ("global", "m_tail", "global"),
        ("global", "m_tail", "shared"),
        ("shared", "m_tail", "global"),
        ("shared", "m_tail", "shared"),
        ("reload_swz", "m_tail", "global"),
        ("reload_swz", "m_tail", "shared"),
    ]
    offsets = {route: 0 for route in route_order}
    while len(candidates) < limit:
        progressed = False
        for route in route_order:
            items = pools.get(route) or []
            offset = offsets[route]
            if offset >= len(items):
                continue
            candidates.append(items[offset])
            offsets[route] = offset + 1
            progressed = True
            if len(candidates) >= limit:
                break
        if not progressed:
            break
    return candidates


def run_bench(python: str, out_json: Path, out_log: Path, timeout: int) -> tuple[int, dict[str, Any] | None, str]:
    cmd = [
        python,
        "tools/bench.py",
        "--quick",
        "--bench-warmup",
        "3",
        "--bench-rep",
        "10",
        "--bench-trials",
        "2",
        "--max-bench-trials",
        "2",
        "--no-adaptive-trials",
        "--json-out",
        str(out_json),
    ]
    with out_log.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    payload = None
    if out_json.exists():
        try:
            payload = json.loads(out_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None
    return proc.returncode, payload, " ".join(cmd)


def summarize_payload(payload: dict[str, Any] | None) -> tuple[bool, float | None, float | None, str]:
    if not payload:
        return False, None, None, "missing_json"
    correctness = payload.get("correctness") or {}
    passed = bool(correctness.get("passed"))
    kernel = payload.get("kernel") or {}
    median = kernel.get("median_ms")
    spread = kernel.get("spread_pct")
    err = payload.get("error")
    reason = "ok" if passed else json.dumps(err or correctness, ensure_ascii=False)[:500]
    return passed, (float(median) if median is not None else None), (float(spread) if spread is not None else None), reason


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--base", type=Path, default=KERNEL_CU)
    parser.add_argument("--python", type=str, default=DEFAULT_PYTHON)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    run_dir = args.run_dir or (WORKSPACE / f"rolesplit_iter_{int(time.time())}")
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cand_dir = run_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"

    base = args.base.read_text(encoding="utf-8")
    if "float* ratio_s = upd_s + CHUNK * BLOCK_DV;" not in base:
        raise SystemExit("base source must include the ratio_s role-split handoff")

    candidates = generate_candidates(args.iterations)
    best_ms: float | None = None
    best_idx: int | None = None
    best_src: str | None = None
    records: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    for idx, cand in enumerate(candidates, start=1):
        src = apply_candidate(base, cand)
        digest = sha256_text(src)
        if digest in seen_hashes:
            record = {"idx": idx, "status": "skip_duplicate", "candidate": asdict(cand), "sha256": digest}
            records.append(record)
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            continue
        seen_hashes.add(digest)

        cand_src = cand_dir / f"cand_{idx:04d}.cu"
        cand_src.write_text(src, encoding="utf-8")
        KERNEL_CU.write_text(src, encoding="utf-8")

        out_json = run_dir / f"cand_{idx:04d}.json"
        out_log = run_dir / f"cand_{idx:04d}.log"
        started = time.time()
        try:
            rc, payload, cmd = run_bench(args.python, out_json, out_log, args.timeout)
            elapsed = time.time() - started
            passed, median, spread, reason = summarize_payload(payload)
            status = "pass" if rc == 0 and passed else "fail"
        except subprocess.TimeoutExpired:
            elapsed = time.time() - started
            rc, payload, cmd = 124, None, "timeout"
            passed, median, spread, reason = False, None, None, "timeout"
            status = "timeout"

        kept = False
        if passed and median is not None and (best_ms is None or median < best_ms * 0.99):
            best_ms = median
            best_idx = idx
            best_src = src
            kept = True
            (run_dir / "best.cu").write_text(src, encoding="utf-8")

        record = {
            "idx": idx,
            "status": status,
            "kept_as_quick_best": kept,
            "candidate": asdict(cand),
            "shared_bytes": shared_bytes(cand),
            "sha256": digest,
            "returncode": rc,
            "passed": passed,
            "median_ms": median,
            "spread_pct": spread,
            "reason": reason,
            "elapsed_s": elapsed,
            "json": str(out_json.relative_to(ROOT)),
            "log": str(out_log.relative_to(ROOT)),
            "source": str(cand_src.relative_to(ROOT)),
            "cmd": cmd,
        }
        records.append(record)
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"[{idx:03d}/{len(candidates):03d}] {status} "
            f"median={median} spread={spread} kept={kept} "
            f"shared={shared_bytes(cand)} {asdict(cand)}",
            flush=True,
        )

    if best_src is not None:
        KERNEL_CU.write_text(best_src, encoding="utf-8")

    summary = {
        "run_dir": str(run_dir.relative_to(ROOT)),
        "iterations_requested": args.iterations,
        "iterations_generated": len(candidates),
        "records": len(records),
        "best_idx": best_idx,
        "best_ms": best_ms,
        "best_source": str((run_dir / "best.cu").relative_to(ROOT)) if best_src is not None else None,
        "records_path": str(results_path.relative_to(ROOT)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
