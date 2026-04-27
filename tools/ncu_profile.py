#!/usr/bin/env python3
"""
ncu_profile.py -- Nsight Compute wrapper with artifact-friendly modes.

Modes:
  targeted: collect a compact metric set for fast diagnosis
  full:     collect the broader metric set
  import:   import an existing .ncu-rep and emit text summaries
  diff:     compare two CSV exports
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from .runtime import build_python_cmd
except ImportError:
    from runtime import build_python_cmd


SKILL_METRICS: dict[str, list[str]] = {
    "roofline": [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained",
        "sm__sass_thread_inst_executed_op_hfma_pred_on.sum.peak_sustained",
    ],
    "memory": [
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
        "l1tex__t_sector_hit_rate.pct",
        "lts__t_sector_hit_rate.pct",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
        "memory_l2_theoretical_sectors_global",
        "memory_l2_theoretical_sectors_global_ideal",
    ],
    "warp_stall": [
        "smsp__warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
        "smsp__warps_issue_stalled_wait_per_issue_active.ratio",
        "smsp__warps_issue_stalled_mio_throttle_per_issue_active.ratio",
        "smsp__warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
        "smsp__warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
        "smsp__warps_issue_stalled_barrier_per_issue_active.ratio",
        "smsp__warps_issue_stalled_membar_per_issue_active.ratio",
        "smsp__warps_issue_stalled_not_selected_per_issue_active.ratio",
        "smsp__warps_issue_stalled_sleeping_per_issue_active.ratio",
        "smsp__warps_issue_stalled_tex_throttle_per_issue_active.ratio",
        "smsp__warps_issue_stalled_no_instruction_per_issue_active.ratio",
    ],
    "occupancy": [
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__warps_active.avg.per_cycle_active",
        "launch__registers_per_thread",
        "launch__shared_mem_per_block_static",
        "launch__shared_mem_per_block_dynamic",
        "launch__block_size",
        "launch__grid_size",
        "launch__occupancy_limit_registers",
        "launch__occupancy_limit_shared_mem",
        "launch__occupancy_limit_warps",
        "launch__occupancy_limit_blocks",
        "launch__waves_per_multiprocessor",
    ],
    "instruction": [
        "sm__inst_executed.sum",
        "sm__inst_executed_pipe_tensor.sum",
        "sm__inst_executed_pipe_fp16.sum",
        "sm__inst_executed_pipe_fp32.sum",
        "sm__inst_executed_pipe_fp64.sum",
        "sm__inst_executed_pipe_lsu.sum",
        "smsp__inst_executed.avg.per_cycle_active",
    ],
}

ALL_SKILLS = list(SKILL_METRICS.keys())
TARGETED_SKILLS = ["roofline", "memory", "warp_stall", "occupancy"]
STALL_METRIC_PREFIX = "smsp__warps_issue_stalled_"


def _find_ncu() -> str | None:
    return shutil.which("ncu")


def _safe_float(val: str) -> float | None:
    if not val:
        return None
    val = val.strip().replace(",", "").replace("%", "")
    try:
        return float(val)
    except ValueError:
        return None


def collect_metrics(skills: list[str]) -> list[str]:
    metrics = []
    for skill in skills:
        metrics.extend(SKILL_METRICS.get(skill, []))
    return sorted(set(metrics))


def _get_kernel_launch_cmd(kernel_file: str, gpu: int = 0) -> list[str]:
    repo_root = str(Path(kernel_file).resolve().parent)
    return build_python_cmd(
        "-c",
        f"""
import importlib.util, os, sys
os.chdir({repo_root!r})
if {repo_root!r} not in sys.path:
    sys.path.insert(0, {repo_root!r})
spec = importlib.util.spec_from_file_location("kernel_mod", {kernel_file!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
import torch
torch.cuda.set_device({gpu})
kernel_type = getattr(mod, "KERNEL_TYPE", None)
if not kernel_type:
    raise RuntimeError("kernel module has no KERNEL_TYPE attribute")
from kernel_configs import KERNEL_CONFIGS
cfg = KERNEL_CONFIGS[kernel_type]
sizes = cfg["test_sizes"]
size = next((sz for label, sz in sizes if label == "large"), sizes[-1][1])
dtype = cfg["test_dtypes"][0]
inputs = cfg["input_generator"](size, dtype, "cpu", seed=42)
inputs = {{
    key: (value.cuda() if hasattr(value, "cuda") else value)
    for key, value in inputs.items()
}}
torch.cuda.synchronize()
for _ in range(3):
    mod.kernel_fn(**inputs)
torch.cuda.synchronize()
""",
    )


def build_ncu_cmd(
    kernel_file: str,
    metrics: list[str],
    csv_out: str,
    rep_out: str | None,
    gpu: int,
    launch_skip: int,
    launch_count: int,
    extra_args: list[str] | None = None,
) -> list[str]:
    ncu = _find_ncu()
    if ncu is None:
        raise RuntimeError("ncu_not_found")
    cmd = [
        ncu,
        "--csv",
        "--page",
        "raw",
        "--log-file",
        csv_out,
        "--target-processes",
        "all",
        "--kernel-name-base",
        "demangled",
        "--launch-skip",
        str(launch_skip),
        "--launch-count",
        str(launch_count),
    ]
    if metrics:
        cmd += ["--metrics", ",".join(metrics)]
    if rep_out:
        cmd += ["-o", rep_out]
    if extra_args:
        cmd += extra_args
    cmd += _get_kernel_launch_cmd(kernel_file, gpu=gpu)
    return cmd


def run_command(cmd: list[str], stdout_path: str | None = None, stderr_path: str | None = None) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if stdout_path:
        Path(stdout_path).write_text(result.stdout or "", encoding="utf-8")
    if stderr_path:
        Path(stderr_path).write_text(result.stderr or "", encoding="utf-8")
    return result


def parse_ncu_csv(csv_path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    try:
        content = Path(csv_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"ncu_error: CSV file not found: {csv_path}")
        return rows

    lines = content.strip().splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('"ID"') or line.startswith("ID"):
            header_idx = i
            break
        if "Metric Name" in line or "metric_name" in line.lower():
            header_idx = i
            break
    if header_idx is None:
        print("ncu_warning: Could not find CSV header in NCU output")
        return rows

    reader = csv.DictReader(lines[header_idx:])
    for row in reader:
        rows.append(dict(row))
    return rows


def analyze_rows(rows: list[dict[str, str]]) -> dict[str, str]:
    results: dict[str, str] = {}
    metric_vals: dict[str, float] = {}

    wide_rows = [row for row in rows if row.get("Kernel Name")]
    if wide_rows:
        for row in wide_rows:
            for name, val_str in row.items():
                if not name or "__" not in name and not name.startswith("launch__"):
                    continue
                val = _safe_float(val_str)
                if val is None:
                    continue
                metric_vals[name] = max(metric_vals.get(name, val), val)
    else:
        for row in rows:
            name = row.get("Metric Name", row.get("metric_name", ""))
            val_str = row.get("Metric Value", row.get("metric_value", row.get("Average", "")))
            val = _safe_float(val_str)
            if name and val is not None:
                metric_vals[name] = max(metric_vals.get(name, val), val)
    if not metric_vals:
        return results

    sm_pct = metric_vals.get("sm__throughput.avg.pct_of_peak_sustained_elapsed")
    mem_pct = metric_vals.get("dram__throughput.avg.pct_of_peak_sustained_elapsed")
    if sm_pct is not None and mem_pct is not None:
        if mem_pct > sm_pct:
            results["ncu_bottleneck"] = f"memory_bound (sm={sm_pct:.1f}%, dram={mem_pct:.1f}%)"
        else:
            results["ncu_bottleneck"] = f"compute_bound (sm={sm_pct:.1f}%, dram={mem_pct:.1f}%)"

    stalls: list[tuple[str, float]] = []
    for name, val in metric_vals.items():
        if STALL_METRIC_PREFIX in name:
            short = name.replace(STALL_METRIC_PREFIX, "").replace("_per_issue_active.ratio", "")
            stalls.append((short, val))
    if stalls:
        stalls.sort(key=lambda item: item[1], reverse=True)
        top = stalls[0]
        results["ncu_top_stall"] = f"{top[0]} ({top[1]:.2f})"
        results["ncu_stall_breakdown"] = ", ".join(f"{name}={val:.2f}" for name, val in stalls[:5])

    occ = metric_vals.get("sm__warps_active.avg.pct_of_peak_sustained_active")
    if occ is not None:
        results["ncu_occupancy"] = f"{occ:.1f}%"

    regs = metric_vals.get("launch__registers_per_thread")
    if regs is not None:
        results["ncu_registers_per_thread"] = f"{int(regs)}"

    l1_hit = metric_vals.get("l1tex__t_sector_hit_rate.pct")
    if l1_hit is not None:
        results["ncu_l1_hit_rate"] = f"{l1_hit:.1f}%"

    l2_hit = metric_vals.get("lts__t_sector_hit_rate.pct")
    if l2_hit is not None:
        results["ncu_l2_hit_rate"] = f"{l2_hit:.1f}%"

    dram_read = metric_vals.get("dram__bytes_read.sum")
    dram_write = metric_vals.get("dram__bytes_write.sum")
    if dram_read is not None and dram_write is not None:
        results["ncu_dram_traffic_gb"] = f"{(dram_read + dram_write) / 1e9:.3f}"

    ld_conflicts = metric_vals.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum")
    st_conflicts = metric_vals.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum")
    if ld_conflicts is not None or st_conflicts is not None:
        results["ncu_smem_bank_conflicts"] = str(int((ld_conflicts or 0) + (st_conflicts or 0)))

    actual = metric_vals.get("memory_l2_theoretical_sectors_global")
    ideal = metric_vals.get("memory_l2_theoretical_sectors_global_ideal")
    if actual is not None and ideal is not None and actual > 0:
        results["ncu_coalescing_efficiency"] = f"{ideal / actual * 100:.1f}%"

    tc_inst = metric_vals.get("sm__inst_executed_pipe_tensor.sum")
    total_inst = metric_vals.get("sm__inst_executed.sum")
    if tc_inst is not None and total_inst is not None and total_inst > 0:
        results["ncu_tensor_core_pct"] = f"{tc_inst / total_inst * 100:.1f}%"

    ipc = metric_vals.get("smsp__inst_executed.avg.per_cycle_active")
    if ipc is not None:
        results["ncu_ipc"] = f"{ipc:.2f}"

    findings: list[str] = []
    actions: list[str] = []
    if stalls and stalls[0][1] > 0.3:
        top_stall = stalls[0][0]
        if top_stall == "long_scoreboard":
            findings.append("High long scoreboard stalls: memory latency dominates")
            actions.append("Add prefetching/pipelining (num_stages), reduce memory accesses, improve L2 locality")
        elif top_stall == "wait":
            findings.append("High wait stalls: barrier synchronization overhead")
            actions.append("Reduce synchronization frequency and rebalance shared-memory access")
        elif top_stall == "mio_throttle":
            findings.append("High MIO throttle: memory instruction queue full")
            actions.append("Reduce outstanding memory ops or increase compute per byte")
        elif top_stall == "math_pipe_throttle":
            findings.append("High math pipe throttle: compute pipeline saturated")
            actions.append("Look for tensor-core use, fusion, or algorithmic simplification")
        elif top_stall == "short_scoreboard":
            findings.append("High short scoreboard stalls: shared memory / L1 latency")
            actions.append("Check bank conflicts and shared-memory access density")

    if occ is not None and occ < 50:
        findings.append(f"Low occupancy ({occ:.0f}%)")
        if regs is not None and regs > 64:
            actions.append(f"Reduce register pressure ({int(regs)} regs/thread)")
        else:
            actions.append("Increase active blocks or reduce shared memory per block")
    if l1_hit is not None and l1_hit < 30:
        findings.append(f"Low L1 hit rate ({l1_hit:.0f}%)")
        actions.append("Improve spatial locality or add shared-memory tiling")
    if l2_hit is not None and l2_hit < 50:
        findings.append(f"Low L2 hit rate ({l2_hit:.0f}%)")
        actions.append("Improve tile ordering and cache reuse")

    for idx, finding in enumerate(findings, start=1):
        results[f"ncu_finding_{idx}"] = finding
    for idx, action in enumerate(actions, start=1):
        results[f"ncu_action_{idx}"] = action
    return results


def format_analysis_text(results: dict[str, str], title: str) -> str:
    lines = [f"=== {title} ==="]
    for key, value in sorted(results.items()):
        lines.append(f"{key}: {value}")
    lines.append(f"=== END {title} ===")
    return "\n".join(lines) + "\n"


def write_summary_and_details(results: dict[str, str], summary_out: str | None, details_out: str | None) -> None:
    summary_lines = [
        f"ncu_bottleneck: {results.get('ncu_bottleneck', '')}",
        f"ncu_top_stall: {results.get('ncu_top_stall', '')}",
        f"ncu_occupancy: {results.get('ncu_occupancy', '')}",
        f"ncu_l1_hit_rate: {results.get('ncu_l1_hit_rate', '')}",
        f"ncu_l2_hit_rate: {results.get('ncu_l2_hit_rate', '')}",
    ]
    details_text = format_analysis_text(results, "NCU ANALYSIS")
    if summary_out:
        Path(summary_out).write_text("\n".join(summary_lines).strip() + "\n", encoding="utf-8")
    if details_out:
        Path(details_out).write_text(details_text, encoding="utf-8")


def run_profile_mode(
    mode: str,
    kernel_file: str,
    output_prefix: str,
    gpu: int,
    skills: list[str],
    extra_args: list[str] | None,
    launch_skip: int,
    launch_count: int,
    stdout_out: str | None,
    stderr_out: str | None,
    summary_out: str | None,
    details_out: str | None,
) -> int:
    metrics = collect_metrics(skills)
    csv_path = f"{output_prefix}.csv"
    rep_path = f"{output_prefix}.ncu-rep"
    cmd = build_ncu_cmd(
        kernel_file=kernel_file,
        metrics=metrics,
        csv_out=csv_path,
        rep_out=output_prefix,
        gpu=gpu,
        launch_skip=launch_skip,
        launch_count=launch_count,
        extra_args=extra_args,
    )
    print(f"ncu_mode: {mode}")
    print(f"ncu_command: {' '.join(cmd)}")
    result = run_command(cmd, stdout_path=stdout_out, stderr_path=stderr_out)
    print(f"ncu_csv: {csv_path}")
    print(f"ncu_report_path: {rep_path}")
    if result.returncode != 0:
        print(f"ncu_error: ncu exited with code {result.returncode}")
        return result.returncode
    rows = parse_ncu_csv(csv_path)
    print(f"ncu_parsed_rows: {len(rows)}")
    analysis = analyze_rows(rows)
    analysis["ncu_mode"] = mode
    analysis["ncu_csv"] = csv_path
    analysis["ncu_report_path"] = rep_path
    write_summary_and_details(analysis, summary_out, details_out)
    print(format_analysis_text(analysis, "NCU ANALYSIS").rstrip())
    return 0


def import_ncu_report(
    rep_path: str,
    csv_out: str,
    summary_out: str | None,
    details_out: str | None,
    stdout_out: str | None,
    stderr_out: str | None,
) -> int:
    ncu = _find_ncu()
    if ncu is None:
        print("ncu_error: ncu not found")
        return 1
    cmd = [ncu, "--import", rep_path, "--csv", "--page", "raw", "--log-file", csv_out]
    print(f"ncu_mode: import")
    print(f"ncu_command: {' '.join(cmd)}")
    result = run_command(cmd, stdout_path=stdout_out, stderr_path=stderr_out)
    if result.returncode != 0:
        print(f"ncu_error: import failed with code {result.returncode}")
        return result.returncode
    rows = parse_ncu_csv(csv_out)
    print(f"ncu_parsed_rows: {len(rows)}")
    analysis = analyze_rows(rows)
    analysis["ncu_mode"] = "import"
    analysis["ncu_csv"] = csv_out
    analysis["ncu_report_path"] = rep_path
    write_summary_and_details(analysis, summary_out, details_out)
    print(format_analysis_text(analysis, "NCU ANALYSIS").rstrip())
    return 0


def diff_profiles(before_csv: str, after_csv: str) -> None:
    before = parse_ncu_csv(before_csv)
    after = parse_ncu_csv(after_csv)
    before_metrics: dict[str, float] = {}
    after_metrics: dict[str, float] = {}

    for row in before:
        name = row.get("Metric Name", row.get("metric_name", ""))
        val = _safe_float(row.get("Metric Value", row.get("metric_value", row.get("Average", ""))))
        if name and val is not None:
            before_metrics[name] = val
    for row in after:
        name = row.get("Metric Name", row.get("metric_name", ""))
        val = _safe_float(row.get("Metric Value", row.get("metric_value", row.get("Average", ""))))
        if name and val is not None:
            after_metrics[name] = val

    all_keys = sorted(set(before_metrics) | set(after_metrics))
    print("\n=== NCU DIFF ===")
    print(f"{'Metric':<70} {'Before':>12} {'After':>12} {'Delta':>12} {'Change':>8}")
    print("-" * 116)
    significant = []
    for key in all_keys:
        bv = before_metrics.get(key)
        av = after_metrics.get(key)
        if bv is None or av is None:
            continue
        delta = av - bv
        pct = (delta / bv * 100) if bv != 0 else 0
        if abs(pct) > 1 or abs(delta) > 0.01:
            significant.append((key, bv, av, pct))
    significant.sort(key=lambda item: abs(item[3]), reverse=True)
    for key, bv, av, pct in significant[:30]:
        delta = av - bv
        direction = "+" if delta > 0 else ""
        short_key = key if len(key) <= 68 else key[:65] + "..."
        print(f"{short_key:<70} {bv:>12.2f} {av:>12.2f} {direction}{delta:>11.2f} {direction}{pct:>6.1f}%")
    print("=== END NCU DIFF ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="NCU profiling wrapper for cuda-evolve")
    parser.add_argument("--mode", choices=["targeted", "full", "import"], default="targeted")
    parser.add_argument("--skills", type=str, default=None)
    parser.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"))
    parser.add_argument("--kernel-file", type=str, default="kernel.py")
    parser.add_argument("--rep-file", type=str, default=None)
    parser.add_argument("--output-prefix", type=str, default="./workspace/ncu_reports/ncu_profile")
    parser.add_argument("--summary-out", type=str, default=None)
    parser.add_argument("--details-out", type=str, default=None)
    parser.add_argument("--stdout-out", type=str, default=None)
    parser.add_argument("--stderr-out", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--launch-skip", type=int, default=0)
    parser.add_argument("--launch-count", type=int, default=1)
    parser.add_argument("--ncu-args", type=str, default="")
    args = parser.parse_args()

    if args.diff:
        diff_profiles(args.diff[0], args.diff[1])
        return

    extra_args = args.ncu_args.split() if args.ncu_args else None
    output_prefix = args.output_prefix
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "import":
        if not args.rep_file:
            print("ERROR: --rep-file is required for --mode import")
            sys.exit(1)
        csv_out = f"{output_prefix}.csv"
        rc = import_ncu_report(
            rep_path=args.rep_file,
            csv_out=csv_out,
            summary_out=args.summary_out,
            details_out=args.details_out,
            stdout_out=args.stdout_out,
            stderr_out=args.stderr_out,
        )
        sys.exit(rc)

    kernel_file = os.path.abspath(args.kernel_file)
    if not os.path.exists(kernel_file):
        print(f"ERROR: kernel file not found: {kernel_file}")
        sys.exit(1)

    if args.skills:
        skills = [s.strip() for s in args.skills.split(",") if s.strip()]
    elif args.mode == "targeted":
        skills = TARGETED_SKILLS
    else:
        skills = ALL_SKILLS

    invalid = [skill for skill in skills if skill not in SKILL_METRICS]
    if invalid:
        print(f"WARNING: unknown skills ignored: {', '.join(invalid)}")
        skills = [skill for skill in skills if skill in SKILL_METRICS]
    if not skills:
        skills = TARGETED_SKILLS if args.mode == "targeted" else ALL_SKILLS

    rc = run_profile_mode(
        mode=args.mode,
        kernel_file=kernel_file,
        output_prefix=output_prefix,
        gpu=args.gpu,
        skills=skills,
        extra_args=extra_args,
        launch_skip=args.launch_skip,
        launch_count=args.launch_count,
        stdout_out=args.stdout_out,
        stderr_out=args.stderr_out,
        summary_out=args.summary_out,
        details_out=args.details_out,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
