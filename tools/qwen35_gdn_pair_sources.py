#!/usr/bin/env python3
"""Same-process A/B benchmark for two qwen35moe_gdn_prefill CUDA sources."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernel_configs import qwen35moe_gdn_prefill as cfg  # noqa: E402
from tools.bench import _do_bench_pair  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, py_path: Path):
    spec = importlib.util.spec_from_file_location(name, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {py_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare_module(run_dir: Path, label: str, source: Path):
    mod_dir = run_dir / label
    mod_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "kernel.py", mod_dir / "kernel.py")
    shutil.copyfile(source, mod_dir / "kernel.cu")
    return load_module(f"qwen35_pair_{label}_{sha256_file(source)[:12]}", mod_dir / "kernel.py")


def load_sizes(size_filter: str) -> list[tuple[str, dict[str, int]]]:
    data = tomllib.loads((ROOT / "kernel_configs" / "qwen35moe_gdn_prefill.toml").read_text())
    out: list[tuple[str, dict[str, int]]] = []
    wanted = None if size_filter == "all" else {x.strip() for x in size_filter.split(",") if x.strip()}
    for item in data["test_sizes"]:
        label = str(item["label"])
        if wanted is not None and label not in wanted:
            continue
        out.append((label, {k: int(v) for k, v in item["params"].items()}))
    return out


def max_abs_pair(a_out: list[torch.Tensor], b_out: list[torch.Tensor]) -> float:
    vals = []
    for a, b in zip(a_out, b_out, strict=True):
        vals.append(float((a - b).abs().max().item()))
    return max(vals) if vals else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True, help="Current/baseline CUDA source")
    parser.add_argument("--b", type=Path, required=True, help="Candidate CUDA source")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--sizes", type=str, default="all")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--max-trials", type=int, default=31)
    parser.add_argument("--target-ci-pct", type=float, default=1.0)
    parser.add_argument("--stability-threshold-pct", type=float, default=1.5)
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    a_src = args.a if args.a.is_absolute() else ROOT / args.a
    b_src = args.b if args.b.is_absolute() else ROOT / args.b
    mod_a = prepare_module(run_dir, "a", a_src)
    mod_b = prepare_module(run_dir, "b", b_src)

    records: list[dict[str, Any]] = []
    ratios: list[float] = []
    for label, size in load_sizes(args.sizes):
        inputs = cfg.input_generator(size, torch.float32, "cuda")
        torch.cuda.synchronize()
        a_out = mod_a.kernel_fn(**inputs)
        b_out = mod_b.kernel_fn(**inputs)
        torch.cuda.synchronize()
        max_abs = max_abs_pair(a_out, b_out)

        def run_a():
            return mod_a.kernel_fn(**inputs)

        def run_b():
            return mod_b.kernel_fn(**inputs)

        b_stats, a_stats, pair_stats = _do_bench_pair(
            run_b,
            run_a,
            warmup=args.warmup,
            rep=args.rep,
            trials=args.trials,
            stability_threshold_pct=args.stability_threshold_pct,
            max_trials=args.max_trials,
            target_ci_pct=args.target_ci_pct,
            adaptive_trials=True,
        )
        ratio = float(pair_stats["median"])
        ratios.append(ratio)
        rec = {
            "label": label,
            "size": size,
            "max_abs_a_b": max_abs,
            "a_median_ms": float(a_stats["median_ms"]),
            "b_median_ms": float(b_stats["median_ms"]),
            "a_over_b_median": ratio,
            "a_stats": a_stats,
            "b_stats": b_stats,
            "paired": pair_stats,
        }
        records.append(rec)
        print(
            f"{label}: a={rec['a_median_ms']:.6f} ms "
            f"b={rec['b_median_ms']:.6f} ms "
            f"a/b={ratio:.5f} max_abs={max_abs:.3e}",
            flush=True,
        )

    geomean = 1.0
    for ratio in ratios:
        geomean *= ratio
    geomean = geomean ** (1.0 / len(ratios)) if ratios else 0.0
    payload = {
        "a_source": str(a_src.relative_to(ROOT)),
        "b_source": str(b_src.relative_to(ROOT)),
        "a_sha256": sha256_file(a_src),
        "b_sha256": sha256_file(b_src),
        "warmup": args.warmup,
        "rep": args.rep,
        "trials": args.trials,
        "max_trials": args.max_trials,
        "target_ci_pct": args.target_ci_pct,
        "stability_threshold_pct": args.stability_threshold_pct,
        "geomean_a_over_b": geomean,
        "min_a_over_b": min(ratios) if ratios else 0.0,
        "max_a_over_b": max(ratios) if ratios else 0.0,
        "records": records,
    }
    out = args.json_out if args.json_out.is_absolute() else ROOT / args.json_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
