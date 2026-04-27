"""Lazy registry for benchmark configurations."""

from __future__ import annotations

import importlib
import pathlib
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from kernel_configs._utils import DTYPE_MAP

_PKG_DIR = pathlib.Path(__file__).resolve().parent


def _load_toml(path: pathlib.Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _parse_sizes(raw: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    return [(entry["label"], entry["params"]) for entry in raw]


def _parse_dtypes(raw: list[str]) -> list[Any]:
    out = []
    for name in raw:
        if name not in DTYPE_MAP:
            raise ValueError(f"Unknown dtype string '{name}'. Valid: {list(DTYPE_MAP)}")
        out.append(DTYPE_MAP[name])
    return out


def _parse_tolerances(raw: dict[str, Any]) -> dict[Any, dict[str, float]]:
    out = {}
    for name, tol in raw.items():
        dt = DTYPE_MAP.get(name)
        if dt is None:
            raise ValueError(f"Unknown dtype '{name}' in tolerances section")
        out[dt] = {"atol": float(tol["atol"]), "rtol": float(tol["rtol"])}
    return out


def _build_config(toml_path: pathlib.Path) -> dict[str, Any]:
    data = _load_toml(toml_path)
    stem = toml_path.stem
    module = importlib.import_module(f"kernel_configs.{stem}")

    cfg: dict[str, Any] = {}
    meta = data.get("meta", {})
    if meta.get("multi_output", False):
        cfg["multi_output"] = True

    cfg["meta"] = meta
    cfg["test_sizes"] = _parse_sizes(data["test_sizes"])
    dtype_entries = data.get("test_dtypes", meta.get("test_dtypes"))
    if dtype_entries is None:
        raise KeyError("test_dtypes")
    cfg["test_dtypes"] = _parse_dtypes(dtype_entries)
    cfg["tolerances"] = _parse_tolerances(data["tolerances"])
    edge_raw = data.get("edge_sizes", [])
    cfg["edge_sizes"] = _parse_sizes(edge_raw) if edge_raw else []
    cfg["input_generator"] = module.input_generator
    cfg["reference_fn"] = module.reference_fn
    cfg["flops_fn"] = module.flops_fn
    cfg["bytes_fn"] = module.bytes_fn
    if hasattr(module, "numerical_stability_cases"):
        cfg["numerical_stability_cases"] = module.numerical_stability_cases
    if hasattr(module, "optimization_traits"):
        cfg["optimization_traits"] = module.optimization_traits
    if hasattr(module, "KERNEL_OPT_CHARACTERISTICS"):
        cfg["kernel_opt_characteristics"] = getattr(module, "KERNEL_OPT_CHARACTERISTICS")
    return cfg


def get_kernel_config(kernel_type: str) -> dict[str, Any]:
    toml_path = _PKG_DIR / f"{kernel_type}.toml"
    py_path = _PKG_DIR / f"{kernel_type}.py"
    if not toml_path.exists():
        raise KeyError(kernel_type)
    if not py_path.exists():
        raise FileNotFoundError(f"Kernel config '{toml_path.name}' has no companion '{py_path.name}'")
    return _build_config(toml_path)


def list_kernel_types() -> list[str]:
    return sorted(path.stem for path in _PKG_DIR.glob("*.toml"))


def _discover_best_effort() -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for kernel_type in list_kernel_types():
        try:
            configs[kernel_type] = get_kernel_config(kernel_type)
        except Exception:
            continue
    return configs


KERNEL_CONFIGS = _discover_best_effort()
