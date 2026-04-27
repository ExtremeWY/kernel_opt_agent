"""Mock profiler backend for end-to-end local validation."""

from __future__ import annotations

import json
from pathlib import Path

from profilers.base import ProfilerBackend


class MockProfilerBackend(ProfilerBackend):
    backend_name = "mock_profiler"

    def collect(self, kernel_file: str, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "profile_report.txt"
        payload = {
            "kernel_file": kernel_file,
            "profile_compute_util": "62%",
            "profile_memory_util": "48%",
            "profile_l1_hit_rate": "71%",
            "profile_l2_hit_rate": "89%",
            "profile_occupancy": "75%",
            "profile_register_pressure": "moderate",
            "profile_spill_pressure": "low",
            "profile_top_stall": "memory_dependency",
            "profile_vectorization_efficiency": "80%",
            "profile_coalescing_efficiency": "93%",
        }
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return report_path

    def analyze(self, report_path: Path) -> dict[str, str]:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        return {
            "profile_compute_util": payload["profile_compute_util"],
            "profile_memory_util": payload["profile_memory_util"],
            "profile_l1_hit_rate": payload["profile_l1_hit_rate"],
            "profile_l2_hit_rate": payload["profile_l2_hit_rate"],
            "profile_occupancy": payload["profile_occupancy"],
            "profile_register_pressure": payload["profile_register_pressure"],
            "profile_spill_pressure": payload["profile_spill_pressure"],
            "profile_top_stall": payload["profile_top_stall"],
            "profile_vectorization_efficiency": payload["profile_vectorization_efficiency"],
            "profile_coalescing_efficiency": payload["profile_coalescing_efficiency"],
        }
