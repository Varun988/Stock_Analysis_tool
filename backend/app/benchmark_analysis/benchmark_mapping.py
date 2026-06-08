from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BENCHMARK_MAPPING_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "benchmark_mapping.json"
)


def load_benchmark_mapping() -> dict[str, dict[str, Any]]:
    if not BENCHMARK_MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark mapping file not found: {BENCHMARK_MAPPING_FILE}"
        )

    with BENCHMARK_MAPPING_FILE.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError("benchmark_mapping.json must contain an object.")

    normalized_mapping: dict[str, dict[str, Any]] = {}

    for benchmark_key, config in payload.items():
        if not isinstance(config, dict):
            continue

        normalized_mapping[str(benchmark_key)] = dict(config)

    return normalized_mapping


def get_benchmark_config(benchmark_key: str | None) -> dict[str, Any] | None:
    if not benchmark_key:
        return None

    mapping = load_benchmark_mapping()
    return mapping.get(str(benchmark_key))