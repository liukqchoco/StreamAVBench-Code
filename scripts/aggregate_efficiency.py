#!/usr/bin/env python3
"""Aggregate end-to-end FPS and TTFC fields from an output manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from streamav_eval.efficiency import aggregate_efficiency


def _load(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            model_id = value.get("model_id")
            if not isinstance(model_id, str) or not model_id:
                raise ValueError(f"{path}:{line_number}: invalid model_id")
            grouped[model_id].append(value)
    if not grouped:
        raise ValueError(f"{path}: no records")
    return grouped


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cases-per-track", type=int, default=160)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        grouped = _load(args.manifest)
        available = []
        unavailable = []
        for model_id, records in sorted(grouped.items()):
            has_runtime = [
                isinstance(record.get("runtime"), dict) for record in records
            ]
            if any(has_runtime) and not all(has_runtime):
                raise ValueError(
                    f"{model_id}: runtime must be present for every case or none"
                )
            if all(has_runtime):
                available.append(
                    aggregate_efficiency(
                        records,
                        expected_cases_per_track=args.expected_cases_per_track,
                    )
                )
            else:
                unavailable.append(model_id)
        result = {
            "schema_version": "streamavbench-efficiency-collection-1.0.0",
            "models": available,
            "unavailable_model_ids": unavailable,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
