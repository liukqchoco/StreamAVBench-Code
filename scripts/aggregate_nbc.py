#!/usr/bin/env python3
"""Combine complete Progressive and Interactive NBC summaries."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _read_summary(
    path: Path, expected_track: str, expected_cases: int
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    expected = {
        "metric_id": "NBC",
        "track": expected_track,
        "complete": True,
        "failed_cases": 0,
        "expected_cases": expected_cases,
        "observed_cases": expected_cases,
        "computed_cases": expected_cases,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"{path}: {key} must be {item!r}")
    if value.get("failed_case_ids") != []:
        raise ValueError(f"{path}: failed_case_ids must be empty")
    for field in ("score", "algorithm_score_1_5", "computed_cases"):
        item = value.get(field)
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise ValueError(f"{path}: invalid {field}={item!r}")
    return value


def _weighted_mean(values: Mapping[str, Mapping[str, Any]], field: str) -> float:
    counts = [int(value["computed_cases"]) for value in values.values()]
    if any(count <= 0 for count in counts):
        raise ValueError("NBC summaries must contain at least one computed case")
    return math.fsum(
        float(value[field]) * count
        for value, count in zip(values.values(), counts, strict=True)
    ) / sum(counts)


def combine(
    progressive_path: Path,
    interactive_path: Path,
    *,
    expected_cases_per_track: int = 160,
) -> dict[str, Any]:
    if expected_cases_per_track <= 0:
        raise ValueError("expected_cases_per_track must be positive")
    tracks = {
        "progressive": _read_summary(
            progressive_path, "progressive", expected_cases_per_track
        ),
        "interactive": _read_summary(
            interactive_path, "interactive", expected_cases_per_track
        ),
    }
    model_ids = {str(value.get("model_id", "")) for value in tracks.values()}
    protocol_hashes = {
        str(value.get("protocol_sha256", "")) for value in tracks.values()
    }
    if len(model_ids) != 1 or "" in model_ids:
        raise ValueError("track summaries must have the same non-empty model_id")
    if len(protocol_hashes) != 1 or "" in protocol_hashes:
        raise ValueError("track summaries must use the same scoring protocol")
    return {
        "schema_version": "streamavbench-nbc-combined-1.0.0",
        "metric_id": "NBC",
        "metric_name": "Native Boundary Continuity",
        "model_id": model_ids.pop(),
        "score": _weighted_mean(tracks, "score"),
        "algorithm_score_1_5": _weighted_mean(tracks, "algorithm_score_1_5"),
        "computed_cases": sum(
            int(value["computed_cases"]) for value in tracks.values()
        ),
        "protocol_sha256": protocol_hashes.pop(),
        "aggregation": "equal-weight arithmetic mean over all cases in both tracks",
        "tracks": {
            name: {
                "score": float(value["score"]),
                "computed_cases": int(value["computed_cases"]),
                "summary": str(path.resolve()),
            }
            for (name, value), path in zip(
                tracks.items(), (progressive_path, interactive_path), strict=True
            )
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progressive-summary", type=Path, required=True)
    parser.add_argument("--interactive-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cases-per-track", type=int, default=160)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = combine(
            args.progressive_summary,
            args.interactive_summary,
            expected_cases_per_track=args.expected_cases_per_track,
        )
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
