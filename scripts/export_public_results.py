#!/usr/bin/env python3
"""Export complete evaluation artifacts to 32-dimension JSON and CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from streamav_eval.leaderboard import build_public_results


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--efficiency", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--nbc",
        type=Path,
        action="append",
        default=[],
        help="Combined NBC result for one model; repeat as needed.",
    )
    parser.add_argument(
        "--nbc-unavailable-model",
        action="append",
        default=[],
        metavar="MODEL_ID",
        help=(
            "Model whose interface does not expose native boundaries; repeat as needed."
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        efficiency_artifact = _object(args.efficiency)
        if (
            efficiency_artifact.get("schema_version")
            != "streamavbench-efficiency-collection-1.0.0"
        ):
            raise ValueError("efficiency artifact has an unsupported schema")
        efficiency = efficiency_artifact.get("models")
        if not isinstance(efficiency, list):
            raise ValueError("efficiency artifact must contain a models array")
        unavailable_efficiency = efficiency_artifact.get("unavailable_model_ids")
        if not isinstance(unavailable_efficiency, list) or not all(
            isinstance(value, str) for value in unavailable_efficiency
        ):
            raise ValueError(
                "efficiency artifact must contain an unavailable_model_ids array"
            )
        result = build_public_results(
            _object(args.report),
            efficiency_models=efficiency,
            efficiency_unavailable_model_ids=unavailable_efficiency,
            nbc_models=[_object(path) for path in args.nbc],
            nbc_unavailable_model_ids=args.nbc_unavailable_model,
            dataset_root=args.dataset_root,
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        fieldnames = ["model_id", *result["dimensions"]]
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(result["models"])
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "models": len(result["models"]),
                "dimensions": len(result["dimensions"]),
                "json": str(args.output_json),
                "csv": str(args.output_csv),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
