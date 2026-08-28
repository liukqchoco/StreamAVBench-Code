"""Command-line entry point for validation, planning, execution, and reporting."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import load_config
from .contracts import ContractError
from .manifest import load_manifest
from .plan import job_to_dict, plan_manifest
from .registry import CanonicalRegistry
from .validation import validate_media


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="streamav-eval",
        description=(
            "Offline StreamAV-Bench evaluator. Config files are parsed as JSON; "
            "a .yaml suffix does not enable YAML syntax."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    registry_parser = subparsers.add_parser(
        "registry", help="inspect the external canonical benchmark registry"
    )
    registry_commands = registry_parser.add_subparsers(
        dest="registry_command", required=True
    )
    summary_parser = registry_commands.add_parser(
        "summary", help="print registry counts as JSON"
    )
    summary_parser.add_argument(
        "--dataset-root",
        type=Path,
        help="path to the separately downloaded benchmark dataset",
    )
    for command, help_text in (
        ("validate", "validate manifest, registry, and media contracts"),
        ("plan", "emit canonical metric jobs as JSONL"),
        ("run", "execute jobs through streamav_eval.runner when installed"),
        ("aggregate", "aggregate results through streamav_eval.report when installed"),
    ):
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument(
            "config",
            type=Path,
            help="configuration containing JSON, even when named *.yaml",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "registry":
            print(
                json.dumps(
                    _registry_summary(dataset_root=args.dataset_root),
                    sort_keys=True,
                )
            )
        elif args.command == "validate":
            summary = _validate(args.config)
            print(json.dumps(summary, sort_keys=True))
        elif args.command == "plan":
            for job in _plan(args.config):
                print(json.dumps(job_to_dict(job), sort_keys=True))
        elif args.command == "run":
            return _delegate("runner", args.config)
        elif args.command == "aggregate":
            return _delegate("report", args.config)
        else:  # argparse enforces this; retain a fail-closed programmatic guard.
            raise ContractError(f"unknown command {args.command!r}")
    except (ContractError, OSError) as exc:
        print(f"streamav-eval: {exc}", file=sys.stderr)
        return 2
    return 0


def _registry_summary(dataset_root: str | Path | None = None) -> dict[str, Any]:
    registry = CanonicalRegistry.load(dataset_root=dataset_root)
    cases = registry.values()
    tracks = Counter(case.track.value for case in cases)
    durations = Counter(f"{case.duration_seconds:g}" for case in cases)
    return {
        "cases": len(cases),
        "tracks": dict(sorted(tracks.items())),
        "prompts": sum(len(case.prompts) for case in cases),
        "durations_seconds": {
            duration: count
            for duration, count in sorted(
                durations.items(), key=lambda item: float(item[0])
            )
        },
    }


def _load_inputs(config_path: Path) -> tuple[Any, CanonicalRegistry, list[Any]]:
    config = load_config(config_path)
    registry = CanonicalRegistry.load(dataset_root=config.dataset_root)
    records = load_manifest(config.manifest_path, fields=config.manifest_fields)
    for record in records:
        case = registry.get(record.case_id)
        declared_track = record.extra.get("track")
        if declared_track is not None and declared_track != case.track.value:
            raise ContractError(
                f"{record.case_id}: manifest track must be {case.track.value!r}"
            )
    return config, registry, records


def _validate(config_path: Path) -> dict[str, Any]:
    config, registry, records = _load_inputs(config_path)
    tracks: dict[str, int] = {"progressive": 0, "interactive": 0}
    for record in records:
        case = registry.get(record.case_id)
        validation = validate_media(
            {
                "video_path": str(record.video_path),
                "audio_path": (
                    str(record.audio_path) if record.audio_path is not None else None
                ),
                "duration_seconds": case.duration_seconds,
                "expected_modalities": {"video": True, "audio": True},
            },
            tolerance_seconds=config.duration_tolerance_seconds,
            ffprobe=config.ffprobe,
        )
        if not validation.valid:
            message = (
                validation.error.get("message", "media validation failed")
                if validation.error is not None
                else "media validation failed"
            )
            raise ContractError(f"{record.case_id}: {message}")
        tracks[case.track.value] += 1
    return {
        "status": "valid",
        "run_id": config.run_id,
        "records": len(records),
        "tracks": tracks,
    }


def _plan(config_path: Path) -> tuple[Any, ...]:
    config, registry, records = _load_inputs(config_path)
    return plan_manifest(
        run_id=config.run_id,
        records=records,
        registry=registry,
    )


def _delegate(module_name: str, config_path: Path) -> int:
    qualified = f"streamav_eval.{module_name}"
    try:
        module = importlib.import_module(qualified)
    except ModuleNotFoundError as exc:
        if exc.name != qualified:
            raise
        raise ContractError(
            f"{module_name} interface is unavailable: module {qualified!r} "
            "is not installed"
        ) from exc
    entrypoint = getattr(module, "main", None)
    if not callable(entrypoint):
        raise ContractError(
            f"{module_name} interface is unavailable: {qualified}.main is missing"
        )
    try:
        result = entrypoint([str(config_path)])
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    if result is None:
        return 0
    if isinstance(result, bool) or not isinstance(result, int):
        raise ContractError(f"{qualified}.main must return an integer or None")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
