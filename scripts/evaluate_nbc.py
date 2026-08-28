#!/usr/bin/env python3
"""Evaluate Native Boundary Continuity (NBC) from a public output manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from streamav_eval.native_boundary_continuity import (
    BoundaryGeometry,
    aggregate_case_records,
    applicable_boundary_frames,
    boundary_sample_phase,
    canonical_json_sha256,
    evaluate_case,
)
from streamav_eval.registry import CanonicalRegistry

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "nbc_protocol.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--track",
        choices=("progressive", "interactive"),
        required=True,
    )
    parser.add_argument("--protocol-config", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--profile",
        help="Profile ID from the protocol. Inferred for published benchmark models.",
    )
    parser.add_argument("--output-fps", type=int)
    parser.add_argument("--first-chunk-frames", type=int)
    parser.add_argument("--subsequent-chunk-frames", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 4) // 2)),
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffmpeg-threads", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--expected-cases", type=int, default=160)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield line_number, value


def _geometry(
    args: argparse.Namespace, protocol: Mapping[str, Any]
) -> BoundaryGeometry:
    profile_id = args.profile
    if profile_id is None:
        method = _method_config(protocol, args.model_id)
        if isinstance(method, Mapping):
            profile_id = method.get("profile")
    profiles = protocol.get("profiles")
    profile = profiles.get(profile_id) if isinstance(profiles, Mapping) else None
    if isinstance(profile, Mapping):
        return BoundaryGeometry(
            output_fps=int(profile["output_fps"]),
            first_chunk_frames=int(profile["first_chunk_frames"]),
            subsequent_chunk_frames=int(profile["subsequent_chunk_frames"]),
        )

    supplied = (
        args.output_fps,
        args.first_chunk_frames,
        args.subsequent_chunk_frames,
    )
    if any(value is None for value in supplied):
        raise ValueError(
            "unknown model profile; provide --profile or all three geometry flags"
        )
    return BoundaryGeometry(
        output_fps=int(args.output_fps),
        first_chunk_frames=int(args.first_chunk_frames),
        subsequent_chunk_frames=int(args.subsequent_chunk_frames),
    )


def _method_config(
    protocol: Mapping[str, Any], model_id: str
) -> Mapping[str, Any] | None:
    methods = protocol.get("methods")
    if not isinstance(methods, Mapping):
        return None
    direct = methods.get(model_id)
    if isinstance(direct, Mapping):
        return direct
    normalized = re.sub(r"[^a-z0-9]+", "", model_id.casefold())
    matches = [
        value
        for key, value in methods.items()
        if isinstance(value, Mapping)
        and normalized
        in {
            re.sub(r"[^a-z0-9]+", "", str(key).casefold()),
            re.sub(
                r"[^a-z0-9]+",
                "",
                str(value.get("display_name", "")).casefold(),
            ),
        }
    ]
    if len(matches) > 1:
        raise ValueError(f"ambiguous model profile for {model_id!r}")
    return matches[0] if matches else None


def build_tasks(
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    geometry: BoundaryGeometry,
) -> list[dict[str, Any]]:
    detector = protocol.get("detector")
    if not isinstance(detector, Mapping):
        raise ValueError("protocol detector must be an object")
    evaluator_fps = int(detector["evaluator_fps"])
    window_seconds = float(detector["window_seconds"])
    if geometry.output_fps % evaluator_fps:
        raise ValueError("output FPS must be divisible by evaluator FPS")

    protocol_sha256 = canonical_json_sha256(protocol)
    registry = CanonicalRegistry.load(dataset_root=args.dataset_root)
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, row in _read_jsonl(args.manifest):
        if row.get("model_id") != args.model_id:
            continue
        case_id = str(row.get("case_id", ""))
        if not case_id:
            raise ValueError(f"{args.manifest}:{line_number}: missing case_id")
        if case_id not in registry:
            raise ValueError(
                f"{args.manifest}:{line_number}: unknown case_id {case_id!r}"
            )
        benchmark_case = registry.get(case_id)
        row_track = row.get("track", benchmark_case.track.value)
        if row_track != benchmark_case.track.value:
            raise ValueError(
                f"{args.manifest}:{line_number}: {case_id} track must be "
                f"{benchmark_case.track.value!r}"
            )
        if benchmark_case.track.value != args.track:
            continue
        if case_id in seen:
            raise ValueError(
                f"{args.manifest}:{line_number}: duplicate case_id {case_id!r}"
            )
        seen.add(case_id)
        video_path = Path(str(row.get("video_path", ""))).expanduser()
        if not video_path.is_absolute():
            video_path = (args.manifest.parent / video_path).resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"{case_id}: video not found: {video_path}")
        duration = float(row.get("duration_seconds", 180.0))
        if abs(duration - benchmark_case.duration_seconds) > 1e-6:
            raise ValueError(
                f"{args.manifest}:{line_number}: {case_id} duration must be "
                f"{benchmark_case.duration_seconds:g} seconds"
            )
        boundaries = geometry.boundary_frames(duration)
        evaluation_frames = round(duration * geometry.output_fps)
        phase = boundary_sample_phase(
            boundaries,
            output_fps=geometry.output_fps,
            evaluator_fps=evaluator_fps,
            window_seconds=window_seconds,
        )
        applicable = applicable_boundary_frames(
            boundaries,
            evaluation_frame_count=evaluation_frames,
            output_fps=geometry.output_fps,
            window_seconds=window_seconds,
        )
        if not applicable:
            raise ValueError(f"{case_id}: no complete native-boundary windows")
        stat = video_path.stat()
        tasks.append(
            {
                "method_id": args.model_id,
                "case_id": case_id,
                "job_id": str(row.get("job_id") or f"{args.model_id}__{case_id}"),
                "track": args.track,
                "video_path": str(video_path),
                "input_fingerprint": {
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                },
                "protocol_sha256": protocol_sha256,
                "duration_seconds": duration,
                "output_fps": geometry.output_fps,
                "sample_phase_frames": phase,
                "evaluation_frame_count": evaluation_frames,
                "boundary_frames": list(boundaries),
                "evaluator_fps": evaluator_fps,
                "width": int(detector["width"]),
                "window_seconds": window_seconds,
                "ffmpeg": args.ffmpeg,
                "ffmpeg_threads": args.ffmpeg_threads,
                "timeout_seconds": args.timeout_seconds,
            }
        )
    if not tasks:
        raise ValueError("manifest contains no rows for the selected model and track")
    if len(tasks) != args.expected_cases:
        raise ValueError(
            f"expected {args.expected_cases} {args.track} cases for "
            f"{args.model_id}, found {len(tasks)}"
        )
    return sorted(tasks, key=lambda value: str(value["case_id"]))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.workers <= 0
        or args.ffmpeg_threads <= 0
        or args.timeout_seconds <= 0
        or args.expected_cases <= 0
    ):
        raise ValueError("worker counts, expected cases, and timeout must be positive")
    protocol = _read_json_object(args.protocol_config)
    if protocol.get("metric_id") != "NBC":
        raise ValueError("protocol metric_id must be NBC")
    geometry = _geometry(args, protocol)
    tasks = build_tasks(args, protocol, geometry)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "metric_id": "NBC",
        "model_id": args.model_id,
        "track": args.track,
        "manifest": str(args.manifest.resolve()),
        "protocol_sha256": canonical_json_sha256(protocol),
        "case_count": len(tasks),
        "expected_cases": args.expected_cases,
        "geometry": {
            "output_fps": geometry.output_fps,
            "first_chunk_frames": geometry.first_chunk_frames,
            "subsequent_chunk_frames": geometry.subsequent_chunk_frames,
        },
    }
    _write_json(args.output_dir / "audit.json", audit)
    if args.audit_only:
        return audit

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(evaluate_case, tasks))
    records_path = args.output_dir / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    summary = aggregate_case_records(records, expected_cases=args.expected_cases)
    summary.update(
        {
            "track": args.track,
            "model_id": args.model_id,
            "protocol_sha256": canonical_json_sha256(protocol),
        }
    )
    _write_json(args.output_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    try:
        summary = run(parse_args(argv))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("complete", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
