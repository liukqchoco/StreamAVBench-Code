"""End-to-end FPS and time-to-first-chunk aggregation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any

TRACKS = ("progressive", "interactive")


def case_efficiency(record: Mapping[str, Any]) -> dict[str, float | str]:
    """Validate one manifest record and compute its end-to-end FPS."""

    model_id = _text(record.get("model_id"), "model_id")
    case_id = _text(record.get("case_id"), "case_id")
    inferred_track = {
        "P": "progressive",
        "I": "interactive",
    }.get(case_id.partition("-")[0])
    if inferred_track is None:
        raise ValueError(f"{case_id}: cannot infer track from case_id")
    track_value = record.get("track")
    track = inferred_track if track_value is None else _text(track_value, "track")
    if track not in TRACKS or track != inferred_track:
        raise ValueError(f"{case_id}: track must match its case_id prefix")
    runtime = record.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError(f"{case_id}: runtime must be an object")
    frames = _positive(runtime.get("generated_video_frames"), "generated_video_frames")
    elapsed = _positive(
        runtime.get("generation_time_seconds"), "generation_time_seconds"
    )
    ttfc = _nonnegative(
        runtime.get("time_to_first_chunk_seconds"),
        "time_to_first_chunk_seconds",
    )
    if ttfc > elapsed:
        raise ValueError(
            f"{case_id}: time_to_first_chunk_seconds cannot exceed "
            "generation_time_seconds"
        )
    return {
        "model_id": model_id,
        "case_id": case_id,
        "track": track,
        "fps": frames / elapsed,
        "ttfc_seconds": ttfc,
    }


def aggregate_efficiency(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_cases_per_track: int = 160,
) -> dict[str, Any]:
    """Average case-level FPS and TTFC within each track and over both tracks."""

    if expected_cases_per_track <= 0:
        raise ValueError("expected_cases_per_track must be positive")
    cases = [case_efficiency(record) for record in records]
    if not cases:
        raise ValueError("efficiency aggregation requires at least one record")
    model_ids = {str(row["model_id"]) for row in cases}
    if len(model_ids) != 1:
        raise ValueError("efficiency records must belong to exactly one model")
    seen: set[tuple[str, str]] = set()
    by_track: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in cases:
        key = str(row["track"]), str(row["case_id"])
        if key in seen:
            raise ValueError(f"duplicate efficiency record: {key}")
        seen.add(key)
        by_track[key[0]].append(row)
    missing = sorted(set(TRACKS) - set(by_track))
    if missing:
        raise ValueError(f"efficiency records are missing tracks: {missing}")
    incomplete = {
        track: len(by_track[track])
        for track in TRACKS
        if len(by_track[track]) != expected_cases_per_track
    }
    if incomplete:
        raise ValueError(
            "efficiency case counts must equal "
            f"{expected_cases_per_track} per track; got {incomplete}"
        )

    track_results = {
        track: {
            "case_count": len(rows),
            "fps": _mean(float(row["fps"]) for row in rows),
            "ttfc_seconds": _mean(float(row["ttfc_seconds"]) for row in rows),
        }
        for track, rows in sorted(by_track.items())
    }
    return {
        "schema_version": "streamavbench-efficiency-1.0.0",
        "model_id": model_ids.pop(),
        "fps": _mean(float(row["fps"]) for row in cases),
        "ttfc_seconds": _mean(float(row["ttfc_seconds"]) for row in cases),
        "case_count": len(cases),
        "tracks": track_results,
        "aggregation": "case-level arithmetic mean within each track and over both tracks",
    }


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise ValueError("mean requires at least one value")
    return math.fsum(items) / len(items)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _positive(value: Any, field: str) -> float:
    number = _number(value, field)
    if number <= 0:
        raise ValueError(f"{field} must be positive")
    return number


def _nonnegative(value: Any, field: str) -> float:
    number = _number(value, field)
    if number < 0:
        raise ValueError(f"{field} must be non-negative")
    return number


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number
