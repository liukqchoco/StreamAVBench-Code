"""Fast native-boundary technical stability evaluation."""

from __future__ import annotations

import hashlib
import io
import json
import math
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from streamav_eval.workers.interactive.pvc import score_boundary_diagnostics

METRIC_ID = "NBC"
SCHEMA_VERSION = "streamavbench.nbc.v1"


@dataclass(frozen=True, slots=True)
class BoundaryGeometry:
    """Fixed native generation-unit geometry for one method profile."""

    output_fps: int
    first_chunk_frames: int
    subsequent_chunk_frames: int

    def __post_init__(self) -> None:
        if self.output_fps <= 0:
            raise ValueError("output_fps must be positive")
        if self.first_chunk_frames <= 0 or self.subsequent_chunk_frames <= 0:
            raise ValueError("native chunk frame counts must be positive")

    def boundary_frames(self, duration_seconds: float) -> tuple[int, ...]:
        target_frames = round(duration_seconds * self.output_fps)
        if not math.isclose(
            target_frames / self.output_fps,
            duration_seconds,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"{duration_seconds:g}s is not integral at {self.output_fps} FPS"
            )
        boundaries: list[int] = []
        cursor = self.first_chunk_frames
        while cursor < target_frames:
            boundaries.append(cursor)
            cursor += self.subsequent_chunk_frames
        return tuple(boundaries)


@dataclass(frozen=True, slots=True)
class FrameStatistics:
    """Brightness and adjacent differences on the boundary-aligned 8-FPS grid."""

    brightness: tuple[float, ...]
    stride_differences: tuple[float, ...]
    decoded_frames: int
    frame_width: int
    frame_height: int


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def applicable_boundary_frames(
    boundary_frames: Iterable[int],
    *,
    evaluation_frame_count: int,
    output_fps: int,
    window_seconds: float = 4.0,
) -> tuple[int, ...]:
    half_window = _half_window_frames(output_fps, window_seconds)
    return tuple(
        boundary
        for boundary in boundary_frames
        if boundary - half_window >= 0
        and boundary + half_window <= evaluation_frame_count
    )


def boundary_sample_phase(
    boundary_frames: Sequence[int],
    *,
    output_fps: int,
    evaluator_fps: int = 8,
    window_seconds: float = 4.0,
) -> int:
    """Return the native-frame phase needed for boundary-aligned 8-FPS sampling."""

    if output_fps % evaluator_fps:
        raise ValueError("output_fps must be divisible by evaluator_fps")
    if not boundary_frames:
        raise ValueError("cannot derive a sampling phase without boundaries")
    stride = output_fps // evaluator_fps
    half_window = _half_window_frames(output_fps, window_seconds)
    phases = {(boundary - half_window) % stride for boundary in boundary_frames}
    if len(phases) != 1:
        raise ValueError(
            "native boundaries do not share one evaluator sampling phase: "
            f"{sorted(phases)}"
        )
    return next(iter(phases))


def decode_frame_statistics(
    video_path: str | Path,
    *,
    output_fps: int,
    evaluator_fps: int = 8,
    sample_phase_frames: int = 0,
    width: int = 160,
    ffmpeg: str = "ffmpeg",
    ffmpeg_threads: int = 2,
    timeout_seconds: float = 600.0,
) -> FrameStatistics:
    """Decode once at native FPS and retain only statistics needed by all windows."""

    if output_fps % evaluator_fps:
        raise ValueError("output_fps must be divisible by evaluator_fps")
    if width <= 0 or ffmpeg_threads <= 0 or timeout_seconds <= 0:
        raise ValueError("width, ffmpeg_threads, and timeout_seconds must be positive")
    stride = output_fps // evaluator_fps
    if not 0 <= sample_phase_frames < stride:
        raise ValueError(
            f"sample_phase_frames must be in [0, {stride}), got {sample_phase_frames}"
        )
    select = (
        f"select='not(mod(n-{sample_phase_frames}\\,{stride}))',"
        f"setpts=N/({evaluator_fps}*TB),format=gray,scale={width}:-1"
    )
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-threads",
        str(ffmpeg_threads),
        "-i",
        str(video_path),
        "-an",
        "-vf",
        select,
        "-r",
        str(evaluator_fps),
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        timeout=timeout_seconds,
    )
    if completed.returncode:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg failed for {video_path} with code {completed.returncode}: "
            f"{message[-2000:]}"
        )

    import numpy as np
    from PIL import Image

    previous: Any | None = None
    brightness: list[float] = []
    stride_differences: list[float] = []
    frame_width = frame_height = 0
    for encoded in iter_concatenated_jpegs(completed.stdout):
        with Image.open(io.BytesIO(encoded)) as image:
            frame = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        if frame.ndim != 2:
            raise RuntimeError("native-boundary decoder produced a non-grayscale frame")
        height, current_width = map(int, frame.shape)
        if not brightness:
            frame_width, frame_height = current_width, height
        elif (current_width, height) != (frame_width, frame_height):
            raise RuntimeError("native-boundary decoder changed frame dimensions")
        brightness.append(float(frame.mean()))
        if previous is None:
            stride_differences.append(float("nan"))
        else:
            stride_differences.append(float(np.mean(np.abs(frame - previous))))
        previous = frame

    if not brightness:
        raise RuntimeError(f"ffmpeg decoded no frames from {video_path}")
    return FrameStatistics(
        brightness=tuple(brightness),
        stride_differences=tuple(stride_differences),
        decoded_frames=len(brightness),
        frame_width=frame_width,
        frame_height=frame_height,
    )


def iter_concatenated_jpegs(payload: bytes) -> Iterable[bytes]:
    """Yield JPEG images from FFmpeg's image2pipe byte stream."""

    cursor = 0
    yielded = 0
    while True:
        start = payload.find(b"\xff\xd8", cursor)
        if start < 0:
            break
        end = payload.find(b"\xff\xd9", start + 2)
        if end < 0:
            raise RuntimeError("truncated JPEG in FFmpeg image2pipe output")
        yielded += 1
        cursor = end + 2
        yield payload[start:cursor]
    if not yielded and payload:
        raise RuntimeError("FFmpeg image2pipe output contained no JPEG frames")
    if payload[cursor:].strip(b"\x00\r\n\t "):
        raise RuntimeError("unexpected bytes after final JPEG frame")


def score_native_boundaries(
    statistics: FrameStatistics,
    boundary_frames: Sequence[int],
    *,
    evaluation_frame_count: int,
    output_fps: int,
    evaluator_fps: int = 8,
    sample_phase_frames: int = 0,
    window_seconds: float = 4.0,
) -> tuple[dict[str, Any], ...]:
    """Score every complete model-native boundary window."""

    if output_fps % evaluator_fps:
        raise ValueError("output_fps must be divisible by evaluator_fps")
    stride = output_fps // evaluator_fps
    if not 0 <= sample_phase_frames < stride:
        raise ValueError(
            f"sample_phase_frames must be in [0, {stride}), got {sample_phase_frames}"
        )
    expected_timeline_samples = evaluation_frame_count // stride
    if statistics.decoded_frames < expected_timeline_samples:
        raise ValueError(
            "decoded video is shorter than the frozen evaluation duration: "
            f"{statistics.decoded_frames} < {expected_timeline_samples} samples"
        )
    half_window = _half_window_frames(output_fps, window_seconds)
    expected_samples = round(window_seconds * evaluator_fps)
    results: list[dict[str, Any]] = []
    for boundary_index, boundary in enumerate(boundary_frames):
        start = boundary - half_window
        end = boundary + half_window
        if start < 0 or end > evaluation_frame_count:
            continue
        if start % stride != sample_phase_frames or end % stride != sample_phase_frames:
            raise ValueError(
                f"boundary {boundary} is incompatible with sampling phase "
                f"{sample_phase_frames}"
            )
        sampled_start = (start - sample_phase_frames) // stride
        sampled_end = (end - sample_phase_frames) // stride
        sampled_indices = tuple(range(sampled_start, sampled_end))
        if len(sampled_indices) != expected_samples or sampled_start < 0:
            raise AssertionError(
                f"boundary {boundary} produced {len(sampled_indices)} samples; "
                f"expected {expected_samples}"
            )
        brightness = [statistics.brightness[index] for index in sampled_indices]
        differences = [
            statistics.stride_differences[index] for index in sampled_indices[1:]
        ]
        if any(not math.isfinite(value) for value in differences):
            raise AssertionError(
                "complete boundary window contains undefined differences"
            )
        results.append(
            {
                "boundary_index": boundary_index,
                "boundary_frame": boundary,
                "boundary_seconds": round(boundary / output_fps, 6),
                "window_start_seconds": round(start / output_fps, 6),
                "window_end_seconds": round(end / output_fps, 6),
                **score_window_diagnostics(
                    brightness,
                    differences,
                    evaluator_fps=evaluator_fps,
                ),
            }
        )
    return tuple(results)


def score_window_diagnostics(
    brightness: Sequence[float],
    differences: Sequence[float],
    *,
    evaluator_fps: int = 8,
) -> dict[str, float | int]:
    """Compute four technical diagnostics for one sampled window."""

    if not brightness:
        raise ValueError("boundary window has no frames")
    if len(differences) != len(brightness) - 1:
        raise ValueError("boundary window differences must connect adjacent frames")
    black_ratio = sum(value < 0.05 for value in brightness) / len(brightness)
    duplicate_flags = [difference < 0.01 for difference in differences]
    duplicate_ratio = (
        sum(duplicate_flags) / len(duplicate_flags) if duplicate_flags else 0.0
    )
    freeze_seconds = _max_true_run(duplicate_flags) / evaluator_fps
    flash_count = sum(
        abs(brightness[index] - brightness[index - 1]) > 0.45
        for index in range(1, len(brightness))
    )
    scored = score_boundary_diagnostics(
        black_ratio,
        duplicate_ratio,
        freeze_seconds,
        int(flash_count),
    )
    return {
        "black_frame_ratio": round(black_ratio, 4),
        "flash_count": int(flash_count),
        "duplicate_frame_ratio": round(duplicate_ratio, 4),
        "freeze_max_sec": round(freeze_seconds, 4),
        **scored,
    }


def evaluate_case(task: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one manifest task; return a serializable success/failure record."""

    started = time.monotonic()
    common = {
        "schema_version": SCHEMA_VERSION,
        "metric_id": METRIC_ID,
        "method_id": str(task["method_id"]),
        "case_id": str(task["case_id"]),
        "job_id": str(task["job_id"]),
        "track": str(task["track"]),
        "video_path": str(task["video_path"]),
        "input_fingerprint": dict(task["input_fingerprint"]),
        "protocol_sha256": str(task["protocol_sha256"]),
    }
    try:
        video_path = Path(str(task["video_path"]))
        _validate_input_fingerprint(video_path, task["input_fingerprint"])
        output_fps = int(task["output_fps"])
        sample_phase_frames = int(task["sample_phase_frames"])
        evaluation_frames = int(task["evaluation_frame_count"])
        statistics = decode_frame_statistics(
            video_path,
            output_fps=output_fps,
            evaluator_fps=int(task["evaluator_fps"]),
            sample_phase_frames=sample_phase_frames,
            width=int(task["width"]),
            ffmpeg=str(task["ffmpeg"]),
            ffmpeg_threads=int(task["ffmpeg_threads"]),
            timeout_seconds=float(task["timeout_seconds"]),
        )
        _validate_input_fingerprint(video_path, task["input_fingerprint"])
        boundaries = score_native_boundaries(
            statistics,
            tuple(int(value) for value in task["boundary_frames"]),
            evaluation_frame_count=evaluation_frames,
            output_fps=output_fps,
            evaluator_fps=int(task["evaluator_fps"]),
            sample_phase_frames=sample_phase_frames,
            window_seconds=float(task["window_seconds"]),
        )
        if not boundaries:
            raise ValueError("case has no complete native-boundary windows")
        component_case_means = {
            name: _mean(float(row[name]) for row in boundaries)
            for name in (
                "black_frame_ratio",
                "duplicate_frame_ratio",
                "freeze_max_sec",
                "flash_count",
            )
        }
        return {
            **common,
            "status": "computed",
            "duration_seconds": float(task["duration_seconds"]),
            "output_fps": output_fps,
            "sample_phase_frames": sample_phase_frames,
            "evaluation_frame_count": evaluation_frames,
            "decoded_frame_count": statistics.decoded_frames,
            "decoded_resolution": [
                statistics.frame_width,
                statistics.frame_height,
            ],
            "native_boundary_count": len(task["boundary_frames"]),
            "applicable_boundary_count": len(boundaries),
            "boundary_coverage": len(boundaries) / len(task["boundary_frames"]),
            "score": _mean(
                float(row["technical_boundary_score"]) for row in boundaries
            ),
            "algorithm_score_1_5": _mean(
                float(row["algorithm_score"]) for row in boundaries
            ),
            "component_case_means": component_case_means,
            "boundaries": list(boundaries),
            "elapsed_seconds": time.monotonic() - started,
        }
    except Exception as exc:  # Keep long formal runs resumable case by case.
        return {
            **common,
            "status": "failed",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "elapsed_seconds": time.monotonic() - started,
        }


def aggregate_case_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_cases: int,
) -> dict[str, Any]:
    """Macro-average boundaries within cases, then cases within the method."""

    latest: dict[str, Mapping[str, Any]] = {}
    for record in records:
        case_id = str(record["case_id"])
        latest[case_id] = record
    computed = [
        record for record in latest.values() if record.get("status") == "computed"
    ]
    failed = [
        record for record in latest.values() if record.get("status") != "computed"
    ]
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "metric_id": METRIC_ID,
        "aggregation": (
            "arithmetic mean over all applicable native boundaries within each "
            "case, followed by an equal-weight arithmetic mean over cases"
        ),
        "expected_cases": expected_cases,
        "observed_cases": len(latest),
        "computed_cases": len(computed),
        "failed_cases": len(failed),
        "complete": len(computed) == expected_cases and not failed,
        "failed_case_ids": sorted(str(record["case_id"]) for record in failed),
    }
    if not computed:
        summary["score"] = None
        return summary
    summary.update(
        {
            "method_id": str(computed[0]["method_id"]),
            "score": _mean(float(record["score"]) for record in computed),
            "algorithm_score_1_5": _mean(
                float(record["algorithm_score_1_5"]) for record in computed
            ),
            "native_boundary_count": sum(
                int(record["native_boundary_count"]) for record in computed
            ),
            "applicable_boundary_count": sum(
                int(record["applicable_boundary_count"]) for record in computed
            ),
            "case_score_min": min(float(record["score"]) for record in computed),
            "case_score_max": max(float(record["score"]) for record in computed),
            "component_method_means": {
                name: _mean(
                    float(record["component_case_means"][name]) for record in computed
                )
                for name in (
                    "black_frame_ratio",
                    "duplicate_frame_ratio",
                    "freeze_max_sec",
                    "flash_count",
                )
            },
            "elapsed_seconds_sum": sum(
                float(record["elapsed_seconds"]) for record in computed
            ),
        }
    )
    return summary


def _half_window_frames(output_fps: int, window_seconds: float) -> int:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    half = output_fps * window_seconds / 2.0
    rounded = round(half)
    if not math.isclose(half, rounded, abs_tol=1e-9):
        raise ValueError("half-window must contain an integral number of native frames")
    return rounded


def _validate_input_fingerprint(
    path: Path,
    fingerprint: Mapping[str, Any],
) -> None:
    stat = path.stat()
    expected = (int(fingerprint["size"]), int(fingerprint["mtime_ns"]))
    observed = (stat.st_size, stat.st_mtime_ns)
    if observed != expected:
        raise RuntimeError(
            f"input changed after manifest creation: expected {expected}, got {observed}"
        )


def _max_true_run(values: Sequence[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot average an empty sequence")
    return sum(materialized) / len(materialized)
