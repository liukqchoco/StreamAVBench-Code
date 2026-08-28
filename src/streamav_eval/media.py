"""Offline ffprobe adapter and media validation."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from .contracts import ContractError, MediaMetadata

FFPROBE_ENTRIES = (
    "format=duration:"
    "stream=index,codec_type,duration,duration_ts,time_base,start_time,start_pts"
)


def probe_media(
    path: str | Path,
    *,
    ffprobe: str = "ffprobe",
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    require_audio: bool = True,
) -> MediaMetadata:
    """Probe one local file; never invokes a shell or network service."""
    media_path = Path(path)
    command = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_entries",
        FFPROBE_ENTRIES,
        str(media_path),
    ]
    try:
        completed = runner(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"ffprobe failed for {media_path}: {exc}") from exc
    try:
        payload = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ContractError(f"ffprobe returned invalid JSON for {media_path}") from exc
    return parse_ffprobe(payload, path=media_path, require_audio=require_audio)


ffprobe_media = probe_media


def parse_ffprobe(
    payload: Mapping[str, Any],
    *,
    path: str | Path = "<memory>",
    require_audio: bool = True,
) -> MediaMetadata:
    """Convert ffprobe JSON into the strict media metadata contract."""
    format_data = payload.get("format")
    streams = payload.get("streams")
    if not isinstance(format_data, Mapping) or not isinstance(streams, Sequence):
        raise ContractError("ffprobe payload requires format and streams")
    video = _first_stream(streams, "video")
    audio = _optional_stream(streams, "audio")
    if audio is None and require_audio:
        raise ContractError("ffprobe payload has no audio stream")
    return MediaMetadata(
        path=Path(path),
        container_duration=_number(format_data.get("duration"), "format.duration"),
        video_duration=_stream_duration(video, "video"),
        audio_duration=(
            _stream_duration(audio, "audio") if audio is not None else None
        ),
        video_start_pts=_start_pts_seconds(video, "video"),
        audio_start_pts=(
            _start_pts_seconds(audio, "audio") if audio is not None else None
        ),
    )


def validate_duration(
    metadata: MediaMetadata,
    *,
    expected_seconds: float = 180.0,
    tolerance_seconds: float = 0.5,
) -> None:
    """Require container, video, and audio durations within an inclusive tolerance."""
    if tolerance_seconds < 0:
        raise ContractError("tolerance_seconds must be non-negative")
    durations = {
        "container": metadata.container_duration,
        "video": metadata.video_duration,
        "audio": metadata.audio_duration,
    }
    failures = {
        name: value
        for name, value in durations.items()
        if value is not None and abs(value - expected_seconds) > tolerance_seconds
    }
    if failures:
        details = ", ".join(f"{name}={value:.6f}s" for name, value in failures.items())
        raise ContractError(
            f"media duration outside {expected_seconds:.3f}s ± "
            f"{tolerance_seconds:.3f}s: {details}"
        )


def _first_stream(streams: Sequence[Any], codec_type: str) -> Mapping[str, Any]:
    stream = _optional_stream(streams, codec_type)
    if stream is not None:
        return stream
    raise ContractError(f"ffprobe payload has no {codec_type} stream")


def _optional_stream(
    streams: Sequence[Any], codec_type: str
) -> Mapping[str, Any] | None:
    for stream in streams:
        if isinstance(stream, Mapping) and stream.get("codec_type") == codec_type:
            return stream
    return None


def _stream_duration(stream: Mapping[str, Any], label: str) -> float:
    duration = stream.get("duration")
    if duration not in (None, "N/A"):
        return _number(duration, f"{label}.duration")
    duration_ts, time_base = stream.get("duration_ts"), stream.get("time_base")
    if duration_ts not in (None, "N/A") and time_base not in (None, "N/A"):
        return _number(duration_ts, f"{label}.duration_ts") * float(
            _fraction(time_base, f"{label}.time_base")
        )
    raise ContractError(f"{label} stream lacks duration")


def _start_pts_seconds(stream: Mapping[str, Any], label: str) -> float:
    start_pts, time_base = stream.get("start_pts"), stream.get("time_base")
    if start_pts not in (None, "N/A") and time_base not in (None, "N/A"):
        return _number(start_pts, f"{label}.start_pts") * float(
            _fraction(time_base, f"{label}.time_base")
        )
    start_time = stream.get("start_time")
    if start_time not in (None, "N/A"):
        return _number(start_time, f"{label}.start_time")
    raise ContractError(f"{label} stream lacks start PTS")


def _number(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be numeric") from exc


def _fraction(value: Any, name: str) -> Fraction:
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ContractError(f"{name} must be a non-zero rational") from exc
