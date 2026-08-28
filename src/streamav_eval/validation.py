"""Side-effect-free media validation and failure classification."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .contracts import MediaMetadata
from .media import parse_ffprobe, probe_media

Probe = Callable[[str | Path], MediaMetadata | Mapping[str, Any]]
SilenceProbe = Callable[[str | Path], bool | Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class ValidationResult:
    status: str
    video_path: str | None
    audio_path: str | None
    audio_source: str | None
    metadata: Mapping[str, Any] | None = None
    error: Mapping[str, str] | None = None

    @property
    def valid(self) -> bool:
        return self.status == "computed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_media(
    record: Mapping[str, Any],
    *,
    probe: Probe = probe_media,
    audio_probe: Callable[[str | Path], Any] | None = None,
    silence_probe: SilenceProbe | None = None,
    tolerance_seconds: float = 0.5,
    base_dir: str | Path | None = None,
    ffprobe: str = "ffprobe",
) -> ValidationResult:
    """Validate expected local modalities, classifying media defects as generation failures.

    A null sidecar audio path means that workers should extract audio from the
    muxed video.  The validator never extracts, downloads, or evaluates media.
    """

    video, audio = _media_paths(record, base_dir)
    expected = record.get("expected_modalities", {})
    expected = expected if isinstance(expected, Mapping) else {}
    expect_video = bool(expected.get("video", True))
    expect_audio = bool(expected.get("audio", True))
    expected_duration = _positive_number(
        record.get("duration_seconds", 180.0), "duration_seconds"
    )

    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds must be non-negative")
    if expect_video and (video is None or not video.is_file()):
        return _generation_failure(video, audio, "missing_video", "video is missing")
    if audio is not None and expect_audio and not audio.is_file():
        return _generation_failure(video, audio, "missing_audio", "audio is missing")
    if video is None:
        return _generation_failure(video, audio, "missing_video", "video is missing")

    try:
        probed = (
            probe_media(video, ffprobe=ffprobe, require_audio=False)
            if audio is not None and probe is probe_media
            else probe_media(video, ffprobe=ffprobe)
            if probe is probe_media
            else probe(video)
        )
        metadata = _metadata(probed, video)
    except Exception as exc:
        return _generation_failure(video, audio, "corrupt_media", str(exc), exc)

    sidecar_duration: float | None = None
    if expect_audio and audio is not None:
        selected_audio_probe = audio_probe
        if selected_audio_probe is None:
            selected_audio_probe = (
                (lambda path: _probe_audio_duration(path, ffprobe=ffprobe))
                if probe is probe_media
                else probe
            )
        try:
            sidecar_duration = _audio_duration(selected_audio_probe(audio))
        except Exception as exc:
            return _generation_failure(video, audio, "corrupt_audio", str(exc), exc)

    duration_errors = {
        name: value
        for name, value in (
            ("container", metadata.container_duration),
            ("video", metadata.video_duration),
            (
                "audio",
                sidecar_duration
                if sidecar_duration is not None
                else metadata.audio_duration,
            ),
        )
        if (name != "audio" or expect_audio)
        and abs(value - expected_duration) > tolerance_seconds
    }
    if duration_errors:
        detail = ", ".join(
            f"{name}={value:.6f}s" for name, value in duration_errors.items()
        )
        return _generation_failure(
            video,
            audio,
            "duration_mismatch",
            f"outside {expected_duration:.3f}s ± {tolerance_seconds:.3f}s: {detail}",
        )

    audio_source = (
        "sidecar" if audio is not None else ("muxed" if expect_audio else None)
    )
    if expect_audio and silence_probe is not None:
        audio_input = audio if audio is not None else video
        try:
            silence = silence_probe(audio_input)
            silent = (
                bool(silence.get("silent"))
                if isinstance(silence, Mapping)
                else bool(silence)
            )
        except Exception as exc:
            return _generation_failure(
                video, audio, "audio_probe_failed", str(exc), exc
            )
        if silent:
            return _generation_failure(
                video, audio, "silent_audio", "expected audio is effectively silent"
            )

    return ValidationResult(
        status="computed",
        video_path=str(video),
        audio_path=str(audio) if audio is not None else None,
        audio_source=audio_source,
        metadata={
            "container_duration": metadata.container_duration,
            "video_duration": metadata.video_duration,
            "audio_duration": (
                sidecar_duration
                if sidecar_duration is not None
                else metadata.audio_duration
            ),
            "video_start_pts": metadata.video_start_pts,
            "audio_start_pts": metadata.audio_start_pts,
        },
    )


validate_record = validate_media


def run_evaluator(
    record: Mapping[str, Any],
    evaluator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    **validation_kwargs: Any,
) -> dict[str, Any]:
    """Validate then evaluate, keeping evaluator exceptions in their own class."""

    validation = validate_media(record, **validation_kwargs)
    if not validation.valid:
        return validation.to_dict()
    request = dict(record)
    request.update(
        {
            "video_path": validation.video_path,
            "audio_path": validation.audio_path,
            "audio_source": validation.audio_source,
        }
    )
    try:
        return dict(evaluator(request))
    except Exception as exc:
        return {
            "status": "evaluator_failure",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def _media_paths(
    record: Mapping[str, Any], base_dir: str | Path | None
) -> tuple[Path | None, Path | None]:
    media = record.get("media", {})
    media = media if isinstance(media, Mapping) else {}
    video = media.get("video", record.get("video_path"))
    audio = media.get("audio", record.get("audio_path"))
    base = Path(base_dir) if base_dir is not None else None

    def resolve(value: Any) -> Path | None:
        if value is None:
            return None
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() or base is None else (base / path).resolve()

    return resolve(video), resolve(audio)


def _metadata(value: MediaMetadata | Mapping[str, Any], path: Path) -> MediaMetadata:
    if isinstance(value, MediaMetadata):
        return value
    if "format" in value and "streams" in value:
        return parse_ffprobe(value, path=path)
    aliases = {
        "container_duration": value.get("container_duration", value.get("duration")),
        "video_duration": value.get("video_duration"),
        "audio_duration": value.get("audio_duration"),
        "video_start_pts": value.get("video_start_pts", 0.0),
        "audio_start_pts": value.get("audio_start_pts", 0.0),
    }
    converted = {
        key: float(item) if item is not None else None for key, item in aliases.items()
    }
    return MediaMetadata(path=path, **converted)


def _probe_audio_duration(path: str | Path, *, ffprobe: str = "ffprobe") -> float:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_entries",
            "format=duration:stream=codec_type,duration",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return _audio_duration(json.loads(completed.stdout))


def _audio_duration(value: Any) -> float:
    if isinstance(value, MediaMetadata):
        if value.audio_duration is None:
            raise ValueError("audio probe did not report an audio stream")
        return value.audio_duration
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, Mapping):
        direct = value.get("audio_duration")
        if direct not in (None, "N/A"):
            return float(direct)
        streams = value.get("streams", ())
        if isinstance(streams, (list, tuple)):
            for stream in streams:
                if isinstance(stream, Mapping) and stream.get("codec_type") == "audio":
                    duration = stream.get("duration")
                    if duration not in (None, "N/A"):
                        return float(duration)
        format_value = value.get("format")
        if isinstance(format_value, Mapping) and format_value.get("duration") not in (
            None,
            "N/A",
        ):
            return float(format_value["duration"])
    raise ValueError("audio probe did not report a duration")


def _generation_failure(
    video: Path | None,
    audio: Path | None,
    code: str,
    message: str,
    exc: BaseException | None = None,
) -> ValidationResult:
    return ValidationResult(
        status="generation_failure",
        video_path=str(video) if video is not None else None,
        audio_path=str(audio) if audio is not None else None,
        audio_source="sidecar" if audio is not None else ("muxed" if video else None),
        error={
            "code": code,
            "type": type(exc).__name__ if exc is not None else "MediaValidationError",
            "message": message,
        },
    )


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)
