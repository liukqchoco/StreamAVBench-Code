"""Separate 2-fps video-only and synchronized-AV preview builders."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from .routing import MediaMode, MLLMMetric, route_for

CommandRunner = Callable[[Sequence[str]], object]


def video_only_command(
    source: str | Path,
    destination: str | Path,
    *,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    fps: float = 2.0,
    width: int | None = None,
    grayscale: bool = False,
    crf: int | None = None,
) -> tuple[str, ...]:
    """Build a preview command that explicitly strips every audio stream."""

    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-ss",
        _seconds(start_seconds),
        "-i",
        str(source),
    ]
    if duration_seconds is not None:
        command.extend(("-t", _seconds(duration_seconds)))
    filters = [f"fps={_seconds(fps)}"]
    if width is not None:
        filters.append(f"scale={width}:-2")
    if grayscale:
        filters.append("format=gray")
    command.extend(
        [
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            ",".join(filters),
            "-c:v",
            "libx264",
        ]
    )
    if crf is not None:
        command.extend(("-crf", str(crf)))
    command.extend(
        (
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        )
    )
    return tuple(command)


def synchronized_av_command(
    source: str | Path,
    destination: str | Path,
    *,
    audio_source: str | Path | None = None,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    fps: float = 2.0,
    width: int | None = None,
    crf: int | None = None,
    audio_bitrate: str | None = None,
) -> tuple[str, ...]:
    """Build a preview command retaining the source audio-video timeline."""

    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-ss",
        _seconds(start_seconds),
        "-i",
        str(source),
    ]
    if audio_source is not None:
        command.extend(
            (
                "-ss",
                _seconds(start_seconds),
                "-i",
                str(audio_source),
            )
        )
    if duration_seconds is not None:
        command.extend(("-t", _seconds(duration_seconds)))
    filters = [f"fps={_seconds(fps)}"]
    if width is not None:
        filters.append(f"scale={width}:-2")
    command.extend(
        [
            "-map",
            "0:v:0",
            "-map",
            "1:a:0" if audio_source is not None else "0:a:0",
            "-vf",
            ",".join(filters),
            "-c:v",
            "libx264",
        ]
    )
    if crf is not None:
        command.extend(("-crf", str(crf)))
    command.extend(
        [
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
        ]
    )
    if audio_bitrate is not None:
        command.extend(("-b:a", audio_bitrate))
    command.extend(
        (
            "-movflags",
            "+faststart",
            str(destination),
        )
    )
    return tuple(command)


def audio_only_command(
    source: str | Path,
    destination: str | Path,
    *,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> tuple[str, ...]:
    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-ss",
        _seconds(start_seconds),
        "-i",
        str(source),
    ]
    if duration_seconds is not None:
        command.extend(("-t", _seconds(duration_seconds)))
    command.extend(
        (
            "-map",
            "0:a:0",
            "-vn",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(destination),
        )
    )
    return tuple(command)


def compact_inline_preview(
    source: str | Path,
    destination: str | Path,
    *,
    synchronized_av: bool,
    fps: float = 2.0,
    crf: int = 32,
) -> Path:
    """Re-encode an oversized preview without changing its timeline or FPS."""

    source_path = Path(source)
    destination_path = Path(destination)
    if synchronized_av:
        command = synchronized_av_command(
            source_path,
            destination_path,
            fps=fps,
            width=480,
            crf=crf,
            audio_bitrate="96k",
        )
    else:
        command = video_only_command(
            source_path,
            destination_path,
            fps=fps,
            width=480,
            crf=crf,
        )
    subprocess.run(command, check=True, capture_output=True)
    if not destination_path.is_file() or destination_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced no compact preview: {destination_path}")
    return destination_path


def build_media(
    metric: MLLMMetric | str,
    source: str | Path,
    destination: str | Path,
    *,
    audio_source: str | Path | None = None,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    runner: CommandRunner | None = None,
) -> Path:
    """Materialize the modality required by a scoring route."""

    route = route_for(metric)
    if route.media_mode is MediaMode.NONE:
        raise ValueError(f"{route.metric.value} does not accept media")
    if start_seconds < 0:
        raise ValueError("start_seconds must be non-negative")
    duration_seconds = float(route.expected_duration_s or 0)
    if end_seconds is not None:
        if end_seconds <= start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        requested_duration = end_seconds - start_seconds
        if abs(requested_duration - duration_seconds) > 1e-6:
            raise ValueError(
                f"{route.metric.value} requires a fixed {duration_seconds:g}s range; "
                f"got {requested_duration:g}s"
            )
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    audio_source_path = Path(audio_source) if audio_source is not None else None
    if audio_source_path is not None and not audio_source_path.is_file():
        raise FileNotFoundError(audio_source_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if route.media_mode is MediaMode.VIDEO_ONLY:
        full_rollout = duration_seconds >= 180.0
        command = video_only_command(
            source_path,
            destination_path,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            fps=6.0 if route.metric is MLLMMetric.PVC else 2.0,
            width=480 if route.metric is MLLMMetric.PVC or full_rollout else None,
            crf=28 if full_rollout else None,
        )
    elif route.media_mode is MediaMode.AUDIO_ONLY:
        command = audio_only_command(
            audio_source_path or source_path,
            destination_path,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )
    else:
        full_rollout = duration_seconds >= 180.0
        command = synchronized_av_command(
            source_path,
            destination_path,
            audio_source=audio_source_path,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            width=480 if full_rollout else None,
            crf=28 if full_rollout else None,
            audio_bitrate="96k" if full_rollout else None,
        )
    if runner is None:
        subprocess.run(command, check=True, capture_output=True)
    else:
        runner(command)
    if runner is None and (
        not destination_path.is_file() or destination_path.stat().st_size == 0
    ):
        raise RuntimeError(f"ffmpeg produced no preview: {destination_path}")
    return destination_path


def build_vq_media(*args: object, **kwargs: object) -> Path:
    return build_media(MLLMMetric.VQ, *args, **kwargs)


def build_aq_media(*args: object, **kwargs: object) -> Path:
    return build_media(MLLMMetric.AQ, *args, **kwargs)


def build_vif_media(*args: object, **kwargs: object) -> Path:
    return build_media(MLLMMetric.VIF, *args, **kwargs)


def build_aif_media(*args: object, **kwargs: object) -> Path:
    return build_media(MLLMMetric.AIF, *args, **kwargs)


def build_p0_vif_media(*args: object, **kwargs: object) -> Path:
    return build_media(MLLMMetric.P0_VIF, *args, **kwargs)


def build_p0_aif_media(*args: object, **kwargs: object) -> Path:
    return build_media(MLLMMetric.P0_AIF, *args, **kwargs)


def _seconds(value: float) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError("time values must be non-negative numbers")
    return f"{float(value):g}"
