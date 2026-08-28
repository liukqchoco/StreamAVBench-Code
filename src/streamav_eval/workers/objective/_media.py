"""Small media adapters; heavy packages are imported only when invoked."""

from __future__ import annotations

import contextlib
import math
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class SampledFrames:
    frames: Sequence[Any]
    timestamps_seconds: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PreparedMedia:
    video_path: str | None
    audio_path: str | None
    duration_seconds: float


class CommandRunner(Protocol):
    def __call__(self, command: Sequence[str], **kwargs: Any) -> Any: ...


class MediaPreparer(Protocol):
    def prepare(
        self,
        *,
        video_path: str | None,
        audio_path: str | None,
        start_seconds: float,
        end_seconds: float,
        include_video: bool,
        include_audio: bool,
    ) -> AbstractContextManager[PreparedMedia]: ...


class FFmpegMediaPreparer:
    """Create self-contained interval files for whole-file model loaders."""

    def __init__(
        self,
        *,
        ffmpeg: str = "ffmpeg",
        runner: CommandRunner = subprocess.run,
        video_fps: float | None = None,
        min_video_side: int | None = None,
        audio_hz: int | None = None,
        minimum_duration_seconds: float | None = None,
    ) -> None:
        for value, name in (
            (video_fps, "video_fps"),
            (min_video_side, "min_video_side"),
            (audio_hz, "audio_hz"),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if minimum_duration_seconds is not None and (
            not math.isfinite(minimum_duration_seconds) or minimum_duration_seconds <= 0
        ):
            raise ValueError("minimum_duration_seconds must be finite and positive")
        self.ffmpeg = ffmpeg
        self.runner = runner
        self.video_fps = video_fps
        self.min_video_side = min_video_side
        self.audio_hz = audio_hz
        self.minimum_duration_seconds = minimum_duration_seconds

    @contextlib.contextmanager
    def prepare(
        self,
        *,
        video_path: str | None,
        audio_path: str | None,
        start_seconds: float,
        end_seconds: float,
        include_video: bool,
        include_audio: bool,
    ) -> Iterator[PreparedMedia]:
        duration = end_seconds - start_seconds
        if start_seconds < 0 or not math.isfinite(duration) or duration <= 0:
            raise ValueError("media interval must be finite and non-empty")
        if include_video and video_path is None:
            raise ValueError("video_path is required for video preparation")
        if include_audio and audio_path is None and video_path is None:
            raise ValueError("audio_path or muxed video_path is required")

        with tempfile.TemporaryDirectory(prefix="streamav-window-") as directory:
            root = Path(directory)
            output_duration = max(duration, self.minimum_duration_seconds or duration)
            pad_duration = output_duration - duration
            prepared_video: str | None = None
            prepared_audio: str | None = None
            if include_video:
                video_output = root / "interval.mp4"
                command = self._base_command(start_seconds, duration, video_path or "")
                if audio_path is not None:
                    command.extend(
                        [
                            "-ss",
                            _seconds(start_seconds),
                            "-t",
                            _seconds(duration),
                            "-i",
                            audio_path,
                        ]
                    )
                video_filters: list[str] = []
                if self.min_video_side is not None:
                    side = self.min_video_side
                    video_filters.append(
                        f"scale='if(gte(iw,ih),-2,{side})':'if(gte(iw,ih),{side},-2)'"
                    )
                if self.video_fps is not None:
                    video_filters.append(f"fps={self.video_fps:g}")
                if pad_duration > 1e-9:
                    video_filters.append(
                        f"tpad=stop_mode=clone:stop_duration={_seconds(pad_duration)}"
                    )
                command.extend(
                    [
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0" if audio_path is not None else "0:a?",
                    ]
                )
                if video_filters:
                    command.extend(["-vf", ",".join(video_filters)])
                if pad_duration > 1e-9:
                    command.extend(
                        [
                            "-af",
                            f"apad=pad_dur={_seconds(pad_duration)}",
                        ]
                    )
                if self.audio_hz is not None:
                    command.extend(["-ar", str(self.audio_hz)])
                command.extend(
                    [
                        "-c:v",
                        "libx264",
                        "-c:a",
                        "aac",
                        "-t",
                        _seconds(output_duration),
                        "-movflags",
                        "+faststart",
                        str(video_output),
                    ]
                )
                self._run(command)
                prepared_video = str(video_output)
            if include_audio:
                audio_output = root / "interval.wav"
                audio_source = audio_path or video_path
                command = self._base_command(
                    start_seconds, duration, audio_source or ""
                )
                command.extend(["-map", "0:a:0", "-vn"])
                if pad_duration > 1e-9:
                    command.extend(
                        [
                            "-af",
                            f"apad=pad_dur={_seconds(pad_duration)}",
                        ]
                    )
                if self.audio_hz is not None:
                    command.extend(["-ar", str(self.audio_hz)])
                command.extend(
                    [
                        "-c:a",
                        "pcm_s16le",
                        "-t",
                        _seconds(output_duration),
                        str(audio_output),
                    ]
                )
                self._run(command)
                prepared_audio = str(audio_output)
            yield PreparedMedia(prepared_video, prepared_audio, output_duration)

    def _base_command(
        self, start_seconds: float, duration_seconds: float, input_path: str
    ) -> list[str]:
        return [
            self.ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            _seconds(start_seconds),
            "-t",
            _seconds(duration_seconds),
            "-i",
            input_path,
        ]

    def _run(self, command: Sequence[str]) -> None:
        self.runner(
            list(command), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )


class OpenCVFrameSampler:
    """Decode nearest source frames on a deterministic output-fps grid."""

    def sample(
        self,
        video_path: str,
        fps: float,
        *,
        start_seconds: float = 0.0,
        end_seconds: float | None = None,
    ) -> SampledFrames:
        if fps <= 0 or not math.isfinite(fps):
            raise ValueError("sample fps must be positive and finite")
        if start_seconds < 0 or not math.isfinite(start_seconds):
            raise ValueError("start_seconds must be non-negative and finite")
        if end_seconds is not None and (
            not math.isfinite(end_seconds) or end_seconds <= start_seconds
        ):
            raise ValueError(
                "end_seconds must be finite and greater than start_seconds"
            )
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV sampling requires opencv-python; use the metric conda env"
            ) from exc
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 0:
            capture.release()
            raise RuntimeError(f"video reports invalid fps: {video_path}")
        frames: list[Any] = []
        times: list[float] = []
        start_index = max(0, math.ceil(start_seconds * source_fps - 1e-9))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)
        index = start_index
        next_sample_time = start_seconds
        source_half_frame = 0.5 / source_fps
        while True:
            timestamp = index / source_fps
            if end_seconds is not None and timestamp >= end_seconds - 1e-9:
                break
            ok, bgr = capture.read()
            if not ok:
                break
            if timestamp + source_half_frame >= next_sample_time:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                times.append(timestamp)
                next_sample_time += 1.0 / fps
            index += 1
        capture.release()
        if not frames:
            raise RuntimeError(f"video contains no decodable frames: {video_path}")
        return SampledFrames(frames, tuple(times))


def absolute_intervals(
    intervals: Sequence[Sequence[float]], start_seconds: float
) -> list[list[float]]:
    return [
        [start_seconds + float(interval[0]), start_seconds + float(interval[1])]
        for interval in intervals
    ]


def _seconds(value: float) -> str:
    return f"{value:.9f}".rstrip("0").rstrip(".") or "0"


def short_side_cap(frame: Any, cap: int = 512) -> Any:
    """Resize a PIL image or ndarray only when its short side exceeds ``cap``."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("frame resizing requires Pillow") from exc

    image = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
    width, height = image.size
    if min(width, height) <= cap:
        return image
    scale = cap / min(width, height)
    return image.resize(
        (round(width * scale), round(height * scale)), resample=Image.Resampling.LANCZOS
    )
