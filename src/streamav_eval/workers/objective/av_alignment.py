"""Audio-video embedding alignment with raw cosine scoring."""

from __future__ import annotations

import argparse
import contextlib
import inspect
import math
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..protocol import WorkerRequest, WorkerResult, serve_jsonl
from ._media import FFmpegMediaPreparer, MediaPreparer, absolute_intervals

METRIC = "AVAlign"
CLIP_DURATION_SECONDS = 2.0
VIDEO_CLIPS = 5
AUDIO_CLIPS = 3
EXPECTED_INTERVAL_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class AlignmentOutput:
    video_embedding: Any
    audio_embedding: Any
    video_intervals_seconds: tuple[tuple[float, float], ...]
    audio_intervals_seconds: tuple[tuple[float, float], ...]


class AlignmentBackend(Protocol):
    def embed(self, video_path: str, audio_path: str) -> AlignmentOutput: ...


class EmbeddingBackend:
    """Lazy, no-download multimodal embedding backend."""

    def __init__(
        self,
        *,
        source_dir: str,
        checkpoint: str,
        device: str = "cuda",
    ) -> None:
        self.source_dir = Path(source_dir)
        self.checkpoint = Path(checkpoint)
        self.device = device
        self._runtime: tuple[Any, Any, Any, Any, Any] | None = None

    def _load(self) -> tuple[Any, Any, Any, Any, Any]:
        if self._runtime is not None:
            return self._runtime
        if not self.source_dir.is_dir():
            raise FileNotFoundError(
                f"alignment source directory not found: {self.source_dir}"
            )
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"alignment checkpoint not found: {self.checkpoint}"
            )
        with _temporary_sys_path(self.source_dir):
            try:
                import torch
                import torchaudio
                from imagebind import data
                from imagebind.models import imagebind_model
                from imagebind.models.imagebind_model import ModalityType
                from pytorchvideo.data.encoded_video import EncodedVideo
            except ImportError as exc:
                raise RuntimeError(
                    "alignment scoring requires the configured source tree and "
                    "its torch runtime"
                ) from exc
        _install_encoded_video_compatibility(data, EncodedVideo)
        model = imagebind_model.imagebind_huge(pretrained=False)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        model.load_state_dict(state)
        model.to(self.device).eval()
        self._runtime = (
            torch,
            torchaudio,
            data,
            ModalityType,
            (model, EncodedVideo),
        )
        return self._runtime

    def embed(self, video_path: str, audio_path: str) -> AlignmentOutput:
        torch, torchaudio, data, modality, runtime = self._load()
        model, encoded_video = runtime
        video = encoded_video.from_path(
            video_path,
            decoder="decord",
            decode_audio=False,
        )
        waveform, sample_rate = torchaudio.load(audio_path)
        video_intervals = _sample_intervals(
            data,
            float(video.duration),
            VIDEO_CLIPS,
        )
        audio_intervals = _sample_intervals(
            data,
            float(waveform.shape[1]) / float(sample_rate),
            AUDIO_CLIPS,
        )
        inputs = {
            modality.VISION: data.load_and_transform_video_data(
                [video_path],
                self.device,
                clip_duration=int(CLIP_DURATION_SECONDS),
                clips_per_video=VIDEO_CLIPS,
            ),
            modality.AUDIO: data.load_and_transform_audio_data(
                [audio_path],
                self.device,
                clip_duration=int(CLIP_DURATION_SECONDS),
                clips_per_video=AUDIO_CLIPS,
            ),
        }
        with torch.no_grad():
            embeddings = model(inputs)
        return AlignmentOutput(
            embeddings[modality.VISION],
            embeddings[modality.AUDIO],
            video_intervals,
            audio_intervals,
        )


class AVAlignmentWorker:
    metric = METRIC

    def __init__(
        self,
        *,
        backend: AlignmentBackend | None = None,
        source_dir: str | None = None,
        checkpoint: str | None = None,
        device: str = "cuda",
        media_preparer: MediaPreparer | None = None,
    ) -> None:
        self.backend = backend or EmbeddingBackend(
            source_dir=source_dir or "",
            checkpoint=checkpoint or "",
            device=device,
        )
        self.media_preparer = media_preparer or FFmpegMediaPreparer()

    def evaluate(self, request: WorkerRequest) -> WorkerResult:
        start, end = request.require_interval_duration(EXPECTED_INTERVAL_SECONDS)
        request.require_video()
        with self.media_preparer.prepare(
            video_path=request.video_path,
            audio_path=request.audio_path,
            start_seconds=start,
            end_seconds=end,
            include_video=True,
            include_audio=True,
        ) as media:
            if media.video_path is None or media.audio_path is None:
                raise RuntimeError(
                    "media preparer did not produce synchronized AV files"
                )
            output = self.backend.embed(media.video_path, media.audio_path)
        _validate_local_intervals(
            output.video_intervals_seconds,
            EXPECTED_INTERVAL_SECONDS,
            "video",
        )
        _validate_local_intervals(
            output.audio_intervals_seconds,
            EXPECTED_INTERVAL_SECONDS,
            "audio",
        )
        score = normalized_cosine(
            output.video_embedding,
            output.audio_embedding,
        )
        return WorkerResult.ok(
            request,
            scores={"av_alignment": score},
            artifacts={
                "video_sample_intervals_seconds": absolute_intervals(
                    output.video_intervals_seconds,
                    start,
                ),
                "audio_sample_intervals_seconds": absolute_intervals(
                    output.audio_intervals_seconds,
                    start,
                ),
                "sample_interval_seconds": [start, end],
                "interval_index": request.interval_index,
            },
            protocol={
                "clip_duration_seconds": CLIP_DURATION_SECONDS,
                "video_clips": VIDEO_CLIPS,
                "audio_clips": AUDIO_CLIPS,
                "score": "raw_l2_normalized_cosine",
                "score_transform": "none",
                "input_isolation": "synchronized_ffmpeg_interval_files",
                "loader_compatibility": "ignore_unsupported_video_sample_rate",
            },
        )


def normalized_cosine(first: Any, second: Any) -> float:
    left = _flatten(first)
    right = _flatten(second)
    if len(left) != len(right) or not left:
        raise RuntimeError("embeddings must be non-empty and equal length")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise RuntimeError("embedding has zero norm")
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )


def _install_encoded_video_compatibility(
    data_module: Any,
    encoded_video: Any,
) -> bool:
    """Ignore a loader keyword absent from older video backends."""

    parameters = inspect.signature(encoded_video.from_path).parameters.values()
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return False

    class CompatibleEncodedVideo:
        @classmethod
        def from_path(cls, *args: Any, **kwargs: Any) -> Any:
            kwargs.pop("sample_rate", None)
            return encoded_video.from_path(*args, **kwargs)

    data_module.EncodedVideo = CompatibleEncodedVideo
    return True


def _validate_local_intervals(
    intervals: Sequence[Sequence[float]],
    duration: float,
    modality: str,
) -> None:
    for interval in intervals:
        if (
            len(interval) != 2
            or float(interval[0]) < 0
            or float(interval[1]) <= float(interval[0])
            or float(interval[1]) > duration + 1e-6
        ):
            raise RuntimeError(
                f"{modality} sample interval falls outside prepared input"
            )


def _flatten(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()

    def visit(item: Any) -> Iterator[float]:
        if isinstance(item, (list, tuple)):
            for child in item:
                yield from visit(child)
        else:
            yield float(item)

    return list(visit(value))


def _sample_intervals(
    data_module: Any,
    duration: float,
    clips_per_video: int,
) -> tuple[tuple[float, float], ...]:
    sampler = data_module.ConstantClipsPerVideoSampler(
        clip_duration=CLIP_DURATION_SECONDS,
        clips_per_video=clips_per_video,
    )
    return tuple(
        (float(start), float(end))
        for start, end in data_module.get_clip_timepoints(sampler, duration)
    )


@contextlib.contextmanager
def _temporary_sys_path(path: Path) -> Iterator[None]:
    value = str(path.resolve())
    sys.path.insert(0, value)
    try:
        yield
    finally:
        if value in sys.path:
            sys.path.remove(value)


def build_command(
    *,
    python: str = "python",
    source_dir: str,
    checkpoint: str,
    device: str = "cuda",
) -> list[str]:
    return [
        python,
        "-m",
        "streamav_eval.workers.objective.av_alignment",
        "--source-dir",
        source_dir,
        "--checkpoint",
        checkpoint,
        "--device",
        device,
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    return serve_jsonl(
        lambda: AVAlignmentWorker(
            source_dir=args.source_dir,
            checkpoint=args.checkpoint,
            device=args.device,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
