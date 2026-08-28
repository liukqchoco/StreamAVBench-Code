"""Perceptual audio-quality evaluator."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from ..protocol import WorkerRequest, WorkerResult, serve_jsonl
from ._media import FFmpegMediaPreparer, MediaPreparer

METRIC = "PQ"
EXPECTED_DURATION_SECONDS = 30.0


class Predictor(Protocol):
    def forward(
        self,
        items: Sequence[Mapping[str, Any]],
    ) -> Sequence[Mapping[str, Any]]: ...


class AudioQualityPredictor:
    """Lazy wrapper around the configured audio-quality model."""

    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = Path(checkpoint)
        self._predictor: Predictor | None = None

    def _load(self) -> Predictor:
        if self._predictor is not None:
            return self._predictor
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"audio-quality checkpoint not found: {self.checkpoint}"
            )
        try:
            from audiobox_aesthetics.infer import initialize_predictor
        except ImportError as exc:
            raise RuntimeError(
                "the configured audio-quality package is unavailable"
            ) from exc
        self._predictor = initialize_predictor(str(self.checkpoint))
        return self._predictor

    def forward(
        self,
        items: Sequence[Mapping[str, Any]],
    ) -> Sequence[Mapping[str, Any]]:
        return self._load().forward(items)


class AudioQualityWorker:
    metric = METRIC

    def __init__(
        self,
        *,
        predictor: Predictor | None = None,
        checkpoint: str | None = None,
        media_preparer: MediaPreparer | None = None,
    ) -> None:
        self.predictor = predictor or AudioQualityPredictor(checkpoint or "")
        self.media_preparer = media_preparer or FFmpegMediaPreparer()

    def evaluate(self, request: WorkerRequest) -> WorkerResult:
        start, end = request.require_interval_duration(EXPECTED_DURATION_SECONDS)
        if request.audio_path is None:
            request.require_video()
        with self.media_preparer.prepare(
            video_path=request.video_path,
            audio_path=request.audio_path,
            start_seconds=start,
            end_seconds=end,
            include_video=False,
            include_audio=True,
        ) as media:
            if media.audio_path is None:
                raise RuntimeError("media preparer did not produce an audio file")
            outputs = self.predictor.forward([{"path": media.audio_path}]) or []
        if len(outputs) != 1 or not isinstance(outputs[0], Mapping):
            raise RuntimeError(
                "audio-quality predictor must return exactly one score object"
            )
        if "PQ" not in outputs[0]:
            raise RuntimeError("audio-quality predictor output is missing PQ")
        pq = float(outputs[0]["PQ"])
        return WorkerResult.ok(
            request,
            scores={"pq": pq},
            artifacts={
                "source_audio_path": request.audio_path or request.video_path,
                "audio_source": (
                    "external_audio_path"
                    if request.audio_path is not None
                    else "demuxed_from_video"
                ),
                "sample_interval_seconds": [start, end],
                "interval_index": request.interval_index,
            },
            protocol={
                "axis": "PQ",
                "other_axes_ignored": ["CE", "CU", "PC"],
                "audio_policy": "single_full_length_30s_input_no_chunking",
                "expected_duration_seconds": EXPECTED_DURATION_SECONDS,
                "input_isolation": "ffmpeg_interval_to_temporary_wav",
            },
        )


def build_command(
    *,
    python: str = "python",
    checkpoint: str,
) -> list[str]:
    return [
        python,
        "-m",
        "streamav_eval.workers.objective.audio_quality",
        "--checkpoint",
        checkpoint,
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args(argv)
    return serve_jsonl(lambda: AudioQualityWorker(checkpoint=args.checkpoint))


if __name__ == "__main__":
    raise SystemExit(main())
