"""Frame-sampled visual-quality evaluator."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from ..protocol import WorkerRequest, WorkerResult, serve_jsonl
from ._media import OpenCVFrameSampler, SampledFrames

METRIC = "VA"
SAMPLE_FPS = 2.0
ENCODER_ARCH = "ViT-L/14"


class FrameSampler(Protocol):
    def sample(
        self,
        video_path: str,
        fps: float,
        *,
        start_seconds: float,
        end_seconds: float,
    ) -> SampledFrames: ...


class FrameScorer(Protocol):
    def score(self, frames: Sequence[Any]) -> Sequence[float]: ...


class FeatureScorer:
    """Lazy image-encoder and linear-head scorer."""

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str = "cuda",
        encoder_cache_root: str | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.encoder_cache_root = encoder_cache_root
        self._loaded: tuple[Any, Any, Any, Any] | None = None

    def _load(self) -> tuple[Any, Any, Any, Any]:
        if self._loaded is not None:
            return self._loaded
        try:
            import clip
            import torch
            import torch.nn as nn
            import torch.nn.functional as functional
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "visual-quality scoring requires torch, Pillow, and the "
                "configured image encoder"
            ) from exc
        checkpoint = Path(self.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"quality checkpoint not found: {checkpoint}")
        if self.encoder_cache_root is None:
            raise ValueError(
                "encoder_cache_root must point to the prepopulated model cache"
            )
        encoder_checkpoint = Path(self.encoder_cache_root) / "ViT-L-14.pt"
        if not encoder_checkpoint.is_file():
            raise FileNotFoundError(
                f"image encoder checkpoint not found: {encoder_checkpoint}"
            )
        encoder, preprocess = clip.load(
            str(encoder_checkpoint),
            device=self.device,
        )
        head = nn.Linear(768, 1)
        head.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        head.to(self.device).eval()
        encoder.eval()
        self._loaded = (torch, functional, Image, (encoder, preprocess, head))
        return self._loaded

    def score(self, frames: Sequence[Any]) -> Sequence[float]:
        torch, functional, image_type, models = self._load()
        encoder, preprocess, head = models
        values: list[float] = []
        with torch.no_grad():
            for frame in frames:
                image = (
                    frame
                    if isinstance(frame, image_type.Image)
                    else image_type.fromarray(frame)
                )
                tensor = preprocess(image).unsqueeze(0).to(self.device)
                features = functional.normalize(
                    encoder.encode_image(tensor).float(),
                    dim=-1,
                    p=2,
                )
                values.append(float(head(features).item()))
        return values


class VisualQualityWorker:
    metric = METRIC

    def __init__(
        self,
        *,
        sampler: FrameSampler | None = None,
        scorer: FrameScorer | None = None,
        checkpoint: str | None = None,
        device: str = "cuda",
        encoder_cache_root: str | None = None,
    ) -> None:
        self.sampler = sampler or OpenCVFrameSampler()
        self.scorer = scorer or FeatureScorer(
            checkpoint or "",
            device=device,
            encoder_cache_root=encoder_cache_root,
        )

    def evaluate(self, request: WorkerRequest) -> WorkerResult:
        start, end = request.interval(default_duration_seconds=30.0)
        sampled = self.sampler.sample(
            request.require_video(),
            SAMPLE_FPS,
            start_seconds=start,
            end_seconds=end,
        )
        raw_scores = [float(value) for value in self.scorer.score(sampled.frames)]
        if len(raw_scores) != len(sampled.frames) or not raw_scores:
            raise RuntimeError(
                "visual-quality scorer must return one score per sampled frame"
            )
        normalized = [value / 10.0 for value in raw_scores]
        return WorkerResult.ok(
            request,
            scores={"aesthetic": sum(normalized) / len(normalized)},
            artifacts={
                "sample_timestamps_seconds": list(sampled.timestamps_seconds),
                "frame_count": len(normalized),
                "sample_interval_seconds": [start, end],
                "interval_index": request.interval_index,
            },
            protocol={
                "sample_fps": SAMPLE_FPS,
                "encoder_architecture": ENCODER_ARCH,
                "frame_transform": "linear_head_score/10",
                "aggregation": "arithmetic_mean",
                "interval_boundary": "half_open_no_cross_boundary",
            },
        )


def build_command(
    *,
    python: str = "python",
    checkpoint: str,
    encoder_cache_root: str,
    device: str = "cuda",
) -> list[str]:
    return [
        python,
        "-m",
        "streamav_eval.workers.objective.visual_quality",
        "--checkpoint",
        checkpoint,
        "--encoder-cache-root",
        encoder_cache_root,
        "--device",
        device,
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder-cache-root", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    return serve_jsonl(
        lambda: VisualQualityWorker(
            checkpoint=args.checkpoint,
            encoder_cache_root=args.encoder_cache_root,
            device=args.device,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
