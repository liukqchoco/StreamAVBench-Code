"""Endpoint audio-video synchronization offsets."""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from ..protocol import WorkerRequest, WorkerResult, serve_jsonl
from ._media import FFmpegMediaPreparer, MediaPreparer

METRIC = "AVSync"
WINDOW_SECONDS = 4.8
VIDEO_FPS = 25
AUDIO_HZ = 16_000
MIN_SIDE = 256
EXPECTED_INTERVAL_SECONDS = 30.0


class OffsetBackend(Protocol):
    def predict_offset(
        self,
        video_path: str,
        *,
        start_seconds: float,
        duration_seconds: float,
    ) -> float: ...


class OffsetModelBackend:
    """Lazy, no-download synchronization model backend."""

    def __init__(
        self,
        *,
        source_dir: str,
        config_path: str,
        checkpoint: str,
        device: str = "cuda",
    ) -> None:
        self.source_dir = Path(source_dir)
        self.config_path = Path(config_path)
        self.checkpoint = Path(checkpoint)
        self.device_name = device
        self._runtime: tuple[Any, ...] | None = None

    def _load(self) -> tuple[Any, ...]:
        if self._runtime is not None:
            return self._runtime
        if not self.source_dir.is_dir():
            raise FileNotFoundError(
                f"synchronization source directory not found: {self.source_dir}"
            )
        for path in (self.config_path, self.checkpoint):
            if not path.is_file():
                raise FileNotFoundError(f"synchronization model file not found: {path}")
        try:
            import torch
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise RuntimeError(
                "synchronization scoring requires torch and omegaconf"
            ) from exc
        with contextlib.ExitStack() as stack:
            for path in self._runtime_paths():
                stack.enter_context(_temporary_sys_path(path))
            try:
                from dataset.dataset_utils import get_video_and_audio
                from dataset.transforms import make_class_grid
                from scripts.train_utils import (
                    get_model,
                    get_transforms,
                    prepare_inputs,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "cannot import the configured synchronization source tree"
                ) from exc
            cfg = OmegaConf.load(self.config_path)
            cfg.model.params.afeat_extractor.params.ckpt_path = None
            cfg.model.params.vfeat_extractor.params.ckpt_path = None
            cfg.model.params.transformer.target = (
                cfg.model.params.transformer.target.replace(
                    ".modules.feature_selector.",
                    ".sync_model.",
                )
            )
            device = torch.device(self.device_name)
            _, model = get_model(cfg, device)
            state = torch.load(
                self.checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            model.load_state_dict(state["model"])
            model.eval()
            transforms = get_transforms(cfg, ["test"])
            num_classes = (
                cfg.model.params.transformer.params.off_head_cfg.params.out_features
            )
            grid = make_class_grid(
                -cfg.data.max_off_sec,
                cfg.data.max_off_sec,
                num_classes,
            )
        self._runtime = (
            torch,
            get_video_and_audio,
            prepare_inputs,
            model,
            transforms,
            grid,
            cfg,
            device,
        )
        return self._runtime

    def predict_offset(
        self,
        video_path: str,
        *,
        start_seconds: float,
        duration_seconds: float,
    ) -> float:
        if abs(duration_seconds - WINDOW_SECONDS) > 1e-6:
            raise ValueError(f"endpoint window must be {WINDOW_SECONDS} seconds")
        (
            torch,
            get_video_and_audio,
            prepare_inputs,
            model,
            transforms,
            grid,
            cfg,
            device,
        ) = self._load()
        with contextlib.ExitStack() as stack:
            for path in self._runtime_paths():
                stack.enter_context(_temporary_sys_path(path))
            rgb, audio, meta = get_video_and_audio(video_path, get_meta=True)
            item = {
                "video": rgb,
                "audio": audio,
                "meta": meta,
                "path": video_path,
                "split": "test",
                "targets": {
                    "v_start_i_sec": start_seconds,
                    "offset_sec": 0.0,
                },
            }
            item = transforms["test"](item)
            batch = torch.utils.data.default_collate([item])
            aud, vid, _ = prepare_inputs(batch, device)
            use_half = bool(cfg.training.use_half_precision) and device.type == "cuda"
            with (
                torch.no_grad(),
                torch.autocast(
                    device.type,
                    enabled=use_half,
                ),
            ):
                _, logits = model(vid, aud)
        predicted_class = int(torch.argmax(logits[0]).item())
        return float(grid[predicted_class].item())

    def _runtime_paths(self) -> tuple[Path, ...]:
        return (
            self.source_dir,
            self.source_dir / "model" / "modules" / "feat_extractors" / "visual",
        )

    def manifest(self) -> Mapping[str, Any]:
        return {
            "implementation": "checkpoint_offset_classifier",
            "source_dir": str(self.source_dir.resolve()),
            "config_path": str(self.config_path.resolve()),
            "checkpoint_path": str(self.checkpoint.resolve()),
            "preprocessing": {
                "video_fps": VIDEO_FPS,
                "audio_hz": AUDIO_HZ,
                "short_side": MIN_SIDE,
                "window_seconds": WINDOW_SECONDS,
            },
        }


class AVSynchronizationWorker:
    metric = METRIC

    def __init__(
        self,
        *,
        backend: OffsetBackend | None = None,
        source_dir: str | None = None,
        config_path: str | None = None,
        checkpoint: str | None = None,
        device: str = "cuda",
        manifest: Mapping[str, Any] | None = None,
        media_preparer: MediaPreparer | None = None,
    ) -> None:
        self.backend = backend or OffsetModelBackend(
            source_dir=source_dir or "",
            config_path=config_path or "",
            checkpoint=checkpoint or "",
            device=device,
        )
        self.explicit_manifest = dict(manifest or {})
        self.media_preparer = media_preparer or FFmpegMediaPreparer(
            video_fps=VIDEO_FPS,
            min_video_side=MIN_SIDE,
            audio_hz=AUDIO_HZ,
            minimum_duration_seconds=10.0,
        )

    def evaluate(self, request: WorkerRequest) -> WorkerResult:
        request.require_video()
        interval_start, interval_end = request.require_interval_duration(
            EXPECTED_INTERVAL_SECONDS
        )
        local_starts = (0.0, EXPECTED_INTERVAL_SECONDS - WINDOW_SECONDS)
        raw: list[float] = []
        for local_start in local_starts:
            absolute_start = interval_start + local_start
            with self.media_preparer.prepare(
                video_path=request.video_path,
                audio_path=request.audio_path,
                start_seconds=absolute_start,
                end_seconds=absolute_start + WINDOW_SECONDS,
                include_video=True,
                include_audio=False,
            ) as media:
                if media.video_path is None:
                    raise RuntimeError(
                        "media preparer did not produce an AV video file"
                    )
                raw.append(
                    float(
                        self.backend.predict_offset(
                            media.video_path,
                            start_seconds=0.0,
                            duration_seconds=WINDOW_SECONDS,
                        )
                    )
                )
        signed = list(raw)
        absolute = [abs(value) for value in raw]
        absolute_intervals = [
            [
                interval_start + start,
                interval_start + start + WINDOW_SECONDS,
            ]
            for start in local_starts
        ]
        endpoint_details = [
            {
                "endpoint": endpoint,
                "local_start_seconds": local_start,
                "sample_interval_seconds": absolute_interval,
                "raw_offset_seconds": raw_value,
                "signed_offset_seconds": signed_value,
                "absolute_offset_seconds": absolute_value,
            }
            for (
                endpoint,
                local_start,
                absolute_interval,
                raw_value,
                signed_value,
                absolute_value,
            ) in zip(
                ("first", "last"),
                local_starts,
                absolute_intervals,
                raw,
                signed,
                absolute,
                strict=True,
            )
        ]
        backend_manifest = getattr(self.backend, "manifest", None)
        manifest = (
            dict(backend_manifest())
            if callable(backend_manifest)
            else self.explicit_manifest
        )
        return WorkerResult.ok(
            request,
            scores={
                "first_signed_offset_seconds": signed[0],
                "last_signed_offset_seconds": signed[1],
                "first_abs_offset_seconds": absolute[0],
                "last_abs_offset_seconds": absolute[1],
                "mean_abs_offset_seconds": sum(absolute) / 2.0,
            },
            artifacts={
                "sample_intervals_seconds": absolute_intervals,
                "endpoint_offset_details": endpoint_details,
                "sample_interval_seconds": [interval_start, interval_end],
                "interval_index": request.interval_index,
                "checkpoint_preprocessing_manifest": manifest,
            },
            protocol={
                "endpoint_windows": "first_and_last",
                "window_seconds": WINDOW_SECONDS,
                "offset_sign_convention": (
                    "positive means audio starts earlier than video"
                ),
                "reported_offsets": ["signed", "absolute"],
                "raw_offset_field": "predicted_class_grid_seconds",
                "input_isolation": "endpoint_windows_to_temporary_av_files",
            },
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
    config_path: str,
    checkpoint: str,
    device: str = "cuda",
) -> list[str]:
    return [
        python,
        "-m",
        "streamav_eval.workers.objective.av_synchronization",
        "--source-dir",
        source_dir,
        "--config",
        config_path,
        "--checkpoint",
        checkpoint,
        "--device",
        device,
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    return serve_jsonl(
        lambda: AVSynchronizationWorker(
            source_dir=args.source_dir,
            config_path=args.config,
            checkpoint=args.checkpoint,
            device=args.device,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
