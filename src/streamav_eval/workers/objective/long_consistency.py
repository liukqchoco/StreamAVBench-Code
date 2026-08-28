"""Full-rollout subject and background consistency evaluator."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from ..protocol import WorkerContractError, WorkerRequest, WorkerResult

DIMENSIONS = {
    "SC": "subject_consistency",
    "BC": "background_consistency",
}
FULL_ROLLOUT_SECONDS = 180.0
EVALUATION_FPS = 2.0
MAX_DIAGNOSTIC_CHARS = 8_000


class ConsistencyRunner:
    """Invoke the configured long-video consistency evaluator."""

    def __init__(
        self,
        source_dir: str,
        *,
        entrypoint_path: str,
        full_json_path: str,
        python: str = "python",
        asset_root: str | None = None,
        timeout_seconds: float | None = None,
        evaluation_fps: float = EVALUATION_FPS,
        ffmpeg: str = "ffmpeg",
        process_runner: Callable[
            ..., subprocess.CompletedProcess[str]
        ] = subprocess.run,
        media_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.source_dir = Path(source_dir).resolve()
        self.python = python
        self.entrypoint_path = Path(entrypoint_path).resolve()
        self.full_json_path = Path(full_json_path).resolve()
        self.timeout_seconds = timeout_seconds
        if not math.isfinite(evaluation_fps) or evaluation_fps <= 0:
            raise ValueError("evaluation_fps must be positive and finite")
        self.evaluation_fps = float(evaluation_fps)
        self.ffmpeg = ffmpeg
        self.asset_root = Path(asset_root).resolve() if asset_root else None
        self.process_runner = process_runner
        self.media_runner = media_runner

    def evaluate(
        self,
        video_path: str,
        dimension: str,
    ) -> tuple[float, dict[str, Any]]:
        return self.evaluate_many([video_path], dimension)[0]

    def evaluate_many(
        self,
        video_paths: Sequence[str],
        dimension: str,
    ) -> list[tuple[float, dict[str, Any]]]:
        if not video_paths:
            return []
        if not self.entrypoint_path.is_file():
            raise FileNotFoundError(
                f"consistency evaluator entry point not found: {self.entrypoint_path}"
            )
        if not self.full_json_path.is_file():
            raise FileNotFoundError(
                f"consistency metadata file not found: {self.full_json_path}"
            )
        source_videos = [Path(video_path).resolve() for video_path in video_paths]
        for source_video in source_videos:
            if not source_video.is_file():
                raise FileNotFoundError(f"video not found: {source_video}")
        environment = os.environ.copy()
        if self.asset_root is not None:
            _require_offline_assets(self.asset_root, dimension)
            environment.update(_offline_environment(self.asset_root))
        with tempfile.TemporaryDirectory(prefix="streamav-consistency-") as temporary:
            root = Path(temporary)
            videos = root / "videos"
            output = root / "results"
            videos.mkdir()
            output.mkdir()
            sampled_names: list[str] = []
            for index, source_video in enumerate(source_videos):
                sampled_name = f"case{index:06d}.mp4"
                self._sample_video(source_video, videos / sampled_name)
                sampled_names.append(sampled_name)
            command = [
                self.python,
                str(self.entrypoint_path),
                "--videos_path",
                str(videos),
                "--dimension",
                dimension,
                "--mode",
                "long_custom_input",
                "--dev_flag",
                "--load_ckpt_from_local",
                "True",
                "--output_path",
                str(output),
                "--full_json_dir",
                str(self.full_json_path),
            ]
            completed = self.process_runner(
                command,
                cwd=str(self.source_dir),
                text=True,
                capture_output=True,
                check=False,
                env=environment,
                timeout=self.timeout_seconds,
            )
            if completed.returncode:
                message = _diagnostic_tail(
                    completed.stderr.strip() or completed.stdout.strip()
                )
                raise RuntimeError(
                    "consistency evaluator exited with "
                    f"{completed.returncode}: {message}"
                )
            result_files = sorted(output.glob("*_eval_results.json"))
            if len(result_files) != 1:
                raise RuntimeError(
                    "consistency evaluator must emit exactly one result file"
                )
            payload = json.loads(result_files[0].read_text(encoding="utf-8"))
            aggregate = _aggregate_score(payload, dimension)
            details = _detailed_scores(payload, dimension)
            results: list[tuple[float, dict[str, Any]]] = []
            for sampled_name in sampled_names:
                detail = details.get(Path(sampled_name).stem)
                if detail is None:
                    raise RuntimeError(f"{dimension} result lacks {sampled_name}")
                score = _finite_score(detail.get("video_results"), dimension)
                results.append(
                    (
                        score,
                        {
                            "result": detail,
                            "batch_score": aggregate,
                            "result_filename": result_files[0].name,
                            "stdout_tail": _diagnostic_tail(completed.stdout),
                            "input_sampling_fps": self.evaluation_fps,
                            "batch_size": len(sampled_names),
                        },
                    )
                )
            return results

    def _sample_video(self, source: Path, destination: Path) -> None:
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            f"fps={self.evaluation_fps:g}",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
        completed = self.media_runner(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=self.timeout_seconds,
        )
        if completed.returncode or not destination.is_file():
            message = _diagnostic_tail(
                completed.stderr.strip() or completed.stdout.strip()
            )
            raise RuntimeError(
                "failed to standardize SC/BC input to "
                f"{self.evaluation_fps:g} FPS: {message}"
            )


class LongConsistencyWorker:
    def __init__(self, metric: str, *, backend: ConsistencyRunner) -> None:
        if metric not in DIMENSIONS:
            raise ValueError(f"consistency metric must be SC or BC, got {metric!r}")
        self.metric = metric
        self.backend = backend

    def evaluate(self, request: WorkerRequest) -> WorkerResult:
        return self.evaluate_many([request])[0]

    def evaluate_many(
        self,
        requests: Sequence[WorkerRequest],
    ) -> list[WorkerResult]:
        if not requests:
            return []
        for request in requests:
            self._validate_request(request)
        dimension = DIMENSIONS[self.metric]
        evaluated = self.backend.evaluate_many(
            [request.require_video() for request in requests],
            dimension,
        )
        if len(evaluated) != len(requests):
            raise RuntimeError(
                f"{dimension} returned {len(evaluated)} results "
                f"for {len(requests)} requests"
            )
        return [
            WorkerResult.ok(
                request,
                scores={dimension: score},
                artifacts=artifacts,
                protocol={
                    "implementation": "full_rollout_consistency",
                    "dimension": dimension,
                    "mode": "long_custom_input",
                    "duration_seconds": FULL_ROLLOUT_SECONDS,
                    "input_sampling_fps": self.backend.evaluation_fps,
                    "sampling": (
                        "uniform temporal sampling before slow-fast clip "
                        "construction, feature extraction, and fusion"
                    ),
                    "batched_invocation": True,
                },
            )
            for request, (score, artifacts) in zip(
                requests,
                evaluated,
                strict=True,
            )
        ]

    def _validate_request(self, request: WorkerRequest) -> None:
        if request.metric != self.metric:
            raise WorkerContractError(
                f"worker handles {self.metric!r}, got {request.metric!r}"
            )
        if request.start_seconds not in (None, 0, 0.0):
            raise WorkerContractError(
                "SC/BC requires the full rollout from zero seconds"
            )
        end = (
            request.end_seconds
            if request.end_seconds is not None
            else request.duration_seconds
        )
        if end is None or not math.isclose(
            end,
            FULL_ROLLOUT_SECONDS,
            abs_tol=1e-6,
        ):
            raise WorkerContractError("SC/BC requires the full 180-second rollout")
        if request.duration_seconds is not None and not math.isclose(
            request.duration_seconds,
            FULL_ROLLOUT_SECONDS,
            abs_tol=1e-6,
        ):
            raise WorkerContractError("SC/BC duration_seconds must be 180")
        request.require_video()


def _aggregate_score(payload: Mapping[str, Any], dimension: str) -> float:
    if not isinstance(payload, Mapping) or dimension not in payload:
        raise RuntimeError(f"result lacks dimension {dimension!r}")
    value = payload[dimension]
    aggregate = value[0] if isinstance(value, list) and value else value
    if isinstance(aggregate, bool) or not isinstance(
        aggregate,
        (int, float),
    ):
        raise RuntimeError(f"{dimension} aggregate is not numeric: {aggregate!r}")
    return _finite_score(aggregate, dimension)


def _finite_score(value: Any, dimension: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{dimension} score is not numeric: {value!r}")
    score = float(value)
    if not math.isfinite(score):
        raise RuntimeError(f"{dimension} aggregate is not finite")
    return score


def _detailed_scores(
    payload: Mapping[str, Any],
    dimension: str,
) -> dict[str, Mapping[str, Any]]:
    value = payload.get(dimension)
    if not isinstance(value, list) or len(value) < 2 or not isinstance(value[1], list):
        raise RuntimeError(f"{dimension} result lacks per-video details")
    result: dict[str, Mapping[str, Any]] = {}
    for item in value[1]:
        if not isinstance(item, Mapping):
            continue
        video_path = item.get("video_path")
        if isinstance(video_path, str):
            result[Path(video_path).stem] = item
    return result


def _diagnostic_tail(value: str) -> str:
    if len(value) <= MAX_DIAGNOSTIC_CHARS:
        return value
    omitted = len(value) - MAX_DIAGNOSTIC_CHARS
    return f"[... {omitted} characters omitted ...]\n{value[-MAX_DIAGNOSTIC_CHARS:]}"


def _offline_environment(asset_root: Path) -> dict[str, str]:
    root = asset_root / "long-consistency"
    return {
        "HOME": str(root / "home"),
        "TORCH_HOME": str(root / "torch"),
        "VBENCH_CACHE_DIR": str(root / "home" / ".cache" / "vbench"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def _require_offline_assets(asset_root: Path, dimension: str) -> None:
    root = asset_root / "long-consistency"
    cache = root / "home" / ".cache"
    if dimension == "subject_consistency":
        required = (
            cache
            / "vbench"
            / "dino_model"
            / "facebookresearch_dino_main",
            cache / "vbench" / "dino_model" / "dino_vitbase16_pretrain.pth",
            root / "torch" / "hub" / "facebookresearch_dinov2_main",
            root
            / "torch"
            / "hub"
            / "checkpoints"
            / "dinov2_vitb14_pretrain.pth",
        )
    else:
        required = (
            cache / "vbench" / "clip_model" / "ViT-B-32.pt",
            cache / "facebookresearch_dino_main",
            cache / "checkpoints" / "dino_vitbase16_pretrain.pth",
            cache / "dino_vitb16_pretrain.pth",
            cache / "open_clip_vitb16_pretrain.pth.tar",
            cache / "clip_vitb16_pretrain.pth.tar",
            cache / "ensemble_lora",
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "long-consistency offline assets are incomplete: " + ", ".join(missing)
        )


def serve_batched_jsonl(
    worker: LongConsistencyWorker,
    *,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> int:
    requests: list[WorkerRequest] = []
    exit_code = 0
    for raw in input_stream:
        if not raw.strip():
            continue
        try:
            requests.append(WorkerRequest.from_json(raw))
        except Exception as exc:
            exit_code = 1
            invalid = WorkerRequest(
                request_id="<invalid>",
                metric=worker.metric,
            )
            output_stream.write(WorkerResult.failed(invalid, exc).to_json() + "\n")
    if requests:
        try:
            results = worker.evaluate_many(requests)
        except Exception as exc:
            exit_code = 1
            results = [WorkerResult.failed(request, exc) for request in requests]
        for result in results:
            output_stream.write(result.to_json() + "\n")
    output_stream.flush()
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", required=True, choices=sorted(DIMENSIONS))
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--full-json-path", required=True)
    parser.add_argument("--asset-root")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument(
        "--evaluation-fps",
        type=float,
        default=EVALUATION_FPS,
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args(argv)
    return serve_batched_jsonl(
        LongConsistencyWorker(
            args.metric,
            backend=ConsistencyRunner(
                args.source_dir,
                entrypoint_path=args.entrypoint,
                full_json_path=args.full_json_path,
                python=args.python,
                asset_root=args.asset_root,
                timeout_seconds=args.timeout_seconds,
                evaluation_fps=args.evaluation_fps,
                ffmpeg=args.ffmpeg,
            ),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
