"""Isolated JSONL MLLM worker backed by the guarded Gemini client."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, TextIO

from streamav_eval.gemini import (
    GeminiClient,
    GeminiConfig,
    GeminiKeyPool,
    GeminiResult,
    InlineMedia,
    RequestTooLargeError,
    require_live_transport,
)
from streamav_eval.workers.protocol import WorkerRequest, WorkerResult

from .media import build_media, compact_inline_preview
from .prompts import FrozenPromptLoader
from .routing import (
    FIXED_USER_PROMPT,
    MediaMode,
    MLLMMetric,
    route_for,
    route_protocol_sha256,
)
from .schemas import require_exact_keys
from .scoring import (
    score_quality,
    score_shared_instruction,
)

DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "resources" / "prompts"
DEFAULT_KEYS_PATH = Path.cwd() / ".secrets" / "api_keys.txt"
DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"
QUALITY_USER_PROMPT = FIXED_USER_PROMPT
INLINE_MEDIA_RAW_BUDGET_BYTES = 14 * 1024 * 1024
CRITERION_METRICS = {
    MLLMMetric.VIF,
    MLLMMetric.AIF,
    MLLMMetric.VID_EARLY,
    MLLMMetric.VID_LATE,
    MLLMMetric.AID_EARLY,
    MLLMMetric.AID_LATE,
    MLLMMetric.P0_VIF,
    MLLMMetric.P0_AIF,
}
INTERACTIVE_METRICS = {
    MLLMMetric.VUF,
    MLLMMetric.AUF,
    MLLMMetric.VSR,
    MLLMMetric.ASR,
    MLLMMetric.HDF_ADJ,
    MLLMMetric.HDF_LR,
    MLLMMetric.PVC,
    MLLMMetric.PAC,
}


class JsonClient(Protocol):
    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: Mapping[str, Any],
        media: tuple[InlineMedia, ...] = (),
    ) -> GeminiResult: ...


ClientFactory = Callable[[GeminiConfig], JsonClient]
MediaBuilder = Callable[..., Path]


class MLLMWorker:
    """Evaluates every StreamAV MLLM route from one isolated process."""

    metric = "mllm"

    def __init__(
        self,
        *,
        client: JsonClient | None = None,
        client_factory: ClientFactory = GeminiClient,
        key_pool: GeminiKeyPool | None = None,
        prompts_dir: str | Path = DEFAULT_PROMPTS_DIR,
        media_builder: MediaBuilder = build_media,
    ) -> None:
        self._injected_client = client
        self._client_factory = client_factory
        self._key_pool = key_pool
        self._prompts = FrozenPromptLoader(prompts_dir)
        self._media_builder = media_builder

    def evaluate(self, request: WorkerRequest) -> WorkerResult:
        route = route_for(request.metric)
        expected_protocol = request.options.get("evaluator_protocol_sha256")
        actual_protocol = route_protocol_sha256(
            route.metric,
            prompts_dir=self._prompts.prompts_dir,
        )
        if expected_protocol != actual_protocol:
            raise ValueError(
                "request evaluator_protocol_sha256 does not match the frozen "
                f"{request.metric} prompt and schema"
            )
        if route.media_mode is MediaMode.NONE:
            raise ValueError(
                "checklist routes are internal and cannot be worker metrics"
            )
        if route.metric in CRITERION_METRICS and request.options.get("criteria") == []:
            return WorkerResult(
                request_id=request.request_id,
                metric=request.metric,
                status="not_applicable",
                artifacts={"benchmark_defined": True},
                protocol={
                    "model": _gemini_model(),
                    "media_mode": route.media_mode.value,
                    "reason": "no benchmark-applicable criterion",
                },
            )
        client, key_slot = self._client_for(request.request_id)
        start_seconds, end_seconds = _request_range(request, route.expected_duration_s)
        with tempfile.TemporaryDirectory(prefix="streamav-mllm-") as temporary:
            media = self._build_media_inputs(
                request,
                route.metric,
                Path(temporary),
                start_seconds,
                end_seconds,
            )
            if route.metric in CRITERION_METRICS:
                return self._criterion_result(
                    client,
                    request,
                    route.metric,
                    media,
                    start_seconds,
                    end_seconds,
                    key_slot,
                )
            if route.metric in INTERACTIVE_METRICS:
                return self._interactive_result(
                    client,
                    request,
                    route.metric,
                    media,
                    start_seconds,
                    end_seconds,
                    key_slot,
                )
            if route.metric in {MLLMMetric.VQ, MLLMMetric.AQ}:
                result = self._quality_response(client, route.metric, media)
                response = _result_value(result)
                scored = score_quality(route.metric.value, response)
            else:
                raise AssertionError(f"unhandled MLLM route {route.metric.value}")

        scores = {
            name: float(value)
            for name, value in scored["dimensions"].items()
            if value is not None
        }
        if scored["mean"] is not None:
            scores["mean"] = float(scored["mean"])
        artifacts: dict[str, Any] = {
            "response": dict(response),
            "gemini": _gemini_artifact(result),
        }
        return WorkerResult.ok(
            request,
            scores=scores,
            artifacts=artifacts,
            protocol={
                "model": _gemini_model(),
                "media_mode": route.media_mode.value,
                "sample_fps": 2,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "encoded_duration_seconds": route.expected_duration_s,
                "key_slot": key_slot,
            },
        )

    def _build_media_inputs(
        self,
        request: WorkerRequest,
        metric: MLLMMetric,
        temporary: Path,
        start_seconds: float,
        end_seconds: float,
    ) -> tuple[InlineMedia, ...]:
        route = route_for(metric)
        windows = request.options.get("media_windows")
        if windows is None:
            windows = [
                {
                    "role": "current",
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                }
            ]
        if not isinstance(windows, list) or not windows:
            raise ValueError("options.media_windows must be a non-empty array")
        media_paths: list[Path] = []
        suffix = ".m4a" if route.media_mode is MediaMode.AUDIO_ONLY else ".mp4"
        mime_type = (
            "audio/mp4" if route.media_mode is MediaMode.AUDIO_ONLY else "video/mp4"
        )
        preview_fps = (
            None
            if route.media_mode is MediaMode.AUDIO_ONLY
            else 6.0
            if metric is MLLMMetric.PVC
            else 2.0
        )
        for index, window in enumerate(windows):
            if not isinstance(window, Mapping):
                raise ValueError("each media window must be an object")
            start, end = window.get("start_seconds"), window.get("end_seconds")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                raise ValueError("media window bounds must be numeric")
            path = temporary / f"{index:02d}-{window.get('role', 'clip')}{suffix}"
            self._media_builder(
                metric,
                request.require_video(),
                path,
                audio_source=request.audio_path,
                start_seconds=float(start),
                end_seconds=float(end),
            )
            media_paths.append(path)
        if (
            route.media_mode is not MediaMode.AUDIO_ONLY
            and sum(path.stat().st_size for path in media_paths)
            > INLINE_MEDIA_RAW_BUDGET_BYTES
        ):
            compact_paths = []
            for index, path in enumerate(media_paths):
                compact_paths.append(
                    compact_inline_preview(
                        path,
                        temporary / f"{index:02d}-compact.mp4",
                        synchronized_av=(route.media_mode is MediaMode.SYNCHRONIZED_AV),
                        fps=preview_fps or 2.0,
                    )
                )
            media_paths = compact_paths
            compact_size = sum(path.stat().st_size for path in media_paths)
            if compact_size > INLINE_MEDIA_RAW_BUDGET_BYTES:
                raise RequestTooLargeError(
                    "compacted inline media is "
                    f"{compact_size} bytes; limit is "
                    f"{INLINE_MEDIA_RAW_BUDGET_BYTES} bytes"
                )
        media = [
            InlineMedia.from_path(
                path,
                mime_type=mime_type,
                video_fps=preview_fps,
            )
            for path in media_paths
        ]
        return tuple(media)

    def _criterion_result(
        self,
        client: JsonClient,
        request: WorkerRequest,
        metric: MLLMMetric,
        media: tuple[InlineMedia, ...],
        start_seconds: float,
        end_seconds: float,
        key_slot: int | None,
    ) -> WorkerResult:
        case_id = request.options.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("shared instruction metrics require options.case_id")
        supplied = request.options.get("criteria")
        if not isinstance(supplied, list):
            raise ValueError("shared instruction metrics require options.criteria")
        criteria = [dict(item) for item in supplied if isinstance(item, Mapping)]
        if len(criteria) != len(supplied):
            raise ValueError("options.criteria must contain only objects")
        if not criteria:
            raise ValueError(f"{case_id} has no applicable instruction criteria")
        route = route_for(metric)
        system_prompt = self._prompts.render(
            route.system_prompt_file,
            {
                "criteria": json.dumps(
                    criteria, ensure_ascii=False, sort_keys=True, indent=2
                )
            },
        )
        result = client.generate_json(
            system_prompt=system_prompt,
            user_prompt=QUALITY_USER_PROMPT,
            response_schema=route.response_schema,
            media=media,
        )
        response = _result_value(result)
        scored = score_shared_instruction(response, criteria)
        means = scored["modality_means"]
        modality = _criterion_modality(metric)
        mean = means[modality]
        if mean is None:
            raise ValueError(f"{metric.value} has no applicable criteria")
        scores = {"mean": float(mean)}
        return WorkerResult.ok(
            request,
            scores=scores,
            artifacts={
                "response": dict(response),
                "criterion_scores": scored["criteria"],
                "gemini": _gemini_artifact(result),
            },
            protocol={
                "model": _gemini_model(),
                "media_mode": route.media_mode.value,
                "sample_fps": 2,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "encoded_duration_seconds": route.expected_duration_s,
                "key_slot": key_slot,
                "shared_criterion_call": False,
                "criterion_modality": _criterion_modality(metric),
            },
        )

    def _interactive_result(
        self,
        client: JsonClient,
        request: WorkerRequest,
        metric: MLLMMetric,
        media: tuple[InlineMedia, ...],
        start_seconds: float,
        end_seconds: float,
        key_slot: int | None,
    ) -> WorkerResult:
        route = route_for(metric)
        replacements = {
            name: str(request.options[name])
            for name in (
                "visual_update",
                "audio_update",
                "current_update",
                "source_prompt",
            )
            if name in request.options
        }
        system_prompt = self._prompts.render(route.system_prompt_file, replacements)
        result = client.generate_json(
            system_prompt=system_prompt,
            user_prompt=QUALITY_USER_PROMPT,
            response_schema=route.response_schema,
            media=media,
        )
        response = _result_value(result)
        protocol = {
            "model": _gemini_model(),
            "media_mode": route.media_mode.value,
            "sample_fps": (
                None
                if route.media_mode is MediaMode.AUDIO_ONLY
                else 6
                if metric is MLLMMetric.PVC
                else 2
            ),
            "sampled_timestamp_step_seconds": (
                0.5 if metric in {MLLMMetric.VUF, MLLMMetric.AUF} else None
            ),
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "key_slot": key_slot,
        }
        try:
            scores, artifacts, not_applicable = _score_interactive(
                metric,
                response,
                sampled_timestamps=request.options.get("sampled_timestamps_seconds"),
            )
            artifacts["gemini"] = _gemini_artifact(result)
        except (KeyError, TypeError, ValueError) as exc:
            return WorkerResult(
                request_id=request.request_id,
                metric=request.metric,
                status="error",
                artifacts={
                    "response": dict(response),
                    "gemini": _gemini_artifact(result),
                },
                protocol=protocol,
                error={"type": type(exc).__name__, "message": str(exc)},
            )
        if not_applicable:
            return WorkerResult(
                request_id=request.request_id,
                metric=request.metric,
                status="not_applicable",
                artifacts=artifacts,
                protocol=protocol,
            )
        return WorkerResult.ok(
            request,
            scores=scores,
            artifacts=artifacts,
            protocol=protocol,
        )

    def _quality_response(
        self,
        client: JsonClient,
        metric: MLLMMetric,
        media: tuple[InlineMedia, ...],
    ) -> GeminiResult:
        route = route_for(metric)
        result = client.generate_json(
            system_prompt=self._prompts.read(route.system_prompt_file),
            user_prompt=QUALITY_USER_PROMPT,
            response_schema=route.response_schema,
            media=media,
        )
        return result

    def _client_for(self, request_id: str) -> tuple[JsonClient, int | None]:
        if self._injected_client is not None:
            return self._injected_client, None
        require_live_transport()
        pool = self._key_pool or GeminiKeyPool.from_environment(
            default_path=DEFAULT_KEYS_PATH
        )
        selection = pool.for_work(request_id)
        model = _gemini_model()
        sse = os.environ.get("STREAMAV_GEMINI_SSE") == "1"
        config = GeminiConfig(
            api_key=selection.key,
            model=model,
            sse=sse,
        )
        return self._client_factory(config), selection.slot


def _gemini_model() -> str:
    return os.environ.get("STREAMAV_GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def _criterion_modality(metric: MLLMMetric) -> str | None:
    if metric in {
        MLLMMetric.VIF,
        MLLMMetric.VID_EARLY,
        MLLMMetric.VID_LATE,
        MLLMMetric.P0_VIF,
    }:
        return "video"
    if metric in {
        MLLMMetric.AIF,
        MLLMMetric.AID_EARLY,
        MLLMMetric.AID_LATE,
        MLLMMetric.P0_AIF,
    }:
        return "audio"
    return None


def _score_interactive(
    metric: MLLMMetric,
    response: Mapping[str, Any],
    *,
    sampled_timestamps: Any = None,
) -> tuple[dict[str, float], dict[str, Any], bool]:
    artifacts: dict[str, Any] = {"response": dict(response)}
    if metric in {MLLMMetric.VUF, MLLMMetric.AUF}:
        expected = metric.value
        require_exact_keys(
            response,
            (
                "metric",
                "started",
                "fulfillment_score",
                "onset_latency_s",
                "target_achievement_latency_s",
                "evidence",
                "reason",
            ),
            context=expected,
        )
        if response.get("metric") != expected:
            raise ValueError(f"{expected} response metric must equal {expected!r}")
        started = response.get("started")
        if not isinstance(started, bool):
            raise ValueError(f"{expected}.started must be boolean")
        allowed_timestamps = _timestamp_grid(sampled_timestamps)
        fulfillment = _one_to_five(response.get("fulfillment_score"), expected)
        onset = _latency(
            response.get("onset_latency_s"),
            "onset_latency_s",
            allowed_timestamps,
        )
        target = _latency(
            response.get("target_achievement_latency_s"),
            "target_achievement_latency_s",
            allowed_timestamps,
        )
        if not started and (onset is not None or target is not None):
            raise ValueError("non-started updates must have null latency fields")
        if started and onset is None:
            raise ValueError("started updates require onset_latency_s")
        if target is not None and onset is not None and target < onset:
            raise ValueError("target achievement cannot precede onset")
        if target is not None and fulfillment < 3.0:
            raise ValueError(
                "target achievement requires fulfillment_score of at least 3"
            )
        _nonempty_string(response.get("evidence"), f"{expected}.evidence")
        _nonempty_string(response.get("reason"), f"{expected}.reason")
        scores = {"fulfillment_score": fulfillment}
        if onset is not None:
            scores["onset_latency_s"] = onset
        if target is not None:
            scores["target_achievement_latency_s"] = target
        artifacts["timing"] = {
            "started": started,
            "onset_latency_s": onset,
            "target_achievement_latency_s": target,
        }
        return scores, artifacts, False
    if metric is MLLMMetric.VSR:
        require_exact_keys(
            response,
            ("subject_retention", "environment_retention"),
            context="VSR",
        )
        values: dict[str, float] = {}
        for field in ("subject_retention", "environment_retention"):
            item = response.get(field)
            if not isinstance(item, Mapping):
                raise ValueError(f"VSR.{field} must be an object")
            require_exact_keys(item, ("score", "reason"), context=f"VSR.{field}")
            _nonempty_string(item.get("reason"), f"VSR.{field}.reason")
            values[field] = _one_to_five(item.get("score"), f"VSR.{field}")
        values["mean"] = sum(values.values()) / len(values)
        return values, artifacts, False
    if metric is MLLMMetric.ASR:
        require_exact_keys(response, ("audio_retention_score", "reason"), context="ASR")
        _nonempty_string(response.get("reason"), "ASR.reason")
        return (
            {
                "mean": _one_to_five(
                    response.get("audio_retention_score"),
                    "ASR.audio_retention_score",
                )
            },
            artifacts,
            False,
        )
    if metric in {MLLMMetric.HDF_ADJ, MLLMMetric.HDF_LR}:
        require_exact_keys(
            response,
            ("source_state_established", "dependency_following_score", "reason"),
            context="HDF",
        )
        _nonempty_string(response.get("reason"), "HDF.reason")
        established = response.get("source_state_established")
        score = response.get("dependency_following_score")
        if not isinstance(established, bool):
            raise ValueError("HDF.source_state_established must be boolean")
        if not established:
            if score is not None:
                raise ValueError("unestablished HDF source requires null score")
            artifacts["source_state_established"] = False
            return {}, artifacts, True
        if score is None:
            raise ValueError("established HDF source requires a score")
        artifacts["source_state_established"] = True
        return {"mean": _one_to_five(score, "HDF")}, artifacts, False
    if metric is MLLMMetric.PVC:
        require_exact_keys(
            response,
            (
                "score",
                "generation_break",
                "deformation",
                "object_disappear",
                "reason",
            ),
            context="PVC",
        )
        for field in ("generation_break", "deformation", "object_disappear"):
            if not isinstance(response.get(field), bool):
                raise ValueError(f"PVC.{field} must be boolean")
        _nonempty_string(response.get("reason"), "PVC.reason")
        return (
            {"mllm_score": _one_to_five(response.get("score"), "PVC")},
            artifacts,
            False,
        )
    if metric is MLLMMetric.PAC:
        require_exact_keys(
            response,
            ("score", "audio_break", "audio_artifact", "reason"),
            context="PAC",
        )
        for field in ("audio_break", "audio_artifact"):
            if not isinstance(response.get(field), bool):
                raise ValueError(f"PAC.{field} must be boolean")
        _nonempty_string(response.get("reason"), "PAC.reason")
        return {"score": _one_to_five(response.get("score"), "PAC")}, artifacts, False
    raise ValueError(f"unsupported Interactive metric {metric.value}")


def _one_to_five(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in range(1, 6)
    ):
        raise ValueError(f"{field} must be an integer from 1 to 5")
    return float(value)


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _timestamp_grid(value: Any) -> tuple[float, ...]:
    if value is None:
        return tuple(index / 2 for index in range(60))
    if not isinstance(value, list) or not value:
        raise ValueError("sampled_timestamps_seconds must be a non-empty array")
    result = tuple(float(item) for item in value)
    if result != tuple(sorted(set(result))):
        raise ValueError("sampled_timestamps_seconds must be sorted and unique")
    return result


def _latency(
    value: Any, field: str, allowed_timestamps: tuple[float, ...]
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric or null")
    result = float(value)
    if not any(abs(result - timestamp) <= 1e-9 for timestamp in allowed_timestamps):
        raise ValueError(f"{field} must equal an observed 0.5s sampled timestamp")
    return result


def serve_mllm_jsonl(
    worker: MLLMWorker,
    *,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> int:
    """Serve independent MLLM requests and always emit one JSON result per line."""

    exit_code = 0
    for raw in input_stream:
        if not raw.strip():
            continue
        request: WorkerRequest | None = None
        try:
            request = WorkerRequest.from_json(raw)
            result = worker.evaluate(request)
        except Exception as exc:
            exit_code = 1
            request = request or WorkerRequest("<invalid>", "<unknown>")
            result = WorkerResult.failed(request, exc)
        output_stream.write(result.to_json() + "\n")
        output_stream.flush()
    return exit_code


def main() -> int:
    return serve_mllm_jsonl(MLLMWorker())


def _result_value(result: GeminiResult) -> Mapping[str, Any]:
    value = result.value
    if not isinstance(value, Mapping):
        raise ValueError("Gemini JSON result must be an object")
    return value


def _gemini_artifact(result: Any) -> dict[str, Any]:
    usage = getattr(result, "usage", {})
    return {
        "request_id": getattr(result, "request_id", None),
        "model_version": getattr(result, "model_version", None),
        "attempts": int(getattr(result, "attempts", 1)),
        "usage": dict(usage) if isinstance(usage, Mapping) else {},
    }


def _request_range(
    request: WorkerRequest, expected_duration_s: int | None
) -> tuple[float, float]:
    if expected_duration_s is None:
        raise ValueError(f"{request.metric} has no media duration")
    return request.require_interval_duration(float(expected_duration_s))


if __name__ == "__main__":
    raise SystemExit(main())
