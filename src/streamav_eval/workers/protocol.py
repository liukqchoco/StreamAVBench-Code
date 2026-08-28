"""JSON request/result contract shared by isolated metric workers."""

from __future__ import annotations

import json
import math
import sys
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TextIO


class WorkerContractError(ValueError):
    """Raised when a worker request is malformed."""


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    request_id: str
    metric: str
    video_path: str | None = None
    audio_path: str | None = None
    duration_seconds: float | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    start_seconds: float = 0.0
    end_seconds: float | None = None
    interval_index: int = 0

    def __post_init__(self) -> None:
        _validate_request_fields(
            duration_seconds=self.duration_seconds,
            start_seconds=self.start_seconds,
            end_seconds=self.end_seconds,
            interval_index=self.interval_index,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WorkerRequest:
        if not isinstance(value, Mapping):
            raise WorkerContractError("request must be a JSON object")
        request_id = value.get("request_id")
        metric = value.get("metric")
        if not isinstance(request_id, str) or not request_id.strip():
            raise WorkerContractError("request_id must be a non-empty string")
        if not isinstance(metric, str) or not metric.strip():
            raise WorkerContractError("metric must be a non-empty string")
        options = value.get("options", {})
        if not isinstance(options, Mapping):
            raise WorkerContractError("options must be an object")
        duration = _optional_finite_number(
            value.get("duration_seconds"), "duration_seconds", positive=True
        )
        start = _optional_finite_number(
            value.get("start_seconds", 0.0), "start_seconds", nonnegative=True
        )
        end = _optional_finite_number(
            value.get("end_seconds"), "end_seconds", nonnegative=True
        )
        interval_index = value.get("interval_index", 0)
        _validate_request_fields(
            duration_seconds=duration,
            start_seconds=start if start is not None else 0.0,
            end_seconds=end,
            interval_index=interval_index,
        )
        video_path = _optional_path(value.get("video_path"), "video_path")
        audio_path = _optional_path(value.get("audio_path"), "audio_path")
        return cls(
            request_id=request_id.strip(),
            metric=metric.strip(),
            video_path=video_path,
            audio_path=audio_path,
            duration_seconds=duration,
            options=dict(options),
            start_seconds=start if start is not None else 0.0,
            end_seconds=end,
            interval_index=interval_index,
        )

    @classmethod
    def from_json(cls, raw: str) -> WorkerRequest:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise WorkerContractError(f"invalid request JSON: {exc.msg}") from exc
        return cls.from_dict(value)

    def require_video(self) -> str:
        if self.video_path is None:
            raise WorkerContractError(f"{self.metric} requires video_path")
        return self.video_path

    def require_audio(self) -> str:
        if self.audio_path is None:
            raise WorkerContractError(f"{self.metric} requires audio_path")
        return self.audio_path

    def interval(
        self, *, default_duration_seconds: float | None = None
    ) -> tuple[float, float]:
        """Return the canonical absolute half-open interval for this request."""
        end = self.end_seconds
        if end is None:
            if default_duration_seconds is not None:
                end = self.start_seconds + default_duration_seconds
            elif self.duration_seconds is not None:
                end = self.duration_seconds
            else:
                raise WorkerContractError(
                    "end_seconds is required when duration_seconds is unavailable"
                )
        if end <= self.start_seconds:
            raise WorkerContractError("end_seconds must be greater than start_seconds")
        if self.duration_seconds is not None and end > self.duration_seconds + 1e-9:
            raise WorkerContractError("end_seconds cannot exceed duration_seconds")
        return self.start_seconds, end

    def require_interval_duration(self, expected_seconds: float) -> tuple[float, float]:
        start, end = self.interval(default_duration_seconds=expected_seconds)
        if not math.isclose(end - start, expected_seconds, abs_tol=1e-6):
            raise WorkerContractError(
                f"requested interval must be exactly {expected_seconds:g} seconds"
            )
        return start, end

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "request_id": self.request_id,
            "metric": self.metric,
            "options": dict(self.options),
        }
        if self.video_path is not None:
            value["video_path"] = self.video_path
        if self.audio_path is not None:
            value["audio_path"] = self.audio_path
        if self.duration_seconds is not None:
            value["duration_seconds"] = self.duration_seconds
        if self.start_seconds != 0.0:
            value["start_seconds"] = self.start_seconds
        if self.end_seconds is not None:
            value["end_seconds"] = self.end_seconds
        if self.interval_index != 0:
            value["interval_index"] = self.interval_index
        return value

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class WorkerResult:
    request_id: str
    metric: str
    status: str
    scores: Mapping[str, float] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    protocol: Mapping[str, Any] = field(default_factory=dict)
    error: Mapping[str, str] | None = None

    @classmethod
    def ok(
        cls,
        request: WorkerRequest,
        *,
        scores: Mapping[str, float],
        artifacts: Mapping[str, Any] | None = None,
        protocol: Mapping[str, Any] | None = None,
    ) -> WorkerResult:
        return cls(
            request_id=request.request_id,
            metric=request.metric,
            status="ok",
            scores={key: float(value) for key, value in scores.items()},
            artifacts=dict(artifacts or {}),
            protocol=dict(protocol or {}),
        )

    @classmethod
    def failed(cls, request: WorkerRequest, exc: BaseException) -> WorkerResult:
        error: dict[str, Any] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        retryable = getattr(exc, "retryable", None)
        if isinstance(retryable, bool):
            error["retryable"] = retryable
        return cls(
            request_id=request.request_id,
            metric=request.metric,
            status="error",
            error=error,
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "request_id": self.request_id,
            "metric": self.metric,
            "status": self.status,
            "scores": dict(self.scores),
            "artifacts": dict(self.artifacts),
            "protocol": dict(self.protocol),
        }
        if self.error is not None:
            value["error"] = dict(self.error)
        return value

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


class ObjectiveWorker(Protocol):
    metric: str

    def evaluate(self, request: WorkerRequest) -> WorkerResult: ...


WorkerFactory = Callable[[], ObjectiveWorker]


def serve_jsonl(
    factory: WorkerFactory,
    *,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> int:
    """Run one lazily-created worker over newline-delimited JSON requests."""
    worker: ObjectiveWorker | None = None
    exit_code = 0
    for raw in input_stream:
        if not raw.strip():
            continue
        request: WorkerRequest | None = None
        try:
            request = WorkerRequest.from_json(raw)
            worker = worker or factory()
            if request.metric != worker.metric:
                raise WorkerContractError(
                    f"worker handles {worker.metric!r}, got {request.metric!r}"
                )
            result = worker.evaluate(request)
        except Exception as exc:  # Worker boundary must return JSON, never a traceback.
            exit_code = 1
            if request is None:
                request = WorkerRequest(
                    request_id="<invalid>",
                    metric=getattr(worker, "metric", "<unknown>"),
                )
            result = WorkerResult.failed(request, exc)
            if bool(request.options.get("debug")):
                result = WorkerResult(
                    request_id=result.request_id,
                    metric=result.metric,
                    status=result.status,
                    scores=result.scores,
                    artifacts={"traceback": traceback.format_exc()},
                    protocol=result.protocol,
                    error=result.error,
                )
        output_stream.write(result.to_json() + "\n")
        output_stream.flush()
    return exit_code


def command_for(module: str, *, python: str = "python") -> list[str]:
    """Build the subprocess command used by the orchestrator."""
    return [python, "-m", module]


def _optional_path(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise WorkerContractError(f"{field_name} must be a non-empty string")
    return str(Path(value).expanduser())


def _optional_finite_number(
    value: object,
    field_name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkerContractError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise WorkerContractError(f"{field_name} must be a finite number")
    if positive and result <= 0:
        raise WorkerContractError(f"{field_name} must be positive")
    if nonnegative and result < 0:
        raise WorkerContractError(f"{field_name} must be non-negative")
    return result


def _validate_request_fields(
    *,
    duration_seconds: object,
    start_seconds: object,
    end_seconds: object,
    interval_index: object,
) -> None:
    duration = _optional_finite_number(
        duration_seconds, "duration_seconds", positive=True
    )
    start = _optional_finite_number(start_seconds, "start_seconds", nonnegative=True)
    end = _optional_finite_number(end_seconds, "end_seconds", nonnegative=True)
    if start is None:
        raise WorkerContractError("start_seconds must be a finite number")
    if end is not None and end <= start:
        raise WorkerContractError("end_seconds must be greater than start_seconds")
    if duration is not None and end is not None and end > duration + 1e-9:
        raise WorkerContractError("end_seconds cannot exceed duration_seconds")
    if (
        isinstance(interval_index, bool)
        or not isinstance(interval_index, int)
        or interval_index < 0
    ):
        raise WorkerContractError("interval_index must be a non-negative integer")
