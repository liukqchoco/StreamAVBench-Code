"""Typed, side-effect-free contracts shared by the orchestrator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """Raised when an input violates an evaluation contract."""


class Track(str, Enum):
    PROGRESSIVE = "progressive"
    INTERACTIVE = "interactive"

    @classmethod
    def parse(cls, value: Track | str) -> Track:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ContractError("track must be a string")
        try:
            return cls(value.strip().lower())
        except ValueError as exc:
            raise ContractError(f"unknown track {value!r}") from exc


class UnitKind(str, Enum):
    SEGMENT = "segment"
    INSTRUCTION_FOLLOWING = "instruction_following"


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    model_id: str
    case_id: str
    video_path: Path
    audio_path: Path | None = None
    source_line: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        model_id = self.model_id.strip()
        if not model_id:
            raise ContractError("model_id must be non-empty")
        if model_id[0] in "=+-@" or any(
            ord(character) < 32 or ord(character) == 127 for character in model_id
        ):
            raise ContractError(
                "model_id must not begin with a spreadsheet formula prefix or "
                "contain control characters"
            )
        if not self.case_id.strip():
            raise ContractError("case_id must be non-empty")
        if not str(self.video_path):
            raise ContractError("video_path must be non-empty")


@dataclass(frozen=True, slots=True)
class Prompt:
    prompt_id: str
    text: str
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RegistryCase:
    case_id: str
    track: Track
    duration_seconds: float
    prompts: tuple[Prompt, ...]
    source_path: Path
    source: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class MediaMetadata:
    path: Path
    container_duration: float
    video_duration: float
    audio_duration: float | None
    video_start_pts: float
    audio_start_pts: float | None

    def __post_init__(self) -> None:
        durations = {
            "container_duration": self.container_duration,
            "video_duration": self.video_duration,
            "audio_duration": self.audio_duration,
        }
        for name, value in durations.items():
            if value is not None and value <= 0:
                raise ContractError(f"{name} must be positive, got {value}")


@dataclass(frozen=True, slots=True)
class EvaluationUnit:
    unit_id: str
    run_id: str
    model_id: str
    case_id: str
    track: Track
    kind: UnitKind
    start_seconds: float
    end_seconds: float
    prompt_ids: tuple[str, ...] = ()
    label: str = ""

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True, slots=True)
class MetricJob:
    """One executable metric request with reproducible input identity."""

    job_id: str
    run_id: str
    model_id: str
    case_id: str
    track: Track
    metric: str
    video_path: Path
    start_seconds: float
    end_seconds: float
    interval: int | None
    input_size_bytes: int
    input_mtime_ns: int
    prompt_ids: tuple[str, ...] = ()
    output_metrics: tuple[str, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)
    audio_path: Path | None = None
    audio_input_size_bytes: int | None = None
    audio_input_mtime_ns: int | None = None

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    def __post_init__(self) -> None:
        if self.end_seconds <= self.start_seconds:
            raise ContractError("metric job end_seconds must exceed start_seconds")
        if self.interval is not None and self.interval not in range(1, 7):
            raise ContractError("metric job interval must be in [1, 6]")
        if self.input_size_bytes < 0 or self.input_mtime_ns < 0:
            raise ContractError("input size and mtime metadata must be non-negative")
        for value, name in (
            (self.audio_input_size_bytes, "audio_input_size_bytes"),
            (self.audio_input_mtime_ns, "audio_input_mtime_ns"),
        ):
            if value is not None and value < 0:
                raise ContractError(f"{name} must be non-negative")
        if self.audio_path is None and (
            self.audio_input_size_bytes is not None
            or self.audio_input_mtime_ns is not None
        ):
            raise ContractError("audio fingerprint requires audio_path")
        if self.audio_path is not None and (
            self.audio_input_size_bytes is None or self.audio_input_mtime_ns is None
        ):
            raise ContractError("audio_path requires a complete audio fingerprint")
        if len(set(self.output_metrics)) != len(self.output_metrics):
            raise ContractError("metric job output_metrics must be unique")


@dataclass(frozen=True, slots=True)
class CanonicalResult:
    """Aggregation-ready representation of a worker result."""

    job_id: str
    model_id: str
    case_id: str
    track: Track
    metric: str
    interval: int | None
    score: float | None
    status: str
    error: Mapping[str, str] | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
