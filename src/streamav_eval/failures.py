"""Failure semantics for StreamAV-Bench result aggregation.

Generation failures and evaluator failures are deliberately different:
generation failures are terminal benchmark outcomes recorded as failure rates
and excluded from metric means, while evaluator failures are retryable
infrastructure problems that block final reporting.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum


class ResultStatus(str, Enum):
    SCORED = "scored"
    NOT_APPLICABLE = "not_applicable"
    GENERATION_FAILURE = "generation_failure"
    EVALUATOR_FAILURE = "evaluator_failure"


class AggregationBlockedError(RuntimeError):
    """Raised when a requested summary is not safe to publish."""


class GenerationFailurePolicyRequired(AggregationBlockedError):
    """Legacy error retained for compatibility with old floor-policy callers."""


class EvaluatorFailurePending(AggregationBlockedError):
    """Raised while evaluator retries remain unresolved."""


FloorPolicy = Mapping[str, float] | Callable[[str], float]


@dataclass(frozen=True)
class FailureCounts:
    generation: int = 0
    evaluator: int = 0


def parse_status(value: ResultStatus | str) -> ResultStatus:
    """Parse a status without treating unknown values as missing."""

    if isinstance(value, ResultStatus):
        return value
    try:
        return ResultStatus(value)
    except ValueError as exc:
        raise ValueError(f"unknown result status: {value!r}") from exc


def metric_floor(metric: str, policy: FloorPolicy | None) -> float:
    """Resolve a legacy floor policy; formal V2 reports no longer use floors."""

    if policy is None:
        raise GenerationFailurePolicyRequired(
            f"generation failure for {metric!r} blocks aggregation until a "
            "metric floor policy is provided"
        )
    try:
        value = policy(metric) if callable(policy) else policy[metric]
    except (KeyError, TypeError) as exc:
        raise GenerationFailurePolicyRequired(
            f"floor policy has no value for metric {metric!r}"
        ) from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"floor for {metric!r} must be numeric")
    return float(value)


def require_no_evaluator_failures(count: int, *, context: str = "") -> None:
    """Block aggregation until all evaluator failures are retried."""

    if count:
        suffix = f" ({context})" if context else ""
        raise EvaluatorFailurePending(
            f"{count} evaluator failure(s) remain unresolved{suffix}"
        )
