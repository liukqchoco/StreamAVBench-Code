"""Normalize worker protocol results into aggregation-ready records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict
from numbers import Real
from typing import Any

from .contracts import CanonicalResult, ContractError, MetricJob
from .failures import ResultStatus
from .metrics import (
    AID_EARLY,
    AID_LATE,
    AIF,
    AQ,
    ASR,
    AUF,
    AVALIGN,
    AVSYNC,
    BC,
    HDF_ADJ,
    HDF_LR,
    P0_AIF,
    P0_VIF,
    PAC,
    PQ,
    PVC,
    PVC_ALGORITHM,
    SC,
    SHARED_IF_EARLY,
    SHARED_IF_FULL,
    SHARED_IF_LATE,
    VA,
    VID_EARLY,
    VID_LATE,
    VIF,
    VQ,
    VSR,
    VUF,
    canonical_metric,
)

# This is deliberately explicit: auxiliary worker values must never become the
# benchmark score merely because they are the only or first numeric field.
SCORE_FIELD_BY_METRIC = {
    VA: "aesthetic",
    VQ: "mean",
    PQ: "pq",
    AQ: "mean",
    AVALIGN: "av_alignment",
    AVSYNC: "mean_abs_offset_seconds",
    VIF: "mean",
    AIF: "mean",
    VID_EARLY: "mean",
    VID_LATE: "mean",
    AID_EARLY: "mean",
    AID_LATE: "mean",
    P0_VIF: "mean",
    P0_AIF: "mean",
    VUF: "fulfillment_score",
    AUF: "fulfillment_score",
    VSR: "mean",
    ASR: "mean",
    HDF_ADJ: "mean",
    HDF_LR: "mean",
    PVC: "score",
    PVC_ALGORITHM: "algorithm_score",
    PAC: "score",
    SC: "subject_consistency",
    BC: "background_consistency",
}

WORKER_STATUS_MAP = {
    "ok": ResultStatus.SCORED.value,
    "error": ResultStatus.EVALUATOR_FAILURE.value,
    "not_applicable": "not_applicable",
}
COMPOSITE_METRICS = frozenset({SHARED_IF_FULL, SHARED_IF_EARLY, SHARED_IF_LATE})


def normalize_worker_result(
    job: MetricJob, result: Mapping[str, Any] | Any
) -> CanonicalResult:
    """Map worker ``ok``/``error`` to canonical score and status fields."""

    value = _mapping(result)
    request_id = value.get("request_id")
    if request_id != job.job_id:
        raise ContractError(
            f"worker request_id {request_id!r} does not match job {job.job_id!r}"
        )
    worker_metric = canonical_metric(str(value.get("metric", "")))
    if worker_metric != job.metric:
        raise ContractError(
            f"worker metric {worker_metric!r} does not match job {job.metric!r}"
        )
    worker_status = value.get("status")
    try:
        status = WORKER_STATUS_MAP[worker_status]
    except (KeyError, TypeError) as exc:
        raise ContractError(f"unknown worker status {worker_status!r}") from exc

    scores = value.get("scores", {})
    if not isinstance(scores, Mapping):
        raise ContractError("worker scores must be an object")
    error = value.get("error")
    if worker_status == "error":
        if not isinstance(error, Mapping):
            raise ContractError("error worker result must include an error object")
        normalized_error = {
            "type": str(error.get("type", "WorkerError")),
            "message": str(error.get("message", "")),
        }
        score = None
    elif worker_status == "not_applicable":
        score = None
        normalized_error = None
    else:
        if job.metric in COMPOSITE_METRICS:
            unexpected = set(scores) - set(job.output_metrics)
            if unexpected:
                raise ContractError(
                    f"{job.metric} returned unexpected composite outputs "
                    f"{sorted(unexpected)}"
                )
            for field, item in scores.items():
                _finite_score(item, field)
            score = None
        else:
            field = SCORE_FIELD_BY_METRIC[job.metric]
            if field not in scores:
                raise ContractError(
                    f"{job.metric} worker result lacks canonical score field {field!r}"
                )
            score = _finite_score(scores[field], field)
        normalized_error = None

    return CanonicalResult(
        job_id=job.job_id,
        model_id=job.model_id,
        case_id=job.case_id,
        track=job.track,
        metric=job.metric,
        interval=job.interval,
        score=score,
        status=status,
        error=normalized_error,
        details={
            "worker_scores": dict(scores),
            "output_metrics": list(job.output_metrics),
            "artifacts": dict(value.get("artifacts", {})),
            "protocol": dict(value.get("protocol", {})),
        },
    )


def result_to_dict(result: CanonicalResult) -> dict[str, Any]:
    value = asdict(result)
    value["track"] = result.track.value
    return value


def _mapping(result: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(result, Mapping):
        return result
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return value
    raise ContractError("worker result must be a mapping or expose to_dict()")


def _finite_score(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ContractError(f"worker score {field!r} must be numeric")
    score = float(value)
    if not math.isfinite(score):
        raise ContractError(f"worker score {field!r} must be finite")
    return score
