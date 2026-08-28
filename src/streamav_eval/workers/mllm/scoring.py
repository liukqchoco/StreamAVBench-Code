"""Deterministic raw-scale MLLM quality and checklist scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .schemas import (
    AIF_CHECKLIST_FIELDS,
    VIF_CHECKLIST_FIELDS,
    require_exact_keys,
)

QUALITY_FIELDS = {
    "VQ": (
        "visual_fidelity",
        "subject_integrity",
        "motion_naturalness",
        "visual_artifacts",
    ),
    "AQ": ("audio_naturalness", "audio_artifacts"),
}
IF_ROOTS = {
    "vif": ("video_instruction_following", VIF_CHECKLIST_FIELDS),
    "p0_vif": ("video_instruction_following", VIF_CHECKLIST_FIELDS),
    "aif": ("audio_instruction_following", AIF_CHECKLIST_FIELDS),
    "p0_aif": ("audio_instruction_following", AIF_CHECKLIST_FIELDS),
}
ANSWER_SCORES = {"yes": 1.0, "partial": 0.5, "no": 0.0}


def null_aware_mean(values: Sequence[float | int | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def score_quality(metric: str, response: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and retain raw 1--5 dimensions plus their raw-scale mean."""

    try:
        fields = QUALITY_FIELDS[metric]
    except KeyError as exc:
        raise ValueError(f"unsupported quality metric: {metric!r}") from exc
    require_exact_keys(response, fields + ("reason",), context=metric)
    scores: dict[str, int] = {}
    for field in fields:
        value = response[field]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"{metric}.{field} must be an integer from 1 to 5")
        scores[field] = value
    reason = response["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"{metric}.reason must be a non-empty string")
    return {
        "dimensions": scores,
        "mean": null_aware_mean(tuple(scores.values())),
        "reason": reason,
    }


def score_instruction_following(
    metric: str, response: Mapping[str, Any]
) -> dict[str, Any]:
    """Map yes/partial/no to 1/.5/0 and average only applicable fields."""

    try:
        root, fields = IF_ROOTS[metric]
    except KeyError as exc:
        raise ValueError(f"unsupported instruction metric: {metric!r}") from exc
    require_exact_keys(response, (root,), context=metric)
    nested = response[root]
    if not isinstance(nested, Mapping):
        raise ValueError(f"{metric}.{root} must be an object")
    require_exact_keys(nested, fields, context=f"{metric}.{root}")
    dimensions: dict[str, float | None] = {}
    judgements: dict[str, Mapping[str, str] | None] = {}
    for field in fields:
        item = nested[field]
        if item is None:
            dimensions[field] = None
            judgements[field] = None
            continue
        if not isinstance(item, Mapping):
            raise ValueError(f"{metric}.{field} must be an object or null")
        require_exact_keys(item, ("answer", "reason"), context=f"{metric}.{field}")
        answer = item["answer"]
        reason = item["reason"]
        if answer not in ANSWER_SCORES:
            raise ValueError(f"{metric}.{field}.answer is invalid: {answer!r}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{metric}.{field}.reason must be non-empty")
        dimensions[field] = ANSWER_SCORES[str(answer)]
        judgements[field] = {"answer": str(answer), "reason": reason}
    return {
        "dimensions": dimensions,
        "mean": null_aware_mean(tuple(dimensions.values())),
        "judgements": judgements,
    }


def score_shared_instruction(
    response: Mapping[str, Any], criteria: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Validate exact criterion coverage and retain modality-specific 1--5 scores."""

    require_exact_keys(response, ("answers",), context="shared instruction response")
    answers = response["answers"]
    if not isinstance(answers, list):
        raise ValueError("shared instruction answers must be an array")
    expected: dict[str, Mapping[str, Any]] = {}
    for criterion in criteria:
        criterion_id = criterion.get("criterion_id")
        modality = criterion.get("modality")
        if (
            not isinstance(criterion_id, str)
            or not criterion_id
            or modality not in {"video", "audio"}
        ):
            raise ValueError("criterion requires criterion_id and video/audio modality")
        if criterion_id in expected:
            raise ValueError(f"duplicate criterion id {criterion_id!r}")
        expected[criterion_id] = criterion

    scored: dict[str, dict[str, Any]] = {}
    for answer in answers:
        if not isinstance(answer, Mapping):
            raise ValueError("shared instruction answer must be an object")
        require_exact_keys(answer, ("id", "reason", "score"), context="answer")
        criterion_id = answer["id"]
        if criterion_id not in expected:
            raise ValueError(f"unexpected criterion id {criterion_id!r}")
        if criterion_id in scored:
            raise ValueError(f"duplicate answer for criterion {criterion_id!r}")
        score = answer["score"]
        reason = answer["reason"]
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"{criterion_id}.score must be an integer from 1 to 5")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{criterion_id}.reason must be non-empty")
        scored[str(criterion_id)] = {
            "score": score,
            "reason": reason,
            "modality": expected[str(criterion_id)]["modality"],
            "dimension": expected[str(criterion_id)].get("dimension"),
        }
    missing = set(expected) - set(scored)
    if missing:
        raise ValueError(f"missing criterion answers: {sorted(missing)}")

    modality_scores = {
        modality: null_aware_mean(
            tuple(
                item["score"]
                for item in scored.values()
                if item["modality"] == modality
            )
        )
        for modality in ("video", "audio")
    }
    return {"criteria": scored, "modality_means": modality_scores}
