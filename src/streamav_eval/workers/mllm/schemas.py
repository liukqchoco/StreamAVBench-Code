"""Strict JSON schemas and validators for StreamAV MLLM calls."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _strict_object(
    properties: Mapping[str, Mapping[str, Any]], required: tuple[str, ...]
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


SCORE_1_TO_5 = {"type": "integer", "enum": [1, 2, 3, 4, 5]}
SHORT_REASON = {"type": "string", "minLength": 1}
NULLABLE_QUESTION = {"type": ["string", "null"]}
IF_ANSWER = {"type": "string", "enum": ["yes", "partial", "no"]}
IF_JUDGEMENT = _strict_object(
    {"answer": IF_ANSWER, "reason": SHORT_REASON}, ("answer", "reason")
)
NULLABLE_IF_JUDGEMENT = {"anyOf": [IF_JUDGEMENT, {"type": "null"}]}

VQ_SCHEMA = _strict_object(
    {
        "visual_fidelity": SCORE_1_TO_5,
        "subject_integrity": SCORE_1_TO_5,
        "motion_naturalness": SCORE_1_TO_5,
        "visual_artifacts": SCORE_1_TO_5,
        "reason": SHORT_REASON,
    },
    (
        "visual_fidelity",
        "subject_integrity",
        "motion_naturalness",
        "visual_artifacts",
        "reason",
    ),
)

AQ_SCHEMA = _strict_object(
    {
        "audio_naturalness": SCORE_1_TO_5,
        "audio_artifacts": SCORE_1_TO_5,
        "reason": SHORT_REASON,
    },
    ("audio_naturalness", "audio_artifacts", "reason"),
)

VIF_CHECKLIST_FIELDS = ("scene", "visual_style", "subjects", "activity")
AIF_CHECKLIST_FIELDS = ("sound_source", "sound_content", "temporal_relation")


def _nested_schema(
    root: str, fields: tuple[str, ...], field_schema: Mapping[str, Any]
) -> dict[str, Any]:
    nested = _strict_object({field: field_schema for field in fields}, fields)
    return _strict_object({root: nested}, (root,))


VIF_CHECKLIST_SCHEMA = _nested_schema(
    "video_instruction_following", VIF_CHECKLIST_FIELDS, NULLABLE_QUESTION
)
AIF_CHECKLIST_SCHEMA = _nested_schema(
    "audio_instruction_following", AIF_CHECKLIST_FIELDS, NULLABLE_QUESTION
)
VIF_JUDGE_SCHEMA = _nested_schema(
    "video_instruction_following", VIF_CHECKLIST_FIELDS, NULLABLE_IF_JUDGEMENT
)
AIF_JUDGE_SCHEMA = _nested_schema(
    "audio_instruction_following", AIF_CHECKLIST_FIELDS, NULLABLE_IF_JUDGEMENT
)

INSTRUCTION_ANSWER_SCHEMA = _strict_object(
    {
        "id": {"type": "string", "minLength": 1},
        "reason": SHORT_REASON,
        "score": SCORE_1_TO_5,
    },
    ("id", "reason", "score"),
)
SHARED_INSTRUCTION_SCHEMA = _strict_object(
    {
        "answers": {
            "type": "array",
            "items": INSTRUCTION_ANSWER_SCHEMA,
            "minItems": 1,
        }
    },
    ("answers",),
)

NULLABLE_SCORE_1_TO_5 = {"anyOf": [SCORE_1_TO_5, {"type": "null"}]}
NULLABLE_LATENCY = {
    "anyOf": [
        {"type": "number", "minimum": 0.0, "maximum": 29.5, "multipleOf": 0.5},
        {"type": "null"},
    ]
}


def _update_fulfillment_schema(metric: str) -> dict[str, Any]:
    return _strict_object(
        {
            "metric": {"type": "string", "enum": [metric]},
            "started": {"type": "boolean"},
            "fulfillment_score": SCORE_1_TO_5,
            "onset_latency_s": NULLABLE_LATENCY,
            "target_achievement_latency_s": NULLABLE_LATENCY,
            "evidence": {"type": "string", "minLength": 1},
            "reason": SHORT_REASON,
        },
        (
            "metric",
            "started",
            "fulfillment_score",
            "onset_latency_s",
            "target_achievement_latency_s",
            "evidence",
            "reason",
        ),
    )


VUF_SCHEMA = _update_fulfillment_schema("VUF")
AUF_SCHEMA = _update_fulfillment_schema("AUF")
RETENTION_FIELD_SCHEMA = _strict_object(
    {"score": SCORE_1_TO_5, "reason": SHORT_REASON},
    ("score", "reason"),
)
VSR_SCHEMA = _strict_object(
    {
        "subject_retention": RETENTION_FIELD_SCHEMA,
        "environment_retention": RETENTION_FIELD_SCHEMA,
    },
    ("subject_retention", "environment_retention"),
)
ASR_SCHEMA = _strict_object(
    {"audio_retention_score": SCORE_1_TO_5, "reason": SHORT_REASON},
    ("audio_retention_score", "reason"),
)
HDF_SCHEMA = _strict_object(
    {
        "source_state_established": {"type": "boolean"},
        "dependency_following_score": NULLABLE_SCORE_1_TO_5,
        "reason": SHORT_REASON,
    },
    ("source_state_established", "dependency_following_score", "reason"),
)
PVC_SCHEMA = _strict_object(
    {
        "score": SCORE_1_TO_5,
        "generation_break": {"type": "boolean"},
        "deformation": {"type": "boolean"},
        "object_disappear": {"type": "boolean"},
        "reason": SHORT_REASON,
    },
    ("score", "generation_break", "deformation", "object_disappear", "reason"),
)
PAC_SCHEMA = _strict_object(
    {
        "score": SCORE_1_TO_5,
        "audio_break": {"type": "boolean"},
        "audio_artifact": {"type": "boolean"},
        "reason": SHORT_REASON,
    },
    ("score", "audio_break", "audio_artifact", "reason"),
)


def require_exact_keys(
    value: Mapping[str, Any], required: tuple[str, ...], *, context: str
) -> None:
    actual = set(value)
    expected = set(required)
    if actual != expected:
        raise ValueError(
            f"{context} keys must be exactly {sorted(expected)}; got {sorted(actual)}"
        )
