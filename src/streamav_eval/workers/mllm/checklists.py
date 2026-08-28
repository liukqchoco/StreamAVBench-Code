"""VIF/AIF checklist request construction and validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .schemas import AIF_CHECKLIST_FIELDS, VIF_CHECKLIST_FIELDS, require_exact_keys


@dataclass(frozen=True, slots=True)
class InstructionCriterion:
    criterion_id: str
    modality: str
    dimension: str
    question: str

    def to_dict(self) -> dict[str, str]:
        return {
            "criterion_id": self.criterion_id,
            "modality": self.modality,
            "dimension": self.dimension,
            "question": self.question,
        }


def criteria_from_record(
    record: Mapping[str, Any], modality: str | None = None
) -> tuple[InstructionCriterion, ...]:
    """Build instruction criteria from one public dataset record."""

    if modality not in {None, "video", "audio"}:
        raise ValueError("modality must be 'video', 'audio', or None")
    checklists = record.get("checklists")
    if not isinstance(checklists, Mapping):
        raise ValueError("dataset record must contain checklists")

    criteria: list[InstructionCriterion] = []
    for current_modality, prefix, root, fields in (
        ("video", "V", "video_instruction_following", VIF_CHECKLIST_FIELDS),
        ("audio", "A", "audio_instruction_following", AIF_CHECKLIST_FIELDS),
    ):
        if modality is not None and current_modality != modality:
            continue
        checklist = {root: checklists.get(current_modality)}
        validate_checklist(current_modality, checklist)
        nested = checklist[root]
        for field in fields:
            question = nested[field]
            if question is not None:
                criteria.append(
                    InstructionCriterion(
                        f"{prefix}.{field}",
                        current_modality,
                        field,
                        question,
                    )
                )
    return tuple(criteria)


def validate_checklist(modality: str, value: Mapping[str, Any]) -> None:
    if modality == "video":
        root, fields = "video_instruction_following", VIF_CHECKLIST_FIELDS
    elif modality == "audio":
        root, fields = "audio_instruction_following", AIF_CHECKLIST_FIELDS
    else:
        raise ValueError("modality must be 'video' or 'audio'")
    require_exact_keys(value, (root,), context=f"{modality} checklist")
    nested = value[root]
    if not isinstance(nested, Mapping):
        raise ValueError(f"{root} must be an object")
    require_exact_keys(nested, fields, context=root)
    for field in fields:
        question = nested[field]
        if question is not None and (
            not isinstance(question, str) or not question.strip()
        ):
            raise ValueError(f"{root}.{field} must be a non-empty string or null")
