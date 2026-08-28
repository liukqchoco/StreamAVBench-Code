"""Canonical benchmark registry loader."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .contracts import ContractError, Prompt, RegistryCase, Track

DATASET_ROOT_ENV = "STREAMAV_DATASET_ROOT"


class CanonicalRegistry:
    def __init__(self, cases: Mapping[str, RegistryCase]) -> None:
        self._cases = dict(cases)

    @classmethod
    def load(
        cls,
        *,
        dataset_root: str | Path | None = None,
    ) -> CanonicalRegistry:
        """Load benchmark records from a separate dataset checkout."""

        root = resolve_dataset_root(dataset_root)
        sources = (
            (Track.PROGRESSIVE, root / "data" / "progressive.json"),
            (Track.INTERACTIVE, root / "data" / "interactive.json"),
        )
        cases: dict[str, RegistryCase] = {}
        for expected_track, path in sources:
            for item in _load_array(path):
                case = _derive_case(item, path, expected_track)
                if case.case_id in cases:
                    raise ContractError(f"duplicate registry case_id {case.case_id}")
                cases[case.case_id] = case
        return cls(cases)

    def get(self, case_id: str) -> RegistryCase:
        try:
            return self._cases[case_id]
        except KeyError as exc:
            raise ContractError(f"unknown canonical case_id {case_id!r}") from exc

    def __contains__(self, case_id: object) -> bool:
        return case_id in self._cases

    def __len__(self) -> int:
        return len(self._cases)

    def values(self) -> tuple[RegistryCase, ...]:
        return tuple(self._cases.values())


def resolve_dataset_root(value: str | Path | None = None) -> Path:
    configured = value if value is not None else os.environ.get(DATASET_ROOT_ENV)
    if configured is None or not str(configured).strip():
        raise ContractError(
            "benchmark data is not bundled with the code; set "
            f"{DATASET_ROOT_ENV} or provide dataset_root"
        )
    root = Path(configured).expanduser().resolve()
    if not root.is_dir():
        raise ContractError(f"dataset root does not exist: {root}")
    return root


def _load_array(path: Path) -> list[Mapping[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load registry {path}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ContractError(f"{path}: registry root must be an array of objects")
    return value


def _derive_case(
    item: Mapping[str, Any],
    source_path: Path,
    expected_track: Track,
) -> RegistryCase:
    case_id = item.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ContractError(f"{source_path}: case_id must be non-empty")
    try:
        track = Track(item.get("track"))
    except ValueError as exc:
        raise ContractError(f"{source_path}:{case_id}: invalid track") from exc
    if track is not expected_track:
        raise ContractError(
            f"{source_path}:{case_id}: track {track} does not match source "
            f"{expected_track}"
        )
    duration = item.get("duration_seconds")
    if not isinstance(duration, (int, float)) or duration <= 0:
        raise ContractError(f"{source_path}:{case_id}: invalid duration_seconds")
    prompts = (
        _progressive_prompts(item, source_path, case_id)
        if track is Track.PROGRESSIVE
        else _interactive_prompts(item, source_path, case_id)
    )
    return RegistryCase(
        case_id=case_id,
        track=track,
        duration_seconds=float(duration),
        prompts=prompts,
        source_path=source_path,
        source=item,
    )


def _progressive_prompts(
    item: Mapping[str, Any],
    source_path: Path,
    case_id: str,
) -> tuple[Prompt, ...]:
    text = item.get("prompt")
    if not isinstance(text, str) or not text.strip():
        raise ContractError(f"{source_path}:{case_id}: prompt must be non-empty")
    return (Prompt(prompt_id="P0", text=text.strip(), payload=item),)


def _interactive_prompts(
    item: Mapping[str, Any],
    source_path: Path,
    case_id: str,
) -> tuple[Prompt, ...]:
    initial_text = item.get("prompt")
    updates = item.get("updates")
    if not isinstance(initial_text, str) or not initial_text.strip():
        raise ContractError(f"{source_path}:{case_id}: prompt must be non-empty")
    if not isinstance(updates, list) or len(updates) != 5:
        raise ContractError(f"{source_path}:{case_id}: requires five runtime updates")
    prompts = [Prompt("P0", initial_text.strip(), item)]
    for position, update in enumerate(updates, start=1):
        if not isinstance(update, Mapping):
            raise ContractError(f"{source_path}:{case_id}: update must be an object")
        prompt_id = update.get("prompt_id")
        text = update.get("prompt")
        if not isinstance(prompt_id, str) or not isinstance(text, str):
            raise ContractError(
                f"{source_path}:{case_id}: update requires prompt_id and prompt"
            )
        if prompt_id != f"P{position}":
            raise ContractError(
                f"{source_path}:{case_id}: runtime prompt IDs must be P1-P5"
            )
        activation = update.get("activation_time_seconds")
        if activation != position * 30:
            raise ContractError(
                f"{source_path}:{case_id}:{prompt_id}: invalid activation time"
            )
        modality = update.get("update_modality")
        if modality not in {"Video-only", "Audio-only", "Joint Audio-Video"}:
            raise ContractError(
                f"{source_path}:{case_id}:{prompt_id}: invalid update modality"
            )
        dependency = update.get("temporal_dependency")
        if dependency not in {"Independent", "Adjacent", "Long-Range"}:
            raise ContractError(
                f"{source_path}:{case_id}:{prompt_id}: invalid temporal dependency"
            )
        if modality == "Joint Audio-Video":
            for field in ("visual_prompt", "audio_prompt"):
                value = update.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ContractError(
                        f"{source_path}:{case_id}:{prompt_id}: missing {field}"
                    )
        prompts.append(Prompt(prompt_id, text.strip(), update))
    return tuple(prompts)
