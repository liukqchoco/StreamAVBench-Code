"""Strict JSONL manifest loading with explicit-only field overrides."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .contracts import ContractError, ManifestRecord


@dataclass(frozen=True, slots=True)
class ManifestFields:
    model_id: str = "model_id"
    case_id: str = "case_id"
    video_path: str = "video_path"
    audio_path: str = "audio_path"


def load_manifest(
    path: str | Path,
    *,
    fields: ManifestFields | None = None,
    resolve_video_paths: bool = True,
) -> list[ManifestRecord]:
    """Load a JSONL manifest without guessing aliases or scanning directories."""
    manifest_path = Path(path)
    selected = fields or ManifestFields()
    records: list[ManifestRecord] = []
    seen: set[tuple[str, str]] = set()

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"{manifest_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(item, dict):
                raise ContractError(
                    f"{manifest_path}:{line_number}: each line must be an object"
                )
            names = (selected.model_id, selected.case_id, selected.video_path)
            missing = [name for name in names if name not in item]
            if missing:
                raise ContractError(
                    f"{manifest_path}:{line_number}: missing fields {missing}; "
                    "aliases require an explicit ManifestFields override"
                )
            model_id = _required_string(item[selected.model_id], selected.model_id)
            case_id = _required_string(item[selected.case_id], selected.case_id)
            raw_video_path = _required_string(
                item[selected.video_path], selected.video_path
            )
            video_path = Path(raw_video_path).expanduser()
            if resolve_video_paths and not video_path.is_absolute():
                video_path = (manifest_path.parent / video_path).resolve()
            raw_audio_path = item.get(selected.audio_path)
            audio_path = None
            if raw_audio_path is not None:
                audio_path = Path(
                    _required_string(raw_audio_path, selected.audio_path)
                ).expanduser()
                if resolve_video_paths and not audio_path.is_absolute():
                    audio_path = (manifest_path.parent / audio_path).resolve()
            key = (model_id, case_id)
            if key in seen:
                raise ContractError(
                    f"{manifest_path}:{line_number}: duplicate model/case pair {key}"
                )
            seen.add(key)
            excluded = {*names, selected.audio_path}
            extra = {key: value for key, value in item.items() if key not in excluded}
            records.append(
                ManifestRecord(
                    model_id=model_id,
                    case_id=case_id,
                    video_path=video_path,
                    audio_path=audio_path,
                    source_line=line_number,
                    extra=extra,
                )
            )
    if not records:
        raise ContractError(f"{manifest_path}: manifest contains no records")
    return records


def iter_manifest(path: str | Path, **kwargs: object) -> Iterable[ManifestRecord]:
    """Compatibility iterator over the strictly loaded records."""
    return iter(load_manifest(path, **kwargs))


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field_name} must be a non-empty string")
    return value.strip()
