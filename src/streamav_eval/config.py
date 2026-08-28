"""Configuration contract for the offline orchestrator."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import ContractError
from .manifest import ManifestFields
from .registry import resolve_dataset_root


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    run_id: str
    manifest_path: Path
    dataset_root: Path
    duration_tolerance_seconds: float = 0.5
    ffprobe: str = "ffprobe"
    manifest_fields: ManifestFields = field(default_factory=ManifestFields)


def load_config(path: str | Path) -> EvaluationConfig:
    """Load a JSON object; a ``.yaml`` suffix does not enable YAML syntax."""
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(
            f"{config_path}: config must contain JSON (even with a .yaml suffix): "
            f"{exc.msg}"
        ) from exc
    except OSError as exc:
        raise ContractError(f"cannot read config {config_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ContractError(f"{config_path}: config root must be an object")
    return config_from_mapping(raw, base_dir=config_path.parent)


def config_from_mapping(
    raw: Mapping[str, Any], *, base_dir: str | Path = "."
) -> EvaluationConfig:
    allowed = {
        "run_id",
        "manifest_path",
        "dataset_root",
        "duration_tolerance_seconds",
        "ffprobe",
        "manifest_fields",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ContractError(f"unknown config fields: {unknown}")
    run_id = _string(raw.get("run_id"), "run_id")
    manifest_path = _string(raw.get("manifest_path"), "manifest_path")
    tolerance = raw.get("duration_tolerance_seconds", 0.5)
    if not isinstance(tolerance, (int, float)) or tolerance < 0:
        raise ContractError("duration_tolerance_seconds must be non-negative")
    ffprobe = _string(raw.get("ffprobe", "ffprobe"), "ffprobe")
    base = Path(base_dir)
    fields_raw = raw.get("manifest_fields", {})
    if not isinstance(fields_raw, Mapping):
        raise ContractError("manifest_fields must be an object")
    field_unknown = sorted(
        set(fields_raw) - {"model_id", "case_id", "video_path", "audio_path"}
    )
    if field_unknown:
        raise ContractError(f"unknown manifest_fields entries: {field_unknown}")
    fields = ManifestFields(
        model_id=_string(fields_raw.get("model_id", "model_id"), "model_id field"),
        case_id=_string(fields_raw.get("case_id", "case_id"), "case_id field"),
        video_path=_string(
            fields_raw.get("video_path", "video_path"), "video_path field"
        ),
        audio_path=_string(
            fields_raw.get("audio_path", "audio_path"), "audio_path field"
        ),
    )
    dataset_root = _optional_path(raw.get("dataset_root"), base, "dataset_root")
    if dataset_root is None:
        dataset_root = resolve_dataset_root()
    return EvaluationConfig(
        run_id=run_id,
        manifest_path=_relative_path(manifest_path, base),
        dataset_root=dataset_root,
        duration_tolerance_seconds=float(tolerance),
        ffprobe=ffprobe,
        manifest_fields=fields,
    )


def _relative_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _optional_path(value: Any, base: Path, name: str) -> Path | None:
    if value is None:
        return None
    return _relative_path(_string(value, name), base)


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a non-empty string")
    return value.strip()
