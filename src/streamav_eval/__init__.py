"""Official StreamAV-Bench offline evaluation toolkit."""

from .contracts import (
    ContractError,
    EvaluationUnit,
    ManifestRecord,
    MediaMetadata,
    Prompt,
    RegistryCase,
    Track,
    UnitKind,
)
from .manifest import ManifestFields, load_manifest
from .media import probe_media, validate_duration
from .registry import CanonicalRegistry
from .units import build_units

__all__ = [
    "CanonicalRegistry",
    "ContractError",
    "EvaluationUnit",
    "ManifestFields",
    "ManifestRecord",
    "MediaMetadata",
    "Prompt",
    "RegistryCase",
    "Track",
    "UnitKind",
    "build_units",
    "load_manifest",
    "probe_media",
    "validate_duration",
]
