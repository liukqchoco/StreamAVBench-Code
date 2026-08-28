"""Canonical StreamAV-Bench metric identifiers and protocol groups."""

from __future__ import annotations

from .contracts import ContractError, Track

VA = "VA"
VQ = "VQ"
PQ = "PQ"
AQ = "AQ"
AVALIGN = "AVAlign"
AVSYNC = "AVSync"
NBC = "NBC"
FPS = "FPS"
TTFC = "TTFC"
VIF = "VIF"
AIF = "AIF"
P0_VIF = "P0-VIF"
P0_AIF = "P0-AIF"
VID_EARLY = "VID-Early"
VID_LATE = "VID-Late"
AID_EARLY = "AID-Early"
AID_LATE = "AID-Late"
VUF = "VUF"
AUF = "AUF"
VSR = "VSR"
ASR = "ASR"
HDF_ADJ = "HDF-Adjacent"
HDF_LR = "HDF-Long-Range"
PVC = "PVC"
PVC_ALGORITHM = "PVC-Algorithm"
PAC = "PAC"
SC = "SC"
BC = "BC"

# Internal evaluator-call IDs.  Each request produces more than one
# publication-facing output, so these must never be expanded into duplicate
# media calls.
SHARED_IF_FULL = "VIF-AIF"
SHARED_IF_EARLY = "VID-AID-Early"
SHARED_IF_LATE = "VID-AID-Late"

INTERVAL_METRICS = (VA, VQ, PQ, AQ, AVALIGN, AVSYNC)
PROGRESSIVE_METRICS = (
    VIF,
    AIF,
    VID_EARLY,
    VID_LATE,
    AID_EARLY,
    AID_LATE,
    SC,
    BC,
)
INTERACTIVE_METRICS = (
    P0_VIF,
    P0_AIF,
    VUF,
    AUF,
    VSR,
    ASR,
    HDF_ADJ,
    HDF_LR,
    PVC,
    PVC_ALGORITHM,
    PAC,
)
OBJECTIVE_METRICS = (
    VA,
    PQ,
    AVALIGN,
    AVSYNC,
    SC,
    BC,
    PVC_ALGORITHM,
)
CANONICAL_METRICS = (
    *INTERVAL_METRICS,
    *PROGRESSIVE_METRICS,
    *INTERACTIVE_METRICS,
    SHARED_IF_FULL,
    SHARED_IF_EARLY,
    SHARED_IF_LATE,
    NBC,
    FPS,
    TTFC,
)

SHARED_DIMENSIONS = (VA, VQ, PQ, AQ, AVALIGN, AVSYNC, NBC, FPS, TTFC)
PROGRESSIVE_DIMENSIONS = (
    VIF,
    AIF,
    "VID",
    "AID",
    "VA-D",
    "VQ-D",
    "PQ-D",
    "AQ-D",
    SC,
    BC,
    "AVAlign-D",
    "AVSync-D",
)
INTERACTIVE_DIMENSIONS = (
    VUF,
    AUF,
    PVC,
    PAC,
    "PVUAR",
    "PAUAR",
    "PVRL",
    "PARL",
    VSR,
    ASR,
    "HDF",
)
PUBLIC_DIMENSIONS = (
    *SHARED_DIMENSIONS,
    *PROGRESSIVE_DIMENSIONS,
    *INTERACTIVE_DIMENSIONS,
)
assert len(PUBLIC_DIMENSIONS) == 32

_ALIASES = {metric.lower().replace("_", "-"): metric for metric in CANONICAL_METRICS}
_ALIASES.update(
    {
        "aesthetic": VA,
        "av-alignment": AVALIGN,
        "avalign": AVALIGN,
        "av-sync": AVSYNC,
        "avsync": AVSYNC,
        "p0-vif": P0_VIF,
        "p0-aif": P0_AIF,
        "subject-consistency": SC,
        "background-consistency": BC,
    }
)


def canonical_metric(value: str) -> str:
    """Return the exact publication-facing metric ID."""

    if not isinstance(value, str) or not value.strip():
        raise ContractError("metric must be a non-empty string")
    key = value.strip().lower().replace("_", "-")
    try:
        return _ALIASES[key]
    except KeyError as exc:
        raise ContractError(f"unknown metric {value!r}") from exc


def instruction_metrics(track: Track | str) -> tuple[str, ...]:
    parsed = Track.parse(track)
    return PROGRESSIVE_METRICS if parsed is Track.PROGRESSIVE else INTERACTIVE_METRICS
