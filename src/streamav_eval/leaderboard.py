"""Publication-facing export of the 32 StreamAV-Bench dimensions."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

from .metrics import PUBLIC_DIMENSIONS
from .registry import CanonicalRegistry

SHARED_EVALUATED = ("VA", "VQ", "PQ", "AQ", "AVAlign", "AVSync")
PROGRESSIVE_DIRECT = ("VIF", "AIF")
DRIFT_NAMES = {"VA": "VA-D", "VQ": "VQ-D", "PQ": "PQ-D", "AQ": "AQ-D"}
INTERACTIVE_DIRECT = ("VUF", "AUF", "PVC", "PAC", "VSR", "ASR", "HDF")

DIRECTIONS = {
    **{metric: "higher" for metric in PUBLIC_DIMENSIONS},
    "AVSync": "lower",
    "VID": "lower",
    "AID": "lower",
    "VA-D": "lower",
    "VQ-D": "lower",
    "PQ-D": "lower",
    "AQ-D": "lower",
    "AVAlign-D": "lower",
    "AVSync-D": "lower",
    "PVRL": "lower",
    "PARL": "lower",
    "TTFC": "lower",
}
UNITS = {
    "AVSync": "seconds",
    "AVSync-D": "seconds",
    "PVRL": "seconds",
    "PARL": "seconds",
    "TTFC": "seconds",
    "PVUAR": "rate",
    "PAUAR": "rate",
    "FPS": "frames_per_second",
}


def build_public_results(
    report: Mapping[str, Any],
    *,
    efficiency_models: Sequence[Mapping[str, Any]],
    efficiency_unavailable_model_ids: Sequence[str] = (),
    nbc_models: Sequence[Mapping[str, Any]] = (),
    nbc_unavailable_model_ids: Sequence[str] = (),
    require_complete: bool = True,
    registry: CanonicalRegistry | None = None,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build one wide row per model without inventing an overall score."""

    if report.get("schema_version") != "streamavbench.report.v2":
        raise ValueError("evaluation report has an unsupported schema")
    if not isinstance(report.get("run_id"), str) or not report["run_id"].strip():
        raise ValueError("evaluation report requires one non-empty run_id")
    if require_complete and report.get("provisional"):
        raise ValueError("refusing to export a provisional evaluation report")
    if require_complete and report.get("failures"):
        raise ValueError("refusing to export a report with failed evaluations")
    efficiency = _unique_by_model(efficiency_models, "efficiency")
    unavailable_efficiency = {str(value) for value in efficiency_unavailable_model_ids}
    if (
        len(unavailable_efficiency) != len(efficiency_unavailable_model_ids)
        or "" in unavailable_efficiency
    ):
        raise ValueError(
            "unavailable efficiency model IDs must be unique and non-empty"
        )
    if set(efficiency) & unavailable_efficiency:
        raise ValueError(
            "a model cannot have both available and unavailable efficiency"
        )
    nbc = _unique_by_model(nbc_models, "NBC")
    unavailable_nbc = {str(value) for value in nbc_unavailable_model_ids}
    if len(unavailable_nbc) != len(nbc_unavailable_model_ids) or "" in unavailable_nbc:
        raise ValueError("unavailable NBC model IDs must be unique and non-empty")
    if set(nbc) & unavailable_nbc:
        raise ValueError("a model cannot have both available and unavailable NBC")
    model_ids = sorted(
        {
            str(row["model_id"])
            for row in report.get("models", ())
            if isinstance(row, Mapping) and row.get("model_id")
        }
    )
    if not model_ids:
        raise ValueError("evaluation report contains no model results")
    for model_id in model_ids:
        _validate_public_model_id(model_id)
    if set(efficiency) | unavailable_efficiency != set(model_ids):
        raise ValueError(
            "available and unavailable efficiency model IDs must partition "
            "the evaluation report"
        )
    if set(nbc) | unavailable_nbc != set(model_ids):
        raise ValueError(
            "available and unavailable NBC model IDs must partition "
            "the evaluation report"
        )
    for model_id, row in efficiency.items():
        _validate_efficiency_row(model_id, row)
    for model_id, row in nbc.items():
        _validate_nbc_row(model_id, row)

    cases = [
        row
        for row in report.get("cases", ())
        if isinstance(row, Mapping) and row.get("metric") in SHARED_EVALUATED
    ]
    model_rows = [row for row in report.get("models", ()) if isinstance(row, Mapping)]
    interactive_rows = [
        row for row in report.get("interactive_metrics", ()) if isinstance(row, Mapping)
    ]
    if require_complete:
        _validate_shared_case_coverage(
            cases,
            model_ids,
            registry or CanonicalRegistry.load(dataset_root=dataset_root),
        )
    rows = []
    for model_id in model_ids:
        scores: dict[str, float | None] = {}
        for metric in SHARED_EVALUATED:
            values = [
                _score(row["value"], f"{model_id}/{metric}")
                for row in cases
                if row.get("model_id") == model_id
                and row.get("metric") == metric
                and row.get("status") == "computed"
                and row.get("value") is not None
            ]
            if not values:
                raise ValueError(f"missing shared metric {metric} for {model_id}")
            scores[metric] = math.fsum(values) / len(values)

        for metric in PROGRESSIVE_DIRECT:
            scores[metric] = _row_value(
                model_rows, model_id, metric, track="progressive"
            )
        for metric in ("VID", "AID"):
            scores[metric] = _row_value(
                report.get("instruction_drift", ()), model_id, metric
            )
        for source, destination in DRIFT_NAMES.items():
            scores[destination] = _row_value(
                report.get("quality_drift", ()),
                model_id,
                source,
                field="quality_drift",
            )
        for metric in ("SC", "BC"):
            scores[metric] = _row_value(
                report.get("visual_consistency", ()), model_id, metric
            )
        for metric in ("AVAlign-D", "AVSync-D"):
            scores[metric] = _row_value(
                report.get("cross_modal_stability", ()), model_id, metric
            )

        interactive = {
            str(row["metric"]): row
            for row in interactive_rows
            if row.get("model_id") == model_id
        }
        for metric in INTERACTIVE_DIRECT:
            scores[metric] = _mapping_value(
                interactive,
                metric,
                "value",
                model_id,
                allow_none=metric == "HDF",
            )
        scores["PVUAR"] = _mapping_value(
            interactive, "VUF", "target_achievement_rate", model_id
        )
        scores["PAUAR"] = _mapping_value(
            interactive, "AUF", "target_achievement_rate", model_id
        )
        scores["PVRL"] = _mapping_value(
            interactive,
            "VUF",
            "conditional_target_latency_s",
            model_id,
            allow_none=True,
        )
        scores["PARL"] = _mapping_value(
            interactive,
            "AUF",
            "conditional_target_latency_s",
            model_id,
            allow_none=True,
        )

        efficiency_row = efficiency.get(model_id)
        scores["FPS"] = (
            _score(efficiency_row.get("fps"), f"{model_id}/FPS")
            if efficiency_row is not None
            else None
        )
        scores["TTFC"] = (
            _score(efficiency_row.get("ttfc_seconds"), f"{model_id}/TTFC")
            if efficiency_row is not None
            else None
        )
        scores["NBC"] = (
            _score(nbc[model_id].get("score"), f"{model_id}/NBC")
            if model_id in nbc
            else None
        )
        if set(scores) != set(PUBLIC_DIMENSIONS):
            missing = sorted(set(PUBLIC_DIMENSIONS) - set(scores))
            extra = sorted(set(scores) - set(PUBLIC_DIMENSIONS))
            raise AssertionError(f"32-dimension mapping mismatch: {missing=}, {extra=}")
        rows.append({"model_id": model_id, **scores})

    return {
        "schema_version": "streamavbench-public-results-1.0.0",
        "run_id": report["run_id"],
        "dimensions": list(PUBLIC_DIMENSIONS),
        "directions": DIRECTIONS,
        "units": UNITS,
        "overall_score": None,
        "overall_score_note": "No overall score is defined by StreamAV-Bench.",
        "models": rows,
    }


def _unique_by_model(
    rows: Iterable[Mapping[str, Any]], label: str
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        model_id = str(row.get("model_id", ""))
        if not model_id:
            raise ValueError(f"{label} row has no model_id")
        if model_id in output:
            raise ValueError(f"duplicate {label} row for {model_id}")
        output[model_id] = row
    return output


def _row_value(
    rows: Iterable[Mapping[str, Any]],
    model_id: str,
    metric: str,
    *,
    field: str = "value",
    track: str | None = None,
) -> float:
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("model_id") == model_id
        and row.get("metric") == metric
        and (track is None or row.get("track") == track)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {model_id}/{metric} row, found {len(matches)}")
    if matches[0].get("provisional"):
        raise ValueError(f"{model_id}/{metric} is provisional")
    return _score(matches[0].get(field), f"{model_id}/{metric}/{field}")


def _mapping_value(
    rows: Mapping[str, Mapping[str, Any]],
    metric: str,
    field: str,
    model_id: str,
    *,
    allow_none: bool = False,
) -> float | None:
    try:
        row = rows[metric]
    except KeyError as exc:
        raise ValueError(f"missing interactive metric {metric} for {model_id}") from exc
    if row.get("provisional"):
        raise ValueError(f"{model_id}/{metric} is provisional")
    value = row.get(field)
    if value is None and allow_none:
        return None
    return _score(value, f"{model_id}/{metric}/{field}")


def _validate_shared_case_coverage(
    cases: Sequence[Mapping[str, Any]],
    model_ids: Sequence[str],
    registry: CanonicalRegistry,
) -> None:
    expected = {(case.track.value, case.case_id) for case in registry.values()}
    if len(expected) != 320:
        raise AssertionError("canonical registry must contain exactly 320 cases")
    for model_id in model_ids:
        for metric in SHARED_EVALUATED:
            rows = [
                row
                for row in cases
                if row.get("model_id") == model_id and row.get("metric") == metric
            ]
            observed = [
                (str(row.get("track")), str(row.get("case_id"))) for row in rows
            ]
            if len(observed) != len(set(observed)):
                raise ValueError(f"duplicate shared cases for {model_id}/{metric}")
            observed_set = set(observed)
            if observed_set != expected:
                missing = len(expected - observed_set)
                extra = len(observed_set - expected)
                raise ValueError(
                    f"{model_id}/{metric} must cover the canonical 320 cases "
                    f"({missing} missing, {extra} unexpected)"
                )
            if any(
                row.get("status") != "computed" or row.get("value") is None
                for row in rows
            ):
                raise ValueError(
                    f"{model_id}/{metric} must have a computed value for every case"
                )


def _validate_efficiency_row(model_id: str, row: Mapping[str, Any]) -> None:
    if row.get("schema_version") != "streamavbench-efficiency-1.0.0":
        raise ValueError(f"{model_id}/efficiency has an unsupported schema")
    if row.get("case_count") != 320:
        raise ValueError(f"{model_id}/efficiency must contain 320 cases")
    tracks = row.get("tracks")
    if not isinstance(tracks, Mapping) or set(tracks) != {
        "progressive",
        "interactive",
    }:
        raise ValueError(f"{model_id}/efficiency must contain both tracks")
    for track in ("progressive", "interactive"):
        value = tracks[track]
        if not isinstance(value, Mapping) or value.get("case_count") != 160:
            raise ValueError(f"{model_id}/efficiency/{track} must contain 160 cases")


def _validate_nbc_row(model_id: str, row: Mapping[str, Any]) -> None:
    if (
        row.get("schema_version") != "streamavbench-nbc-combined-1.0.0"
        or row.get("metric_id") != "NBC"
    ):
        raise ValueError(f"{model_id}/NBC has an unsupported schema")
    if row.get("computed_cases") != 320:
        raise ValueError(f"{model_id}/NBC must contain 320 computed cases")


def _score(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _validate_public_model_id(model_id: str) -> None:
    if model_id[0] in "=+-@" or any(
        ord(character) < 32 or ord(character) == 127 for character in model_id
    ):
        raise ValueError(
            f"unsafe model_id {model_id!r}: formula prefixes and control "
            "characters are not allowed"
        )
