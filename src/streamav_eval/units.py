"""Deterministic temporal evaluation-unit construction."""

from __future__ import annotations

import re

from .contracts import (
    ContractError,
    EvaluationUnit,
    ManifestRecord,
    RegistryCase,
    Track,
    UnitKind,
)

BENCHMARK_DURATION_SECONDS = 180.0
SEGMENT_SECONDS = 30.0


def build_units(
    *,
    run_id: str,
    record: ManifestRecord,
    case: RegistryCase,
) -> tuple[EvaluationUnit, ...]:
    """Build six intervals and one track-specific instruction unit.

    AVSync is an interval metric.  Its endpoint windows are planner options,
    not independent full-video evaluation units.
    """
    if record.case_id != case.case_id:
        raise ContractError(
            f"manifest case {record.case_id!r} does not match registry {case.case_id!r}"
        )
    if abs(case.duration_seconds - BENCHMARK_DURATION_SECONDS) > 1e-9:
        raise ContractError(
            "unit construction requires the canonical 180-second case duration"
        )
    prefix = "__".join(
        _id_component(value, name)
        for value, name in (
            (run_id, "run_id"),
            (record.model_id, "model_id"),
            (case.case_id, "case_id"),
        )
    )

    units: list[EvaluationUnit] = []
    for index in range(6):
        start = index * SEGMENT_SECONDS
        units.append(
            _unit(
                prefix,
                suffix=f"segment-{index + 1:02d}",
                run_id=run_id,
                record=record,
                case=case,
                kind=UnitKind.SEGMENT,
                start=start,
                end=start + SEGMENT_SECONDS,
                label=f"30s segment {index + 1}/6",
            )
        )

    if case.track is Track.PROGRESSIVE:
        if_start, if_end = 0.0, BENCHMARK_DURATION_SECONDS
        if_label = "Progressive full-video instruction following"
    else:
        if_start, if_end = 0.0, SEGMENT_SECONDS
        if_label = "Interactive P0 instruction following"
    units.append(
        _unit(
            prefix,
            suffix="if-p0",
            run_id=run_id,
            record=record,
            case=case,
            kind=UnitKind.INSTRUCTION_FOLLOWING,
            start=if_start,
            end=if_end,
            prompt_ids=("P0",),
            label=if_label,
        )
    )
    return tuple(units)


def _unit(
    prefix: str,
    *,
    suffix: str,
    run_id: str,
    record: ManifestRecord,
    case: RegistryCase,
    kind: UnitKind,
    start: float,
    end: float,
    prompt_ids: tuple[str, ...] = (),
    label: str,
) -> EvaluationUnit:
    return EvaluationUnit(
        unit_id=f"{prefix}__{suffix}",
        run_id=run_id,
        model_id=record.model_id,
        case_id=case.case_id,
        track=case.track,
        kind=kind,
        start_seconds=start,
        end_seconds=end,
        prompt_ids=prompt_ids,
        label=label,
    )


def _id_component(value: str, name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    if not cleaned:
        raise ContractError(f"{name} has no usable ID characters")
    return cleaned
