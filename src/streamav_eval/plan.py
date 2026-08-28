"""Deterministic expansion of source records into metric jobs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from typing import Any

from .contracts import ContractError, ManifestRecord, MetricJob, RegistryCase, Track
from .metrics import (
    AID_EARLY,
    AID_LATE,
    AIF,
    ASR,
    AUF,
    AVSYNC,
    BC,
    HDF_ADJ,
    HDF_LR,
    INTERVAL_METRICS,
    P0_AIF,
    P0_VIF,
    PAC,
    PVC,
    PVC_ALGORITHM,
    SC,
    SHARED_IF_EARLY,
    SHARED_IF_FULL,
    SHARED_IF_LATE,
    VID_EARLY,
    VID_LATE,
    VIF,
    VSR,
    VUF,
    canonical_metric,
)
from .units import build_units
from .workers.interactive.pvc import PVC_ALGORITHM_VERSION
from .workers.mllm.checklists import criteria_from_record
from .workers.mllm.routing import route_protocol_sha256

# Kept here as part of the planner protocol, not as an EvaluationUnit kind.
AVSYNC_WINDOW_SECONDS = 4.8


def plan_jobs(
    *,
    run_id: str,
    record: ManifestRecord,
    case: RegistryCase,
    metrics: Iterable[str] | None = None,
) -> tuple[MetricJob, ...]:
    """Create jobs for the complete protocol or an explicit metric subset."""

    try:
        stat = record.video_path.stat()
    except OSError as exc:
        raise ContractError(
            f"cannot stat input video {record.video_path}: {exc}"
        ) from exc

    units = build_units(run_id=run_id, record=record, case=case)
    segments = [unit for unit in units if unit.kind.value == "segment"]
    instruction = [unit for unit in units if unit.kind.value == "instruction_following"]
    if len(segments) != 6 or len(instruction) != 1:
        raise ContractError("unit protocol must contain six intervals and one IF unit")

    selected = (
        None
        if metrics is None
        else frozenset(canonical_metric(metric) for metric in metrics)
    )
    internal_metrics = {SHARED_IF_FULL, SHARED_IF_EARLY, SHARED_IF_LATE}
    if selected is not None and (unsupported := selected & internal_metrics):
        names = ", ".join(sorted(unsupported))
        raise ContractError(
            f"internal composite metrics cannot be requested directly: {names}"
        )
    jobs: list[MetricJob] = []
    for interval, unit in enumerate(segments, 1):
        for metric in INTERVAL_METRICS:
            if selected is not None and metric not in selected:
                continue
            options: dict[str, Any] = {
                "clip_start_seconds": unit.start_seconds,
                "clip_end_seconds": unit.end_seconds,
                "case_id": case.case_id,
            }
            if metric == AVSYNC:
                options["avsync_windows"] = {
                    "first": {
                        "start_seconds": unit.start_seconds,
                        "end_seconds": unit.start_seconds + AVSYNC_WINDOW_SECONDS,
                    },
                    "last": {
                        "start_seconds": unit.end_seconds - AVSYNC_WINDOW_SECONDS,
                        "end_seconds": unit.end_seconds,
                    },
                    "window_seconds": AVSYNC_WINDOW_SECONDS,
                }
            jobs.append(
                _job(
                    run_id=run_id,
                    record=record,
                    case=case,
                    metric=metric,
                    start=unit.start_seconds,
                    end=unit.end_seconds,
                    interval=interval,
                    size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    options=options,
                )
            )

    if case.track is Track.PROGRESSIVE:
        jobs.extend(
            _progressive_jobs(
                run_id,
                record,
                case,
                stat.st_size,
                stat.st_mtime_ns,
                selected,
            )
        )
    else:
        jobs.extend(
            _interactive_jobs(
                run_id,
                record,
                case,
                stat.st_size,
                stat.st_mtime_ns,
                selected,
            )
        )
    return tuple(jobs)


def _progressive_jobs(
    run_id: str,
    record: ManifestRecord,
    case: RegistryCase,
    size: int,
    mtime_ns: int,
    selected: frozenset[str] | None,
) -> list[MetricJob]:
    jobs: list[MetricJob] = []
    definitions = (
        (VIF, "video", 0.0, 180.0, "full"),
        (AIF, "audio", 0.0, 180.0, "full"),
        (VID_EARLY, "video", 0.0, 30.0, "early"),
        (VID_LATE, "video", 150.0, 180.0, "late"),
        (AID_EARLY, "audio", 0.0, 30.0, "early"),
        (AID_LATE, "audio", 150.0, 180.0, "late"),
    )
    for metric, modality, start, end, phase in definitions:
        if selected is not None and metric not in selected:
            continue
        criteria = [
            criterion.to_dict()
            for criterion in criteria_from_record(case.source, modality)
        ]
        jobs.append(
            _job(
                run_id=run_id,
                record=record,
                case=case,
                metric=metric,
                start=start,
                end=end,
                interval=None,
                size=size,
                mtime_ns=mtime_ns,
                prompt_ids=("P0",),
                options={
                    "case_id": case.case_id,
                    "prompt_ids": ["P0"],
                    "criteria": criteria,
                    "modality": modality,
                    "phase": phase,
                },
            )
        )
    for metric in (SC, BC):
        if selected is None or metric in selected:
            jobs.append(
                _job(
                    run_id=run_id,
                    record=record,
                    case=case,
                    metric=metric,
                    start=0.0,
                    end=180.0,
                    interval=None,
                    size=size,
                    mtime_ns=mtime_ns,
                    options={"case_id": case.case_id, "phase": "full"},
                )
            )
    return jobs


def _interactive_jobs(
    run_id: str,
    record: ManifestRecord,
    case: RegistryCase,
    size: int,
    mtime_ns: int,
    selected: frozenset[str] | None,
) -> list[MetricJob]:
    jobs: list[MetricJob] = []
    for metric, modality in ((P0_VIF, "video"), (P0_AIF, "audio")):
        if selected is not None and metric not in selected:
            continue
        criteria = [
            criterion.to_dict()
            for criterion in criteria_from_record(case.source, modality)
        ]
        jobs.append(
            _job(
                run_id=run_id,
                record=record,
                case=case,
                metric=metric,
                start=0.0,
                end=30.0,
                interval=None,
                size=size,
                mtime_ns=mtime_ns,
                prompt_ids=("P0",),
                options={
                    "case_id": case.case_id,
                    "prompt_ids": ["P0"],
                    "criteria": criteria,
                    "modality": modality,
                    "phase": "p0",
                },
            )
        )

    by_id = {prompt.prompt_id: prompt for prompt in case.prompts}
    for prompt in case.prompts[1:]:
        try:
            prompt_number = int(prompt.prompt_id.removeprefix("P"))
        except ValueError as exc:
            raise ContractError(
                f"invalid runtime prompt id {prompt.prompt_id}"
            ) from exc
        if prompt_number not in range(1, 6):
            raise ContractError("Interactive runtime prompt IDs must be P1-P5")
        interval = prompt_number + 1
        start, end = prompt_number * 30.0, (prompt_number + 1) * 30.0
        modality = str(prompt.payload.get("update_modality"))
        dependency = str(prompt.payload.get("temporal_dependency"))
        update_by_metric = {
            VUF: {
                "visual_update": prompt.payload.get("visual_prompt") or prompt.text
            },
            AUF: {"audio_update": prompt.payload.get("audio_prompt") or prompt.text},
        }
        applicable_uf = {
            "Video-only": (VUF,),
            "Audio-only": (AUF,),
            "Joint Audio-Video": (VUF, AUF),
        }.get(modality)
        if applicable_uf is None:
            raise ContractError(f"unknown update modality {modality!r}")
        for metric in applicable_uf:
            if selected is not None and metric not in selected:
                continue
            jobs.append(
                _interactive_job(
                    run_id,
                    record,
                    case,
                    metric,
                    start,
                    end,
                    interval,
                    size,
                    mtime_ns,
                    prompt,
                    {
                        **update_by_metric[metric],
                        "update_modality": modality,
                        "sampled_timestamps_seconds": [
                            index / 2 for index in range(60)
                        ],
                        "phase": "runtime_update",
                    },
                )
            )

        windows = [
            {"role": "previous", "start_seconds": start - 30.0, "end_seconds": start},
            {"role": "current", "start_seconds": start, "end_seconds": end},
        ]
        if selected is None or VSR in selected:
            jobs.append(
                _interactive_job(
                    run_id,
                    record,
                    case,
                    VSR,
                    start,
                    end,
                    interval,
                    size,
                    mtime_ns,
                    prompt,
                    {
                        "current_update": prompt.text,
                        "media_windows": windows,
                        "phase": "state_retention",
                    },
                )
            )
        if selected is None or ASR in selected:
            jobs.append(
                _interactive_job(
                    run_id,
                    record,
                    case,
                    ASR,
                    start,
                    end,
                    interval,
                    size,
                    mtime_ns,
                    prompt,
                    {
                        "current_update": prompt.text,
                        "media_windows": windows,
                        "phase": "state_retention",
                    },
                )
            )

        if dependency in {"Adjacent", "Long-Range"}:
            metric = HDF_ADJ if dependency == "Adjacent" else HDF_LR
            if selected is None or metric in selected:
                source_id = prompt.payload.get("depends_on")
                if not isinstance(source_id, str) or source_id not in by_id:
                    raise ContractError(
                        f"{case.case_id}:{prompt.prompt_id} has invalid depends_on"
                    )
                source_number = int(source_id.removeprefix("P"))
                source_start = source_number * 30.0
                jobs.append(
                    _interactive_job(
                        run_id,
                        record,
                        case,
                        metric,
                        start,
                        end,
                        interval,
                        size,
                        mtime_ns,
                        prompt,
                        {
                            "source_prompt_id": source_id,
                            "source_prompt": by_id[source_id].text,
                            "current_update": prompt.text,
                            "dependency_class": dependency,
                            "media_windows": [
                                {
                                    "role": "source",
                                    "start_seconds": source_start,
                                    "end_seconds": source_start + 30.0,
                                },
                                {
                                    "role": "current",
                                    "start_seconds": start,
                                    "end_seconds": end,
                                },
                            ],
                            "phase": "history_dependency",
                        },
                    )
                )

        for metric in (PVC, PAC):
            if selected is None or metric in selected:
                jobs.append(
                    _interactive_job(
                        run_id,
                        record,
                        case,
                        metric,
                        start - 2.0,
                        start + 2.0,
                        interval,
                        size,
                        mtime_ns,
                        prompt,
                        {
                            "current_update": prompt.text,
                            "boundary_seconds": start,
                            "phase": "transition_boundary",
                            **(
                                {"pvc_algorithm_version": PVC_ALGORITHM_VERSION}
                                if metric == PVC
                                else {}
                            ),
                        },
                    )
                )
        if selected is not None and PVC_ALGORITHM in selected:
            jobs.append(
                _interactive_job(
                    run_id,
                    record,
                    case,
                    PVC_ALGORITHM,
                    start - 2.0,
                    start + 2.0,
                    interval,
                    size,
                    mtime_ns,
                    prompt,
                    {
                        "current_update": prompt.text,
                        "boundary_seconds": start,
                        "phase": "transition_boundary",
                        "pvc_algorithm_version": PVC_ALGORITHM_VERSION,
                    },
                )
            )
    return jobs


def _interactive_job(
    run_id: str,
    record: ManifestRecord,
    case: RegistryCase,
    metric: str,
    start: float,
    end: float,
    interval: int,
    size: int,
    mtime_ns: int,
    prompt: Any,
    extra_options: Mapping[str, Any],
) -> MetricJob:
    return _job(
        run_id=run_id,
        record=record,
        case=case,
        metric=metric,
        start=start,
        end=end,
        interval=interval,
        size=size,
        mtime_ns=mtime_ns,
        prompt_ids=(prompt.prompt_id,),
        options={
            "case_id": case.case_id,
            "prompt_id": prompt.prompt_id,
            "prompt_ids": [prompt.prompt_id],
            **dict(extra_options),
        },
    )


def plan_manifest(
    *,
    run_id: str,
    records: Iterable[ManifestRecord],
    registry: Any,
    metrics: Iterable[str] | None = None,
) -> tuple[MetricJob, ...]:
    """Plan records against an object exposing ``get(case_id)``."""

    return tuple(
        job
        for record in records
        for job in plan_jobs(
            run_id=run_id,
            record=record,
            case=registry.get(record.case_id),
            metrics=metrics,
        )
    )


def job_to_dict(job: MetricJob) -> dict[str, Any]:
    value = asdict(job)
    value["track"] = job.track.value
    value["video_path"] = str(job.video_path)
    if job.audio_path is not None:
        value["audio_path"] = str(job.audio_path)
    else:
        for key in (
            "audio_path",
            "audio_input_size_bytes",
            "audio_input_mtime_ns",
        ):
            value.pop(key, None)
    value["prompt_ids"] = list(job.prompt_ids)
    value["output_metrics"] = list(job.output_metrics)
    return value


def _job(
    *,
    run_id: str,
    record: ManifestRecord,
    case: RegistryCase,
    metric: str,
    start: float,
    end: float,
    interval: int | None,
    size: int,
    mtime_ns: int,
    prompt_ids: tuple[str, ...] = (),
    output_metrics: tuple[str, ...] = (),
    options: dict[str, Any],
) -> MetricJob:
    try:
        evaluator_protocol_sha256 = route_protocol_sha256(metric)
    except ValueError:
        pass
    else:
        options = {
            **options,
            "evaluator_protocol_sha256": evaluator_protocol_sha256,
        }
    audio_stat = None
    if record.audio_path is not None:
        try:
            audio_stat = record.audio_path.stat()
        except OSError as exc:
            raise ContractError(
                f"cannot stat input audio {record.audio_path}: {exc}"
            ) from exc
    scope = (
        f"i{interval:02d}"
        if interval is not None
        else ("full180" if case.track is Track.PROGRESSIVE else "first30")
    )
    components = [
        run_id,
        record.model_id,
        case.track.value,
        case.case_id,
        metric,
        scope,
        f"size-{size}",
        f"mtime-{mtime_ns}",
    ]
    if audio_stat is not None:
        components.extend(
            (
                f"audio-size-{audio_stat.st_size}",
                f"audio-mtime-{audio_stat.st_mtime_ns}",
            )
        )
    identity = {
        "run_id": run_id,
        "model_id": record.model_id,
        "track": case.track.value,
        "case_id": case.case_id,
        "scope": scope,
        "start_seconds": start,
        "end_seconds": end,
        "interval": interval,
        "input_size_bytes": size,
        "input_mtime_ns": mtime_ns,
        "audio_input_size_bytes": (
            audio_stat.st_size if audio_stat is not None else None
        ),
        "audio_input_mtime_ns": (
            audio_stat.st_mtime_ns if audio_stat is not None else None
        ),
        "prompt_ids": list(prompt_ids),
        "output_metrics": list(output_metrics),
        "options": options,
    }
    digest = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    components.append(f"spec-{digest}")
    job_id = "__".join(_id_component(value) for value in components)
    return MetricJob(
        job_id=job_id,
        run_id=run_id,
        model_id=record.model_id,
        case_id=case.case_id,
        track=case.track,
        metric=metric,
        video_path=record.video_path,
        start_seconds=start,
        end_seconds=end,
        interval=interval,
        input_size_bytes=size,
        input_mtime_ns=mtime_ns,
        prompt_ids=prompt_ids,
        output_metrics=output_metrics,
        options=options,
        audio_path=record.audio_path,
        audio_input_size_bytes=(audio_stat.st_size if audio_stat is not None else None),
        audio_input_mtime_ns=(
            audio_stat.st_mtime_ns if audio_stat is not None else None
        ),
    )


def _id_component(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip("-")
    if not cleaned:
        raise ContractError("job ID component has no usable characters")
    return cleaned
