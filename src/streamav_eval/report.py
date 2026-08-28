"""Deterministic reports derived only from canonical raw records."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .pvc_fusion import materialize_pvc_records
from .registry import CanonicalRegistry, resolve_dataset_root
from .stability import (
    CONSISTENCY_METRICS,
    INSTRUCTION_DRIFT_METRICS,
    SHARED_IF_EARLY,
    SHARED_IF_LATE,
    build_long_horizon_diagnostics,
)

SHARED_IF_FULL = "VIF-AIF"
IF_METRICS = frozenset(
    {
        "VIF",
        "AIF",
        "P0-VIF",
        "P0-AIF",
        "vif",
        "aif",
        "p0_vif",
        "p0_aif",
        "visual_instruction_following",
        "audio_instruction_following",
    }
)
PROGRESSIVE_ENDPOINT_METRICS = frozenset(
    {"VID-Early", "VID-Late", "AID-Early", "AID-Late"}
)
INTERACTIVE_UPDATE_METRICS = frozenset(
    {
        "VUF",
        "AUF",
        "VSR",
        "ASR",
        "HDF-Adjacent",
        "HDF-Long-Range",
        "PVC",
        "PVC-Algorithm",
        "PAC",
    }
)


def read_raw_records(
    paths: str | Path | Iterable[str | Path],
) -> list[dict[str, Any]]:
    selected = [paths] if isinstance(paths, (str, os.PathLike)) else list(paths)
    records: list[dict[str, Any]] = []
    for path in selected:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                value = json.loads(raw)
                if not isinstance(value, Mapping):
                    raise ValueError(
                        f"{path}:{line_number}: raw record must be an object"
                    )
                records.append(_normalize(value))
    return records


def interval_to_case(
    records: Iterable[Mapping[str, Any]],
    *,
    floor_policy: Mapping[str, float] | None = None,
    expected_intervals: int = 6,
) -> list[dict[str, Any]]:
    """Reduce interval metrics while passing direct case metrics through."""

    del floor_policy
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    direct: list[dict[str, Any]] = []
    for source in records:
        record = _normalize(source)
        metric = str(record["metric"])
        if metric == SHARED_IF_FULL:
            direct.extend(_shared_if_case_records(record))
            continue
        if metric in {
            *INSTRUCTION_DRIFT_METRICS,
            SHARED_IF_EARLY,
            SHARED_IF_LATE,
            *PROGRESSIVE_ENDPOINT_METRICS,
            *INTERACTIVE_UPDATE_METRICS,
        }:
            continue
        if _is_direct_case(record):
            direct.append(_case_record(record))
            continue
        key = (
            str(record["model_id"]),
            str(record["track"]),
            str(record["case_id"]),
            metric,
        )
        groups.setdefault(key, []).append(record)

    cases = direct
    for (model_id, track, case_id, metric), group in sorted(groups.items()):
        intervals: dict[int, dict[str, Any]] = {}
        for record in group:
            index = _interval_index(record)
            if index in intervals:
                raise ValueError(
                    f"duplicate interval {index} for "
                    f"{(model_id, track, case_id, metric)}"
                )
            intervals[index] = record
        counts = _status_counts(group)
        missing = sorted(set(range(1, expected_intervals + 1)) - set(intervals))
        generation = counts["generation_failure"] > 0
        values = [
            _score(record["value"])
            for record in group
            if record["status"] == "computed"
        ]
        provisional = bool(missing or counts["evaluator_failure"])
        if missing or counts["evaluator_failure"]:
            status = "evaluator_failure"
        elif generation:
            status = "generation_failure"
        elif values:
            status = "computed"
        else:
            status = "not_applicable"
        cases.append(
            {
                "model_id": model_id,
                "track": track,
                "case_id": case_id,
                "metric": metric,
                "status": status,
                "value": (
                    math.fsum(values) / len(values)
                    if values and not generation
                    else None
                ),
                "intervals_expected": expected_intervals,
                "intervals_seen": len(intervals),
                "missing_intervals": missing,
                "coverage": counts,
                "provisional": provisional,
            }
        )
    return sorted(
        cases,
        key=lambda item: tuple(
            str(item[key]) for key in ("model_id", "track", "metric", "case_id")
        ),
    )


def aggregate_models(
    case_records: Iterable[Mapping[str, Any]],
    *,
    expected_cases: int | Mapping[str, int] = 160,
    floor_policy: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Macro-average strict ``(model, track, metric)`` groups with coverage."""

    del floor_policy
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for source in case_records:
        record = _normalize(source)
        key = (
            str(record["model_id"]),
            str(record["track"]),
            str(record["metric"]),
        )
        groups.setdefault(key, []).append(record)
    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        model_id, track, metric = key
        case_ids = [str(record["case_id"]) for record in group]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError(f"duplicate case in aggregate {key}")
        expected = (
            int(expected_cases.get(track, expected_cases.get(track.lower(), 160)))
            if isinstance(expected_cases, Mapping)
            else int(expected_cases)
        )
        counts = _status_counts(group)
        values = [
            _score(record["value"])
            for record in group
            if record["status"] == "computed"
        ]
        complete = (
            len(group) == expected
            and counts["evaluator_failure"] == 0
            and not any(bool(record.get("provisional")) for record in group)
        )
        output.append(
            {
                "model_id": model_id,
                "track": track,
                "metric": metric,
                "value": math.fsum(values) / len(values) if values else None,
                "coverage": {
                    "planned": expected,
                    "observed": len(group),
                    "scored": len(values),
                    "excluded_generation_failures": counts["generation_failure"],
                    **counts,
                },
                "complete": complete,
                "provisional": not complete,
            }
        )
    return output


def quality_drift(
    raw_records: Iterable[Mapping[str, Any]],
    *,
    expected_cases: int | Mapping[str, int] = 160,
    floor_policy: Mapping[str, float] | None = None,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> list[dict[str, Any]]:
    return build_long_horizon_diagnostics(
        raw_records,
        expected_cases=expected_cases,
        floor_policy=floor_policy,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=seed,
        bootstrap_confidence=confidence,
    )["quality_drift"]


def build_interactive_diagnostics(
    raw_records: Iterable[Mapping[str, Any]],
    *,
    registry: CanonicalRegistry | None = None,
    dataset_root: str | Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Aggregate update-level Interactive metrics with conditional coverage."""

    records = [
        _normalize(record)
        for record in materialize_pvc_records(raw_records)
        if str(record.get("track", "")).lower() == "interactive"
        and str(record.get("metric", record.get("metric_id", "")))
        in INTERACTIVE_UPDATE_METRICS
    ]
    if not records:
        return {"interactive_cases": [], "interactive_metrics": []}
    expected = _interactive_expected_prompts(
        registry or CanonicalRegistry.load(dataset_root=dataset_root)
    )
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            str(record["model_id"]),
            str(record["metric"]),
            str(record["case_id"]),
        )
        grouped.setdefault(key, []).append(record)
    case_rows: list[dict[str, Any]] = []
    for (model_id, metric, case_id), group in sorted(grouped.items()):
        prompt_ids = [str(row.get("prompt_id", "")) for row in group]
        if not all(prompt_ids) or len(prompt_ids) != len(set(prompt_ids)):
            raise ValueError(
                f"invalid or duplicate update IDs for {(model_id, metric, case_id)}"
            )
        planned = expected.get((metric, case_id), ())
        missing = sorted(set(planned) - set(prompt_ids))
        counts = _status_counts(group)
        values = [_score(row["value"]) for row in group if row["status"] == "computed"]
        required_retention_missing = (
            metric in {"VSR", "ASR"} and counts["not_applicable"] > 0
        )
        failed = bool(
            missing or counts["evaluator_failure"] or required_retention_missing
        )
        row: dict[str, Any] = {
            "model_id": model_id,
            "track": "interactive",
            "case_id": case_id,
            "metric": metric,
            "status": (
                "evaluator_failure"
                if failed
                else "computed"
                if values
                else "not_applicable"
            ),
            "value": math.fsum(values) / len(values) if values else None,
            "coverage": {
                "planned_updates": len(planned),
                "observed_updates": len(group),
                "scored_updates": len(values),
                "missing_prompt_ids": missing,
                **counts,
            },
            "provisional": failed,
        }
        if metric in {"VUF", "AUF"}:
            timings = [
                artifacts.get("timing")
                for item in group
                if item["status"] == "computed"
                if isinstance((artifacts := item.get("artifacts")), Mapping)
            ]
            started = [
                item
                for item in timings
                if isinstance(item, Mapping) and item.get("started")
            ]
            targets = [
                item
                for item in timings
                if isinstance(item, Mapping)
                and item.get("target_achievement_latency_s") is not None
            ]
            onset_values = [
                _score(item["onset_latency_s"])
                for item in started
                if item.get("onset_latency_s") is not None
            ]
            target_values = [
                _score(item["target_achievement_latency_s"]) for item in targets
            ]
            denominator = len(timings)
            row.update(
                {
                    "response_rate": len(started) / denominator
                    if denominator
                    else None,
                    "target_achievement_rate": (
                        len(targets) / denominator if denominator else None
                    ),
                    "conditional_onset_latency_s": (
                        math.fsum(onset_values) / len(onset_values)
                        if onset_values
                        else None
                    ),
                    "conditional_target_latency_s": (
                        math.fsum(target_values) / len(target_values)
                        if target_values
                        else None
                    ),
                }
            )
        if metric.startswith("HDF-"):
            established = [
                artifacts["source_state_established"]
                for item in group
                if item["status"] in {"computed", "not_applicable"}
                if isinstance((artifacts := item.get("artifacts")), Mapping)
                and isinstance(artifacts.get("source_state_established"), bool)
            ]
            row["source_establishment_rate"] = (
                sum(established) / len(established) if established else None
            )
        if metric == "VSR":
            for output_name, response_name in (
                ("subject_retention", "subject_retention"),
                ("environment_retention", "environment_retention"),
            ):
                component_values = []
                for item in group:
                    if item["status"] != "computed":
                        continue
                    artifacts = item.get("artifacts")
                    response = (
                        artifacts.get("response")
                        if isinstance(artifacts, Mapping)
                        else None
                    )
                    component = (
                        response.get(response_name)
                        if isinstance(response, Mapping)
                        else None
                    )
                    score = (
                        component.get("score")
                        if isinstance(component, Mapping)
                        else None
                    )
                    if score is None:
                        raise ValueError(
                            f"computed VSR record lacks numeric {response_name}"
                        )
                    component_values.append(_score(score))
                row[output_name] = (
                    math.fsum(component_values) / len(component_values)
                    if component_values
                    else None
                )
        if metric == "PVC":
            for field in ("algorithm_score", "mllm_score"):
                component_values = [
                    _score(scores[field])
                    for item in group
                    if item["status"] == "computed"
                    if isinstance((scores := item.get("worker_scores")), Mapping)
                    and field in scores
                ]
                row[field] = (
                    math.fsum(component_values) / len(component_values)
                    if component_values
                    else None
                )
        case_rows.append(row)

    hdf_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in case_rows:
        if str(row["metric"]).startswith("HDF-"):
            hdf_groups.setdefault((row["model_id"], row["case_id"]), []).append(row)
    for (model_id, case_id), group in sorted(hdf_groups.items()):
        scored_updates = [int(row["coverage"]["scored_updates"]) for row in group]
        scored_total = sum(scored_updates)
        observed_total = sum(int(row["coverage"]["observed_updates"]) for row in group)
        value = (
            math.fsum(
                float(row["value"]) * count
                for row, count in zip(group, scored_updates, strict=True)
                if row["value"] is not None
            )
            / scored_total
            if scored_total
            else None
        )
        establishment_total = math.fsum(
            float(row["source_establishment_rate"])
            * int(row["coverage"]["observed_updates"])
            for row in group
            if row.get("source_establishment_rate") is not None
        )
        provisional = any(bool(row["provisional"]) for row in group)
        case_rows.append(
            {
                "model_id": model_id,
                "track": "interactive",
                "case_id": case_id,
                "metric": "HDF",
                "status": (
                    "evaluator_failure"
                    if provisional
                    else "computed"
                    if value is not None
                    else "not_applicable"
                ),
                "value": value,
                "source_establishment_rate": (
                    establishment_total / observed_total if observed_total else None
                ),
                "coverage": {
                    "planned_updates": sum(
                        int(row["coverage"]["planned_updates"]) for row in group
                    ),
                    "observed_updates": observed_total,
                    "scored_updates": scored_total,
                    "missing_prompt_ids": [
                        prompt_id
                        for row in group
                        for prompt_id in row["coverage"]["missing_prompt_ids"]
                    ],
                },
                "provisional": provisional,
            }
        )

    model_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in case_rows:
        model_groups.setdefault((row["model_id"], row["metric"]), []).append(row)
    model_rows: list[dict[str, Any]] = []
    for (model_id, metric), group in sorted(model_groups.items()):
        values = [
            _score(row["value"])
            for row in group
            if row["status"] == "computed" and row["value"] is not None
        ]
        eligible_cases = (
            len(
                {
                    case_id
                    for (name, case_id), prompt_ids in expected.items()
                    if name.startswith("HDF-") and prompt_ids
                }
            )
            if metric == "HDF"
            else sum(
                bool(prompt_ids)
                for (name, _case_id), prompt_ids in expected.items()
                if name == metric
            )
        )
        complete = len(group) == eligible_cases and not any(
            row["provisional"] for row in group
        )
        model = {
            "model_id": model_id,
            "track": "interactive",
            "metric": metric,
            "value": math.fsum(values) / len(values) if values else None,
            "coverage": {
                "planned_cases": eligible_cases,
                "observed_cases": len(group),
                "scored_cases": len(values),
                "not_applicable_cases": sum(
                    row["status"] == "not_applicable" for row in group
                ),
            },
            "complete": complete,
            "provisional": not complete,
        }
        for field in (
            "response_rate",
            "target_achievement_rate",
            "conditional_onset_latency_s",
            "conditional_target_latency_s",
            "source_establishment_rate",
            "subject_retention",
            "environment_retention",
            "algorithm_score",
            "mllm_score",
        ):
            field_values = [
                _score(row[field]) for row in group if row.get(field) is not None
            ]
            if field_values:
                model[field] = math.fsum(field_values) / len(field_values)
        model_rows.append(model)
    return {
        "interactive_cases": case_rows,
        "interactive_metrics": model_rows,
    }


def _interactive_expected_prompts(
    registry: CanonicalRegistry,
) -> dict[tuple[str, str], tuple[str, ...]]:
    output: dict[tuple[str, str], tuple[str, ...]] = {}
    for case in registry.values():
        if case.track.value != "interactive":
            continue
        selected: dict[str, list[str]] = {
            metric: [] for metric in INTERACTIVE_UPDATE_METRICS
        }
        for prompt in case.prompts[1:]:
            modality = prompt.payload.get("update_modality")
            dependency = prompt.payload.get("temporal_dependency")
            if modality in {"Video-only", "Joint Audio-Video"}:
                selected["VUF"].append(prompt.prompt_id)
            if modality in {"Audio-only", "Joint Audio-Video"}:
                selected["AUF"].append(prompt.prompt_id)
            for metric in ("VSR", "ASR", "PVC", "PVC-Algorithm", "PAC"):
                selected[metric].append(prompt.prompt_id)
            if dependency == "Adjacent":
                selected["HDF-Adjacent"].append(prompt.prompt_id)
            elif dependency == "Long-Range":
                selected["HDF-Long-Range"].append(prompt.prompt_id)
        for metric, prompt_ids in selected.items():
            output[(metric, case.case_id)] = tuple(prompt_ids)
    return output


def build_report(
    raw_records: Iterable[Mapping[str, Any]],
    *,
    expected_cases: int | Mapping[str, int] = 160,
    floor_policy: Mapping[str, float] | None = None,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    bootstrap_confidence: float = 0.95,
    registry: CanonicalRegistry | None = None,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    normalized = [_normalize(record) for record in raw_records]
    run_ids = {str(record.get("run_id", "")) for record in normalized}
    if len(run_ids) > 1:
        raise ValueError(
            "raw records from multiple run_id values cannot be aggregated together"
        )
    report_run_id = next(iter(run_ids), "")
    raw = _latest_attempts(materialize_pvc_records(normalized))
    cases = interval_to_case(
        raw + _quality_subdimension_records(raw),
        floor_policy=floor_policy,
    )
    models = aggregate_models(
        cases, expected_cases=expected_cases, floor_policy=floor_policy
    )
    long_horizon = build_long_horizon_diagnostics(
        raw,
        expected_cases=expected_cases,
        floor_policy=floor_policy,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_confidence=bootstrap_confidence,
    )
    interactive = build_interactive_diagnostics(
        raw,
        registry=registry,
        dataset_root=dataset_root,
    )
    cases.extend(
        row
        for row in interactive["interactive_cases"]
        if row["metric"] != "PVC-Algorithm"
    )
    models.extend(
        row
        for row in interactive["interactive_metrics"]
        if row["metric"] != "PVC-Algorithm"
    )
    diagnostic_rows = [
        row
        for key in (
            "trajectories",
            "trajectory_statistics",
            "quality_drift",
            "cross_modal_stability",
            "instruction_drift",
            "visual_consistency",
        )
        for row in long_horizon[key]
    ]
    return {
        "schema_version": "streamavbench.report.v2",
        "run_id": report_run_id or None,
        "provisional": any(item["provisional"] for item in models)
        or any(item["provisional"] for item in diagnostic_rows),
        "cases": cases,
        "models": models,
        **long_horizon,
        **interactive,
        "failures": [
            record
            for record in raw
            if record["status"] in {"generation_failure", "evaluator_failure"}
        ],
    }


generate_report = build_report


def _quality_subdimension_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        metric = str(record.get("metric"))
        if metric not in {"VQ", "AQ"}:
            continue
        scores = record.get("worker_scores")
        if not isinstance(scores, Mapping):
            continue
        for name, value in scores.items():
            if (
                name == "mean"
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                continue
            output.append(
                {
                    **record,
                    "metric": f"{metric}.{name}",
                    "value": float(value),
                }
            )
    return output


def write_report_artifacts(
    output_dir: str | Path, report: Mapping[str, Any]
) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": root / "report.json",
        "cases_jsonl": root / "cases.jsonl",
        "models_jsonl": root / "models.jsonl",
        "models_csv": root / "models.csv",
        "failures_jsonl": root / "failures.jsonl",
    }
    _atomic_text(
        paths["report"],
        json.dumps(dict(report), indent=2, sort_keys=True) + "\n",
    )
    for name, key in (
        ("cases_jsonl", "cases"),
        ("models_jsonl", "models"),
        ("failures_jsonl", "failures"),
    ):
        rows = report.get(key, [])
        _atomic_text(
            paths[name],
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        )
    _atomic_csv(
        paths["models_csv"],
        [_flatten_model(row) for row in report.get("models", [])],
    )
    return paths


def _normalize(source: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(source)
    record["metric"] = record.get("metric", record.get("metric_id"))
    record["value"] = record.get("value", record.get("score"))
    status = str(record.get("status", "computed"))
    record["status"] = {"ok": "computed", "scored": "computed"}.get(status, status)
    for key in ("model_id", "track", "case_id", "metric"):
        if record.get(key) in (None, ""):
            raise ValueError(f"record is missing {key}")
    if record["status"] not in {
        "computed",
        "not_applicable",
        "generation_failure",
        "evaluator_failure",
    }:
        raise ValueError(f"unknown status {record['status']!r}")
    return record


def _latest_attempts(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], tuple[int, int, dict[str, Any]]] = {}
    unkeyed: list[tuple[int, dict[str, Any]]] = []
    for position, record in enumerate(records):
        job_id = record.get("job_id")
        if job_id in (None, ""):
            unkeyed.append((position, record))
            continue
        key = (str(record.get("run_id", "")), str(job_id))
        attempt = record.get("attempts", 0)
        rank = (
            int(attempt)
            if isinstance(attempt, int) and not isinstance(attempt, bool)
            else 0
        )
        previous = latest.get(key)
        if previous is None or (rank, position) >= previous[:2]:
            latest[key] = (rank, position, record)
    selected = unkeyed + [
        (position, record) for _rank, position, record in latest.values()
    ]
    return [record for _, record in sorted(selected, key=lambda item: item[0])]


def _shared_if_case_records(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    outputs = record.get("outputs")
    status = str(record["status"])
    rows: list[dict[str, Any]] = []
    for metric in ("VIF", "AIF"):
        value = (
            outputs.get(metric)
            if status == "computed" and isinstance(outputs, Mapping)
            else None
        )
        if status == "computed" and value is None:
            raise ValueError(f"shared VIF-AIF result lacks {metric}")
        rows.append(
            {
                "model_id": record["model_id"],
                "track": record["track"],
                "case_id": record["case_id"],
                "metric": metric,
                "status": status,
                "value": _score(value) if value is not None else None,
                "coverage": _status_counts([record]),
                "provisional": status == "evaluator_failure",
            }
        )
    return rows


def _case_record(record: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "model_id": record["model_id"],
        "track": record["track"],
        "case_id": record["case_id"],
        "metric": record["metric"],
        "status": record["status"],
        "value": (_score(record["value"]) if record["status"] == "computed" else None),
        "coverage": _status_counts([record]),
        "provisional": record["status"] == "evaluator_failure",
    }
    for key in ("artifacts", "protocol", "worker_scores", "phase", "prompt_id"):
        if key in record:
            row[key] = record[key]
    return row


def _is_direct_case(record: Mapping[str, Any]) -> bool:
    return (
        str(record["metric"]) in IF_METRICS | set(CONSISTENCY_METRICS)
        or record.get("interval", record.get("interval_index")) is None
    )


def _interval_index(record: Mapping[str, Any]) -> int:
    value = record.get("interval", record.get("interval_index"))
    if isinstance(value, Mapping):
        value = value.get("index")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("interval must be an integer")
    return value


def _status_counts(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    counts = {
        "computed": 0,
        "not_applicable": 0,
        "generation_failure": 0,
        "evaluator_failure": 0,
    }
    for record in records:
        counts[str(record["status"])] += 1
    return counts


def _score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"score must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"score must be finite, got {value!r}")
    return result


def _flatten_model(row: Mapping[str, Any]) -> dict[str, Any]:
    flat = {key: value for key, value in row.items() if key != "coverage"}
    coverage = row.get("coverage", {})
    if isinstance(coverage, Mapping):
        flat.update({f"coverage_{key}": value for key, value in coverage.items()})
    return flat


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        _atomic_text(path, "")
        return
    fieldnames = sorted({key for row in rows for key in row})
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="streamav-eval aggregate")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, Mapping):
            raise ValueError("report config root must be an object")
        _validate_report_config(config)
        base = args.config.resolve().parent
        raw_paths = [_config_relative_path(path, base) for path in config["raw_paths"]]
        records = read_raw_records(raw_paths)
        report = build_report(
            records,
            expected_cases=config.get("expected_cases", 160),
            floor_policy=config.get("floor_policy"),
            bootstrap_samples=int(config.get("bootstrap_samples", 10_000)),
            bootstrap_seed=int(config.get("bootstrap_seed", 0)),
            bootstrap_confidence=float(config.get("bootstrap_confidence", 0.95)),
            dataset_root=(
                _config_relative_path(config["dataset_root"], base)
                if "dataset_root" in config
                else resolve_dataset_root()
            ),
        )
        write_report_artifacts(
            _config_relative_path(config["output_dir"], base),
            report,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


def _config_relative_path(value: Any, base: Path) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        raise ValueError("configured paths must be non-empty strings")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _validate_report_config(config: Mapping[str, Any]) -> None:
    allowed = {
        "raw_paths",
        "output_dir",
        "dataset_root",
        "expected_cases",
        "floor_policy",
        "bootstrap_samples",
        "bootstrap_seed",
        "bootstrap_confidence",
    }
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"unknown report config fields: {unknown}")
    raw_paths = config.get("raw_paths")
    if (
        isinstance(raw_paths, (str, bytes))
        or not isinstance(raw_paths, Sequence)
        or not raw_paths
        or not all(isinstance(path, str) and path for path in raw_paths)
        or len(set(raw_paths)) != len(raw_paths)
    ):
        raise ValueError("raw_paths must be a non-empty array of unique strings")
    if not isinstance(config.get("output_dir"), str) or not config["output_dir"]:
        raise ValueError("output_dir must be a non-empty string")
    if "dataset_root" in config and (
        not isinstance(config["dataset_root"], str) or not config["dataset_root"]
    ):
        raise ValueError("dataset_root must be a non-empty string")
    samples = config.get("bootstrap_samples", 10_000)
    seed = config.get("bootstrap_seed", 0)
    confidence = config.get("bootstrap_confidence", 0.95)
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap_seed must be an integer")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 < confidence < 1
    ):
        raise ValueError("bootstrap_confidence must be between zero and one")


if __name__ == "__main__":
    raise SystemExit(main())
