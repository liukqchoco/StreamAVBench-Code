"""Authoritative derived math for Progressive long-horizon diagnostics."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from numbers import Real
from typing import Any

QUALITY_METRICS = (
    "VA",
    "VQ",
    "VQ.visual_fidelity",
    "VQ.subject_integrity",
    "VQ.motion_naturalness",
    "VQ.visual_artifacts",
    "PQ",
    "AQ",
    "AQ.audio_naturalness",
    "AQ.audio_artifacts",
)
CROSS_MODAL_METRICS = ("AVAlign", "AVSync")
INSTRUCTION_DRIFT_METRICS = ("VID", "AID")
CONSISTENCY_METRICS = ("SC", "BC")
SHARED_IF_EARLY = "VID-AID-Early"
SHARED_IF_LATE = "VID-AID-Late"
ENDPOINT_SOURCE_METRICS = {
    "VID-Early": ("VID", "early"),
    "VID-Late": ("VID", "late"),
    "AID-Early": ("AID", "early"),
    "AID-Late": ("AID", "late"),
}
TRAJECTORY_SECONDS = (30, 60, 90, 120, 150, 180)
_TERMINAL_STATUSES = {
    "computed",
    "not_applicable",
    "generation_failure",
    "evaluator_failure",
}


def paired_bootstrap_ci(
    values: Sequence[Real],
    *,
    samples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap case indices after each value has been paired within a case."""

    numeric = [_score(value) for value in values]
    _validate_bootstrap(numeric, samples, confidence)
    rng = random.Random(seed)
    size = len(numeric)
    means = [
        math.fsum(numeric[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    ]
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    return _quantile(means, alpha), _quantile(means, 1.0 - alpha)


def build_long_horizon_diagnostics(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_cases: int | Mapping[str, int] = 160,
    floor_policy: Mapping[str, float] | None = None,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    bootstrap_confidence: float = 0.95,
) -> dict[str, list[dict[str, Any]]]:
    """Derive every Progressive stability result from canonical raw records."""

    del floor_policy  # Generation failures are reported and excluded, never floored.
    progressive = [
        record
        for source in records
        if (record := _normalize(source))["track"].lower() == "progressive"
    ]
    expected = _expected_progressive(expected_cases)
    trajectories, trajectory_statistics, quality, cross_modal = _interval_diagnostics(
        progressive + _quality_subdimension_records(progressive),
        expected=expected,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        confidence=bootstrap_confidence,
    )
    instruction_cases, instruction_models = _instruction_drift(
        progressive,
        expected=expected,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        confidence=bootstrap_confidence,
    )
    consistency = _direct_diagnostics(
        progressive,
        expected=expected,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        confidence=bootstrap_confidence,
    )
    return {
        "trajectories": trajectories,
        "trajectory_statistics": trajectory_statistics,
        "quality_drift": quality,
        "cross_modal_stability": cross_modal,
        "instruction_drift_cases": instruction_cases,
        "instruction_drift": instruction_models,
        "visual_consistency": consistency,
    }


def _quality_subdimension_records(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        metric = record["metric"]
        if metric not in {"VQ", "AQ"}:
            continue
        scores = record.get("worker_scores")
        if not isinstance(scores, Mapping):
            continue
        for name, value in scores.items():
            if name == "mean":
                continue
            if isinstance(value, Real) and not isinstance(value, bool):
                output.append(
                    {
                        **record,
                        "metric": f"{metric}.{name}",
                        "value": float(value),
                    }
                )
    return output


def _interval_diagnostics(
    records: Sequence[dict[str, Any]],
    *,
    expected: int,
    samples: int,
    seed: int,
    confidence: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    groups: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = {}
    statuses: dict[tuple[str, str], dict[str, int]] = {}
    observed: dict[tuple[str, str], set[str]] = {}
    allowed = set(QUALITY_METRICS) | set(CROSS_MODAL_METRICS)
    for record in records:
        metric = record["metric"]
        if metric not in allowed:
            continue
        interval = _interval(record)
        key = (record["model_id"], metric, record["case_id"])
        if interval in groups.setdefault(key, {}):
            raise ValueError(f"duplicate interval {interval} for {key}")
        groups[key][interval] = record
        model_key = key[:2]
        observed.setdefault(model_key, set()).add(record["case_id"])
        statuses.setdefault(model_key, _empty_status_counts())[record["status"]] += 1

    trajectories: list[dict[str, Any]] = []
    trajectory_statistics: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    cross_modal: list[dict[str, Any]] = []
    for model_id, metric in sorted(observed):
        case_ids = sorted(observed[(model_id, metric)])
        complete_vectors: list[list[float]] = []
        endpoint_pairs: list[tuple[float, float]] = []
        unresolved_cases = not_applicable_cases = generation_cases = 0
        for case_id in case_ids:
            by_interval = groups[(model_id, metric, case_id)]
            resolved: list[float] = []
            case_unresolved = case_na = case_generation = False
            for interval in range(1, 7):
                record = by_interval.get(interval)
                if record is None:
                    case_unresolved = True
                    continue
                value, reason = _resolved_value(record)
                if reason == "not_applicable":
                    case_na = True
                elif reason == "generation_failure":
                    case_generation = True
                elif reason is not None:
                    case_unresolved = True
                if value is not None:
                    resolved.append(value)
            if case_na:
                not_applicable_cases += 1
            if case_generation:
                generation_cases += 1
            if case_na or case_generation:
                continue
            if case_unresolved or len(resolved) != 6:
                unresolved_cases += 1
                continue
            complete_vectors.append(resolved)
            endpoint_pairs.append((resolved[0], resolved[-1]))

        counts = statuses[(model_id, metric)]
        coverage = {
            "planned_cases": expected,
            "observed_cases": len(case_ids),
            "complete_six_point_cases": len(complete_vectors),
            "paired_endpoint_cases": len(endpoint_pairs),
            "unresolved_cases": unresolved_cases,
            "not_applicable_cases": not_applicable_cases,
            "excluded_generation_failure_cases": generation_cases,
            "raw_intervals_observed": sum(counts.values()),
            **counts,
        }
        complete = len(case_ids) == expected and unresolved_cases == 0
        trajectories.append(
            {
                "model_id": model_id,
                "track": "progressive",
                "metric": metric,
                "points": _trajectory_points(
                    complete_vectors,
                    samples=samples,
                    seed=seed,
                    confidence=confidence,
                ),
                "coverage": coverage,
                "complete": complete,
                "provisional": not complete,
            }
        )
        trajectory_statistics.append(
            _trajectory_statistics_row(
                model_id,
                metric,
                complete_vectors,
                coverage,
                complete,
            )
        )
        if metric in QUALITY_METRICS:
            drift_row = _difference_row(
                model_id,
                metric,
                "quality_drift",
                [abs(early - late) for early, late in endpoint_pairs],
                coverage,
                complete,
                samples,
                seed,
                confidence,
                direction="absolute_endpoint_difference",
                positive_means="greater_drift",
            )
            drift_row["signed_endpoint_change"] = (
                math.fsum(late - early for early, late in endpoint_pairs)
                / len(endpoint_pairs)
                if endpoint_pairs
                else None
            )
            quality.append(drift_row)
        else:
            drift_row = _difference_row(
                model_id,
                "AVAlign-D" if metric == "AVAlign" else "AVSync-D",
                "value",
                [abs(late - early) for early, late in endpoint_pairs],
                coverage,
                complete,
                samples,
                seed,
                confidence,
                direction="absolute_endpoint_difference",
                positive_means="greater_drift",
            )
            drift_row["signed_endpoint_change"] = (
                math.fsum(late - early for early, late in endpoint_pairs)
                / len(endpoint_pairs)
                if endpoint_pairs
                else None
            )
            cross_modal.append(drift_row)
    return trajectories, trajectory_statistics, quality, cross_modal


def _instruction_drift(
    records: Sequence[dict[str, Any]],
    *,
    expected: int,
    samples: int,
    seed: int,
    confidence: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    endpoints: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    status_counts: dict[tuple[str, str], dict[str, int]] = {}
    observed: dict[tuple[str, str], set[str]] = {}
    for record in records:
        source_metric = record["metric"]
        if source_metric not in {
            *INSTRUCTION_DRIFT_METRICS,
            SHARED_IF_EARLY,
            SHARED_IF_LATE,
            *ENDPOINT_SOURCE_METRICS,
        }:
            continue
        derived = (
            (ENDPOINT_SOURCE_METRICS[source_metric][0],)
            if source_metric in ENDPOINT_SOURCE_METRICS
            else INSTRUCTION_DRIFT_METRICS
            if source_metric in {SHARED_IF_EARLY, SHARED_IF_LATE}
            else (source_metric,)
        )
        for metric in derived:
            key = (record["model_id"], metric, record["case_id"])
            endpoint = _endpoint(record)
            if endpoint in endpoints.setdefault(key, {}):
                raise ValueError(f"duplicate {endpoint} endpoint for {key}")
            endpoints[key][endpoint] = record
            model_key = key[:2]
            observed.setdefault(model_key, set()).add(record["case_id"])
            status_counts.setdefault(model_key, _empty_status_counts())[
                record["status"]
            ] += 1

    case_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for model_key in sorted(observed):
        model_id, metric = model_key
        valid: list[dict[str, Any]] = []
        missing = mismatch = out_of_range = 0
        n_a = generation = evaluator = 0
        for case_id in sorted(observed[model_key]):
            pair = endpoints[(model_id, metric, case_id)]
            if set(pair) != {"early", "late"}:
                missing += 1
                continue
            statuses = {pair[name]["status"] for name in ("early", "late")}
            if "evaluator_failure" in statuses:
                evaluator += 1
                continue
            if "generation_failure" in statuses:
                generation += 1
                continue
            if "not_applicable" in statuses:
                n_a += 1
                continue
            early = _criterion_scores(pair["early"], metric)
            late = _criterion_scores(pair["late"], metric)
            if not early and not late:
                n_a += 1
                continue
            if set(early) != set(late):
                mismatch += 1
                continue
            if any(
                not 1.0 <= value <= 5.0 for value in (*early.values(), *late.values())
            ):
                out_of_range += 1
                continue
            criterion_ids = sorted(early)
            changes = [abs(early[name] - late[name]) for name in criterion_ids]
            row = {
                "model_id": model_id,
                "track": "progressive",
                "case_id": case_id,
                "metric": metric,
                "value": math.fsum(changes) / len(changes),
                "early_alignment": math.fsum(early.values()) / len(early),
                "late_alignment": math.fsum(late.values()) / len(late),
                "criterion_count": len(criterion_ids),
                "criterion_ids": criterion_ids,
                "range": [0.0, 4.0],
                "status": "computed",
            }
            valid.append(row)
            case_rows.append(row)

        coverage = {
            "planned_cases": expected,
            "observed_cases": len(observed[model_key]),
            "paired_cases": len(valid),
            "missing_endpoint_cases": missing,
            "criterion_mismatch_cases": mismatch,
            "out_of_range_cases": out_of_range,
            "not_applicable_cases": n_a,
            "generation_failure_cases": generation,
            "evaluator_failure_cases": evaluator,
            **status_counts[model_key],
        }
        complete = len(valid) + n_a + generation == expected and not any(
            (missing, mismatch, out_of_range, evaluator)
        )
        drift = [row["value"] for row in valid]
        early_values = [row["early_alignment"] for row in valid]
        late_values = [row["late_alignment"] for row in valid]
        drift_ci = _optional_ci(drift, samples, seed, confidence)
        early_ci = _optional_ci(early_values, samples, seed, confidence)
        late_ci = _optional_ci(late_values, samples, seed, confidence)
        model_rows.append(
            {
                "model_id": model_id,
                "track": "progressive",
                "metric": metric,
                "value": math.fsum(drift) / len(drift) if drift else None,
                "early_alignment": (
                    math.fsum(early_values) / len(early_values)
                    if early_values
                    else None
                ),
                "late_alignment": (
                    math.fsum(late_values) / len(late_values) if late_values else None
                ),
                "ci_low": drift_ci[0],
                "ci_high": drift_ci[1],
                "early_ci_low": early_ci[0],
                "early_ci_high": early_ci[1],
                "late_ci_low": late_ci[0],
                "late_ci_high": late_ci[1],
                "confidence": confidence,
                "bootstrap_samples": samples,
                "seed": seed,
                "range": [0.0, 4.0],
                "coverage": coverage,
                "complete": complete,
                "provisional": not complete,
            }
        )
    return case_rows, model_rows


def _direct_diagnostics(
    records: Sequence[dict[str, Any]],
    *,
    expected: int,
    samples: int,
    seed: int,
    confidence: float,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for record in records:
        if record["metric"] not in CONSISTENCY_METRICS:
            continue
        key = (record["model_id"], record["metric"])
        if record["case_id"] in groups.setdefault(key, {}):
            raise ValueError(
                f"duplicate consistency case {record['case_id']} for {key}"
            )
        groups[key][record["case_id"]] = record
    output: list[dict[str, Any]] = []
    for (model_id, metric), cases in sorted(groups.items()):
        counts = _empty_status_counts()
        values: list[float] = []
        for record in cases.values():
            counts[record["status"]] += 1
            if record["status"] == "computed":
                values.append(_score(record["value"]))
        low, high = _optional_ci(values, samples, seed, confidence)
        complete = (
            len(cases) == expected
            and counts["evaluator_failure"] == 0
            and counts["computed"]
            + counts["not_applicable"]
            + counts["generation_failure"]
            == expected
        )
        output.append(
            {
                "model_id": model_id,
                "track": "progressive",
                "metric": metric,
                "value": math.fsum(values) / len(values) if values else None,
                "ci_low": low,
                "ci_high": high,
                "confidence": confidence,
                "bootstrap_samples": samples,
                "seed": seed,
                "coverage": {
                    "planned_cases": expected,
                    "observed_cases": len(cases),
                    **counts,
                },
                "complete": complete,
                "provisional": not complete,
                "protocol": "full-rollout consistency",
            }
        )
    return output


def _trajectory_statistics_row(
    model_id: str,
    metric: str,
    vectors: Sequence[Sequence[float]],
    coverage: Mapping[str, Any],
    complete: bool,
) -> dict[str, Any]:
    degradation_slopes = [_case_degradation_slope(metric, vector) for vector in vectors]
    trajectory_variations = [_case_trajectory_variation(vector) for vector in vectors]
    return {
        "model_id": model_id,
        "track": "progressive",
        "metric": metric,
        "degradation_slope_per_100s": (
            math.fsum(degradation_slopes) / len(degradation_slopes)
            if degradation_slopes
            else None
        ),
        "trajectory_variation": (
            math.fsum(trajectory_variations) / len(trajectory_variations)
            if trajectory_variations
            else None
        ),
        "n_cases": len(vectors),
        "aggregation": "compute_per_case_then_macro_average",
        "slope_direction": "positive_means_degradation",
        "variation_interpretation": "greater_means_more_adjacent_interval_change",
        "coverage": dict(coverage),
        "complete": complete,
        "provisional": not complete,
    }


def _case_degradation_slope(metric: str, values: Sequence[float]) -> float:
    if len(values) != len(TRAJECTORY_SECONDS):
        raise ValueError("trajectory slope requires exactly six interval scores")
    scores = [_score(value) for value in values]
    mean_time = math.fsum(TRAJECTORY_SECONDS) / len(TRAJECTORY_SECONDS)
    mean_score = math.fsum(scores) / len(scores)
    denominator = math.fsum(
        (seconds - mean_time) ** 2 for seconds in TRAJECTORY_SECONDS
    )
    raw_slope = (
        math.fsum(
            (seconds - mean_time) * (score - mean_score)
            for seconds, score in zip(TRAJECTORY_SECONDS, scores, strict=True)
        )
        / denominator
    )
    orientation = 1.0 if metric == "AVSync" else -1.0
    return orientation * 100.0 * raw_slope


def _case_trajectory_variation(values: Sequence[float]) -> float:
    if len(values) != len(TRAJECTORY_SECONDS):
        raise ValueError("trajectory variation requires exactly six interval scores")
    scores = [_score(value) for value in values]
    return math.fsum(
        abs(current - previous)
        for previous, current in zip(scores, scores[1:], strict=False)
    ) / (len(scores) - 1)


def _trajectory_points(
    vectors: Sequence[Sequence[float]],
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> list[dict[str, Any]]:
    if not vectors:
        return []
    _validate_bootstrap([1.0], samples, confidence)
    size = len(vectors)
    rng = random.Random(seed)
    draws = [[rng.randrange(size) for _ in range(size)] for _ in range(samples)]
    points: list[dict[str, Any]] = []
    for index, seconds in enumerate(TRAJECTORY_SECONDS):
        values = [vector[index] for vector in vectors]
        boot = sorted(
            math.fsum(values[case_index] for case_index in draw) / size
            for draw in draws
        )
        alpha = (1.0 - confidence) / 2.0
        points.append(
            {
                "interval": index + 1,
                "endpoint_seconds": seconds,
                "value": math.fsum(values) / size,
                "ci_low": _quantile(boot, alpha),
                "ci_high": _quantile(boot, 1.0 - alpha),
                "n_cases": size,
            }
        )
    return points


def _difference_row(
    model_id: str,
    metric: str,
    value_field: str,
    values: Sequence[float],
    coverage: Mapping[str, Any],
    complete: bool,
    samples: int,
    seed: int,
    confidence: float,
    *,
    direction: str,
    positive_means: str = "degradation",
) -> dict[str, Any]:
    low, high = _optional_ci(values, samples, seed, confidence)
    return {
        "model_id": model_id,
        "track": "progressive",
        "metric": metric,
        value_field: math.fsum(values) / len(values) if values else None,
        "ci_low": low,
        "ci_high": high,
        "confidence": confidence,
        "bootstrap_samples": samples,
        "seed": seed,
        "n_cases": len(values),
        "direction": direction,
        "positive_means": positive_means,
        "coverage": dict(coverage),
        "complete": complete,
        "provisional": not complete,
    }


def _resolved_value(record: Mapping[str, Any]) -> tuple[float | None, str | None]:
    status = record["status"]
    if status == "computed":
        return _score(record["value"]), None
    return None, status


def _criterion_scores(record: Mapping[str, Any], metric: str) -> dict[str, float]:
    candidates: list[Any] = [record.get("criterion_scores"), record.get("subscores")]
    details = record.get("details")
    if isinstance(details, Mapping):
        artifacts = details.get("artifacts")
        candidates.extend(
            [
                details.get("criterion_scores"),
                details.get("subscores"),
                (
                    artifacts.get("criterion_scores")
                    if isinstance(artifacts, Mapping)
                    else None
                ),
                (
                    artifacts.get("subscores")
                    if isinstance(artifacts, Mapping)
                    else None
                ),
            ]
        )
    artifacts = record.get("artifacts")
    if isinstance(artifacts, Mapping):
        candidates.extend(
            [artifacts.get("criterion_scores"), artifacts.get("subscores")]
        )
    modality = "video" if metric == "VID" else "audio"
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            alternate = "visual" if modality == "video" else modality
            if modality in candidate:
                candidate = candidate[modality]
            elif alternate in candidate:
                candidate = candidate[alternate]
        parsed = _parse_scores(candidate, modality=modality)
        if parsed is not None:
            return parsed
    raise ValueError(
        f"{metric} endpoint requires criterion_scores/subscores with criterion IDs"
    )


def _parse_scores(value: Any, *, modality: str | None) -> dict[str, float] | None:
    if isinstance(value, Mapping):
        if "criterion_scores" in value:
            return _parse_scores(value["criterion_scores"], modality=modality)
        parsed: dict[str, float] = {}
        for key, item in value.items():
            if key in {"mean", "reason", "overall_reason"}:
                continue
            if (
                modality is not None
                and isinstance(item, Mapping)
                and item.get("modality")
                not in {None, modality, "visual" if modality == "video" else modality}
            ):
                continue
            score = item.get("score") if isinstance(item, Mapping) else item
            if isinstance(score, Real) and not isinstance(score, bool):
                parsed[str(key)] = _score(score)
        return parsed or None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parsed = {}
        for item in value:
            if not isinstance(item, Mapping):
                return None
            criterion_id = item.get("criterion_id", item.get("id"))
            score = item.get("score")
            if modality is not None and item.get("modality") not in {
                None,
                modality,
                "visual" if modality == "video" else modality,
            }:
                continue
            if criterion_id in (None, "") or score is None:
                return None
            key = str(criterion_id)
            if key in parsed:
                raise ValueError(f"duplicate criterion ID {key!r}")
            parsed[key] = _score(score)
        return parsed or None
    return None


def _endpoint(record: Mapping[str, Any]) -> str:
    source = str(record.get("metric"))
    if source in ENDPOINT_SOURCE_METRICS:
        return ENDPOINT_SOURCE_METRICS[source][1]
    if record.get("metric") == SHARED_IF_EARLY:
        return "early"
    if record.get("metric") == SHARED_IF_LATE:
        return "late"
    value = record.get("endpoint", record.get("phase"))
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"early", "first", "start"}:
            return "early"
        if lowered in {"late", "last", "end"}:
            return "late"
    interval = record.get("interval", record.get("interval_index"))
    if isinstance(interval, Mapping):
        interval = interval.get("index")
    if interval == 1:
        return "early"
    if interval == 6:
        return "late"
    raise ValueError("VID/AID record requires early/late endpoint or interval 1/6")


def _interval(record: Mapping[str, Any]) -> int:
    value = record.get("interval", record.get("interval_index"))
    if isinstance(value, Mapping):
        value = value.get("index")
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in range(1, 7)
    ):
        raise ValueError("trajectory record requires interval in [1, 6]")
    return value


def _normalize(source: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(source)
    record["metric"] = str(record.get("metric", record.get("metric_id", "")))
    record["value"] = record.get("value", record.get("score"))
    status = str(record.get("status", "computed"))
    record["status"] = {"ok": "computed", "scored": "computed"}.get(status, status)
    for field in ("model_id", "track", "case_id", "metric"):
        if record.get(field) in (None, ""):
            raise ValueError(f"record is missing {field}")
        record[field] = str(record[field])
    if record["status"] not in _TERMINAL_STATUSES:
        raise ValueError(f"unknown status {record['status']!r}")
    return record


def _expected_progressive(expected: int | Mapping[str, int]) -> int:
    value = (
        expected.get("progressive", expected.get("Progressive", 160))
        if isinstance(expected, Mapping)
        else expected
    )
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("expected Progressive cases must be a positive integer")
    return value


def _empty_status_counts() -> dict[str, int]:
    return {status: 0 for status in sorted(_TERMINAL_STATUSES)}


def _optional_ci(
    values: Sequence[Real], samples: int, seed: int, confidence: float
) -> tuple[float | None, float | None]:
    return (
        paired_bootstrap_ci(values, samples=samples, seed=seed, confidence=confidence)
        if values
        else (None, None)
    )


def _validate_bootstrap(
    values: Sequence[float], samples: int, confidence: float
) -> None:
    if not values:
        raise ValueError("paired bootstrap requires at least one case")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("bootstrap samples must be a positive integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"score must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"score must be finite, got {value!r}")
    return result
