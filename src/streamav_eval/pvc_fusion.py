"""Join independently computed PVC algorithm and MLLM components."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .workers.interactive.pvc import PVC_ALGORITHM_VERSION, fuse_pvc

PVCKey = tuple[str, str, str, str, float, int, int, str]


def read_algorithm_records(
    paths: Iterable[str | Path],
) -> list[dict[str, Any]]:
    import json

    records: list[dict[str, Any]] = []
    for source in paths:
        path = Path(source)
        if not path.is_file():
            raise ValueError(f"PVC algorithm raw file does not exist: {path}")
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"{path}:{line_number}: expected object")
                records.append(dict(value))
    return records


def build_algorithm_index(
    records: Iterable[Mapping[str, Any]],
) -> dict[PVCKey, dict[str, Any]]:
    index: dict[PVCKey, dict[str, Any]] = {}
    for source in records:
        if _metric(source) != "PVC-Algorithm":
            continue
        if str(source.get("status")) not in {"computed", "scored"}:
            continue
        score = _component_score(source, "algorithm_score")
        if score is None:
            raise ValueError(
                f"computed PVC-Algorithm record {pvc_key(source)} lacks score"
            )
        record = dict(source)
        record["_algorithm_score"] = score
        key = pvc_key(record)
        previous = index.get(key)
        if previous is not None:
            previous_attempt = int(previous.get("attempts", 0))
            current_attempt = int(record.get("attempts", 0))
            if current_attempt < previous_attempt:
                continue
            previous_score = float(previous["_algorithm_score"])
            if current_attempt == previous_attempt and previous_score != score:
                raise ValueError(
                    f"conflicting PVC-Algorithm scores for {key}: "
                    f"{previous_score} vs {score}"
                )
        index[key] = record
    return index


def fuse_pvc_record(
    source: Mapping[str, Any],
    algorithm_index: Mapping[PVCKey, Mapping[str, Any]],
) -> dict[str, Any]:
    record = dict(source)
    if _metric(record) != "PVC":
        return record
    mllm_score = _component_score(record, "mllm_score")
    if mllm_score is None:
        return record
    algorithm = algorithm_index.get(pvc_key(record))
    if algorithm is None:
        return record
    _validate_boundary(record, algorithm)
    algorithm_score = _component_score(algorithm, "algorithm_score")
    if algorithm_score is None:
        algorithm_score = float(algorithm["_algorithm_score"])
    score = fuse_pvc(algorithm_score, mllm_score)
    worker_scores = dict(record.get("worker_scores", {}))
    worker_scores.update(
        {
            "algorithm_score": algorithm_score,
            "mllm_score": mllm_score,
            "score": score,
        }
    )
    artifacts = dict(record.get("artifacts", {}))
    algorithm_artifacts = algorithm.get("artifacts")
    if isinstance(algorithm_artifacts, Mapping):
        diagnostics = algorithm_artifacts.get("technical_diagnostics")
        if isinstance(diagnostics, Mapping):
            artifacts["technical_diagnostics"] = dict(diagnostics)
    artifacts["fusion"] = {"algorithm_weight": 0.70, "mllm_weight": 0.30}
    record.update(
        {
            "status": "computed",
            "value": score,
            "worker_scores": worker_scores,
            "artifacts": artifacts,
            "provisional": False,
        }
    )
    record.pop("error", None)
    return record


def materialize_pvc_records(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    values = [dict(record) for record in records]
    index = build_algorithm_index(values)
    return [fuse_pvc_record(record, index) for record in values]


def pvc_key(record: Mapping[str, Any]) -> PVCKey:
    prompt_id = record.get("prompt_id")
    options = record.get("options")
    if (not isinstance(prompt_id, str) or not prompt_id) and isinstance(
        options, Mapping
    ):
        prompt_id = options.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        prompt_ids = record.get("prompt_ids")
        if (
            isinstance(prompt_ids, list)
            and len(prompt_ids) == 1
            and isinstance(prompt_ids[0], str)
        ):
            prompt_id = prompt_ids[0]
    run_id = record.get("run_id")
    model_id, case_id = record.get("model_id"), record.get("case_id")
    boundary = record.get("boundary_seconds")
    if boundary is None and isinstance(options, Mapping):
        boundary = options.get("boundary_seconds")
    algorithm_version = record.get("pvc_algorithm_version")
    if algorithm_version is None and isinstance(options, Mapping):
        algorithm_version = options.get("pvc_algorithm_version")
    if not all(
        isinstance(value, str) and value
        for value in (run_id, model_id, case_id, prompt_id)
    ):
        raise ValueError("PVC records require run_id, model_id, case_id, and prompt_id")
    if isinstance(boundary, bool) or not isinstance(boundary, (int, float)):
        raise ValueError("PVC records require numeric boundary_seconds")
    if algorithm_version != PVC_ALGORITHM_VERSION:
        raise ValueError(
            "PVC records require the current pvc_algorithm_version "
            f"{PVC_ALGORITHM_VERSION!r}"
        )
    size = record.get("input_size_bytes")
    mtime = record.get("input_mtime_ns")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or isinstance(mtime, bool)
        or not isinstance(mtime, int)
        or mtime < 0
    ):
        raise ValueError("PVC records require a valid video input fingerprint")
    return (
        run_id,
        model_id,
        case_id,
        prompt_id,
        float(boundary),
        size,
        mtime,
        algorithm_version,
    )


def _metric(record: Mapping[str, Any]) -> str:
    return str(record.get("metric", record.get("metric_id", "")))


def _component_score(record: Mapping[str, Any], field: str) -> float | None:
    scores = record.get("worker_scores")
    if isinstance(scores, Mapping) and field in scores:
        return float(scores[field])
    if field == "algorithm_score":
        value = record.get("_algorithm_score", record.get("value"))
        if value is not None:
            return float(value)
    return None


def _validate_boundary(mllm: Mapping[str, Any], algorithm: Mapping[str, Any]) -> None:
    left, right = mllm.get("boundary_seconds"), algorithm.get("boundary_seconds")
    if left is not None and right is not None and float(left) != float(right):
        raise ValueError(
            f"PVC component boundary mismatch: MLLM={left}, algorithm={right}"
        )
