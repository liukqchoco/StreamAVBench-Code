"""Resumable subprocess coordinator for immutable JSONL evaluation plans."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .normalize import SCORE_FIELD_BY_METRIC
from .pvc_fusion import (
    build_algorithm_index,
    fuse_pvc_record,
    pvc_key,
    read_algorithm_records,
)
from .validation import validate_media
from .worker_config import WorkerConfig, load_worker_config

RunProcess = Callable[..., subprocess.CompletedProcess[str]]


def read_plan(path: str | Path) -> list[dict[str, Any]]:
    records = _read_jsonl(path)
    seen: set[str] = set()
    for index, record in enumerate(records, 1):
        job_id = _job_id(record)
        if job_id in seen:
            raise ValueError(f"duplicate job_id {job_id!r} at plan line {index}")
        seen.add(job_id)
    return records


load_plan = read_plan


def completed_job_ids(path: str | Path, run_id: str) -> set[str]:
    source = Path(path)
    if not source.exists():
        return set()
    return _completed_record_job_ids(_read_jsonl(source), run_id)


def _completed_record_job_ids(
    records: Sequence[Mapping[str, Any]],
    run_id: str,
    *,
    job_spec_digests: Mapping[str, str] | None = None,
    worker_config_digests: Mapping[str, str] | None = None,
) -> set[str]:
    completed: set[str] = set()
    for record in records:
        if str(record.get("run_id", "")) != run_id:
            continue
        job_id = _job_id(record)
        if job_spec_digests is not None and (
            job_id not in job_spec_digests
            or record.get("job_spec_digest") != job_spec_digests[job_id]
        ):
            continue
        if worker_config_digests is not None and (
            job_id not in worker_config_digests
            or record.get("worker_config_digest") != worker_config_digests[job_id]
        ):
            continue
        if str(record.get("status", "")) in {
            "computed",
            "scored",
            "not_applicable",
        }:
            completed.add(job_id)
    return completed


def run_plan(
    plan_path: str | Path,
    checkpoint_config: str | Path | Mapping[str, WorkerConfig],
    raw_path: str | Path,
    *,
    run_id: str | None = None,
    process_runner: RunProcess = subprocess.run,
    validate_inputs: bool = False,
    duration_tolerance_seconds: float = 0.5,
    shard_index: int = 0,
    shard_count: int = 1,
    pvc_algorithm_raw_paths: Sequence[str | Path] = (),
) -> list[dict[str, Any]]:
    """Run outstanding jobs with bounded batches and isolated retries."""

    plan = read_plan(plan_path)
    _validate_shard(shard_index, shard_count)
    configs = (
        dict(checkpoint_config)
        if isinstance(checkpoint_config, Mapping)
        else load_worker_config(checkpoint_config)
    )
    selected_run = run_id or _infer_run_id(plan)
    selected_jobs = [
        job
        for job in plan
        if str(job.get("run_id", selected_run)) == selected_run
        and _job_shard(_job_id(job), shard_count) == shard_index
    ]
    prepared_jobs: list[dict[str, Any]] = []
    job_spec_digests: dict[str, str] = {}
    worker_config_digests: dict[str, str] = {}
    for source in selected_jobs:
        job = dict(source)
        config = _select_worker(job, configs)
        job_id = _job_id(job)
        job_spec_digests[job_id] = _job_spec_digest(job)
        worker_config_digests[job_id] = _worker_config_digest(config)
        job["_job_spec_digest"] = job_spec_digests[job_id]
        job["_worker_config_digest"] = worker_config_digests[job_id]
        prepared_jobs.append(job)

    algorithm_records = read_algorithm_records(pvc_algorithm_raw_paths)
    raw_records = _read_jsonl(raw_path) if Path(raw_path).is_file() else []
    done = _completed_record_job_ids(
        raw_records,
        selected_run,
        job_spec_digests=job_spec_digests,
        worker_config_digests=worker_config_digests,
    )
    pending = [job for job in prepared_jobs if _job_id(job) not in done]
    for job in pending:
        _require_input_fingerprints(job)
    algorithm_records.extend(raw_records)
    algorithm_index = build_algorithm_index(algorithm_records)
    missing_pvc = [
        pvc_key(job)
        for job in pending
        if str(job.get("metric_id", job.get("metric", ""))) == "PVC"
        and pvc_key(job) not in algorithm_index
    ]
    if missing_pvc:
        preview = ", ".join(
            "/".join(str(component) for component in key) for key in missing_pvc[:5]
        )
        raise ValueError(
            "PVC MLLM jobs require precomputed PVC-Algorithm raw records; "
            f"missing {len(missing_pvc)} boundaries ({preview})"
        )
    emitted: list[dict[str, Any]] = []
    attempts = previous_attempt_counts(raw_path, selected_run)

    if validate_inputs:
        validation_cache: dict[tuple[str, str | None], Mapping[str, Any]] = {}
        valid_pending: list[dict[str, Any]] = []
        for job in pending:
            video_path = str(job.get("video_path", ""))
            raw_audio_path = job.get("audio_path")
            audio_path = (
                str(raw_audio_path) if raw_audio_path not in (None, "") else None
            )
            cache_key = (video_path, audio_path)
            validation = validation_cache.get(cache_key)
            if validation is None:
                validation = validate_media(
                    {
                        "video_path": video_path,
                        "audio_path": audio_path,
                        "duration_seconds": 180.0,
                        "expected_modalities": {"video": True, "audio": True},
                    },
                    tolerance_seconds=duration_tolerance_seconds,
                ).to_dict()
                validation_cache[cache_key] = validation
            if validation["status"] == "computed":
                valid_pending.append(job)
                continue
            record = canonical_record(
                job,
                {
                    "status": "generation_failure",
                    "error": validation.get("error"),
                    "artifacts": {"media_validation": validation},
                },
                run_id=selected_run,
                attempt=attempts.get(_job_id(job), 0) + 1,
            )
            append_raw_record(raw_path, record)
            emitted.append(record)
            attempts[_job_id(job)] = int(record["attempts"])
        pending = valid_pending

    groups: dict[str, tuple[WorkerConfig, list[Mapping[str, Any]]]] = {}
    for job in pending:
        config = _select_worker(job, configs)
        groups.setdefault(config.worker_id, (config, []))[1].append(job)

    retries_remaining = {
        worker_id: config.max_retries for worker_id, (config, _jobs) in groups.items()
    }
    active_groups = groups
    while active_groups:
        retry_groups: dict[str, tuple[WorkerConfig, list[Mapping[str, Any]]]] = {}
        for config, jobs, records in _execute_groups(
            active_groups,
            selected_run,
            attempts,
            process_runner=process_runner,
        ):
            algorithm_records.extend(
                record
                for record in records
                if str(record.get("metric_id", record.get("metric", "")))
                == "PVC-Algorithm"
            )
            algorithm_index = build_algorithm_index(algorithm_records)
            records = [fuse_pvc_record(record, algorithm_index) for record in records]
            append_raw_records(raw_path, records)
            emitted.extend(records)
            by_job = {_job_id(job): job for job in jobs}
            for record in records:
                job_id = str(record["job_id"])
                attempts[job_id] = int(record["attempts"])
                if (
                    record["status"] == "evaluator_failure"
                    and retries_remaining[config.worker_id] > 0
                    and not (
                        isinstance(record.get("error"), Mapping)
                        and record["error"].get("retryable") is False
                    )
                ):
                    retry_groups.setdefault(config.worker_id, (config, []))[1].append(
                        by_job[job_id]
                    )
        for worker_id in retry_groups:
            retries_remaining[worker_id] -= 1
        active_groups = retry_groups
    return emitted


def _execute_groups(
    groups: Mapping[str, tuple[WorkerConfig, list[Mapping[str, Any]]]],
    run_id: str,
    attempts: Mapping[str, int],
    *,
    process_runner: RunProcess,
) -> Iterable[tuple[WorkerConfig, list[Mapping[str, Any]], list[dict[str, Any]]]]:
    tasks: list[tuple[WorkerConfig, list[Mapping[str, Any]], str | None]] = []
    resource_limits: dict[str, int] = {}
    for worker_id, (config, _jobs) in groups.items():
        resource = config.concurrency_group or worker_id
        limit = config.concurrency_limit or config.max_parallel
        previous = resource_limits.setdefault(resource, limit)
        if previous != limit:
            raise ValueError(
                f"concurrency group {resource!r} has conflicting limits "
                f"{previous} and {limit}"
            )
    semaphores = {
        resource: threading.Semaphore(limit)
        for resource, limit in resource_limits.items()
    }
    device_locks: dict[str, threading.Lock] = {}
    for config, jobs in groups.values():
        size = 1 if config.mode == "single" else config.batch_size
        for index, batch in enumerate(_chunks(jobs, size)):
            device = (
                config.cuda_devices[index % len(config.cuda_devices)]
                if config.cuda_devices
                else None
            )
            if device is not None:
                device_locks.setdefault(device, threading.Lock())
            tasks.append((config, batch, device))

    def run_task(
        config: WorkerConfig,
        jobs: list[Mapping[str, Any]],
        device: str | None,
    ) -> tuple[WorkerConfig, list[Mapping[str, Any]], list[dict[str, Any]]]:
        semaphore = semaphores[config.concurrency_group or config.worker_id]
        device_lock = device_locks.get(device) if device is not None else None
        with semaphore:
            if device_lock is None:
                records = execute_worker(
                    jobs,
                    config,
                    run_id,
                    process_runner=process_runner,
                    attempts=attempts,
                )
            else:
                with device_lock:
                    records = execute_worker(
                        jobs,
                        config,
                        run_id,
                        process_runner=process_runner,
                        attempts=attempts,
                        runtime_environment={"CUDA_VISIBLE_DEVICES": device},
                    )
        return config, jobs, records

    max_workers = max(1, sum(resource_limits.values()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_task, config, jobs, device)
            for config, jobs, device in tasks
        ]
        for future in concurrent.futures.as_completed(futures):
            yield future.result()


def execute_worker(
    jobs: Sequence[Mapping[str, Any]],
    config: WorkerConfig,
    run_id: str,
    *,
    process_runner: RunProcess = subprocess.run,
    attempts: Mapping[str, int] | None = None,
    runtime_environment: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    requests = [_worker_request(job, config) for job in jobs]
    stdin = (
        "\n".join(json.dumps(request, sort_keys=True) for request in requests) + "\n"
    )
    try:
        completed = process_runner(
            list(config.command),
            input=stdin,
            text=True,
            capture_output=True,
            check=False,
            timeout=config.timeout_seconds,
            env=config.subprocess_env(runtime_overrides=runtime_environment),
        )
        outputs = _parse_worker_outputs(completed.stdout)
        matched = _match_worker_outputs(jobs, outputs)
        diagnostics = _diagnostic_tail(completed.stderr.strip())
        records: list[dict[str, Any]] = []
        for job, output in zip(jobs, matched, strict=True):
            attempt = (attempts or {}).get(_job_id(job), 0) + 1
            if output is None:
                output = {
                    "status": "evaluator_failure",
                    "error": {
                        "type": "MissingWorkerOutput",
                        "message": (
                            f"worker exited with code {completed.returncode} "
                            f"without a matching result"
                            + (f": {diagnostics}" if diagnostics else "")
                        ),
                    },
                }
            records.append(
                canonical_record(job, output, run_id=run_id, attempt=attempt)
            )
        return records
    except Exception as exc:
        return [
            canonical_record(
                job,
                {
                    "status": "evaluator_failure",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
                run_id=run_id,
                attempt=(attempts or {}).get(_job_id(job), 0) + 1,
            )
            for job in jobs
        ]


def canonical_record(
    job: Mapping[str, Any],
    worker_output: Mapping[str, Any],
    *,
    run_id: str,
    returncode: int = 0,
    attempt: int | None = None,
) -> dict[str, Any]:
    """Merge worker output with immutable plan identity into raw-result v2 form."""

    status = str(worker_output.get("status", "evaluator_failure"))
    status = {
        "ok": "computed",
        "scored": "computed",
        "error": "evaluator_failure",
    }.get(status, status)
    if status not in {
        "computed",
        "not_applicable",
        "generation_failure",
        "evaluator_failure",
    }:
        worker_output = {
            **worker_output,
            "error": {
                "type": "WorkerContractError",
                "message": f"unsupported worker status: {status!r}",
            },
        }
        status = "evaluator_failure"
    metric = str(job.get("metric_id", job.get("metric", "")))
    output_metrics = job.get("output_metrics")
    is_composite = isinstance(output_metrics, list) and bool(output_metrics)
    scores = worker_output.get("scores")
    value = None
    if status == "computed":
        contract_error: str | None = None
        if not isinstance(scores, Mapping):
            contract_error = "computed worker result requires a scores object"
        else:
            try:
                normalized_scores = {
                    str(name): _finite_worker_score(item, str(name))
                    for name, item in scores.items()
                }
            except ValueError as exc:
                contract_error = str(exc)
            else:
                scores = normalized_scores
                if is_composite:
                    expected = set(output_metrics)
                    if set(scores) != expected:
                        contract_error = (
                            "composite worker result must contain exactly "
                            f"{sorted(expected)}, got {sorted(scores)}"
                        )
                elif metric == "PVC" and "mllm_score" in scores:
                    # PVC is completed only after the independently computed
                    # deterministic component is joined below.
                    value = None
                else:
                    canonical_field = SCORE_FIELD_BY_METRIC.get(metric)
                    if canonical_field is None:
                        contract_error = (
                            f"metric {metric!r} has no canonical score-field contract"
                        )
                    elif canonical_field not in scores:
                        contract_error = (
                            f"{metric} worker result lacks canonical score field "
                            f"{canonical_field!r}"
                        )
                    else:
                        value = scores[canonical_field]
        if contract_error is not None:
            status = "evaluator_failure"
            worker_output = {
                **worker_output,
                "error": {
                    "type": "WorkerContractError",
                    "message": contract_error,
                },
            }
            value = None
    if (
        status == "computed"
        and metric == "PVC"
        and value is None
        and isinstance(worker_output.get("scores"), Mapping)
        and "mllm_score" in worker_output["scores"]
    ):
        status = "evaluator_failure"
        worker_output = {
            **worker_output,
            "error": {
                "type": "MissingPVCAlgorithmDependency",
                "message": "PVC MLLM score requires a matching PVC-Algorithm record",
            },
        }
    interval = job.get("interval", job.get("interval_index"))
    record: dict[str, Any] = {
        # v2 adds worker_scores and evaluator metadata. Report readers remain
        # field-based and continue to accept all existing raw-result.v1 rows.
        "schema_version": "streamavbench.raw-result.v2",
        "run_id": run_id,
        "job_id": _job_id(job),
        "model_id": job.get("model_id"),
        "track": job.get("track"),
        "case_id": job.get("case_id"),
        "metric_id": job.get("metric_id", job.get("metric")),
        "status": status,
        "value": value if status == "computed" else None,
        "attempts": (
            attempt if attempt is not None else int(job.get("attempts", 0)) + 1
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    record["job_spec_digest"] = (
        job["_job_spec_digest"]
        if isinstance(job.get("_job_spec_digest"), str)
        else _job_spec_digest(job)
    )
    if isinstance(job.get("_worker_config_digest"), str):
        record["worker_config_digest"] = job["_worker_config_digest"]
    for key in (
        "input_size_bytes",
        "input_mtime_ns",
        "audio_input_size_bytes",
        "audio_input_mtime_ns",
    ):
        if key in job and job[key] is not None:
            record[key] = job[key]
    if interval is not None:
        record["interval"] = interval
    if is_composite:
        record["output_metrics"] = list(output_metrics)
        scores = worker_output.get("scores")
        record["outputs"] = dict(scores) if isinstance(scores, Mapping) else {}
    if isinstance(scores, Mapping):
        record["worker_scores"] = dict(scores)
    for key in ("scale", "implementation", "assets"):
        if key in job:
            record[key] = job[key]
    for key in ("subscores", "protocol", "artifacts", "error"):
        if key in worker_output:
            record[key] = worker_output[key]
    options = job.get("options")
    if isinstance(options, Mapping):
        for key in (
            "phase",
            "prompt_id",
            "prompt_ids",
            "modality",
            "update_modality",
            "dependency_class",
            "source_prompt_id",
            "boundary_seconds",
            "media_windows",
            "sampled_timestamps_seconds",
            "evaluator_protocol_sha256",
            "pvc_algorithm_version",
        ):
            if key in options:
                record[key] = options[key]
    if status == "evaluator_failure" and "error" not in record:
        record["error"] = {
            "type": "WorkerProcessError",
            "message": f"worker exited with code {returncode}",
        }
    record["provisional"] = status == "evaluator_failure"
    return record


def append_raw_record(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one canonical record and fsync it."""

    append_raw_records(path, [record])


def append_raw_records(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Append a completed batch with one writer and one fsync."""

    if not records:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(dict(record), sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def previous_attempt_counts(path: str | Path, run_id: str) -> dict[str, int]:
    source = Path(path)
    if not source.exists():
        return {}
    counts: dict[str, int] = {}
    for record in _read_jsonl(source):
        if str(record.get("run_id", "")) != run_id:
            continue
        job_id = _job_id(record)
        value = record.get("attempts", 0)
        if isinstance(value, int) and not isinstance(value, bool):
            counts[job_id] = max(counts.get(job_id, 0), value)
    return counts


def _worker_request(job: Mapping[str, Any], config: WorkerConfig) -> dict[str, Any]:
    request = (
        dict(job.get("request", {})) if isinstance(job.get("request"), Mapping) else {}
    )
    request.setdefault("request_id", _job_id(job))
    request.setdefault("metric", job.get("metric_id", job.get("metric")))
    for key in (
        "video_path",
        "audio_path",
        "duration_seconds",
        "start_seconds",
        "end_seconds",
        "options",
    ):
        if key in job:
            request.setdefault(key, job[key])
    interval = job.get("interval", job.get("interval_index"))
    if interval is not None:
        request.setdefault("interval_index", interval)
    media = job.get("media")
    if isinstance(media, Mapping):
        request.setdefault("video_path", media.get("video"))
        request.setdefault("audio_path", media.get("audio"))
        if media.get("audio") is None and media.get("video") is not None:
            options = dict(request.get("options", {}))
            options.setdefault("audio_source", "muxed")
            options.setdefault("extract_audio", True)
            request["options"] = options
    options = dict(request.get("options", {}))
    if config.checkpoints:
        options.setdefault("checkpoints", dict(config.checkpoints))
    request["options"] = options
    return request


def _select_worker(
    job: Mapping[str, Any], configs: Mapping[str, WorkerConfig]
) -> WorkerConfig:
    requested = job.get(
        "worker_id", job.get("worker", job.get("metric_id", job.get("metric")))
    )
    if requested in configs:
        return configs[str(requested)]
    suffix = [
        config for key, config in configs.items() if key.rsplit(".", 1)[-1] == requested
    ]
    if len(suffix) == 1:
        return suffix[0]
    raise KeyError(f"no unambiguous worker configuration for {requested!r}")


def _parse_worker_outputs(stdout: str) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for line_number, line in enumerate(stdout.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"worker stdout line {line_number} is not valid JSON"
            ) from exc
        if not isinstance(value, Mapping):
            raise RuntimeError(
                f"worker stdout line {line_number} must be a JSON object"
            )
        outputs.append(dict(value))
    return outputs


def _match_worker_outputs(
    jobs: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any] | None]:
    if not outputs:
        return [None] * len(jobs)
    identified = [
        (str(request_id), output)
        for output in outputs
        if isinstance((request_id := output.get("request_id")), str) and request_id
    ]
    if len(identified) != len(outputs):
        raise RuntimeError("every worker output must include a request_id")
    identified_ids = [request_id for request_id, _output in identified]
    if len(set(identified_ids)) != len(identified_ids):
        raise RuntimeError("worker returned duplicate request_id values")
    expected_ids = {_job_id(job) for job in jobs}
    unexpected = sorted(set(identified_ids) - expected_ids)
    if unexpected:
        raise RuntimeError(
            f"worker returned unexpected request_id values: {unexpected}"
        )
    by_request = dict(identified)
    matched = [by_request.get(_job_id(job)) for job in jobs]
    for job, output in zip(jobs, matched, strict=True):
        if output is None:
            continue
        expected_metric = str(job.get("metric_id", job.get("metric", "")))
        if output.get("metric") != expected_metric:
            raise RuntimeError(
                f"worker returned metric {output.get('metric')!r} for "
                f"{_job_id(job)}; expected {expected_metric!r}"
            )
    return matched


def _chunks(
    values: Sequence[Mapping[str, Any]], size: int
) -> Iterable[list[Mapping[str, Any]]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def _validate_shard(index: int, count: int) -> None:
    if (
        isinstance(index, bool)
        or isinstance(count, bool)
        or not isinstance(index, int)
        or not isinstance(count, int)
        or count <= 0
        or index < 0
        or index >= count
    ):
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")


def _job_shard(job_id: str, count: int) -> int:
    digest = hashlib.sha256(job_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count


def _diagnostic_tail(value: str, limit: int = 4_000) -> str:
    if len(value) <= limit:
        return value
    return f"[... {len(value) - limit} characters omitted ...]\n{value[-limit:]}"


def _infer_run_id(plan: Sequence[Mapping[str, Any]]) -> str:
    values = {str(job["run_id"]) for job in plan if job.get("run_id") not in (None, "")}
    if len(values) != 1:
        raise ValueError(
            "run_id must be supplied when the plan does not contain exactly one"
        )
    return values.pop()


def _require_input_fingerprints(job: Mapping[str, Any]) -> None:
    paths = (
        (
            "video",
            job.get("video_path"),
            job.get("input_size_bytes"),
            job.get("input_mtime_ns"),
        ),
        (
            "audio",
            job.get("audio_path"),
            job.get("audio_input_size_bytes"),
            job.get("audio_input_mtime_ns"),
        ),
    )
    for label, raw_path, raw_size, raw_mtime in paths:
        if raw_size is None and raw_mtime is None:
            continue
        if raw_path in (None, ""):
            raise ValueError(f"{_job_id(job)} has an incomplete {label} fingerprint")
        if (
            isinstance(raw_size, bool)
            or not isinstance(raw_size, int)
            or isinstance(raw_mtime, bool)
            or not isinstance(raw_mtime, int)
        ):
            raise ValueError(f"{_job_id(job)} has an invalid {label} fingerprint")
        path = Path(str(raw_path))
        try:
            stat = path.stat()
        except OSError as exc:
            raise ValueError(
                f"{_job_id(job)} cannot stat {label} input: {exc}"
            ) from exc
        if (stat.st_size, stat.st_mtime_ns) != (raw_size, raw_mtime):
            raise ValueError(f"{_job_id(job)} {label} input changed after planning")


def _job_id(record: Mapping[str, Any]) -> str:
    value = record.get("job_id", record.get("request_id", record.get("unit_id")))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("record requires a non-empty job_id")
    return value.strip()


def _job_spec_digest(job: Mapping[str, Any]) -> str:
    payload = {
        str(key): value for key, value in job.items() if not str(key).startswith("_")
    }
    return _canonical_digest(payload)


def _worker_config_digest(config: WorkerConfig) -> str:
    return _canonical_digest(
        {
            "worker_id": config.worker_id,
            "command": list(config.command),
            "mode": config.mode,
            "network": config.network,
            "environment": dict(config.environment),
            "checkpoints": dict(config.checkpoints),
            "timeout_seconds": config.timeout_seconds,
            "batch_size": config.batch_size,
            "max_parallel": config.max_parallel,
            "max_retries": config.max_retries,
            "cuda_devices": list(config.cuda_devices),
            "concurrency_group": config.concurrency_group,
            "concurrency_limit": config.concurrency_limit,
        }
    )


def _canonical_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_worker_score(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"worker score {field!r} must be numeric")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"worker score {field!r} must be finite")
    return score


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number}: expected object")
            records.append(dict(value))
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="streamav-eval run")
    parser.add_argument("config", type=Path, help="JSON run configuration")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, Mapping):
            raise ValueError("run config root must be an object")
        _validate_run_config(config)
        base = args.config.resolve().parent
        emitted = run_plan(
            _config_relative_path(config["plan_path"], base),
            _config_relative_path(config["worker_config"], base),
            _config_relative_path(config["raw_path"], base),
            run_id=config.get("run_id"),
            validate_inputs=config.get("validate_inputs", True),
            duration_tolerance_seconds=float(
                config.get("duration_tolerance_seconds", 0.5)
            ),
            shard_index=int(config.get("shard_index", 0)),
            shard_count=int(config.get("shard_count", 1)),
            pvc_algorithm_raw_paths=tuple(
                _config_relative_path(path, base)
                for path in config.get("pvc_algorithm_raw_paths", ())
            ),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    latest_status: dict[str, str] = {}
    for record in emitted:
        latest_status[_job_id(record)] = str(record.get("status", ""))
    return (
        1
        if any(
            status in {"generation_failure", "evaluator_failure"}
            for status in latest_status.values()
        )
        else 0
    )


def _config_relative_path(value: Any, base: Path) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        raise ValueError("configured paths must be non-empty strings")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _validate_run_config(config: Mapping[str, Any]) -> None:
    allowed = {
        "run_id",
        "plan_path",
        "worker_config",
        "raw_path",
        "validate_inputs",
        "duration_tolerance_seconds",
        "shard_index",
        "shard_count",
        "pvc_algorithm_raw_paths",
    }
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"unknown run config fields: {unknown}")
    for name in ("plan_path", "worker_config", "raw_path"):
        if not isinstance(config.get(name), str) or not config[name]:
            raise ValueError(f"{name} must be a non-empty string")
    run_id = config.get("run_id")
    if run_id is not None and (not isinstance(run_id, str) or not run_id):
        raise ValueError("run_id must be a non-empty string when provided")
    if not isinstance(config.get("validate_inputs", True), bool):
        raise ValueError("validate_inputs must be a boolean")
    tolerance = config.get("duration_tolerance_seconds", 0.5)
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or tolerance < 0
    ):
        raise ValueError("duration_tolerance_seconds must be non-negative")
    for name, minimum in (("shard_index", 0), ("shard_count", 1)):
        value = config.get(name, minimum)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    paths = config.get("pvc_algorithm_raw_paths", ())
    if (
        isinstance(paths, (str, bytes))
        or not isinstance(paths, Sequence)
        or not all(isinstance(path, str) and path for path in paths)
    ):
        raise ValueError("pvc_algorithm_raw_paths must be a string array")


if __name__ == "__main__":
    raise SystemExit(main())
