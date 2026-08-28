#!/usr/bin/env python3
"""Reuse valid PVC-Algorithm records and compute only missing plan boundaries."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from streamav_eval.pvc_fusion import pvc_key
from streamav_eval.runner import (
    _job_spec_digest,
    append_raw_record,
    canonical_record,
    read_plan,
)
from streamav_eval.workers.objective.pvc_algorithm import PVCAlgorithmWorker
from streamav_eval.workers.protocol import WorkerRequest, WorkerResult


def _fingerprint(job: Mapping[str, Any]) -> tuple[int, int]:
    size = job.get("input_size_bytes")
    mtime = job.get("input_mtime_ns")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or isinstance(mtime, bool)
        or not isinstance(mtime, int)
        or mtime < 0
    ):
        raise ValueError(f"{job.get('job_id')}: invalid video input fingerprint")
    return size, mtime


def _validate_video_fingerprint(job: Mapping[str, Any]) -> None:
    path = Path(str(job["video_path"]))
    expected = _fingerprint(job)
    stat = path.stat()
    observed = stat.st_size, stat.st_mtime_ns
    if observed != expected:
        raise ValueError(
            f"{path}: input changed after planning; expected {expected}, got {observed}"
        )


def _algorithm_job(pvc_job: Mapping[str, Any]) -> dict[str, Any]:
    job = dict(pvc_job)
    job["metric"] = "PVC-Algorithm"
    if "metric_id" in job:
        job["metric_id"] = "PVC-Algorithm"
    job["job_id"] = f"{job['job_id']}__PVC-Algorithm"
    job["_job_spec_digest"] = _job_spec_digest(job)
    return job


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--seed-raw", action="append", type=Path, default=[])
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    plan_jobs = [
        job
        for job in read_plan(args.plan)
        if job.get("metric", job.get("metric_id")) == "PVC"
    ]
    for job in plan_jobs:
        _validate_video_fingerprint(job)
    expected = {pvc_key(job): (_fingerprint(job), job) for job in plan_jobs}
    if len(expected) != len(plan_jobs):
        raise ValueError("PVC plan contains duplicate dependency keys")
    if not expected:
        raise SystemExit("MLLM plan contains no PVC jobs")

    available: dict[
        tuple[str, str, str, str, float, int, int, str], dict[str, Any]
    ] = {}
    for record in _read_records(args.output):
        if record.get(
            "metric_id", record.get("metric")
        ) == "PVC-Algorithm" and record.get("status") in {"computed", "scored"}:
            key = pvc_key(record)
            if key in expected and _fingerprint(record) == expected[key][0]:
                available[key] = record
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed_path in args.seed_raw:
        for record in _read_records(seed_path):
            if record.get(
                "metric_id", record.get("metric")
            ) != "PVC-Algorithm" or record.get("status") not in {"computed", "scored"}:
                continue
            key = pvc_key(record)
            if (
                key not in available
                and key in expected
                and _fingerprint(record) == expected[key][0]
            ):
                append_raw_record(args.output, record)
                available[key] = record

    missing = [expected[key][1] for key in sorted(set(expected) - set(available))]
    worker = PVCAlgorithmWorker()

    def evaluate(pvc_job: Mapping[str, Any]) -> dict[str, Any]:
        job = _algorithm_job(pvc_job)
        request = WorkerRequest(
            request_id=str(job["job_id"]),
            metric="PVC-Algorithm",
            video_path=str(job["video_path"]),
            duration_seconds=180.0,
            start_seconds=float(job["start_seconds"]),
            end_seconds=float(job["end_seconds"]),
            interval_index=int(job.get("interval") or 0),
            options=dict(job.get("options", {})),
        )
        try:
            _validate_video_fingerprint(job)
            result = worker.evaluate(request)
            _validate_video_fingerprint(job)
        except Exception as exc:
            result = WorkerResult.failed(request, exc)
        return canonical_record(
            job,
            result.to_dict(),
            run_id=str(job["run_id"]),
            attempt=1,
        )

    failures = 0
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = [executor.submit(evaluate, job) for job in missing]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            record = future.result()
            append_raw_record(args.output, record)
            if record["status"] != "computed":
                failures += 1
            if index % 25 == 0 or index == len(futures):
                print(
                    json.dumps(
                        {
                            "event": "pvc_progress",
                            "finished": index,
                            "scheduled": len(futures),
                            "failures": failures,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    if failures:
        raise SystemExit(f"{failures} PVC-Algorithm jobs failed")
    print(
        json.dumps(
            {
                "event": "pvc_complete",
                "expected": len(expected),
                "reused": len(available),
                "computed": len(missing),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
