"""Standalone PVC technical component."""

from __future__ import annotations

from streamav_eval.workers.interactive.pvc import analyze_boundary_technical
from streamav_eval.workers.protocol import (
    WorkerRequest,
    WorkerResult,
    serve_jsonl,
)

METRIC = "PVC-Algorithm"


class PVCAlgorithmWorker:
    metric = METRIC

    def evaluate(self, request: WorkerRequest) -> WorkerResult:
        start, end = request.require_interval_duration(4.0)
        diagnostics = analyze_boundary_technical(
            request.require_video(),
            start_seconds=start,
            end_seconds=end,
        )
        return WorkerResult.ok(
            request,
            scores={"algorithm_score": float(diagnostics["algorithm_score"])},
            artifacts={"technical_diagnostics": diagnostics},
            protocol={
                "implementation": "boundary_technical_analysis",
                "fps": 8,
                "width": 160,
                "start_seconds": start,
                "end_seconds": end,
                "scale": [1.0, 5.0],
                "pvc_algorithm_weight": 0.70,
            },
        )


def main() -> int:
    return serve_jsonl(PVCAlgorithmWorker)


if __name__ == "__main__":
    raise SystemExit(main())
