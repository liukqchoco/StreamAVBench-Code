#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

EVALUATION_CONFIG="${1:-examples/evaluation.json}"
RUN_CONFIG="${2:-examples/run.json}"
REPORT_CONFIG="${3:-examples/report.json}"

mkdir -p artifacts
python evaluate.py validate "$EVALUATION_CONFIG"
python evaluate.py plan "$EVALUATION_CONFIG" > artifacts/plan.jsonl
python scripts/render_config.py \
  configs/workers.template.json workers.local.json
python scripts/prepare_pvc.py \
  --plan artifacts/plan.jsonl \
  --output artifacts/pvc-algorithm.jsonl
python evaluate.py run "$RUN_CONFIG"
python evaluate.py aggregate "$REPORT_CONFIG"

echo "Evaluation complete: artifacts/report/report.json"
