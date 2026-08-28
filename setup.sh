#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_ENV="streamav"
SYNC_ENV="streamav-sync"
cd "$ROOT"

conda create -y -n "$MAIN_ENV" -c conda-forge \
  python=3.10.12 pip ffmpeg=7.1
conda run -n "$MAIN_ENV" python -m pip install \
  -r "$ROOT/requirements.txt"

conda create -y -n "$SYNC_ENV" -c conda-forge \
  python=3.10.12 pip ffmpeg=7.1
conda run -n "$SYNC_ENV" python -m pip install \
  -r "$ROOT/requirements_sync.txt"

echo "Setup complete. Run: conda activate $MAIN_ENV"
