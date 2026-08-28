#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ASSETS_ROOT="${STREAMAV_ASSETS_ROOT:-$REPO_DIR/.cache/assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
if ! "$PYTHON_BIN" -c "import dreamsim, torch" >/dev/null 2>&1; then
  echo "Error: activate the streamav environment before downloading assets" >&2
  exit 1
fi

download() {
  local output="$1"
  local url="$2"
  local partial="${output}.part"

  mkdir -p "$(dirname "$output")"
  if [ -s "$output" ]; then
    echo "[skip] $output"
    return
  fi

  echo "[download] $output"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 5 --retry-delay 2 --connect-timeout 20 \
      --continue-at - --output "$partial" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$partial" "$url"
  else
    echo "Error: curl or wget is required" >&2
    exit 1
  fi

  if [ ! -s "$partial" ]; then
    echo "Error: downloaded file is empty: $output" >&2
    exit 1
  fi
  mv "$partial" "$output"
}

download \
  "$ASSETS_ROOT/visual_quality/ViT-L-14.pt" \
  "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
download \
  "$ASSETS_ROOT/visual_quality/head.pth" \
  "https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth"
download \
  "$ASSETS_ROOT/audio_quality/model.pt" \
  "https://dl.fbaipublicfiles.com/audiobox-aesthetics/checkpoint.pt"
download \
  "$ASSETS_ROOT/av_alignment/model.pth" \
  "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"
download \
  "$ASSETS_ROOT/av_synchronization/config.yaml" \
  "https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/24-01-04T16-39-21/cfg-24-01-04T16-39-21.yaml"
download \
  "$ASSETS_ROOT/av_synchronization/model.pt" \
  "https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/24-01-04T16-39-21/24-01-04T16-39-21.pt"

LONG_ROOT="$ASSETS_ROOT/long-consistency"
mkdir -p \
  "$LONG_ROOT/home/.cache/vbench" \
  "$LONG_ROOT/torch"
download \
  "$LONG_ROOT/home/.cache/vbench/clip_model/ViT-B-32.pt" \
  "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"

echo "[download] DINO, DINOv2, and DreamSim caches"
HOME="$LONG_ROOT/home" \
TORCH_HOME="$LONG_ROOT/torch" \
VBENCH_CACHE_DIR="$LONG_ROOT/home/.cache/vbench" \
"$PYTHON_BIN" - <<'PY'
import os
import shutil
from pathlib import Path

import torch
from dreamsim import dreamsim

torch.hub.load(
    "facebookresearch/dino:main",
    "dino_vitb16",
    trust_repo=True,
)
torch.hub.load(
    "facebookresearch/dinov2:main",
    "dinov2_vitb14",
    trust_repo=True,
)
dreamsim(
    pretrained=True,
    cache_dir=os.path.expanduser("~/.cache"),
)

hub = Path(torch.hub.get_dir())
vbench_dino = Path(os.environ["VBENCH_CACHE_DIR"]) / "dino_model"
vbench_dino.mkdir(parents=True, exist_ok=True)
shutil.copytree(
    hub / "facebookresearch_dino_main",
    vbench_dino / "facebookresearch_dino_main",
    dirs_exist_ok=True,
)
shutil.copy2(
    hub / "checkpoints" / "dino_vitbase16_pretrain.pth",
    vbench_dino / "dino_vitbase16_pretrain.pth",
)
PY

echo "Assets downloaded into: $ASSETS_ROOT"
