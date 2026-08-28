<div align="center">

# StreamAV-Bench

### A Comprehensive Benchmark for Streaming Audio-Video Generation

Kaiqi Liu · Haoxuan Zeng · Jingqi Liu · Jiacong Fang · Ziqi Cai ·
Yunyao Mao · Henglin Liu · Yu Sheng · Shuchen Weng · Boxin Shi

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://liukqchoco.github.io/StreamAVBench/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2608.26336)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-Hugging%20Face-green)](https://huggingface.co/spaces/StreamAVBench/Leaderboard)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-yellow)](https://huggingface.co/datasets/StreamAVBench/StreamAVBench)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-%3E%3D3.10-3776AB.svg)

</div>

<p align="center">
  <img src="https://liukqchoco.github.io/StreamAVBench/assets/teaser.webp"
       width="96%" alt="StreamAV-Bench benchmark overview">
</p>

## Overview

StreamAV-Bench evaluates streaming audio-video generation through a
**Progressive Track** for instruction adherence and long-horizon stability and
an **Interactive Track** for runtime-update response, state retention, and
history-dependent reuse. It contains 160 scenarios per track, 800 interactive
updates, and 32 evaluation dimensions.

This repository provides the evaluation code. Benchmark prompts and checklists
are distributed through the
[StreamAV-Bench dataset](https://huggingface.co/datasets/StreamAVBench/StreamAVBench).

## Repository Structure

```text
.
├── configs/                 Worker template and NBC protocol
├── docs/                    External dependency guide
├── examples/                Minimal evaluation, run, and report configurations
├── scripts/                 Checkpoint, configuration, aggregation, and export helpers
├── src/streamav_eval/       Evaluation package and command-line implementation
├── third_party/             Minimal evaluator runtime dependencies
├── evaluate.py              Main evaluation command
├── run_evaluation.sh        End-to-end evaluation wrapper
├── setup.sh                 Conda environment setup
├── requirements.txt         Main evaluator environment
├── requirements_sync.txt    Synchronization evaluator environment
├── pyproject.toml           Package metadata and CLI entry point
├── LICENSE                  StreamAV-Bench MIT License
└── NOTICE                   External software and model attribution
```

## Quick Start

Create the two evaluator environments and download the evaluator checkpoints:

```bash
bash setup.sh
conda activate streamav
export STREAMAV_DATASET_ROOT="/path/to/StreamAVBench-Dataset"
export STREAMAV_GEMINI_KEYS_PATH="/path/to/api_keys.txt"
bash scripts/download_assets.sh
```

Put one Gemini API key per line in the key file. See
[docs/dependencies.md](docs/dependencies.md) for checkpoint sources.

## Prepare Model Outputs

Provide one JSON object per generated scenario:

```json
{
  "model_id": "my-model",
  "case_id": "P-0001",
  "video_path": "outputs/my-model/P-0001.mp4",
  "runtime": {
    "generated_video_frames": 2880,
    "generation_time_seconds": 240.0,
    "time_to_first_chunk_seconds": 4.2
  }
}
```

Paths may be absolute or relative to the manifest. The track and expected
duration are read from the benchmark dataset. A full submission contains one
output for every scenario in both tracks.
Include `runtime` for every case of a model when generation timing can be
measured; otherwise omit it from every case. For cascaded systems, timing must
include the full audio-video pipeline.

## Run Evaluation

Copy the example manifest, replace its media paths with your generated videos,
and run:

```bash
bash run_evaluation.sh examples/evaluation.json
```

Results are written to `artifacts/report/report.json`. The script validates the
inputs, builds metric jobs, runs evaluators, and aggregates the report.

Native-boundary continuity and efficiency require generation-time metadata and
are added before public export. The corresponding helpers expose their options
through:

```bash
python scripts/aggregate_efficiency.py --help
python scripts/evaluate_nbc.py --help
python scripts/aggregate_nbc.py --help
```

Export the 32 public dimensions to JSON and CSV with:

```bash
python scripts/export_public_results.py --help
```

## Metrics

StreamAV-Bench reports 32 dimensions and does not define an overall score:

- **Shared:** VA, VQ, PQ, AQ, AVAlign, AVSync, NBC, FPS, and TTFC.
- **Progressive instruction following:** VIF, AIF, VID, and AID.
- **Progressive long-horizon stability:** VA-D, VQ-D, PQ-D, AQ-D, SC, BC,
  AVAlign-D, and AVSync-D.
- **Interactive response and state:** VUF, AUF, PVC, PAC, PVUAR, PAUAR, PVRL,
  PARL, VSR, ASR, and HDF.

SC and BC uniformly sample the full 180-second rollout at 2 FPS before the
VBench-Long slow-fast evaluation. AVSync averages the absolute offsets from the
first and last 4.8-second windows of each 30-second interval. Results are first
aggregated within each case and then averaged uniformly across cases.
VSR and ASR are evaluated for every runtime update.

## Citation

```bibtex
@article{liu2026streamav,
  title={StreamAV-Bench: A Comprehensive Benchmark for Streaming Audio-Video Generation},
  author={Liu, Kaiqi and Zeng, Haoxuan and Liu, Jingqi and Fang, Jiacong and Cai, Ziqi and Mao, Yunyao and Liu, Henglin and Sheng, Yu and Weng, Shuchen and Shi, Boxin},
  journal={arXiv preprint arXiv:2608.26336},
  year={2026}
}
```

## License

The StreamAV-Bench code, documentation, and evaluator prompts are released
under the [MIT License](LICENSE). The
[benchmark dataset](https://huggingface.co/datasets/StreamAVBench/StreamAVBench)
is released separately under CC BY-NC 4.0. Third-party packages, models, and
checkpoints retain their own terms; see [NOTICE](NOTICE).
