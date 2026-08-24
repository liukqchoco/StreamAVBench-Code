<div align="center">

# StreamAV-Bench

### A Comprehensive Benchmark for Streaming Audio-Video Generation

[Project Page](https://liukqchoco.github.io/StreamAVBench/) ·
[Leaderboard](https://huggingface.co/spaces/liukqchoco04/StreamAVBench-Leaderboard)

</div>

> **Release status:** This repository is being prepared for the public release
> of StreamAV-Bench. Evaluation code, benchmark data, and documentation will be
> added here upon release.

## Overview

StreamAV-Bench evaluates streaming audio-video generation through two
complementary tracks:

- **Progressive Track:** evaluates instruction adherence, quality evolution,
  and long-horizon consistency during continuous generation from a global
  prompt.
- **Interactive Track:** evaluates responses to runtime prompt updates,
  transition stability, state retention, and dependencies on earlier
  interactions.

The benchmark contains:

- **320** expert-verified scenarios;
- **160** progressive and **160** interactive scenarios;
- **800** runtime updates;
- **33** fine-grained metrics organized into six evaluation groups;
- evaluations of **13** representative streaming audio-video systems.

## Planned Release

The public release will include:

- benchmark scenarios, prompts, and evaluation checklists;
- the StreamAV-Bench evaluation toolkit;
- metric configurations and evaluation protocols;
- scripts for preparing model outputs and running evaluation;
- result aggregation and leaderboard export utilities;
- reproducibility documentation and example commands.

## Evaluation Groups

StreamAV-Bench organizes its 33 metrics into:

1. Quality and Alignment;
2. Streaming Efficiency;
3. Instruction Fulfillment and Drift;
4. Long-Horizon Quality and Consistency;
5. Interactive Update Response;
6. State Retention and History Dependency.

The leaderboard reports metrics separately rather than introducing an
unsupported overall score. Metric direction, applicability, and missing-value
rules follow the official evaluation protocol.

## News

- **2026-08:** Project page and leaderboard repositories initialized.
- Evaluation code and benchmark data: **coming soon**.

## Citation

Citation information will be added when the paper is publicly released.

## License

License information for the code and benchmark data will be announced with the
public release.
