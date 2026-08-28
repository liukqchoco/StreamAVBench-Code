"""StreamAV-Bench implementation of published boundary diagnostics."""

from __future__ import annotations

import subprocess
import tempfile
from itertools import groupby
from pathlib import Path
from typing import Any

PVC_ALGORITHM_VERSION = "streamavbench.pvc-algorithm.v1"


def analyze_boundary_technical(
    video_path: str | Path,
    fps: int = 8,
    *,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> dict[str, Any]:
    """Measure black frames, repeats, freezes, and flashes in one boundary window."""
    import numpy as np
    from PIL import Image

    with tempfile.TemporaryDirectory(prefix="streamav-pvc-") as temporary:
        pattern = Path(temporary) / "frame_%04d.jpg"
        command = [
            "ffmpeg",
            "-nostdin",
            "-y",
        ]
        if start_seconds is not None:
            command.extend(("-ss", f"{start_seconds:g}"))
        command.extend(
            [
                "-i",
                str(video_path),
            ]
        )
        if end_seconds is not None:
            if start_seconds is None or end_seconds <= start_seconds:
                raise ValueError("technical detector requires a valid interval")
            command.extend(("-t", f"{end_seconds - start_seconds:g}"))
        command.extend(
            [
                "-vf",
                f"fps={fps},format=gray,scale=160:-1",
                str(pattern),
            ]
        )
        subprocess.run(
            command,
            check=True,
            capture_output=True,
        )
        frames = []
        for path in sorted(Path(temporary).glob("frame_*.jpg")):
            with Image.open(path) as image:
                frames.append(np.asarray(image.convert("L"), dtype=np.float32) / 255.0)
    if not frames:
        return {
            "black_frame_ratio": 1.0,
            "flash_count": 0,
            "duplicate_frame_ratio": 0.0,
            "freeze_max_sec": 0.0,
            "technical_boundary_score": 0.0,
            "algorithm_score": 1.0,
        }
    frame_stack = np.stack(frames, axis=0)
    brightness = frame_stack.mean(axis=(1, 2))
    black_ratio = float(np.mean(brightness < 0.05))
    frame_differences = np.abs(np.diff(frame_stack, axis=0)).mean(axis=(1, 2))
    duplicate_flags = frame_differences < 0.01
    duplicate_ratio = float(np.mean(duplicate_flags)) if duplicate_flags.size else 0.0
    freeze_seconds = _longest_true_span(duplicate_flags.tolist()) / fps
    flash_count = int(np.count_nonzero(np.abs(np.diff(brightness)) > 0.45))
    scored = score_boundary_diagnostics(
        black_ratio, duplicate_ratio, freeze_seconds, flash_count
    )
    return {
        "black_frame_ratio": round(black_ratio, 4),
        "flash_count": flash_count,
        "duplicate_frame_ratio": round(duplicate_ratio, 4),
        "freeze_max_sec": round(freeze_seconds, 4),
        **scored,
    }


def fuse_pvc(algorithm_score: float, mllm_score: float) -> float:
    if not 1.0 <= algorithm_score <= 5.0 or not 1.0 <= mllm_score <= 5.0:
        raise ValueError("PVC component scores must be in [1, 5]")
    return round(0.70 * algorithm_score + 0.30 * mllm_score, 4)


def score_boundary_diagnostics(
    black_frame_ratio: float,
    duplicate_frame_ratio: float,
    freeze_max_sec: float,
    flash_count: int,
) -> dict[str, float]:
    penalty = (
        70.0 * black_frame_ratio
        + 45.0 * duplicate_frame_ratio
        + 12.0 * freeze_max_sec
        + 8.0 * flash_count
    )
    technical = round(max(0.0, min(100.0, 100.0 - penalty)), 4)
    return {
        "technical_boundary_score": technical,
        "algorithm_score": 1.0 + 4.0 * technical / 100.0,
    }


def _longest_true_span(values: list[bool]) -> int:
    return max(
        (sum(1 for _ in group) for value, group in groupby(values) if value),
        default=0,
    )
