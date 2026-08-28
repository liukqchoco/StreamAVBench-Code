#!/usr/bin/env python3
"""Render allowlisted ${VARS} in JSON templates without executing code."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PLACEHOLDER = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
ALLOWED = {
    "STREAMAV_ASSETS_ROOT",
    "STREAMAV_CONDA_ENVS_ROOT",
    "STREAMAV_CUDA_DEVICE",
    "STREAMAV_CUDA_DEVICES",
    "STREAMAV_GEMINI_KEYS_PATH",
    "STREAMAV_GPU_PARALLELISM",
    "STREAMAV_MLLM_PARALLELISM",
    "STREAMAV_ROOT",
}


def defaults(root: Path) -> dict[str, str]:
    return {
        "STREAMAV_ROOT": str(root),
        "STREAMAV_ASSETS_ROOT": str(root / ".cache" / "assets"),
        "STREAMAV_CONDA_ENVS_ROOT": str(_default_conda_envs_root()),
        "STREAMAV_CUDA_DEVICE": "0",
        "STREAMAV_CUDA_DEVICES": "0",
        "STREAMAV_GEMINI_KEYS_PATH": str(root / ".secrets" / "api_keys.txt"),
        "STREAMAV_GPU_PARALLELISM": "1",
        "STREAMAV_MLLM_PARALLELISM": "8",
    }


def _default_conda_envs_root() -> Path:
    configured = os.environ.get("CONDA_ENVS_PATH")
    if configured:
        return Path(configured.split(os.pathsep)[0]).expanduser()
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        path = Path(prefix).expanduser()
        return path.parent if path.parent.name == "envs" else path / "envs"
    return Path.home() / ".conda" / "envs"


def render(value: Any, variables: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        if value == "${STREAMAV_CUDA_DEVICES}":
            devices = [
                item.strip()
                for item in variables["STREAMAV_CUDA_DEVICES"].split(",")
                if item.strip()
            ]
            if not devices:
                raise ValueError("STREAMAV_CUDA_DEVICES must not be empty")
            return devices
        if value in {
            "${STREAMAV_GPU_PARALLELISM}",
            "${STREAMAV_MLLM_PARALLELISM}",
        }:
            name = value[2:-1]
            try:
                result = int(variables[name])
            except ValueError as exc:
                raise ValueError(f"{name} must be a positive integer") from exc
            if result <= 0:
                raise ValueError(f"{name} must be a positive integer")
            return result
        names = PLACEHOLDER.findall(value)
        unknown = sorted(set(names) - ALLOWED)
        if unknown:
            raise ValueError(f"unsupported placeholders: {', '.join(unknown)}")
        for name in names:
            value = value.replace("${" + name + "}", variables[name])
        if "${" in value:
            raise ValueError(f"malformed or unsupported placeholder in {value!r}")
        return value
    if isinstance(value, list):
        return [render(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: render(item, variables) for key, item in value.items()}
    return value


def render_file(source: Path, destination: Path, variables: Mapping[str, str]) -> None:
    value = json.loads(source.read_text(encoding="utf-8"))
    rendered = render(value, variables)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(rendered, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    variables = defaults(root)
    for name in ALLOWED:
        if name in os.environ:
            variables[name] = os.environ[name]
    try:
        render_file(args.template, args.output, variables)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
