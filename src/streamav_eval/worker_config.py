"""JSON checkpoint configuration for isolated, offline workers."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


class WorkerConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    worker_id: str
    command: tuple[str, ...]
    mode: str = "single"
    network: bool = False
    environment: Mapping[str, str] = field(default_factory=dict)
    checkpoints: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float | None = None
    batch_size: int = 1
    max_parallel: int = 1
    max_retries: int = 0
    cuda_devices: tuple[str, ...] = ()
    concurrency_group: str | None = None
    concurrency_limit: int | None = None

    def subprocess_env(
        self,
        parent: Mapping[str, str] | None = None,
        runtime_overrides: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        env = dict(os.environ if parent is None else parent)
        for key in tuple(env):
            if _is_sensitive_parent_env_name(key):
                env.pop(key, None)
        env.update(self.environment)
        if runtime_overrides:
            env.update(runtime_overrides)
        env["STREAMAV_NETWORK"] = "enabled" if self.network else "disabled"
        if not self.network:
            env.update(
                {
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "WANDB_MODE": "offline",
                    "NO_PROXY": "*",
                    "no_proxy": "*",
                }
            )
            for key in tuple(env):
                if key.lower() in {
                    "http_proxy",
                    "https_proxy",
                    "all_proxy",
                    "gemini_api_key",
                    "google_api_key",
                }:
                    env.pop(key, None)
        return env


def load_worker_config(path: str | Path) -> dict[str, WorkerConfig]:
    """Load a JSON object; YAML execution and auto-download directives are rejected."""

    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkerConfigError(
            f"cannot load JSON worker config {config_path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise WorkerConfigError("worker config root must be an object")
    root = value.get("workers", value)
    if not isinstance(root, Mapping):
        raise WorkerConfigError("workers must be an object")
    result: dict[str, WorkerConfig] = {}
    for worker_id, raw in _leaf_workers(root):
        config = worker_config_from_mapping(worker_id, raw, config_path.parent)
        result[worker_id] = config
        aliases = raw.get("aliases", ())
        if (
            isinstance(aliases, (str, bytes))
            or not isinstance(aliases, Sequence)
            or not all(isinstance(alias, str) and alias.strip() for alias in aliases)
        ):
            raise WorkerConfigError(
                f"worker {worker_id!r} aliases must be a string array"
            )
        for alias in aliases:
            alias = alias.strip()
            if alias in result:
                raise WorkerConfigError(f"duplicate worker or alias {alias!r}")
            result[alias] = replace(config, worker_id=alias)
    if not result:
        raise WorkerConfigError("worker config contains no workers")
    return result


load_checkpoint_config = load_worker_config


def worker_config_from_mapping(
    worker_id: str, raw: Mapping[str, Any], base_dir: str | Path = "."
) -> WorkerConfig:
    if not isinstance(raw, Mapping):
        raise WorkerConfigError(f"worker {worker_id!r} must be an object")
    if any(key in raw for key in ("download", "download_url", "auto_download")):
        raise WorkerConfigError("workers may not configure downloads")

    module = raw.get("command_module", raw.get("module"))
    command_value = raw.get("command")
    python = str(raw.get("python", "python"))
    if command_value is None:
        if not isinstance(module, str) or not module.strip():
            raise WorkerConfigError(
                f"worker {worker_id!r} needs command or command_module"
            )
        command = (python, "-m", module.strip())
    else:
        if (
            isinstance(command_value, (str, bytes))
            or not isinstance(command_value, Sequence)
            or not command_value
            or not all(isinstance(item, str) and item for item in command_value)
        ):
            raise WorkerConfigError("command must be a non-empty string array")
        command = tuple(command_value)

    mode = str(raw.get("mode", raw.get("protocol", "single"))).lower()
    mode = {"one": "single", "json": "single", "jsonl": "jsonl"}.get(mode, mode)
    if mode not in {"single", "jsonl"}:
        raise WorkerConfigError("worker mode must be single or jsonl")

    network_value = raw.get("network", raw.get("allow_network", False))
    if not isinstance(network_value, bool):
        raise WorkerConfigError("network must be a boolean")
    network = network_value
    is_gemini = (
        "gemini" in worker_id.lower()
        or "mllm" in worker_id.lower()
        or any(
            "gemini" in part.lower() or "workers.mllm" in part.lower()
            for part in command
        )
    )
    explicit_guard = raw.get("network_guard")
    if network and (not is_gemini or explicit_guard != "gemini"):
        raise WorkerConfigError(
            "network is disabled except for a Gemini worker with network_guard='gemini'"
        )

    environment = raw.get("env", raw.get("environment_variables", {}))
    if environment is None:
        environment = {}
    if not isinstance(environment, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in environment.items()
    ):
        raise WorkerConfigError("env must map strings to strings")
    embedded_secrets = sorted(
        key for key, value in environment.items() if value and _is_secret_env_name(key)
    )
    if embedded_secrets:
        raise WorkerConfigError(
            "worker env must reference secret files instead of embedding "
            f"credentials: {embedded_secrets}"
        )

    checkpoints: dict[str, str] = {}
    for key, value in raw.items():
        if not _is_local_asset_key(str(key)) or value is None:
            continue
        if not isinstance(value, (str, os.PathLike)):
            raise WorkerConfigError(f"{key} must be a local path")
        text = str(value)
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", text):
            raise WorkerConfigError(f"{key} must be local; URLs are not allowed")
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = (Path(base_dir) / path).resolve()
        checkpoints[str(key)] = str(path)

    timeout = raw.get("timeout_seconds")
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
    ):
        raise WorkerConfigError("timeout_seconds must be positive")
    batch_size = _positive_int(raw.get("batch_size", 1), "batch_size")
    max_parallel = _positive_int(raw.get("max_parallel", 1), "max_parallel")
    max_retries = _nonnegative_int(raw.get("max_retries", 0), "max_retries")
    devices_raw = raw.get("cuda_devices", ())
    if (
        isinstance(devices_raw, (str, bytes))
        or not isinstance(devices_raw, Sequence)
        or not all(
            isinstance(device, (str, int))
            and not isinstance(device, bool)
            and str(device).strip()
            for device in devices_raw
        )
    ):
        raise WorkerConfigError("cuda_devices must be an array of device IDs")
    cuda_devices = tuple(str(device).strip() for device in devices_raw)
    if len(set(cuda_devices)) != len(cuda_devices):
        raise WorkerConfigError("cuda_devices must not contain duplicates")
    if cuda_devices and max_parallel > len(cuda_devices):
        raise WorkerConfigError(
            "GPU worker max_parallel cannot exceed cuda_devices count"
        )
    concurrency_group_raw = raw.get("concurrency_group")
    if concurrency_group_raw is not None and (
        not isinstance(concurrency_group_raw, str) or not concurrency_group_raw.strip()
    ):
        raise WorkerConfigError("concurrency_group must be a non-empty string")
    concurrency_group = (
        concurrency_group_raw.strip()
        if isinstance(concurrency_group_raw, str)
        else None
    )
    concurrency_limit_raw = raw.get("concurrency_limit")
    concurrency_limit = (
        _positive_int(concurrency_limit_raw, "concurrency_limit")
        if concurrency_limit_raw is not None
        else None
    )
    if (concurrency_group is None) != (concurrency_limit is None):
        raise WorkerConfigError(
            "concurrency_group and concurrency_limit must be configured together"
        )
    return WorkerConfig(
        worker_id=str(worker_id),
        command=command,
        mode=mode,
        network=network,
        environment=dict(environment),
        checkpoints=checkpoints,
        timeout_seconds=float(timeout) if timeout is not None else None,
        batch_size=1 if mode == "single" else batch_size,
        max_parallel=max_parallel,
        max_retries=max_retries,
        cuda_devices=cuda_devices,
        concurrency_group=concurrency_group,
        concurrency_limit=concurrency_limit,
    )


def _leaf_workers(root: Mapping[str, Any], prefix: str = ""):
    for name, value in root.items():
        if not isinstance(value, Mapping):
            continue
        worker_id = f"{prefix}.{name}" if prefix else str(name)
        if any(key in value for key in ("command", "command_module", "module")):
            yield worker_id, value
        else:
            yield from _leaf_workers(value, worker_id)


def _is_local_asset_key(key: str) -> bool:
    lowered = key.lower()
    return any(
        token in lowered
        for token in (
            "checkpoint",
            "config_path",
            "source_dir",
            "download_root",
            "asset_path",
        )
    ) and lowered not in {"checkpoints"}


def _is_secret_env_name(name: str) -> bool:
    upper = name.upper()
    if upper.endswith(("_PATH", "_FILE")):
        return False
    return any(
        token in upper
        for token in (
            "API_KEY",
            "CREDENTIAL",
            "TOKEN",
            "PASSWORD",
            "PASSWD",
            "SECRET",
        )
    )


def _is_sensitive_parent_env_name(name: str) -> bool:
    upper = name.upper()
    return any(
        token in upper
        for token in (
            "API_KEY",
            "APIKEY",
            "CREDENTIAL",
            "KEYS_PATH",
            "KEY_FILE",
            "KEY_PATH",
            "TOKEN",
            "JWT",
            "PASSWORD",
            "PASSWD",
            "SECRET",
            "COOKIE",
        )
    ) or upper in {
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "DOCKER_CONFIG",
        "GIT_ASKPASS",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "KRB5CCNAME",
        "KUBECONFIG",
        "NETRC",
        "SSH_AUTH_SOCK",
    }


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WorkerConfigError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkerConfigError(f"{name} must be a non-negative integer")
    return value
