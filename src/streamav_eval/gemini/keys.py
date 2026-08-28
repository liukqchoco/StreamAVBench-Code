"""Deterministic, secret-safe Gemini API key slotting."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


class KeyPoolError(ValueError):
    """Raised when a key pool cannot be used safely."""


@dataclass(frozen=True, slots=True)
class KeySelection:
    """A selected key and its stable zero-based slot."""

    slot: int
    key: str

    def __repr__(self) -> str:
        return f"KeySelection(slot={self.slot}, key=<redacted>)"


@dataclass(frozen=True, slots=True)
class GeminiKeyPool:
    """An ordered pool with at most 100 stable slots."""

    _keys: tuple[str, ...] = field(repr=False)
    MAX_KEYS = 100
    ENV_PATH = "STREAMAV_GEMINI_KEYS_PATH"
    DEFAULT_PATH = Path(".secrets/api_keys.txt")

    @classmethod
    def from_file(cls, path: str | Path) -> GeminiKeyPool:
        key_path = Path(path).expanduser()
        if not key_path.is_file():
            raise KeyPoolError(f"Gemini key pool does not exist: {key_path}")
        if os.name == "posix" and key_path.stat().st_mode & 0o077:
            raise KeyPoolError(
                f"Gemini key pool permissions are too broad: {key_path}; "
                "run chmod 600 on the file"
            )
        keys = tuple(
            line.strip()
            for line in key_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        return cls.from_keys(keys)

    @classmethod
    def from_environment(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        default_path: str | Path | None = None,
    ) -> GeminiKeyPool:
        """Load the env-selected key file, or the repository-local default."""

        source = os.environ if environ is None else environ
        configured = source.get(cls.ENV_PATH)
        path = (
            Path(configured).expanduser()
            if configured
            else Path(default_path or cls.DEFAULT_PATH).expanduser()
        )
        return cls.from_file(path)

    @classmethod
    def from_keys(cls, keys: tuple[str, ...] | list[str]) -> GeminiKeyPool:
        normalized = tuple(str(key).strip() for key in keys)
        if not normalized or any(not key for key in normalized):
            raise KeyPoolError("Gemini key pool must contain non-empty keys")
        if len(normalized) > cls.MAX_KEYS:
            raise KeyPoolError(f"Gemini key pool supports at most {cls.MAX_KEYS} slots")
        if len(set(normalized)) != len(normalized):
            raise KeyPoolError("Gemini key pool contains duplicate keys")
        return cls(normalized)

    def __len__(self) -> int:
        return len(self._keys)

    def for_slot(self, slot: int) -> KeySelection:
        if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
            raise KeyPoolError("key slot must be a non-negative integer")
        selected = slot % len(self._keys)
        return KeySelection(selected, self._keys[selected])

    def for_work(self, work_id: str | int) -> KeySelection:
        """Map work deterministically, independent of process hash randomization."""

        if isinstance(work_id, bool):
            raise KeyPoolError("work_id must be a string or integer")
        if isinstance(work_id, int):
            if work_id < 0:
                raise KeyPoolError("integer work_id must be non-negative")
            slot = work_id
        elif isinstance(work_id, str) and work_id:
            slot = int.from_bytes(
                hashlib.sha256(work_id.encode("utf-8")).digest()[:8], "big"
            )
        else:
            raise KeyPoolError("work_id must be a non-empty string or integer")
        return self.for_slot(slot)
