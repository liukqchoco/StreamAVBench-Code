"""Read-only loading and exact substitution for frozen MLLM prompts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


class FrozenPromptLoader:
    """Loads only named prompt files and preserves their bytes as text."""

    def __init__(self, prompts_dir: str | Path):
        self.prompts_dir = Path(prompts_dir)

    def read(self, filename: str) -> str:
        if not filename or Path(filename).name != filename:
            raise ValueError(f"invalid frozen prompt filename: {filename!r}")
        path = self.prompts_dir / filename
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"prompt file is empty: {path}")
        return text

    def render(self, filename: str, replacements: Mapping[str, str]) -> str:
        text = self.read(filename)
        for marker, value in replacements.items():
            placeholder = "{" + marker + "}"
            if text.count(placeholder) != 1:
                raise ValueError(
                    f"frozen prompt {filename} must contain {placeholder} exactly once"
                )
            text = text.replace(placeholder, value)
        return text
