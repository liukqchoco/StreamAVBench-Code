"""Minimal Gemini ``generateContent`` client for benchmark judging."""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiError(RuntimeError):
    """Base class for Gemini client failures."""


class RequestTooLargeError(GeminiError):
    """Raised before transport when an inline request exceeds its limit."""


class GeminiResponseError(GeminiError):
    """Raised when Gemini returns no usable strict-JSON candidate."""


class GeminiTransportError(GeminiError):
    """Raised for retryable network or response-decoding failures."""


class LiveTransportDisabledError(GeminiError):
    """Raised when a real Gemini request is attempted in offline mode."""


class GeminiHTTPError(GeminiError):
    def __init__(self, status: int, message: str):
        super().__init__(f"Gemini HTTP {status}: {message}")
        self.status = status

    @property
    def retryable(self) -> bool:
        return self.status in {408, 409, 425, 429} or self.status >= 500


class Transport(Protocol):
    def __call__(
        self, url: str, headers: Mapping[str, str], body: bytes, timeout_s: float
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 5
    initial_delay_s: float = 2.0
    multiplier: float = 2.0
    max_delay_s: float = 30.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if self.initial_delay_s < 0 or self.multiplier < 0 or self.max_delay_s < 0:
            raise ValueError("retry delays must be non-negative")

    def delay_before(self, attempt: int) -> float:
        if attempt <= 1:
            return 0.0
        return min(
            self.initial_delay_s * self.multiplier ** (attempt - 2),
            self.max_delay_s,
        )


@dataclass(frozen=True, slots=True)
class GeminiConfig:
    api_key: str = field(repr=False)
    model: str = "gemini-3.1-pro-preview"
    sse: bool = False
    timeout_s: float = 180.0
    max_request_bytes: int = 20 * 1024 * 1024

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("api_key must be non-empty")
        if not self.model.strip() or "/" in self.model:
            raise ValueError("model must be a non-empty Gemini model ID")
        if self.timeout_s <= 0 or self.max_request_bytes <= 0:
            raise ValueError("timeout and request-size limit must be positive")


@dataclass(frozen=True, slots=True)
class InlineMedia:
    data: bytes
    mime_type: str = "video/mp4"
    video_fps: float | None = 2.0

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        mime_type: str = "video/mp4",
        video_fps: float | None = 2.0,
    ) -> InlineMedia:
        return cls(Path(path).read_bytes(), mime_type, video_fps)

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("inline media must not be empty")
        if not self.mime_type.strip():
            raise ValueError("mime_type must be non-empty")
        if self.mime_type.startswith("video/") and (
            self.video_fps is None or self.video_fps <= 0
        ):
            raise ValueError("video media requires a positive video_fps")


@dataclass(frozen=True, slots=True)
class GeminiResult:
    value: Mapping[str, Any]
    raw_text: str
    request_id: str | None
    usage: Mapping[str, Any]
    attempts: int
    model_version: str | None = None


def urllib_transport(
    url: str, headers: Mapping[str, str], body: bytes, timeout_s: float
) -> Mapping[str, Any]:
    require_live_transport()
    request = urllib.request.Request(
        url, data=body, headers=dict(headers), method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = response.read().decode("utf-8")
            content_type = response.headers.get("Content-Type", "")
            if "text/event-stream" in content_type or payload.lstrip().startswith(
                "data:"
            ):
                return _merge_sse_events(payload)
            return json.loads(payload)
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(payload).get("error", {})
            message = str(detail.get("message") or payload[:500])
        except json.JSONDecodeError:
            message = payload[:500]
        raise GeminiHTTPError(exc.code, message) from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise GeminiTransportError(f"Gemini transport failed: {exc}") from exc


class GeminiClient:
    """Inline-base64 client with strict request shape and injectable transport."""

    def __init__(
        self,
        config: GeminiConfig,
        *,
        transport: Transport | None = None,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.config = config
        self._transport = urllib_transport if transport is None else transport
        self._retry = retry if retry is not None else RetryPolicy()
        self._sleep = sleep

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: Mapping[str, Any],
        media: Sequence[InlineMedia] = (),
    ) -> GeminiResult:
        body = self.build_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=response_schema,
            media=media,
        )
        last_error: Exception | None = None
        for attempt in range(1, self._retry.max_attempts + 1):
            delay = self._retry.delay_before(attempt)
            if delay:
                self._sleep(delay)
            try:
                payload = self._transport(
                    self._url(),
                    {
                        "x-goog-api-key": self.config.api_key,
                        "Accept": (
                            "text/event-stream"
                            if self.config.sse
                            else "application/json"
                        ),
                        "Content-Type": "application/json",
                        "User-Agent": "StreamAVBench/1.0",
                    },
                    body,
                    self.config.timeout_s,
                )
                raw_text = _candidate_text(payload)
                value = _parse_candidate_json(raw_text)
                if not isinstance(value, dict):
                    raise GeminiResponseError("candidate JSON must be an object")
                return GeminiResult(
                    value=value,
                    raw_text=raw_text,
                    request_id=_optional_string(payload.get("responseId")),
                    usage=_mapping(payload.get("usageMetadata")),
                    attempts=attempt,
                    model_version=_optional_string(payload.get("modelVersion")),
                )
            except Exception as exc:
                last_error = exc
                if not _retryable(exc) or attempt == self._retry.max_attempts:
                    raise
        raise GeminiError("Gemini retries exhausted") from last_error

    def build_request(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: Mapping[str, Any],
        media: Sequence[InlineMedia] = (),
    ) -> bytes:
        if not system_prompt.strip() or not user_prompt.strip():
            raise ValueError("system_prompt and user_prompt must be non-empty")
        _require_strict_schema(response_schema)
        parts: list[dict[str, Any]] = [{"text": user_prompt}]
        for item in media:
            part: dict[str, Any] = {
                "inlineData": {
                    "mimeType": item.mime_type,
                    "data": base64.b64encode(item.data).decode("ascii"),
                }
            }
            if item.mime_type.startswith("video/"):
                part["videoMetadata"] = {"fps": item.video_fps}
            parts.append(part)
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": dict(response_schema),
            },
        }
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
        if len(body) > self.config.max_request_bytes:
            raise RequestTooLargeError(
                f"encoded request is {len(body)} bytes; limit is "
                f"{self.config.max_request_bytes} bytes"
            )
        return body

    def _url(self) -> str:
        method = "streamGenerateContent" if self.config.sse else "generateContent"
        url = f"{GEMINI_API_BASE}/models/{self.config.model}:{method}"
        return f"{url}?alt=sse" if self.config.sse else url


def live_transport_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether the process explicitly opted into real Gemini traffic."""

    source = os.environ if environ is None else environ
    return source.get("STREAMAV_ALLOW_GEMINI") == "1"


def require_live_transport(environ: Mapping[str, str] | None = None) -> None:
    """Fail closed unless live Gemini transport was explicitly enabled."""

    if not live_transport_enabled(environ):
        raise LiveTransportDisabledError(
            "live Gemini transport is disabled; set STREAMAV_ALLOW_GEMINI=1 "
            "to explicitly enable it"
        )


def _candidate_text(payload: Mapping[str, Any]) -> str:
    try:
        parts = payload["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GeminiResponseError("response contains no candidate text") from exc
    texts = [
        part["text"]
        for part in parts
        if isinstance(part, Mapping)
        and part.get("thought") is not True
        and isinstance(part.get("text"), str)
    ]
    text = "".join(texts)
    if not text.strip():
        raise GeminiResponseError("candidate text is empty")
    return text


def _merge_sse_events(payload: str) -> Mapping[str, Any]:
    events: list[Mapping[str, Any]] = []
    for line in payload.splitlines():
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw or raw == "[DONE]":
            continue
        event = json.loads(raw)
        if isinstance(event, Mapping):
            events.append(event)
    if not events:
        raise GeminiResponseError("SSE response contains no JSON events")

    merged: dict[str, Any] = {}
    for key in ("responseId", "modelVersion", "usageMetadata"):
        for event in reversed(events):
            if key in event:
                merged[key] = event[key]
                break

    parts: list[Mapping[str, Any]] = []
    final_candidate: dict[str, Any] = {}
    for event in events:
        candidates = event.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            continue
        candidate = candidates[0]
        if not isinstance(candidate, Mapping):
            continue
        final_candidate.update(candidate)
        content = candidate.get("content")
        if not isinstance(content, Mapping):
            continue
        event_parts = content.get("parts")
        if isinstance(event_parts, list):
            parts.extend(part for part in event_parts if isinstance(part, Mapping))
    if parts:
        final_candidate["content"] = {"parts": parts}
        merged["candidates"] = [final_candidate]
    return merged


def _parse_candidate_json(raw_text: str) -> Any:
    text = raw_text.strip()
    if text.startswith("```") and text.endswith("```"):
        first_newline = text.find("\n")
        if first_newline < 0:
            raise GeminiResponseError("fenced candidate contains no JSON body")
        language = text[3:first_newline].strip().lower()
        if language not in {"", "json"}:
            raise GeminiResponseError(
                f"unsupported candidate code-fence language: {language}"
            )
        text = text[first_newline + 1 : -3].strip()
    elif text.startswith("`") and text.endswith("`") and len(text) >= 2:
        text = text[1:-1].strip()
    return json.loads(text)


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, GeminiHTTPError):
        return exc.retryable
    return isinstance(
        exc,
        (
            GeminiResponseError,
            GeminiTransportError,
            json.JSONDecodeError,
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
        ),
    )


def _require_strict_schema(schema: Mapping[str, Any]) -> None:
    """Reject object schemas that allow undeclared output fields."""

    if (
        schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
    ):
        raise ValueError(
            "response_schema must be a strict object with additionalProperties=false"
        )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None
