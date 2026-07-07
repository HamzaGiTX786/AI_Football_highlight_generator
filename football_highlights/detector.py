"""Vision LLM backends for event detection."""

from __future__ import annotations

import base64
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx

from .config import Backend, Settings
from .models import Event, EventType, FrameBatch
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


class DetectorError(RuntimeError):
    """Raised when the LLM backend fails to produce a usable response."""


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class VisionBackend(ABC):
    """Abstract vision LLM backend."""

    name: str = "base"

    @abstractmethod
    def detect(self, batch: FrameBatch) -> list[Event]:
        """Run detection on one frame batch. Returns a list of Event."""


# ---------------------------------------------------------------------------
# Claude (Anthropic)
# ---------------------------------------------------------------------------


class ClaudeBackend(VisionBackend):
    name = "claude"

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise DetectorError(
                "ANTHROPIC_API_KEY is not set. Either set it in .env, "
                "or use --backend ollama for a free local model."
            )
        # Import locally so the dependency is optional at import time
        from anthropic import APIError, Anthropic

        self._client = Anthropic(api_key=api_key)
        self._model = model
        self._api_error = APIError

    def detect(self, batch: FrameBatch) -> list[Event]:
        content: list[dict[str, Any]] = []
        for frame in batch.frames:
            image_data, media_type = _image_to_b64_and_media_type(frame.path)
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    },
                }
            )
        content.append(
            {
                "type": "text",
                "text": USER_PROMPT_TEMPLATE.format(
                    n=len(batch.frames),
                    start=batch.start,
                    end=batch.end,
                    duration=batch.duration,
                )
                + _frame_timestamp_block(batch),
            }
        )

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
        except self._api_error as exc:
            if exc.__class__.__name__ == "RequestTooLargeError":
                raise DetectorError(
                    "Claude rejected this batch because the images are too large. "
                    "Try --frames-per-batch 8, --frame-width 480, or --fps 0.25."
                ) from exc
            raise DetectorError(f"Claude request failed: {exc}") from exc

        text = _extract_text(response)
        return _parse_events_json(text, batch)


# ---------------------------------------------------------------------------
# Ollama (local)
# ---------------------------------------------------------------------------


class OllamaBackend(VisionBackend):
    name = "ollama"

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.Client(timeout=300.0)

    def detect(self, batch: FrameBatch) -> list[Event]:
        images_b64 = [_image_to_b64_and_media_type(f.path)[0] for f in batch.frames]
        prompt = (
            SYSTEM_PROMPT
            + "\n\n"
            + USER_PROMPT_TEMPLATE.format(
                n=len(batch.frames),
                start=batch.start,
                end=batch.end,
                duration=batch.duration,
            )
            + _frame_timestamp_block(batch)
        )

        payload = {
            "model": self._model,
            "prompt": prompt,
            "images": images_b64,
            "stream": False,
            "format": "json",
        }
        try:
            r = self._client.post(f"{self._base_url}/api/generate", json=payload)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise DetectorError(
                f"Ollama request failed: {exc}. "
                f"Make sure `ollama serve` is running and the model "
                f"'{self._model}' is pulled (`ollama pull {self._model}`)."
            ) from exc

        data = r.json()
        text = data.get("response", "")
        return _parse_events_json(text, batch)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_to_b64_and_media_type(path: str | None) -> tuple[str, str]:
    if not path:
        raise DetectorError("Frame has no on-disk path; cannot encode.")
    image_path = Path(path)
    suffix = image_path.suffix.lower()
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix)
    if media_type is None:
        raise DetectorError(f"Unsupported frame image type: {suffix or '(none)'}")
    data = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
    return data, media_type


def _frame_timestamp_block(batch: FrameBatch) -> str:
    """Per-frame timestamp legend so the LLM can map images to seconds."""
    lines = ["", "Frame timestamps (in seconds):"]
    for i, f in enumerate(batch.frames):
        lines.append(f"  Frame {i + 1}: t={f.timestamp:.1f}s")
    return "\n".join(lines)


def _extract_text(response: Any) -> str:
    """Pull the text content out of an Anthropic Message response."""
    parts: list[str] = []
    for block in getattr(response, "content", []):
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    text = "\n".join(parts).strip()
    if not text:
        raise DetectorError("LLM returned an empty response.")
    return text


def _parse_events_json(text: str, batch: FrameBatch) -> list[Event]:
    """Robustly extract the events array from a possibly chatty LLM response."""
    for candidate in _json_candidates(text):
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    else:
        raise DetectorError(f"Could not parse JSON from LLM response:\n{text[:500]}")

    if isinstance(data, list):
        raw_events = data
    else:
        raw_events = data.get("events", [])
    if not isinstance(raw_events, list):
        raise DetectorError("LLM JSON did not contain an 'events' list.")

    events: list[Event] = []
    for raw in raw_events:
        try:
            ev = _coerce_event(raw, batch)
            if ev is not None:
                events.append(ev)
        except Exception as exc:  # noqa: BLE001 - skip bad rows
            # Skip rows the LLM got wrong; don't fail the whole batch.
            print(f"  [warn] skipped malformed event: {exc}")
            continue
    return events


def _json_candidates(text: str) -> list[str]:
    """Return likely complete JSON snippets from a model response."""
    candidates = [text.strip()]

    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    candidates.extend(snippet.strip() for snippet in fenced)

    candidates.extend(_balanced_json_snippets(text, "{", "}"))
    candidates.extend(_balanced_json_snippets(text, "[", "]"))

    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _balanced_json_snippets(text: str, opener: str, closer: str) -> list[str]:
    """Extract complete top-level JSON-looking snippets without greedy regex."""
    snippets: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False

    for i, char in enumerate(text):
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if char == opener:
            if depth == 0:
                start = i
            depth += 1
        elif char == closer and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                snippets.append(text[start : i + 1])
                start = None

    return snippets


def _coerce_event(raw: dict[str, Any], batch: FrameBatch) -> Event | None:
    if "timestamp" not in raw or "event_type" not in raw:
        return None
    try:
        ts = float(raw["timestamp"])
    except (TypeError, ValueError):
        return None

    # Clamp timestamp to the batch window
    ts = max(batch.start, min(ts, batch.end))

    et_raw = str(raw["event_type"]).strip().lower()
    try:
        event_type = EventType(et_raw)
    except ValueError:
        # Map common misspellings
        aliases = {
            "shot": EventType.SHOT_ON_TARGET,
            "shot_saved": EventType.SAVE,
            "corner": EventType.OTHER,
            "foul": EventType.CONTROVERSIAL,
            "offside": EventType.OTHER,
        }
        event_type = aliases.get(et_raw, EventType.OTHER)

    try:
        importance = int(raw.get("importance", 3))
    except (TypeError, ValueError):
        importance = 3
    importance = max(1, min(5, importance))

    description = str(raw.get("description", "")).strip() or "Notable event."
    if len(description) > 500:
        description = description[:497] + "..."

    players = raw.get("players") or []
    teams = raw.get("teams") or []
    if not isinstance(players, list):
        players = [str(players)]
    if not isinstance(teams, list):
        teams = [str(teams)]
    players = [str(p) for p in players][:10]
    teams = [str(t) for t in teams][:4]

    return Event(
        timestamp=ts,
        event_type=event_type,
        importance=importance,
        description=description,
        players=players,
        teams=teams,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_backend(settings: Settings) -> VisionBackend:
    if settings.backend is Backend.CLAUDE:
        return ClaudeBackend(
            api_key=settings.anthropic_api_key or "",
            model=settings.claude_model,
        )
    if settings.backend is Backend.OLLAMA:
        return OllamaBackend(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )
    raise DetectorError(f"Unknown backend: {settings.backend}")
