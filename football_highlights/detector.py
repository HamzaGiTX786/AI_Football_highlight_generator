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
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key)
        self._model = model

    def detect(self, batch: FrameBatch) -> list[Event]:
        content: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": _png_to_b64(f.path),
                },
            }
            for f in batch.frames
        ]
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

        response = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )

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
        images_b64 = [_png_to_b64(f.path) for f in batch.frames]
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


def _png_to_b64(path: str | None) -> str:
    if not path:
        raise DetectorError("Frame has no on-disk path; cannot encode.")
    return base64.standard_b64encode(Path(path).read_bytes()).decode("ascii")


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
    # Try direct JSON parse first
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find a JSON object in the response
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise DetectorError(f"Could not parse JSON from LLM response:\n{text[:500]}")
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            raise DetectorError(
                f"Could not parse JSON from LLM response:\n{text[:500]}"
            ) from exc

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
