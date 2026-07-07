"""Tests for renderer helpers that do not require ffmpeg."""

from __future__ import annotations

import json
from pathlib import Path

from football_highlights.models import Clip, Event, EventType
from football_highlights.renderer import _write_concat_list, write_events_report


def test_write_concat_list_uses_ffmpeg_friendly_paths(tmp_path: Path) -> None:
    clip = tmp_path / "clip one.mp4"
    clip.write_bytes(b"")
    list_file = tmp_path / "concat.txt"

    _write_concat_list([clip], list_file)

    text = list_file.read_text(encoding="utf-8")
    assert text.startswith("file '")
    assert clip.resolve().as_posix() in text


def test_write_events_report_includes_raw_events(tmp_path: Path) -> None:
    clips = [
        Clip(
            start=6,
            end=13,
            event_type=EventType.GOAL,
            importance=5,
            description="Shot into the top corner.",
        )
    ]
    events = [
        Event(
            timestamp=10,
            event_type=EventType.GOAL,
            importance=5,
            description="Shot into the top corner.",
        )
    ]

    path = write_events_report(clips, events, tmp_path / "nested" / "events.json")
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["raw_event_count"] == 1
    assert data["raw_events"][0]["event_type"] == "goal"
    assert data["clips"][0]["event_type"] == "goal"
