"""Tests for Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from football_highlights.models import Clip, Event, EventType


def test_event_validates_importance_bounds() -> None:
    Event(timestamp=10, event_type=EventType.GOAL, importance=5, description="Header in.")
    with pytest.raises(ValidationError):
        Event(timestamp=10, event_type=EventType.GOAL, importance=10, description="x")
    with pytest.raises(ValidationError):
        Event(timestamp=10, event_type=EventType.GOAL, importance=0, description="x")


def test_event_strips_description() -> None:
    e = Event(timestamp=10, event_type=EventType.SHOT_ON_TARGET, importance=2, description="  shot saved  ")
    assert e.description == "shot saved"


def test_event_is_highlight_worthy() -> None:
    e1 = Event(timestamp=0, event_type=EventType.GOAL, importance=5, description="x")
    e2 = Event(timestamp=0, event_type=EventType.SHOT_OFF_TARGET, importance=1, description="x")
    assert e1.is_highlight_worthy(3) is True
    assert e2.is_highlight_worthy(3) is False


def test_clip_duration() -> None:
    c = Clip(start=10, end=15, event_type=EventType.GOAL, importance=5, description="x")
    assert c.duration == 5
    assert "goal" in c.format_label()
    assert "00:10" in c.format_label()
