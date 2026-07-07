"""Tests for detector response parsing."""

from __future__ import annotations

import pytest

from football_highlights.detector import DetectorError, _parse_events_json
from football_highlights.models import EventType, FrameBatch, FrameSample


def _batch() -> FrameBatch:
    return FrameBatch(
        start=10,
        end=20,
        frames=[
            FrameSample(timestamp=10, path="frame_1.png"),
            FrameSample(timestamp=20, path="frame_2.png"),
        ],
    )


def test_parse_events_json_object() -> None:
    events = _parse_events_json(
        """
        {
          "events": [
            {
              "timestamp": 12,
              "event_type": "goal",
              "importance": 5,
              "description": "Shot into the bottom corner.",
              "players": ["9"],
              "teams": ["home"]
            }
          ]
        }
        """,
        _batch(),
    )

    assert len(events) == 1
    assert events[0].event_type is EventType.GOAL
    assert events[0].timestamp == 12


def test_parse_events_json_bare_array() -> None:
    events = _parse_events_json(
        """
        [
          {
            "timestamp": 25,
            "event_type": "foul",
            "importance": "4",
            "description": "Late challenge near the penalty area."
          }
        ]
        """,
        _batch(),
    )

    assert len(events) == 1
    assert events[0].event_type is EventType.CONTROVERSIAL
    assert events[0].timestamp == 20
    assert events[0].importance == 4


def test_parse_events_json_rejects_non_json() -> None:
    with pytest.raises(DetectorError):
        _parse_events_json("No events here.", _batch())
