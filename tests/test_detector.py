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


def test_parse_events_json_uses_first_complete_fenced_block() -> None:
    batch = FrameBatch(
        start=1280,
        end=1310,
        frames=[
            FrameSample(timestamp=1280, path="frame_1.png"),
            FrameSample(timestamp=1310, path="frame_2.png"),
        ],
    )
    response = """
    ```json
    {
      "events": [
        {
          "timestamp": 1351,
          "event_type": "controversial",
          "importance": 3,
          "description": "Possible handball near the touchline.",
          "players": ["11", "3"],
          "teams": ["Barcelona", "Real Madrid"]
        }
      ]
    }
    ```

    Wait, let me recalibrate. The frame timestamps are 1280-1310s.

    ```json
    {
      "events": [
    """

    events = _parse_events_json(response, batch)

    assert len(events) == 1
    assert events[0].event_type is EventType.CONTROVERSIAL
    assert events[0].timestamp == 1310


def test_parse_events_json_rejects_non_json() -> None:
    with pytest.raises(DetectorError):
        _parse_events_json("No events here.", _batch())
