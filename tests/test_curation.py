"""Tests for the curation pipeline (pure logic, no API)."""

from __future__ import annotations

from football_highlights.curation import (
    curate,
    dedupe_overlapping,
    expand_to_clips,
    filter_by_importance,
    merge_close_clips,
    sort_clips,
)
from football_highlights.models import Clip, Event, EventType


def _ev(t: float, et: EventType, imp: int, desc: str = "x") -> Event:
    return Event(timestamp=t, event_type=et, importance=imp, description=desc)


def test_filter_by_importance() -> None:
    events = [
        _ev(10, EventType.GOAL, 5),
        _ev(20, EventType.SHOT_OFF_TARGET, 1),
        _ev(30, EventType.YELLOW_CARD, 3),
    ]
    out = filter_by_importance(events, min_importance=3)
    assert len(out) == 2
    assert {e.event_type for e in out} == {EventType.GOAL, EventType.YELLOW_CARD}


def test_dedupe_keeps_higher_importance() -> None:
    events = [
        _ev(10, EventType.SHOT_ON_TARGET, 2),
        _ev(11, EventType.GOAL, 5),
    ]
    out = dedupe_overlapping(events, min_gap_seconds=10)
    assert len(out) == 1
    assert out[0].event_type is EventType.GOAL


def test_dedupe_keeps_far_apart_events() -> None:
    events = [
        _ev(10, EventType.GOAL, 5),
        _ev(120, EventType.YELLOW_CARD, 3),
    ]
    out = dedupe_overlapping(events, min_gap_seconds=10)
    assert len(out) == 2


def test_expand_to_clips_applies_padding() -> None:
    events = [_ev(60, EventType.GOAL, 5, "header")]
    clips = expand_to_clips(events, padding_before=4, padding_after=3, video_duration=300)
    assert len(clips) == 1
    assert clips[0].start == 56
    assert clips[0].end == 63


def test_expand_clamps_to_video_duration() -> None:
    events = [_ev(298, EventType.GOAL, 5)]
    clips = expand_to_clips(events, padding_before=4, padding_after=3, video_duration=300)
    assert clips[0].end == 300


def test_expand_drops_zero_length_clips() -> None:
    # If padding_after is 0 and the event lands exactly at the start,
    # we get a zero-length clip that should be dropped.
    events = [_ev(0, EventType.GOAL, 5)]
    clips = expand_to_clips(events, padding_before=0, padding_after=0)
    assert clips == []


def test_merge_close_clips() -> None:
    clips = [
        Clip(start=10, end=15, event_type=EventType.GOAL, importance=5, description="a"),
        Clip(start=16, end=20, event_type=EventType.OTHER,
             importance=2, description="b"),
    ]
    out = merge_close_clips(clips, max_gap=2.0)
    assert len(out) == 1
    assert out[0].start == 10
    assert out[0].end == 20
    assert out[0].importance == 5
    assert out[0].source_event_count == 2


def test_merge_does_not_combine_far_clips() -> None:
    clips = [
        Clip(start=10, end=15, event_type=EventType.GOAL, importance=5, description="a"),
        Clip(start=40, end=45, event_type=EventType.YELLOW_CARD, importance=3, description="b"),
    ]
    out = merge_close_clips(clips, max_gap=2.0)
    assert len(out) == 2


def test_sort_clips() -> None:
    clips = [
        Clip(start=30, end=35, event_type=EventType.GOAL, importance=5, description="a"),
        Clip(start=10, end=15, event_type=EventType.YELLOW_CARD, importance=3, description="b"),
    ]
    out = sort_clips(clips)
    assert [c.start for c in out] == [10, 30]


def test_curate_full_pipeline() -> None:
    events = [
        _ev(10, EventType.SHOT_ON_TARGET, 2),       # dropped (low imp)
        _ev(12, EventType.GOAL, 5, "winner"),       # kept, deduped with above
        _ev(11, EventType.SAVE, 3),                 # within dedupe gap of 10
        _ev(60, EventType.YELLOW_CARD, 3, "tackle"),
        _ev(70, EventType.RED_CARD, 5, "violent conduct"),
    ]
    out = curate(
        events,
        min_importance=3,
        dedupe_gap=10.0,
        padding_before=4,
        padding_after=3,
        merge_gap=2.0,
        video_duration=600,
    )
    # We expect: a clip around the goal (kept over save/sot), a clip around the
    # yellow card, and a clip around the red card. The goal and yellow are >2s
    # apart so they don't merge; the yellow and red are >2s apart.
    assert len(out) == 3
    assert out[0].event_type is EventType.GOAL
    assert out[1].event_type is EventType.YELLOW_CARD
    assert out[2].event_type is EventType.RED_CARD
    assert [c.start for c in out] == sorted(c.start for c in out)
