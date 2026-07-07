"""Event curation: filter, dedupe, expand to clips, merge, sort."""

from __future__ import annotations

from .models import Clip, Event


def filter_by_importance(events: list[Event], min_importance: int) -> list[Event]:
    """Drop events below the importance threshold."""
    return [e for e in events if e.is_highlight_worthy(min_importance)]


def dedupe_overlapping(
    events: list[Event], min_gap_seconds: float = 10.0
) -> list[Event]:
    """
    When two events are within `min_gap_seconds` of each other, keep the one
    with the higher importance (and earlier timestamp on tie).
    """
    if not events:
        return []
    sorted_events = sorted(events, key=lambda e: (e.timestamp, -e.importance))
    kept: list[Event] = []
    for ev in sorted_events:
        if not kept:
            kept.append(ev)
            continue
        if ev.timestamp - kept[-1].timestamp < min_gap_seconds:
            if ev.importance > kept[-1].importance:
                kept[-1] = ev
        else:
            kept.append(ev)
    return kept


def expand_to_clips(
    events: list[Event],
    padding_before: float = 4.0,
    padding_after: float = 3.0,
    video_duration: float | None = None,
) -> list[Clip]:
    """Turn each event into a Clip with padding on either side."""
    clips: list[Clip] = []
    for ev in events:
        start = max(0.0, ev.timestamp - padding_before)
        end = ev.timestamp + padding_after
        if video_duration is not None:
            end = min(end, video_duration)
        if end <= start:
            continue
        clips.append(
            Clip(
                start=start,
                end=end,
                event_type=ev.event_type,
                importance=ev.importance,
                description=ev.description,
                source_event_count=1,
            )
        )
    return clips


def merge_close_clips(clips: list[Clip], max_gap: float = 2.0) -> list[Clip]:
    """
    Combine adjacent clips that are within `max_gap` seconds of each other,
    so the final highlight flows naturally without rapid cuts.
    """
    if not clips:
        return []
    sorted_clips = sorted(clips, key=lambda c: c.start)
    merged: list[Clip] = [sorted_clips[0].model_copy(deep=True)]
    for clip in sorted_clips[1:]:
        last = merged[-1]
        if clip.start - last.end <= max_gap:
            last.end = max(last.end, clip.end)
            last.importance = max(last.importance, clip.importance)
            last.source_event_count += 1
            if clip.event_type.value != last.event_type.value and clip.importance >= last.importance:
                # Capture the more interesting event type
                last.event_type = clip.event_type
                last.description = clip.description
        else:
            merged.append(clip.model_copy(deep=True))
    return merged


def sort_clips(clips: list[Clip]) -> list[Clip]:
    return sorted(clips, key=lambda c: c.start)


def curate(
    events: list[Event],
    min_importance: int = 3,
    dedupe_gap: float = 10.0,
    padding_before: float = 4.0,
    padding_after: float = 3.0,
    merge_gap: float = 2.0,
    video_duration: float | None = None,
) -> list[Clip]:
    """Full pipeline: filter -> dedupe -> expand -> merge -> sort."""
    filtered = filter_by_importance(events, min_importance)
    deduped = dedupe_overlapping(filtered, dedupe_gap)
    clips = expand_to_clips(deduped, padding_before, padding_after, video_duration)
    clips = merge_close_clips(clips, merge_gap)
    clips = sort_clips(clips)
    return clips
