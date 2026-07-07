"""Pydantic data models shared across the pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EventType(str, Enum):
    """Taxonomy of football events the detector is asked to identify."""

    GOAL = "goal"
    SHOT_ON_TARGET = "shot_on_target"
    SHOT_OFF_TARGET = "shot_off_target"
    PENALTY = "penalty"
    SAVE = "save"
    YELLOW_CARD = "yellow_card"
    RED_CARD = "red_card"
    SUBSTITUTION = "substitution"
    VAR_CHECK = "var_check"
    DANGEROUS_FREE_KICK = "dangerous_free_kick"
    CONTROVERSIAL = "controversial"
    OTHER = "other"

    @classmethod
    def labels(cls) -> list[str]:
        return [e.value for e in cls]


class Event(BaseModel):
    """A single notable moment in the match."""

    timestamp: float = Field(..., ge=0, description="Seconds into the video")
    event_type: EventType
    importance: int = Field(..., ge=1, le=5, description="1=trivial, 5=match-defining")
    description: str = Field(..., min_length=1, max_length=500)
    players: list[str] = Field(default_factory=list)
    teams: list[str] = Field(default_factory=list)

    @field_validator("description")
    @classmethod
    def _strip_description(cls, v: str) -> str:
        return v.strip()

    def is_highlight_worthy(self, threshold: int) -> bool:
        return self.importance >= threshold


class Clip(BaseModel):
    """A cut range from the source video, derived from one or more events."""

    start: float = Field(..., ge=0)
    end: float = Field(..., ge=0)
    event_type: EventType
    importance: int = Field(..., ge=1, le=5)
    description: str
    source_event_count: int = 1

    @property
    def duration(self) -> float:
        return self.end - self.start

    def format_label(self) -> str:
        mins, secs = divmod(int(self.start), 60)
        return f"[{mins:02d}:{secs:02d}] {self.event_type.value}: {self.description}"


class FrameSample(BaseModel):
    """One sampled frame with its timestamp in the source video."""

    timestamp: float = Field(..., ge=0)
    path: Optional[str] = None
    data: Optional[bytes] = None  # PNG bytes; used when path is None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FrameBatch(BaseModel):
    """A temporal window of frames, sent to the LLM as one request."""

    start: float
    end: float
    frames: list[FrameSample]

    @property
    def duration(self) -> float:
        return self.end - self.start
