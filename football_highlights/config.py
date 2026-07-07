"""Configuration loaded from environment / .env file."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env from the project root if present
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)


class Backend(str, Enum):
    CLAUDE = "claude"
    OLLAMA = "ollama"


class Settings(BaseModel):
    """Runtime settings for the highlight generator."""

    # LLM backend
    backend: Backend = Backend.CLAUDE

    # Claude
    anthropic_api_key: str | None = None
    claude_model: str = "claude-sonnet-4-6"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5vl:7b"

    # Sampling / curation
    sample_fps: float = Field(default=0.5, gt=0, le=10)
    frames_per_batch: int = Field(default=16, ge=1, le=64)
    frame_width: int = Field(default=640, ge=160, le=1920)
    jpeg_quality: int = Field(default=4, ge=2, le=31)
    min_importance: int = Field(default=3, ge=1, le=5)
    padding_before: float = Field(default=4.0, ge=0, le=60)
    padding_after: float = Field(default=3.0, ge=0, le=60)

    # Output
    output_dir: Path = Field(default_factory=lambda: _PROJECT_ROOT / "output")

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables."""
        backend = os.getenv("BACKEND", "claude").strip().lower()
        try:
            backend_enum = Backend(backend)
        except ValueError as exc:
            raise ValueError(
                f"Invalid BACKEND={backend!r}; expected 'claude' or 'ollama'."
            ) from exc

        return cls(
            backend=backend_enum,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            claude_model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b"),
            sample_fps=float(os.getenv("SAMPLE_FPS", "0.5")),
            frames_per_batch=int(os.getenv("FRAMES_PER_BATCH", "16")),
            frame_width=int(os.getenv("FRAME_WIDTH", "640")),
            jpeg_quality=int(os.getenv("JPEG_QUALITY", "4")),
            min_importance=int(os.getenv("MIN_IMPORTANCE", "3")),
            padding_before=float(os.getenv("PADDING_BEFORE", "4")),
            padding_after=float(os.getenv("PADDING_AFTER", "3")),
            output_dir=Path(os.getenv("OUTPUT_DIR", _PROJECT_ROOT / "output")),
        )
