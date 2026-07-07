"""Cut and concatenate highlight clips with ffmpeg."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from .models import Clip, Event


class FFmpegNotFoundError(RuntimeError):
    """Raised when ffmpeg is not on PATH."""


def _ensure_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FFmpegNotFoundError(
            "ffmpeg binary not found. Install it (e.g. `winget install Gyan.FFmpeg` "
            "on Windows, `brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Linux) "
            "and ensure it is on your PATH."
        )
    return path


def _ffmpeg() -> str:
    return _ensure_ffmpeg()


def cut_clip(source: str | Path, start: float, end: float, dest: str | Path) -> None:
    """Cut a single clip from `source` between `start` and `end` (seconds)."""
    if end <= start:
        raise ValueError(f"Clip end must be after start; got start={start}, end={end}")

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start
    cmd = [
        _ffmpeg(),
        "-y",
        "-ss", f"{start:.3f}",
        "-i", str(source),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-loglevel", "error",
        str(dest),
    ]
    subprocess.run(cmd, check=True)


def _write_concat_list(clips: list[Path], list_file: Path) -> None:
    """Write an ffmpeg concat demuxer list."""
    with list_file.open("w", encoding="utf-8") as f:
        for c in clips:
            # ffmpeg's concat demuxer handles forward-slash absolute paths best on Windows.
            safe = c.resolve().as_posix().replace("'", "'\\''")
            f.write(f"file '{safe}'\n")


def concatenate(clips: list[Path], output: str | Path) -> None:
    """Concatenate the given clips into a single MP4 (re-encoded for safety)."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    list_file = output.with_suffix(output.suffix + ".list.txt")
    _write_concat_list(clips, list_file)
    cmd = [
        _ffmpeg(),
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-loglevel", "error",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def render_highlights(
    source_video: str | Path,
    clips: list[Clip],
    output_dir: str | Path,
) -> Path:
    """
    Cut each clip from `source_video`, then concat them into one MP4.
    Returns the path to the final highlight file.
    """
    if not clips:
        raise ValueError("No clips to render - empty highlight list.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = output_dir / f"clips_{timestamp}"
    work_dir.mkdir(parents=True, exist_ok=True)

    clip_paths: list[Path] = []
    for i, clip in enumerate(clips, start=1):
        dest = work_dir / f"clip_{i:03d}.mp4"
        cut_clip(source_video, clip.start, clip.end, dest)
        clip_paths.append(dest)

    final_path = output_dir / f"highlights_{timestamp}.mp4"
    concatenate(clip_paths, final_path)

    # Drop the concat list to keep the output dir tidy
    list_file = final_path.with_suffix(final_path.suffix + ".list.txt")
    if list_file.exists():
        list_file.unlink()

    return final_path


def write_events_report(
    clips: list[Clip],
    raw_events: list[Event] | None,
    output_path: str | Path,
) -> Path:
    """Write a JSON report alongside the highlight video."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "clips": [c.model_dump(mode="json") for c in clips],
        "raw_event_count": len(raw_events) if raw_events is not None else None,
        "raw_events": (
            [event.model_dump(mode="json") for event in raw_events]
            if raw_events is not None
            else None
        ),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
