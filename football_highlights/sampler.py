"""Frame extraction from a video using ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .models import FrameBatch, FrameSample


class FFmpegNotFoundError(RuntimeError):
    """Raised when the ffmpeg binary is not on PATH."""


def _ensure_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FFmpegNotFoundError(
            "ffmpeg binary not found. Install it (e.g. `winget install Gyan.FFmpeg` "
            "on Windows, `brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Linux) "
            "and ensure it is on your PATH."
        )
    return path


def get_duration_seconds(video_path: str | Path) -> float:
    """Return the duration of the video in seconds via ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        # Fallback to ffmpeg -i parsing if ffprobe is missing
        ffprobe = _ensure_ffmpeg()  # raises if missing
    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def sample_frames(
    video_path: str | Path,
    fps: float = 0.5,
    output_dir: str | Path | None = None,
    frame_width: int = 640,
    jpeg_quality: int = 4,
) -> list[FrameSample]:
    """
    Sample frames from a video at a given rate.

    Returns a list of FrameSample with timestamp + path to a JPEG on disk.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if frame_width <= 0:
        raise ValueError("frame_width must be positive")
    if not 2 <= jpeg_quality <= 31:
        raise ValueError("jpeg_quality must be between 2 and 31")

    _ensure_ffmpeg()
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="football_frames_"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JPEG keeps vision requests small enough for cloud API limits.
    pattern = output_dir / "frame_%06d.jpg"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps},scale='min({frame_width},iw)':-2",
        "-q:v", str(jpeg_quality),
        "-loglevel", "error",
        str(pattern),
    ]
    subprocess.run(cmd, check=True)

    # List resulting files in sorted order; derive timestamps from index.
    files = sorted(output_dir.glob("frame_*.jpg"))
    interval = 1.0 / fps
    samples = [
        FrameSample(timestamp=round(i * interval, 3), path=str(p))
        for i, p in enumerate(files)
    ]
    return samples


def batch_frames(
    frames: list[FrameSample],
    batch_size: int = 16,
) -> list[FrameBatch]:
    """Group frames into temporal batches for the LLM."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    batches: list[FrameBatch] = []
    for i in range(0, len(frames), batch_size):
        window = frames[i : i + batch_size]
        start = window[0].timestamp
        end = window[-1].timestamp
        batches.append(FrameBatch(start=start, end=end, frames=window))
    return batches
