"""Command-line entry point: `python -m football_highlights <video>`."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .config import Backend, Settings
from .curation import curate
from .detector import DetectorError, get_backend
from .models import Event
from .renderer import render_highlights, write_events_report
from .sampler import (
    FFmpegNotFoundError,
    batch_frames,
    get_duration_seconds,
    sample_frames,
)

console = Console()


def _print_banner(text: str) -> None:
    console.rule(f"[bold cyan]{text}[/bold cyan]")


def _print_events_table(events: list[Event]) -> None:
    table = Table(title="Detected Events", show_lines=False)
    table.add_column("Time", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("Imp", justify="right")
    table.add_column("Description")

    def _fmt(t: float) -> str:
        m, s = divmod(int(t), 60)
        return f"{m:02d}:{s:02d}"

    for e in sorted(events, key=lambda x: x.timestamp):
        table.add_row(_fmt(e.timestamp), e.event_type.value, str(e.importance), e.description)
    console.print(table)


def _print_clips_table(clips) -> None:
    table = Table(title="Highlight Clips", show_lines=False)
    table.add_column("Start", style="cyan", no_wrap=True)
    table.add_column("End", style="cyan", no_wrap=True)
    table.add_column("Dur", justify="right")
    table.add_column("Type", style="magenta")
    table.add_column("Imp", justify="right")
    table.add_column("Description")

    def _fmt(t: float) -> str:
        m, s = divmod(int(t), 60)
        return f"{m:02d}:{s:02d}"

    for c in clips:
        table.add_row(
            _fmt(c.start),
            _fmt(c.end),
            f"{c.duration:.1f}s",
            c.event_type.value,
            str(c.importance),
            c.description,
        )
    console.print(table)


def _run_pipeline(
    video_path: Path,
    settings: Settings,
    sample_fps: float,
    frames_per_batch: int,
    min_importance: int,
    padding_before: float,
    padding_after: float,
) -> None:
    if not video_path.is_file():
        raise click.ClickException(f"Video not found: {video_path}")

    _print_banner("Stage 1 / 4 - Probing video")
    duration = get_duration_seconds(video_path)
    console.print(
        f"  Video: [bold]{video_path.name}[/bold]\n"
        f"  Duration: [bold]{duration / 60:.1f} min[/bold]\n"
        f"  Backend: [bold]{settings.backend.value}[/bold]  "
        f"(model: [bold]{settings.claude_model if settings.backend is Backend.CLAUDE else settings.ollama_model}[/bold])"
    )

    settings.output_dir.mkdir(parents=True, exist_ok=True)

    _print_banner("Stage 2 / 4 - Sampling frames")
    with tempfile.TemporaryDirectory(prefix="football_frames_") as tmp:
        frames = sample_frames(video_path, fps=sample_fps, output_dir=tmp)
        console.print(f"  Sampled [bold]{len(frames)}[/bold] frames at {sample_fps} fps")
        if not frames:
            raise click.ClickException("No frames extracted - is the video valid?")

        batches = batch_frames(frames, batch_size=frames_per_batch)
        console.print(f"  Split into [bold]{len(batches)}[/bold] batch(es) of up to {frames_per_batch} frames each")

        _print_banner("Stage 3 / 4 - Detecting events")
        backend = get_backend(settings)
        all_events: list[Event] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Analyzing batches", total=len(batches))
            for batch in batches:
                try:
                    events = backend.detect(batch)
                except DetectorError as exc:
                    console.print(f"  [red]Backend error:[/red] {exc}")
                    raise click.ClickException(str(exc))
                all_events.extend(events)
                progress.advance(task)

        console.print(f"  Detected [bold]{len(all_events)}[/bold] raw event(s)")
        if all_events:
            _print_events_table(all_events)

    _print_banner("Stage 4 / 4 - Curating & rendering")
    clips = curate(
        all_events,
        min_importance=min_importance,
        padding_before=padding_before,
        padding_after=padding_after,
        video_duration=duration,
    )
    if not clips:
        console.print(
            "  [yellow]No highlight clips passed the importance threshold "
            f"({min_importance}).[/yellow] Try lowering --min-importance."
        )
        return

    _print_clips_table(clips)

    final_path = render_highlights(video_path, clips, settings.output_dir)
    report_path = write_events_report(clips, all_events, settings.output_dir / "events.json")

    _print_banner("Done")
    console.print(f"  [bold green]Highlight video:[/bold green] {final_path}")
    console.print(f"  [bold green]Events report:  [/bold green] {report_path}")


@click.command(
    help="Generate a highlight reel from a full football match video.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.argument(
    "video",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--backend",
    type=click.Choice([b.value for b in Backend], case_sensitive=False),
    default=None,
    help="Vision LLM backend. Default: from BACKEND in .env (claude).",
)
@click.option(
    "--fps",
    type=float,
    default=None,
    help="Frames per second to sample from the input video (default: 0.5).",
)
@click.option(
    "--frames-per-batch",
    type=int,
    default=None,
    help="How many frames to send to the LLM per request (default: 16).",
)
@click.option(
    "--min-importance",
    type=int,
    default=None,
    help="Drop detected events with importance below this (1-5, default: 3).",
)
@click.option(
    "--padding-before",
    type=float,
    default=None,
    help="Seconds of context before each event (default: 4).",
)
@click.option(
    "--padding-after",
    type=float,
    default=None,
    help="Seconds of context after each event (default: 3).",
)
@click.option(
    "--output",
    "output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where to write the highlight MP4 and report (default: ./output).",
)
def main(
    video: Path,
    backend: str | None,
    fps: float | None,
    frames_per_batch: int | None,
    min_importance: int | None,
    padding_before: float | None,
    padding_after: float | None,
    output_dir: Path | None,
) -> None:
    settings = Settings.from_env()
    if backend is not None:
        settings.backend = Backend(backend.lower())
    if fps is not None:
        settings.sample_fps = fps
    if frames_per_batch is not None:
        settings.frames_per_batch = frames_per_batch
    if min_importance is not None:
        settings.min_importance = min_importance
    if padding_before is not None:
        settings.padding_before = padding_before
    if padding_after is not None:
        settings.padding_after = padding_after
    if output_dir is not None:
        settings.output_dir = output_dir

    try:
        _run_pipeline(
            video_path=video,
            settings=settings,
            sample_fps=settings.sample_fps,
            frames_per_batch=settings.frames_per_batch,
            min_importance=settings.min_importance,
            padding_before=settings.padding_before,
            padding_after=settings.padding_after,
        )
    except FFmpegNotFoundError as exc:
        raise click.ClickException(str(exc))


if __name__ == "__main__":
    main()
