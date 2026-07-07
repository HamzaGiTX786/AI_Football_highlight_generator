# AI Football Highlight Generator

Turn a full football (soccer) match video into a concatenated highlight reel automatically.

The pipeline samples frames with ffmpeg, asks a multi-modal LLM (Claude in the cloud, or Qwen2.5-VL / LLaVA through Ollama locally) to identify notable events, curates them with importance-based filtering and temporal deduplication, then cuts and concatenates the result into one MP4.

## Why This Is Interesting

- Multi-modal AI: real vision-language reasoning, not keyword matching.
- Production-shaped: modular pipeline, typed models, unit tests, and a CLI with progress output.
- Resumable: works on a single match video with no pre-segmentation or training data.
- Cheap to run: free with Ollama locally, or roughly a few dollars per full match with a cloud vision model.

## Architecture

```text
football-match.mp4
        |
        v
+------------------+
| 1. Frame Sampler |  ffmpeg at configurable fps -> PNG frames with timestamps
+------------------+
        |
        v
+------------------+
| 2. Event Detector|  Vision LLM (Claude or Ollama) -> JSON events
+------------------+
        |
        v
+------------------+
| 3. Curation      |  filter -> dedupe -> expand -> merge -> sort
+------------------+
        |
        v
+------------------+
| 4. Renderer      |  ffmpeg cut clips -> concat -> MP4
+------------------+
        |
        v
highlights.mp4 + events.json
```

Each stage lives in its own module:

| File | Purpose |
|---|---|
| `football_highlights/sampler.py` | ffmpeg-based frame extraction and batching |
| `football_highlights/detector.py` | Vision LLM backends for Claude and Ollama |
| `football_highlights/prompts.py` | System and user prompts for event detection |
| `football_highlights/curation.py` | Importance filtering, dedupe, clip expansion, and merging |
| `football_highlights/renderer.py` | ffmpeg clip cutting and concatenation |
| `football_highlights/models.py` | Pydantic data models |
| `football_highlights/config.py` | Settings loaded from `.env` |
| `football_highlights/cli.py` | Click CLI entry point |

## Setup

### 1. Install ffmpeg

The pipeline uses `ffmpeg` and `ffprobe` directly.

- Windows: `winget install Gyan.FFmpeg`
- macOS: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg` on Debian/Ubuntu, or your distro equivalent

Verify with:

```bash
ffmpeg -version
ffprobe -version
```

### 2. Install Python dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure a backend

Copy `.env.example` to `.env`, then choose one backend.

Claude API:

```env
BACKEND=claude
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-sonnet-4-6
```

Ollama:

```bash
ollama pull qwen2.5vl:7b
```

```env
BACKEND=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5vl:7b
```

## Usage

```bash
python -m football_highlights path/to/match.mp4
```

Or, after installing the package:

```bash
football-highlights path/to/match.mp4
```

Useful options:

| Flag | Default | Description |
|---|---:|---|
| `--backend` | `.env` | `claude` or `ollama` |
| `--fps` | `0.5` | Frames per second to sample |
| `--frames-per-batch` | `16` | Frames per LLM request |
| `--min-importance` | `3` | Drop events below this score |
| `--padding-before` | `4` | Seconds before each event |
| `--padding-after` | `3` | Seconds after each event |
| `--output` | `./output` | Destination for MP4 and JSON report |

Examples:

```bash
# Cloud backend
python -m football_highlights ./matches/arsenal_chelsea.mp4 --backend claude

# Local backend
python -m football_highlights ./matches/arsenal_chelsea.mp4 --backend ollama

# Include more borderline moments
python -m football_highlights ./matches/arsenal_chelsea.mp4 --min-importance 2

# Sample more densely for better recall
python -m football_highlights ./matches/arsenal_chelsea.mp4 --fps 1
```

Outputs:

- `output/highlights_YYYYMMDD_HHMMSS.mp4` - final highlight reel
- `output/events.json` - curated clips plus raw detected events

## Development

Run the test suite:

```bash
pytest
```

Most tests are pure logic and do not require API keys or ffmpeg. Full end-to-end testing needs a short sample video plus ffmpeg installed; see `examples/README.md`.

## Limitations And Roadmap

v1 limitations:

- Audio is ignored, so crowd noise and commentary spikes are not used yet.
- Player names are only used when visibly readable; the prompt discourages guessing.
- Low-quality footage and unusual camera angles can reduce detector accuracy.
- There is no team-specific filtering yet.

Planned v2 ideas:

- Audio analysis for crowd reactions.
- Text overlays for event type and timestamp.
- Streamlit web UI for drag-and-drop demos.
- Optional team/player filtering.

## License

MIT - see `LICENSE`.
