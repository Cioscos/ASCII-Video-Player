# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Pixellator is a terminal ASCII-art video player (Python, OpenCV + NumPy, optional audio via `sounddevice` + `moviepy`, `tqdm` for the high-performance precompute progress bar). Comments, docstrings, log messages and user-facing strings are written in **Italian** — keep that convention. `readme.md` (English) and `readme_it.md` (Italian) are user docs and must be kept in sync.

## Running

Set up the venv (Python 3.12, pinned deps in `requirements.txt`; test deps in `requirements-dev.txt`):

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements-dev.txt
```

moviepy is 2.x: import from `moviepy` directly (`from moviepy import AudioFileClip`), not the removed `moviepy.editor`.

```bash
python main.py <video_path> <width> [--fps N] [--batch_size N] [--palette basic|standard|extended|box|custom] \
    [--custom-palette FILE] [--audio] [--no-loop] [--high-performance] [--log_fps] [--log_performance] [--verbose]
```

`run.bat` (gitignored, local) is the typical dev invocation:
`python main.py videos/5.mp4 300 --fps 60 --batch_size 100 --palette extended --high-performance --audio --log_fps`

Sample videos go in `videos/` (untracked). Logs are written to `logs/` (`ascii_video_<timestamp>.log` plus per-process `Reader.log`/`Converter.log`); the console only shows WARNING+ unless `--verbose`.

Tests: `.venv/Scripts/python.exe -m pytest -q` from the repo root. `bench.py <video> <width> [--palette P]` benchmarks decode+resize, conversion, bytes/frame and precompute throughput (baseline and final numbers in `docs/superpowers/bench-*.txt`). Beyond the tests, verification of playback is manual: run the player on a short clip in `videos/`, check `logs/`, and test both modes (with and without `--high-performance`) with a character palette and with `--palette box`.

## Architecture

`main.py` parses args, computes the ASCII height from the first frame (`compute_ascii_height`; the character-aspect correction `CHAR_ASPECT_CORRECTION = 2.25` lives only in `ascii_converter.py`), shows a calibration frame (`calibration_frame.py`), resolves the palette string (`'box'` is a sentinel meaning colored Unicode blocks, not a character set), then dispatches to one of two playback modes. Palettes `basic` (10 chars), `standard` (42, default) and `extended` (67) are in `main.PALETTES`.

**Shared modules** (used by both modes, so rendering changes are made once):
- `ascii_converter.py`: `AsciiConverter` — vectorized NumPy conversion with color run-length, outputs `bytes`; `CONVERTER_VERSION` is part of the cache key, bump it when output changes.
- `terminal_display.py`: `TerminalDisplay` — writes frames with per-row delta updates.
- `fps_stats.py`: `FpsStats` — FPS statistics/graph for `--log_fps`.
- `playback.py`: `FrameDecimator` (effective fps = `min(--fps, source fps)`, so playback speed is always correct; NaN/0 source fps keeps all frames), `PlaybackClock` (absolute deadlines) and `run_playback`, which always runs in the main thread: it skips frames only when drawing is late and rebases the clock when the source is slow. It drives audio sync and loops via `FrameSource.restart()`.

**Standard mode — `pipeline.py` `VideoPipeline`**: reader process (grab/retrieve + resize, decimated by `FrameDecimator`) → converter process (`AsciiConverter`) → main-thread `run_playback` via `pipeline.frame_source()`. Queues are lossless and blocking (`--batch_size` frames per batch). Process functions are top-level and import their deps inside the function body (Windows `spawn` start method) — keep them picklable and self-contained.

**High-performance mode — `precomputer.py`**: `precompute()` converts all frames with a `multiprocessing.Pool` (workers ignore SIGINT) into `cached_frames/<name>_<sha1[:12]>/` containing `frames.bin`, `index.npy` and `metadata.json` (written last, so an interrupted run is never reused). `CachedFrames` reads it via mmap and `CachedFrameSource` feeds `run_playback`. The cache key includes video path, size, mtime, width, height, effective fps, palette and `CONVERTER_VERSION`, so no manual invalidation is needed. Any precompute error falls back to standard mode. Old caches (`frame_*.txt` + `metadata.pkl`) are ignored and can be deleted by hand.

**Audio** (`audio_player.py`): `sounddevice` stream over moviepy-decoded samples. It loops with the video and follows video time: `run_playback` calls `update_video_time(t)` and the stream callback jumps to `t` when drift exceeds `sync_tolerance` (0.1s). A video without an audio track continues without audio.
