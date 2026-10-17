# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Pixellator is a terminal ASCII-art video player (Python, OpenCV + NumPy, optional audio via `sounddevice` + `moviepy`, `tqdm` for high-performance mode). Comments, docstrings, log messages and user-facing strings are written in **Italian** — keep that convention. `readme.md` (English) and `readme_it.md` (Italian) are user docs; the README does not yet document `--high-performance`.

## Running

There is no build step, test suite, or linter config. Set up the venv (Python 3.12, pinned deps in `requirements.txt`):

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

moviepy is 2.x: import from `moviepy` directly (`from moviepy import AudioFileClip`), not the removed `moviepy.editor`.

```bash
python main.py <video_path> <width> [--fps N] [--batch_size N] [--palette basic|standard|extended|box|custom] \
    [--custom-palette FILE] [--audio] [--no-loop] [--high-performance] [--log_fps] [--log_performance] [--verbose]
```

`run.bat` (gitignored, local) is the typical dev invocation:
`python main.py videos/5.mp4 300 --fps 60 --batch_size 100 --palette extended --high-performance --audio --log_fps`

Sample videos go in `videos/` (untracked). Logs are written to `logs/` (`ascii_video_<timestamp>.log` plus per-process `Reader.log`/`Converter.log`/`Renderer.log`); the console only shows WARNING+ unless `--verbose`.

Verification is manual: run the player on a short clip in `videos/` and check `logs/`. When touching conversion code, test both modes (with and without `--high-performance`) and both a character palette and `--palette box`.

## Architecture

`main.py` parses args, checks optional deps (audio, tqdm), computes ASCII height from the first frame (`height = aspect * width / 2.25` — the 2.25 char-aspect correction is hard-coded in several files), shows a calibration frame (`calibration_frame.py`), resolves the palette string (`'box'` is a sentinel meaning colored Unicode blocks, not a character set), then dispatches to one of two independent playback modes:

1. **Standard mode — `pipeline.py` `VideoPipeline`**: real-time, multi-process.
   - `frame_reader_process` (separate process) reads/decimates frames to target FPS and pushes batches of `batch_size` onto `frame_queue`.
   - `frame_converter_process` (separate process) converts to ANSI 256-color ASCII strings using precomputed color lookup tables and vectorized NumPy; pushes to `ascii_queue`.
   - `_frame_renderer_thread` (thread in main process) pops frames, paces to target FPS, writes via `TerminalOutputBuffer`, draws the FPS/frame-time graph when `--log_fps`, and drives audio sync.
   - The reader does not decimate: it reads every source frame at `--fps` pace, so `--fps` below the source rate plays in slow motion (high-performance mode does decimate via `frame_step`).
   - Process functions are top-level and import their deps inside the function body (Windows `spawn` start method) — keep them picklable and self-contained.

2. **High-performance mode — `precomputer.py`**: `FramePrecomputer` converts every frame up front with a thread pool and writes `cached_frames/<video>_w<width>_fps<fps>/frame_NNNNNN.txt` plus `metadata.pkl`; `PrecomputedFramePlayer` then streams those files from disk with its own playback/stats thread. Falls back to standard mode if precomputation fails.
   - The cache key and `is_cache_valid()` do **not** include the palette — after changing the palette (or conversion logic) for the same video/width/fps, delete the matching `cached_frames/` subdir.

**Duplicated conversion logic**: ASCII/box conversion is implemented separately in `pipeline.py` (`frame_converter_process`) and `precomputer.py` (`_convert_frame_to_ascii`, `_convert_frame_to_box_ascii`). Changes to rendering output usually need to be made in both. The same applies to the FPS stats/graph code in the renderer thread vs. `PrecomputedFramePlayer._playback_thread_func`.

**Audio** (`audio_player.py`): `sounddevice` stream over moviepy-decoded samples. Audio follows video: renderers call `update_video_time(t)` and the stream callback jumps to `t` when drift exceeds `sync_tolerance` (0.1s).

`renderer.py` (`AsciiRenderer`) is currently unused; rendering goes through `TerminalOutputBuffer` directly.
