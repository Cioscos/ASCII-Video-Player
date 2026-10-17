# ASCII Video Player

![ASCII Video Preview](https://github.com/Cioscos/ASCII-Video-Player/blob/master/doc/intro.png)

## 📝 Description

**ASCII Video Player** is a high-performance real-time video to ASCII art converter that displays videos in the terminal. It uses an optimized parallel pipeline to convert and display frames, supporting synchronized audio playback and various character palettes.

## ✨ Features

- **ASCII art video playback** in the terminal with color support
- **Parallel processing pipeline** for optimal performance
- **Audio-video synchronization**: audio follows the video clock and resyncs to the current video time when it drifts
- **Multiple ASCII character palettes** (basic, standard, extended, box, or custom)
- **Automatic calibration system** to adapt to terminal dimensions
- **Real-time performance monitoring** with graphical FPS display
- **Deadline-based pacing**: each frame has an absolute deadline; frames are skipped only when drawing falls behind, so playback keeps its timing even on less powerful hardware

## 🔧 Requirements

- Python 3.12 (see `requirements.txt` for pinned versions)
- Terminal with ANSI color support (most modern terminals)
- Python dependencies:
  - OpenCV (`opencv-python`)
  - NumPy
  - tqdm (progress bar of `--high-performance`)
  - For audio (optional):
    - SoundDevice
    - MoviePy

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/ascii-video-player.git
   cd ascii-video-player
   ```

2. Install dependencies:
   ```bash
   # Basic installation
   pip install opencv-python numpy tqdm

   # With audio support
   pip install opencv-python numpy tqdm sounddevice moviepy

   # Or, with pinned versions
   pip install -r requirements.txt
   ```

## 🚀 Usage

The basic syntax is:

```bash
python main.py <video_path> <width> [options]
```

Where:
- `<video_path>`: path to the video file to play
- `<width>`: width of the ASCII output (in characters)

### Examples

Simple playback:
```bash
python main.py my_video.mp4 100
```

Playback with audio and specific frame rate:
```bash
python main.py my_video.mp4 120 --fps 24 --audio
```

Playback with extended palette and performance logging:
```bash
python main.py my_video.mp4 80 --palette extended --log_fps --log_performance
```

### Available Options

| Option | Description |
|---------|-------------|
| `--fps N` | Set target frame rate to N FPS (default: 10). It is capped at the source video's FPS: playback speed is always correct |
| `--audio` | Enable synchronized audio playback |
| `--no-loop` | Disable automatic video looping |
| `--log_fps` | Enable FPS statistics display |
| `--log_performance` | Record detailed performance data |
| `--verbose` | Show detailed log messages in the terminal |
| `--batch_size N` | Batch size for frame processing in standard mode (default: 1) |
| `--high-performance` | Pre-convert the whole video into a binary on-disk cache and play it from there (see below) |
| `--palette TYPE` | Character palette choice: "basic", "standard" (default), "extended", "box" or "custom" |
| `--custom-palette FILE` | File with custom characters when palette="custom" |

## 🏗️ Architecture

Both modes share the same conversion (`ascii_converter.py`), terminal output (`terminal_display.py`) and playback loop (`playback.py`), which always runs in the main process.

**Standard mode** (default), a multi-process pipeline:

1. **Frame Reader Process**: decodes the video, drops frames to reach the effective FPS and resizes them
2. **ASCII Conversion Process**: transforms frames into colored ASCII byte strings
3. **Playback (main process)**: paces frames on absolute deadlines, draws only the rows that changed and keeps audio in sync

**High-performance mode** (`--high-performance`): all frames are converted up front by a process pool into a binary cache, then played back by memory-mapping the cache file.

### High-performance mode

```bash
python main.py my_video.mp4 300 --fps 60 --palette extended --high-performance
```

The cache is stored in `cached_frames/<video>_<hash>/` (`frames.bin`, `index.npy`, `metadata.json`). The hash covers the video path, size and modification time, width, height, effective FPS, palette and the converter version, so changing any of them creates a new cache. If the pre-computation fails, the player falls back to standard mode. Ctrl+C during pre-computation leaves no usable partial cache.

**Caches created by previous versions (`frame_*.txt` + `metadata.pkl`) are ignored: delete the old `cached_frames/` subdirectories by hand.**

### Code Structure

- `main.py`: Application entry point, CLI argument handling, palettes
- `ascii_converter.py`: Vectorized frame to colored ASCII conversion (color run-length, bytes output)
- `terminal_display.py`: Terminal output with per-row delta updates
- `playback.py`: `FrameDecimator`, `PlaybackClock` and the `run_playback` loop
- `fps_stats.py`: FPS statistics and graph
- `pipeline.py`: Standard mode reader/converter processes
- `precomputer.py`: High-performance mode binary cache and pre-computation
- `audio_player.py`: Synchronized audio playback management
- `calibration_frame.py`: Terminal calibration frame generation
- `utils.py`: Utility functions and logging system
- `bench.py`: Converter benchmark

## 🔍 Technical Notes

### Performance

- Frame conversion is vectorized with NumPy and merges runs of equal colors to minimize the bytes written
- Only the terminal rows that changed since the previous frame are rewritten
- Standard mode uses lossless blocking queues between processes, so no frame is lost

### FPS Control

- The effective FPS is `min(--fps, source FPS)`, so the playback speed is always correct
- Frames are paced on absolute deadlines; frames are skipped only when drawing is late
- Audio loops together with the video and re-syncs to the video time

### Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## 📊 Performance Visualization

When the `--log_fps` option is enabled, a real-time graph is displayed showing:
- Current FPS
- Frame rate stability
- Graph of recent frame times with outlier detection

![Performance Graph](https://github.com/Cioscos/ASCII-Video-Player/blob/master/doc/stats.png)

## 🎨 Available Character Palettes

- **Basic**: 10 characters (`" .:-=+*#%@"`)
- **Standard**: 42 characters, more detailed (default)
- **Extended**: 67 characters, maximum detail
- **Box**: Uses colored Unicode blocks for maximum visual density
- **Custom**: User-defined through a text file

## 📄 License

This project is distributed under the MIT License. See the `LICENSE` file for more details.

## 👥 Contributions

Contributions, bug reports, and feature requests are welcome.
To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

---

*Created with ❤️ for ASCII art and optimization enthusiasts*
