# ASCII Video Player

![ASCII Video Preview](https://github.com/Cioscos/ASCII-Video-Player/blob/master/doc/intro.png)

## 📝 Description

**ASCII Video Player** is a high-performance real-time video to ASCII art converter that displays videos in the terminal. It uses an optimized parallel pipeline to convert and display frames, supporting synchronized audio playback and various character palettes.

## ✨ Features

- **ASCII art video playback** in the terminal with color support
- **Parallel processing pipeline** for optimal performance
- **Audio-video synchronization** with adaptive delay management
- **Multiple ASCII character palettes** (basic, standard, extended, box, or custom)
- **Automatic calibration system** to adapt to terminal dimensions
- **Real-time performance monitoring** with graphical FPS display
- **Adaptive load management** to maintain synchronization even on less powerful hardware

## 🔧 Requirements

- Python 3.6+
- Terminal with ANSI color support (most modern terminals)
- Python dependencies:
  - OpenCV (`opencv-python`)
  - NumPy
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
   pip install opencv-python numpy

   # With audio support
   pip install opencv-python numpy sounddevice moviepy
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
| `--fps N` | Set target frame rate to N FPS (default: 10) |
| `--audio` | Enable synchronized audio playback |
| `--no-loop` | Disable automatic video looping |
| `--log_fps` | Enable FPS statistics display |
| `--log_performance` | Record detailed performance data |
| `--verbose` | Show detailed log messages in the terminal |
| `--batch_size N` | Batch size for frame processing (default: 1) |
| `--palette TYPE` | Character palette choice: "basic", "standard", "extended", "box" or "custom" |
| `--custom-palette FILE` | File with custom characters when palette="custom" |

## 🏗️ Architecture

The system uses a multi-process parallel pipeline with:

1. **Frame Reader Process**: extracts frames from the video at a controlled rate
2. **ASCII Conversion Process**: transforms frames into colored ASCII representations
3. **Rendering Thread**: displays ASCII frames on the terminal and manages synchronization

### Code Structure

- `main.py`: Application entry point, CLI argument handling
- `pipeline.py`: Implementation of the parallel video processing pipeline
- `audio_player.py`: Synchronized audio playback management
- `calibration_frame.py`: Terminal calibration frame generation
- `renderer.py`: Optimized ASCII frame rendering
- `terminal_output_buffer.py`: Optimized terminal output buffer
- `utils.py`: Utility functions and logging system

## 🔍 Technical Notes

### Performance

- Frame conversion uses vectorized NumPy algorithms to maximize efficiency
- Optimizations include lookup table pre-calculation, memory allocation minimization, and vectorized operations
- The rendering system uses an optimized buffer to minimize terminal I/O operations
- Adaptive load management automatically detects and adapts to the host machine's processing capacity

### FPS Control

The player attempts to maintain the specified target frame rate using the following strategies:
- Controlled frame extraction rate
- Rendering timing management to maintain consistent FPS
- Real-time performance monitoring and dynamic adaptation

## 📊 Performance Visualization

When the `--log_fps` option is enabled, a real-time graph is displayed showing:
- Current FPS
- Frame rate stability
- Graph of recent frame times with outlier detection

![Performance Graph](https://github.com/Cioscos/ASCII-Video-Player/blob/master/doc/stats.png)

## 🎨 Available Character Palettes

- **Basic**: 10 characters (`" .:-=+*#%@"`)
- **Standard**: 42 characters, more detailed
- **Extended**: 70 characters, maximum detail
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
