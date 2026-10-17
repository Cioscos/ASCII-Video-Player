"""Benchmark del convertitore ASCII: decodifica+resize, conversione, byte per frame, precompute."""
import argparse
import tempfile
import time

import cv2

from ascii_converter import compute_ascii_height
from main import PALETTES as MAIN_PALETTES

PALETTES = {**MAIN_PALETTES, "box": "box", "half": "half"}


def _load_frames(video, width, max_frames):
    """Decodifica i primi N frame in memoria e misura decodifica + resize."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Impossibile aprire il video: {video}")
    frames = []
    elapsed = 0.0
    while len(frames) < max_frames:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        cv2.resize(frame, (width, compute_ascii_height(w, h, width)), interpolation=cv2.INTER_NEAREST)
        elapsed += time.perf_counter() - t0
        frames.append(frame)
    cap.release()
    return frames, elapsed


def _make_converter(width, palette, src_size, truecolor=False):
    """Restituisce la funzione di conversione di ascii_converter per il frame sorgente dato."""
    from ascii_converter import make_converter
    conv = make_converter(width, compute_ascii_height(src_size[1], src_size[0], width), palette, truecolor)
    return conv.convert_frame


def _bench_precompute(video, width, palette, max_frames, truecolor=False):
    """Precompute limitato a N frame (fps sorgente), con cache in una directory temporanea."""
    from precomputer import CachedFrames, precompute

    cap = cv2.VideoCapture(video)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    height = compute_ascii_height(src_w, src_h, width)

    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        cache_dir = precompute(video, width, height, source_fps, None, palette, cache_root=tmp,
                               max_frames=max_frames, truecolor=truecolor)
        elapsed = time.perf_counter() - t0
        if cache_dir is None:
            return 0.0
        cached = CachedFrames.open(cache_dir)
        n = len(cached)
        cached.close()
    return n / elapsed if elapsed > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Benchmark del convertitore ASCII")
    parser.add_argument("video")
    parser.add_argument("width", type=int)
    parser.add_argument("--palette", default="extended", choices=list(PALETTES))
    parser.add_argument("--truecolor", action="store_true", help="Colori a 24 bit")
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()

    palette = PALETTES[args.palette]
    frames, decode_s = _load_frames(args.video, args.width, args.frames)
    if not frames:
        raise SystemExit("Nessun frame decodificato")

    convert = _make_converter(args.width, palette, frames[0].shape[:2], args.truecolor)
    t0 = time.perf_counter()
    outputs = [convert(f) for f in frames]
    convert_s = time.perf_counter() - t0
    n = len(frames)
    avg_bytes = sum(len(o.encode("utf-8") if isinstance(o, str) else o) for o in outputs) / n

    pre_fps = _bench_precompute(args.video, args.width, palette, args.frames, args.truecolor)

    print(f"video={args.video} width={args.width} palette={args.palette} truecolor={args.truecolor} frames={n}")
    print(f"decode+resize ms/frame: {decode_s / n * 1000:.2f}")
    print(f"convert ms/frame: {convert_s / n * 1000:.2f}")
    print(f"bytes/frame: {avg_bytes:.0f}")
    print(f"precompute frame/s: {pre_fps:.1f}")


if __name__ == "__main__":
    main()
