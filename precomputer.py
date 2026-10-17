"""
Cache binaria dei frame ASCII per la modalità high-performance.

Layout di una directory di cache:
- frames.bin: i frame convertiti (bytes UTF-8 con sequenze ANSI) concatenati;
- index.npy: offset int64 di inizio di ogni frame, più l'offset finale;
- metadata.json: scritto per ultimo, solo a pre-elaborazione completata.

Una directory senza metadata.json non è mai considerata valida.
"""
import hashlib
import json
import logging
import mmap
import os

import cv2
import numpy as np
from tqdm import tqdm

from ascii_converter import CONVERTER_VERSION, AsciiConverter
from playback import FrameDecimator

logger = logging.getLogger(__name__)

FRAMES_FILE = "frames.bin"
INDEX_FILE = "index.npy"
METADATA_FILE = "metadata.json"

# Convertitore del processo worker, creato da _init_worker
_worker_converter = None


def cache_dir_for(video_path, width, height, effective_fps, palette, cache_root="cached_frames") -> str:
    """
    Directory di cache per un video e i parametri di conversione.

    La chiave cambia se cambiano il file (percorso, dimensione, data di modifica),
    le dimensioni della griglia, gli fps effettivi, la palette o la versione del convertitore.

    Returns:
        str: "<cache_root>/<nome video>_<sha1[:12]>"
    """
    abs_path = os.path.abspath(video_path)
    st = os.stat(abs_path)
    key = "|".join(str(part) for part in (
        abs_path, st.st_size, st.st_mtime_ns, int(width), int(height),
        round(float(effective_fps), 3), palette, CONVERTER_VERSION,
    ))
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    name = os.path.splitext(os.path.basename(abs_path))[0]
    return os.path.join(cache_root, f"{name}_{digest}")


def _init_worker(width, height, palette):
    """Inizializzatore del pool: crea il convertitore del processo worker."""
    global _worker_converter
    from ascii_converter import AsciiConverter
    _worker_converter = AsciiConverter(width, height, palette)


def _convert_worker(resized):
    """Converte nel processo worker un frame già ridimensionato."""
    return _worker_converter.convert(resized)


def _resized_frames(cap, converter, decimator, max_frames):
    """Generatore dei frame decimati e ridimensionati (nel processo principale)."""
    source_index = 0
    kept = 0
    while max_frames is None or kept < max_frames:
        if not cap.grab():
            return
        keep = decimator.should_keep(source_index)
        source_index += 1
        if not keep:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        kept += 1
        yield converter.resize(frame)


def _remove_quietly(path):
    """Rimuove un file ignorando l'assenza o gli errori."""
    try:
        os.remove(path)
    except OSError:
        pass


def precompute(video_path, width, height, source_fps, target_fps, palette, cache_root="cached_frames",
               processes: int | None = None, max_frames: int | None = None) -> str | None:
    """
    Converte tutti i frame decimati del video e li salva nella cache binaria.

    Decodifica, decimazione e resize avvengono nel processo principale; la conversione
    usa un pool di processi (processes=0: nel processo corrente; None: min(cpu, 8)).
    In caso di qualunque eccezione (anche KeyboardInterrupt) i file parziali vengono
    rimossi e l'eccezione viene rilanciata.

    Returns:
        str | None: Directory della cache, oppure None se il video non è leggibile o vuoto
    """
    decimator = FrameDecimator(source_fps, target_fps)
    effective_fps = decimator.effective_fps
    cache_dir = cache_dir_for(video_path, width, height, effective_fps, palette, cache_root)
    if processes is None:
        processes = min(os.cpu_count() or 1, 8)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Impossibile aprire il video: {video_path}")
        cap.release()
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expected = None
    if total > 0:
        expected = sum(1 for i in range(total) if decimator.should_keep(i))
        if max_frames is not None:
            expected = min(expected, max_frames)

    os.makedirs(cache_dir, exist_ok=True)
    metadata_path = os.path.join(cache_dir, METADATA_FILE)
    frames_path = os.path.join(cache_dir, FRAMES_FILE)
    tmp_path = frames_path + ".tmp"
    index_path = os.path.join(cache_dir, INDEX_FILE)
    # Una cache precedente non deve restare valida durante la nuova scrittura
    _remove_quietly(metadata_path)

    logger.info(f"Pre-elaborazione: {video_path}, {width}x{height}, fps effettivi {effective_fps}, "
                f"processi {processes}, cache {cache_dir}")
    converter = AsciiConverter(width, height, palette)
    offsets = [0]
    try:
        frames_iter = _resized_frames(cap, converter, decimator, max_frames)
        with open(tmp_path, "wb") as out, tqdm(total=expected, desc="Pre-elaborazione", unit="frame") as bar:
            def write(data):
                out.write(data)
                offsets.append(offsets[-1] + len(data))
                bar.update(1)

            if processes > 0:
                from multiprocessing import Pool
                with Pool(processes, initializer=_init_worker, initargs=(width, height, palette)) as pool:
                    for data in pool.imap(_convert_worker, frames_iter, chunksize=8):
                        write(data)
            else:
                for resized in frames_iter:
                    write(converter.convert(resized))

        cap.release()
        if len(offsets) == 1:
            logger.error(f"Nessun frame letto dal video: {video_path}")
            _remove_quietly(tmp_path)
            return None

        np.save(index_path, np.array(offsets, dtype=np.int64))
        os.replace(tmp_path, frames_path)
        metadata = {
            "effective_fps": effective_fps,
            "frames": len(offsets) - 1,
            "width": width,
            "height": height,
            "palette": palette,
            "video_path": os.path.abspath(video_path),
            "converter_version": CONVERTER_VERSION,
        }
        with open(metadata_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)
        os.replace(metadata_path + ".tmp", metadata_path)
    except BaseException:
        cap.release()
        for path in (metadata_path + ".tmp", metadata_path, tmp_path, frames_path, index_path):
            _remove_quietly(path)
        raise

    logger.info(f"Pre-elaborazione completata: {len(offsets) - 1} frame in {cache_dir}")
    return cache_dir


class CachedFrames:
    """
    Accesso in sola lettura a una cache binaria tramite mmap.

    Attributes:
        metadata (dict): Contenuto di metadata.json
    """

    def __init__(self, cache_dir, metadata, offsets, file, mm):
        self.cache_dir = cache_dir
        self.metadata = metadata
        self._offsets = offsets
        self._file = file
        self._mm = mm

    @classmethod
    def open(cls, cache_dir) -> "CachedFrames | None":
        """Apre la cache; None se manca metadata.json o la cache è incompleta."""
        metadata_path = os.path.join(cache_dir, METADATA_FILE)
        if not os.path.isfile(metadata_path):
            return None
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            offsets = [int(o) for o in np.load(os.path.join(cache_dir, INDEX_FILE))]
            file = open(os.path.join(cache_dir, FRAMES_FILE), "rb")
        except (OSError, ValueError) as e:
            logger.warning(f"Cache non valida in {cache_dir}: {e}")
            return None
        try:
            if len(offsets) < 2 or len(offsets) - 1 != metadata.get("frames"):
                raise ValueError("indice e metadati non coerenti")
            mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            if len(mm) != offsets[-1]:
                mm.close()
                raise ValueError("dimensione di frames.bin non coerente con l'indice")
        except (OSError, ValueError) as e:
            file.close()
            logger.warning(f"Cache non valida in {cache_dir}: {e}")
            return None
        return cls(cache_dir, metadata, offsets, file, mm)

    def __len__(self):
        return len(self._offsets) - 1

    def frame(self, i) -> bytes:
        """Restituisce i bytes del frame i."""
        return self._mm[self._offsets[i]:self._offsets[i + 1]]

    def close(self):
        """Chiude mmap e file (necessario su Windows per poter cancellare la cache)."""
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._file is not None:
            self._file.close()
            self._file = None


class CachedFrameSource:
    """FrameSource che legge i frame in ordine da una CachedFrames."""

    def __init__(self, cached: CachedFrames):
        self.cached = cached
        self._next = 0

    def next_frame(self):
        """Restituisce (indice, frame) oppure None a fine cache."""
        if self._next >= len(self.cached):
            return None
        i = self._next
        self._next += 1
        return i, self.cached.frame(i)

    def restart(self):
        """Riparte dal primo frame."""
        self._next = 0
