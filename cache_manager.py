"""
Gestione delle cache binarie in cached_frames/: elenco, cancellazione e limite di spazio.
"""

import json
import logging
import os
import shutil
from dataclasses import dataclass

from ascii_converter import CONVERTER_VERSION

logger = logging.getLogger(__name__)

DEFAULT_LIMIT = "2G"
METADATA_FILE = "metadata.json"

_SUFFIXES = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}


def parse_size(text: str) -> int:
    """Converte '512', '10K', '5m', '2G' in byte (base 1024). ValueError se non valido."""
    value = str(text).strip().upper()
    multiplier = 1
    if value and value[-1] in _SUFFIXES:
        multiplier = _SUFFIXES[value[-1]]
        value = value[:-1]
    if not value.isdigit():
        raise ValueError(f"Dimensione non valida: {text!r}")
    return int(value) * multiplier


@dataclass
class CacheEntry:
    path: str
    video_name: str
    width: int | None
    height: int | None
    fps: float | None
    palette: str | None
    truecolor: bool
    size_bytes: int
    last_used: float
    obsolete: bool


def _dir_size(path: str) -> int:
    total = 0
    for dirpath, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


def _read_entry(path: str) -> CacheEntry:
    name = os.path.basename(path)
    metadata_path = os.path.join(path, METADATA_FILE)
    metadata = None
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            metadata = loaded
    except (OSError, ValueError):
        pass

    try:
        last_used = os.path.getmtime(metadata_path if metadata is not None else path)
    except OSError:
        last_used = 0.0
    size = _dir_size(path)

    if metadata is None:
        return CacheEntry(path, name, None, None, None, None, False, size, last_used, True)

    video_path = metadata.get("video_path")
    version = metadata.get("converter_version")
    return CacheEntry(
        path=path,
        video_name=os.path.basename(video_path) if video_path else name,
        width=metadata.get("width"),
        height=metadata.get("height"),
        fps=metadata.get("effective_fps"),
        palette=metadata.get("palette"),
        truecolor=bool(metadata.get("truecolor", False)),
        size_bytes=size,
        last_used=last_used,
        obsolete=version is not None and version != CONVERTER_VERSION,
    )


def list_caches(root: str) -> list[CacheEntry]:
    """Elenca le cache in root, dalla più recente alla meno recente; [] se root non esiste."""
    if not os.path.isdir(root):
        return []
    entries = [_read_entry(os.path.join(root, n)) for n in os.listdir(root)
               if os.path.isdir(os.path.join(root, n))]
    entries.sort(key=lambda e: e.last_used, reverse=True)
    return entries


def delete_cache(path: str) -> None:
    """Cancella una directory di cache."""
    shutil.rmtree(path)


def enforce_limit(root: str, limit_bytes: int, keep: set[str]) -> list[str]:
    """
    Cancella le cache meno recenti finché il totale è <= limit_bytes.

    Le cache in keep non vengono mai cancellate, nemmeno se da sole superano il limite.
    limit_bytes == 0 disabilita il limite. Ritorna i percorsi cancellati.
    """
    if limit_bytes <= 0:
        return []
    keep_abs = {os.path.abspath(k) for k in keep}
    entries = list_caches(root)
    total = sum(e.size_bytes for e in entries)
    removed = []
    for entry in reversed(entries):  # dalla meno recente
        if total <= limit_bytes:
            break
        if os.path.abspath(entry.path) in keep_abs:
            continue
        try:
            delete_cache(entry.path)
        except OSError as e:
            logger.warning(f"Impossibile cancellare la cache {entry.path}: {e}")
            continue
        total -= entry.size_bytes
        removed.append(entry.path)
    return removed
