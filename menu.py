"""
Menu principale: impostazioni persistenti (settings.json), logica pura del menu (MenuState)
e disegno del testo dello schermo (render_menu).

La logica riceve nomi di tasti (come quelli di KeyReader) e produce lo stato e un'eventuale
azione ("play" o "quit"); il chiamante si occupa di leggere i tasti e di posizionare il cursore.
"""

import json
import os
from dataclasses import asdict, dataclass, fields
from datetime import datetime

from cache_manager import DEFAULT_LIMIT, delete_cache, list_caches, parse_size

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
PALETTE_CHOICES = ["basic", "standard", "extended", "box", "half", "custom"]
SETTINGS_FILE = "settings.json"

MAIN_ITEMS = ["Riproduci", "Video", "Impostazioni", "Cache", "Esci"]
ENTER_PATH = "Inserisci percorso…"
NO_VIDEOS = "Nessun video in videos/"

# (campo di Settings, etichetta) nell'ordine della schermata Impostazioni
SETTINGS_ITEMS = [
    ("width", "Larghezza"),
    ("fps", "FPS"),
    ("palette", "Palette"),
    ("custom_palette", "File palette custom"),
    ("truecolor", "Truecolor"),
    ("high_performance", "High-performance"),
    ("audio", "Audio"),
    ("loop", "Loop"),
    ("log_fps", "Statistiche FPS"),
    ("cache_limit", "Limite cache"),
]
EDIT_FIELDS = {"width", "fps", "custom_palette", "cache_limit"}
BOOL_FIELDS = {"truecolor", "high_performance", "audio", "loop", "log_fps"}
EDIT_LABELS = dict(SETTINGS_ITEMS, video_path="Percorso del video")

HINTS = {
    "main": "↑/↓ muovi  Invio conferma  Esc esci",
    "video": "↑/↓ muovi  Invio conferma  Esc indietro",
    "settings": "↑/↓ muovi  ←/→ cambia  Invio modifica  Esc salva e torna",
    "cache": "↑/↓ muovi  Invio cancella  o cancella obsolete  Esc indietro",
    "confirm": "s conferma  n/Esc annulla",
    "edit": "Digita il valore  Backspace cancella  Invio conferma  Esc annulla",
}


@dataclass
class Settings:
    video_path: str | None = None
    width: int | None = None
    fps: int = 10
    palette: str = "standard"
    custom_palette: str | None = None
    truecolor: bool = False
    high_performance: bool = False
    audio: bool = False
    loop: bool = True
    log_fps: bool = False
    cache_limit: str = DEFAULT_LIMIT

    def to_argv(self) -> list[str]:
        """Argomenti per main.parse_arguments (senza 'main.py')."""
        if self.video_path is None:
            raise ValueError("Nessun video selezionato")
        argv = [self.video_path]
        if self.width is not None:
            argv.append(str(self.width))
        argv += ["--fps", str(self.fps), "--palette", self.palette]
        if self.palette == "custom" and self.custom_palette:
            argv += ["--custom-palette", self.custom_palette]
        if self.truecolor:
            argv.append("--truecolor")
        if self.high_performance:
            argv.append("--high-performance")
        if self.audio:
            argv.append("--audio")
        if not self.loop:
            argv.append("--no-loop")
        if self.log_fps:
            argv.append("--log_fps")
        argv += ["--cache-limit", self.cache_limit]
        return argv


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_settings(s: Settings) -> bool:
    if not isinstance(s.cache_limit, str):
        return False
    try:
        parse_size(s.cache_limit)
    except (ValueError, TypeError):
        return False
    return (
        (s.video_path is None or isinstance(s.video_path, str))
        and (s.width is None or (_is_int(s.width) and s.width > 0))
        and _is_int(s.fps) and s.fps > 0
        and s.palette in PALETTE_CHOICES
        and (s.custom_palette is None or isinstance(s.custom_palette, str))
        and all(isinstance(getattr(s, name), bool) for name in BOOL_FIELDS)
    )


def load_settings(path: str = SETTINGS_FILE) -> tuple[Settings, str | None]:
    """Carica le impostazioni; file mancante → default, file non valido → default e messaggio."""
    if not os.path.exists(path):
        return Settings(), None
    invalid = (Settings(), f"Impostazioni in '{path}' non valide: uso i valori predefiniti")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return invalid
    if not isinstance(data, dict):
        return invalid
    names = {f.name for f in fields(Settings)}
    settings = Settings(**{k: v for k, v in data.items() if k in names})
    if not _valid_settings(settings):
        return invalid
    return settings, None


def save_settings(settings: Settings, path: str = SETTINGS_FILE) -> None:
    """Salva le impostazioni in JSON (OSError in caso di errore)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(settings), f, indent=2, ensure_ascii=False)


def list_videos(folder: str = "videos") -> list[str]:
    """Percorsi dei file video in folder, ordinati per nome; [] se la cartella non esiste."""
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, name) for name in sorted(os.listdir(folder))
            if name.lower().endswith(VIDEO_EXTENSIONS) and os.path.isfile(os.path.join(folder, name))]


class MenuState:
    """Stato del menu guidato dai nomi dei tasti."""

    def __init__(self, settings: Settings, videos: list[str], cache_root: str = "cached_frames",
                 settings_path: str = SETTINGS_FILE):
        self.settings = settings
        self.videos = videos
        self.cache_root = cache_root
        self.settings_path = settings_path
        self.screen = "main"
        self.selected = 0
        self.message: str | None = None
        self.caches = []
        # stato della schermata edit
        self.edit_field: str | None = None
        self.edit_buffer = ""
        self._edit_return: tuple[str, int] = ("main", 0)
        # stato della schermata confirm
        self.confirm_text = ""
        self.confirm_paths: list[str] = []

    # --- navigazione ---

    def item_count(self) -> int:
        if self.screen == "main":
            return len(MAIN_ITEMS)
        if self.screen == "video":
            return len(self.videos) + 1
        if self.screen == "settings":
            return len(SETTINGS_ITEMS)
        if self.screen == "cache":
            return len(self.caches)
        return 0

    def _go(self, screen: str, selected: int = 0) -> None:
        self.screen = screen
        self.selected = selected

    def handle_key(self, key: str) -> str | None:
        """Applica un tasto e restituisce l'azione risultante ("play", "quit" o None)."""
        self.message = None
        if self.screen in ("main", "video", "settings", "cache") and key in ("up", "down"):
            count = self.item_count()
            if count:
                step = -1 if key == "up" else 1
                self.selected = max(0, min(count - 1, self.selected + step))
            return None
        handler = getattr(self, f"_key_{self.screen}")
        return handler(key)

    def _key_main(self, key: str) -> str | None:
        if key == "esc":
            return "quit"
        if key != "enter":
            return None
        item = MAIN_ITEMS[self.selected]
        if item == "Riproduci":
            path = self.settings.video_path
            if not path:
                self.message = "Seleziona prima un video"
            elif not os.path.isfile(path):
                self.message = f"Il file video '{path}' non esiste"
            elif self.settings.palette == "custom" and not self.settings.custom_palette:
                self.message = "Seleziona un file per la palette custom"
            else:
                return "play"
        elif item == "Video":
            current = self.settings.video_path
            self._go("video", self.videos.index(current) if current in self.videos else 0)
        elif item == "Impostazioni":
            self._go("settings")
        elif item == "Cache":
            self._refresh_caches()
            self._go("cache")
        elif item == "Esci":
            return "quit"
        return None

    def _key_video(self, key: str) -> None:
        if key == "esc":
            self._go("main", MAIN_ITEMS.index("Video"))
        elif key == "enter":
            if self.selected < len(self.videos):
                self.settings.video_path = self.videos[self.selected]
                self._go("main", MAIN_ITEMS.index("Video"))
            else:
                self._start_edit("video_path", "", ("main", MAIN_ITEMS.index("Video")))

    def _key_settings(self, key: str) -> None:
        name = SETTINGS_ITEMS[self.selected][0]
        if key == "esc":
            try:
                save_settings(self.settings, self.settings_path)
            except OSError as e:
                self.message = f"Impossibile salvare le impostazioni: {e}"
            self._go("main", MAIN_ITEMS.index("Impostazioni"))
        elif key in ("left", "right"):
            step = -1 if key == "left" else 1
            if name in BOOL_FIELDS:
                setattr(self.settings, name, not getattr(self.settings, name))
            elif name == "palette":
                index = PALETTE_CHOICES.index(self.settings.palette)
                self.settings.palette = PALETTE_CHOICES[(index + step) % len(PALETTE_CHOICES)]
            elif name == "fps":
                self.settings.fps = max(1, self.settings.fps + step)
        elif key == "enter":
            if name in BOOL_FIELDS:
                setattr(self.settings, name, not getattr(self.settings, name))
            elif name == "palette":
                self._key_settings("right")
            elif name in EDIT_FIELDS:
                value = getattr(self.settings, name)
                self._start_edit(name, "" if value is None else str(value), ("settings", self.selected))

    def _key_cache(self, key: str) -> None:
        if key == "esc":
            self._go("main", MAIN_ITEMS.index("Cache"))
        elif key == "enter" and self.caches:
            entry = self.caches[self.selected]
            self._confirm(f"Cancellare {entry.video_name}? (s/n)", [entry.path])
        elif key == "o":
            obsolete = [e.path for e in self.caches if e.obsolete]
            if obsolete:
                self._confirm(f"Cancellare {len(obsolete)} cache obsolete? (s/n)", obsolete)
            else:
                self.message = "Nessuna cache obsoleta"

    def _key_confirm(self, key: str) -> None:
        if key in ("n", "esc"):
            self.screen = "cache"
        elif key == "s":
            errors = []
            for path in self.confirm_paths:
                try:
                    delete_cache(path)
                except OSError as e:
                    errors.append(f"Impossibile cancellare {os.path.basename(path)}: {e}")
            if errors:
                self.message = "; ".join(errors)
            self._refresh_caches()
            self.screen = "cache"
            self.selected = max(0, min(self.selected, len(self.caches) - 1))

    def _key_edit(self, key: str) -> None:
        if key == "esc":
            self._go(*self._edit_return)
        elif key == "backspace":
            self.edit_buffer = self.edit_buffer[:-1]
        elif key == "space":
            self.edit_buffer += " "
        elif key == "enter":
            self._commit_edit()
        elif len(key) == 1:
            self.edit_buffer += key

    # --- supporto ---

    def _start_edit(self, field: str, initial: str, return_to: tuple[str, int]) -> None:
        self.edit_field = field
        self.edit_buffer = initial
        self._edit_return = return_to
        self.screen = "edit"

    def _commit_edit(self) -> None:
        field = self.edit_field
        text = self.edit_buffer.strip()
        if field in ("width", "fps"):
            if field == "width" and not text:
                value = None
            elif text.isascii() and text.isdigit() and int(text) > 0:
                value = int(text)
            else:
                self.message = "Inserisci un numero intero maggiore di 0"
                return
        elif field == "cache_limit":
            try:
                parse_size(text)
            except ValueError:
                self.message = "Limite non valido: usa byte o suffissi K/M/G (es. 2G), 0 = nessun limite"
                return
            value = text
        elif field == "video_path":
            if not text:
                self.message = "Inserisci un percorso"
                return
            value = text
        else:  # custom_palette
            value = text or None
        setattr(self.settings, field, value)
        self._go(*self._edit_return)

    def _confirm(self, text: str, paths: list[str]) -> None:
        self.confirm_text = text
        self.confirm_paths = paths
        self.screen = "confirm"

    def refresh(self) -> None:
        """Rilegge l'elenco dei video e delle cache."""
        self.videos = list_videos()
        self._refresh_caches()

    def _refresh_caches(self) -> None:
        self.caches = list_caches(self.cache_root)


# --- disegno ---

def _format_setting(settings: Settings, name: str) -> str:
    value = getattr(settings, name)
    if name in BOOL_FIELDS:
        return "sì" if value else "no"
    if name == "width":
        return "auto" if value is None else str(value)
    if name == "custom_palette":
        return value or "(nessuno)"
    if name == "cache_limit":
        try:
            return "nessun limite" if parse_size(value) == 0 else str(value)
        except ValueError:
            return str(value)
    return str(value)


def _format_palette(palette: str | None) -> str:
    if palette is None:
        return "?"
    if palette in ("box", "half"):
        return palette
    return f"{len(palette)} caratteri"


def _format_cache(entry) -> str:
    date = datetime.fromtimestamp(entry.last_used).strftime("%Y-%m-%d %H:%M")
    line = f"{entry.video_name}  {entry.size_bytes / 1024 ** 2:.1f} MB  {_format_palette(entry.palette)}"
    if entry.truecolor:
        line += " truecolor"
    line += f"  {date}"
    if entry.obsolete:
        line += "  [obsoleta]"
    return line


def _window(items: list[str], selected: int | None, available: int) -> list[str]:
    """Righe della lista con il marcatore sulla selezionata, scorrendo per mantenerla visibile."""
    lines = [("> " if i == selected else "  ") + item for i, item in enumerate(items)]
    available = max(1, available)
    if len(lines) <= available:
        return lines
    start = 0
    if selected is not None:
        start = max(0, min(selected - available // 2, len(lines) - available))
    return lines[start:start + available]


def render_menu(state: MenuState, cols: int, rows: int) -> str:
    """Testo completo dello schermo del menu, entro cols colonne e rows righe."""
    header = ["Pixellator", ""]
    footer = []
    items: list[str] = []
    selected: int | None = state.selected
    after: list[str] = []

    if state.screen == "main":
        items = MAIN_ITEMS
        after = ["", f"Video: {state.settings.video_path or '(nessuno)'}"]
    elif state.screen == "video":
        header.append("Video")
        items = [os.path.basename(v) for v in state.videos] + [ENTER_PATH]
        if not state.videos:
            header.append(NO_VIDEOS)
    elif state.screen == "settings":
        header.append("Impostazioni")
        items = [f"{label}: {_format_setting(state.settings, name)}" for name, label in SETTINGS_ITEMS]
    elif state.screen in ("cache", "confirm"):
        header.append("Cache")
        if state.caches:
            items = [_format_cache(e) for e in state.caches]
        else:
            header.append("Nessuna cache")
            selected = None
        total = sum(e.size_bytes for e in state.caches) / 1024 ** 2
        after = ["", f"Totale: {total:.1f} MB",
                 f"Limite: {_format_setting(state.settings, 'cache_limit')}"]
        if state.screen == "confirm":
            after += ["", state.confirm_text]
    elif state.screen == "edit":
        header.append(f"Modifica: {EDIT_LABELS[state.edit_field]}")
        items = [state.edit_buffer + "_"]
        selected = 0

    footer.append("")
    if state.message:
        footer.append(state.message)
    footer.append(HINTS[state.screen])

    available = rows - len(header) - len(after) - len(footer)
    lines = header + _window(items, selected, available) + after + footer
    if len(lines) > rows:
        lines = lines[:max(0, rows - 1)] + [lines[-1]]  # la riga dei tasti resta visibile
    return "\n".join(line[:cols] for line in lines[:rows])
