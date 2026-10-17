# Pixellator – Bugfix e Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correggere i 19 bug catalogati (Fase 1) e poi ristrutturare Pixellator attorno a un nucleo condiviso ad alte prestazioni (Fase 2), mantenendo identico l'aspetto visivo.

**Architecture:** Fase 0 crea la rete di sicurezza (test di caratterizzazione e benchmark). Fase 1 corregge i bug nella struttura attuale, introducendo già `playback.py` (decimatore e clock) per non rifare il lavoro dopo. Fase 2 sostituisce la conversione duplicata con `AsciiConverter` vettorizzato (run-length del colore, output bytes), un `TerminalDisplay` con delta per riga, `FpsStats` condiviso, un loop `run_playback` comune alle due modalità e una cache binaria `frames.bin` + `index.npy` letta via `mmap`.

**Tech Stack:** Python 3.12, OpenCV 4.11, NumPy 2.1, tqdm, sounddevice, moviepy 2.x, pytest.

**Spec:** `docs/superpowers/specs/2026-10-17-bugfix-performance-design.md`

## Global Constraints

- Commenti, docstring, log e stringhe utente in **italiano**.
- Funzioni eseguite in processi figli: top-level, picklable, import delle dipendenze dentro il corpo (start method `spawn` su Windows).
- Entrambe le modalità (standard e `--high-performance`) restano supportate.
- Nessuna nuova dipendenza runtime. `numba` NON viene aggiunto da questo piano (vedi Task 16, step di benchmark).
- Aspetto visivo: stessa griglia (carattere, codice colore 256) per cella del converter attuale, per ogni palette tranne `standard` (cambiata da B10).
- Formule di riferimento da preservare: altezza `int(src_h / src_w * width / 2.25)` (minimo 1); colore `16 + 36*min(5, r//43) + 6*min(5, g//43) + min(5, b//43)`; carattere `palette[int(gray / 255.0 * (len(palette) - 1))]` con `gray = cv2.cvtColor(resized, COLOR_BGR2GRAY)`; resize `cv2.INTER_NEAREST`; ogni riga termina con `\033[0m`, righe separate da `\n`.
- Comando test: `.venv/Scripts/python.exe -m pytest -q` dalla root del repo.
- Messaggi di commit in inglese, stile del repo (`Fixed ...`, `Added ...`), chiusi con `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. Video senza traccia audio con `--audio` → si continua senza audio, nessuna eccezione (test in Task 5).
2. Video molto corto (1 frame) in loop → nessun busy-loop infinito se la sorgente restituisce subito `None` dopo `restart()` (test in Task 11).
3. `CAP_PROP_FPS` restituito come 0 o NaN da OpenCV → il decimatore tiene tutti i frame e usa `target_fps` come fps effettivi (test in Task 2).
4. Ctrl+C / eccezione durante il precompute → nessun `metadata.json`, nessun `frames.bin` parziale riusato al run successivo (test in Task 13).
5. Palette custom con caratteri multibyte UTF-8 (es. `░▒▓█`) → output correttamente codificato e griglia uguale al riferimento (test in Task 8).

---

## Fase 0 – Rete di sicurezza

### Task 1: Riferimento di caratterizzazione, fixture e benchmark baseline

**Files:**
- Create: `requirements-dev.txt`, `tests/__init__.py`, `tests/conftest.py`, `tests/reference.py`, `tests/test_reference.py`, `bench.py`, `docs/superpowers/bench-baseline.txt`

**Interfaces:**
- Produces:
  - `tests.reference.reference_grid(frame_bgr: np.ndarray, width: int, palette: str) -> list[list[tuple[str, int]]]` — resize + formule dei Global Constraints; per `palette == 'box'` il carattere è `'█'`.
  - `tests.reference.parse_ansi_grid(data: bytes | str) -> list[list[tuple[str, int]]]` — decodifica l'output: tiene il colore corrente da `\033[38;5;Nm`, ignora `\033[0m`, una riga per `\n`, ogni altro carattere è una cella `(char, colore_corrente)`.
  - fixture `random_frame` (`np.random.default_rng(1234)`, shape `(90, 160, 3)`, `uint8`).
  - fixture `tiny_video(tmp_path) -> str`: file `tiny.avi` codec `MJPG`, 24 fps, 48 frame 64×48, frame i con colore pieno `(i*5 % 256, 100, 200)`.
  - `bench.py`: `python bench.py <video> <width> [--palette P] [--frames N]` (default palette `extended`, N 300). Stampa: `decode+resize ms/frame`, `convert ms/frame`, `bytes/frame`, `precompute frame/s`.

- [ ] **Step 1:** `requirements-dev.txt` con `-r requirements.txt` e `pytest==8.3.4`; installare con `uv pip install --python .venv/Scripts/python.exe -r requirements-dev.txt`.
- [ ] **Step 2: Scrivere i test di caratterizzazione** in `tests/test_reference.py`:

```python
@pytest.mark.parametrize("palette", [" .:-=+*#%@", EXTENDED, "@", "box"])
def test_legacy_precomputer_matches_reference(random_frame, palette):
    pre = FramePrecomputer.__new__(FramePrecomputer)  # evita la creazione di directory
    pre.width, pre.ascii_palette = 120, palette
    out = pre._convert_frame_to_ascii(random_frame)
    assert parse_ansi_grid(out) == reference_grid(random_frame, 120, palette)

def test_tiny_video_fixture(tiny_video):
    cap = cv2.VideoCapture(tiny_video)
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 48
```
(`EXTENDED` = stringa `extended` di `main.py`, importata da lì.)

- [ ] **Step 3:** Implementare `tests/reference.py` e `tests/conftest.py`. Run: `.venv/Scripts/python.exe -m pytest -q tests/test_reference.py` → PASS (il riferimento descrive il comportamento attuale; se fallisce, è il riferimento a essere sbagliato, non il codice).
- [ ] **Step 4:** Implementare `bench.py`. Conversione: decodifica i primi N frame in memoria, poi cronometra solo la conversione (oggi `FramePrecomputer._convert_frame_to_ascii`; dalla Task 8 `AsciiConverter.convert_frame` — il bench usa `ascii_converter` se importabile, altrimenti il legacy). Precompute legacy: sottoclasse locale di `FramePrecomputer` che in `_extract_video_info` limita `actual_frames` a N, con `output_dir` in una `tempfile.TemporaryDirectory()`.
- [ ] **Step 5:** Run: `.venv/Scripts/python.exe bench.py videos/5.mp4 300 --palette extended` e `... --palette box`; salvare entrambe le uscite in `docs/superpowers/bench-baseline.txt`.
- [ ] **Step 6:** Commit (`Added characterization tests and benchmark baseline`).

---

## Fase 1 – Correzione bug

### Task 2: `FrameDecimator` e velocità corretta (B1, B2)

**Files:**
- Create: `playback.py`, `tests/test_decimator.py`
- Modify: `pipeline.py` (`frame_reader_process`, `VideoPipeline.__init__`), `precomputer.py` (`_extract_video_info`, loop di `precompute_frames`), `main.py` (passaggio di `fps_originale`)

**Interfaces:**
- Produces: `playback.FrameDecimator(source_fps: float, target_fps: float | None)` con attributo `effective_fps: float` e metodo `should_keep(source_index: int) -> bool`. `VideoPipeline(..., source_fps: float)` (nuovo parametro keyword). Metadati del precompute: `target_fps` = `effective_fps`.

- [ ] **Step 1: Test** in `tests/test_decimator.py`:

```python
def kept(src, tgt, n): d = FrameDecimator(src, tgt); return [i for i in range(n) if d.should_keep(i)]

def test_30_to_24_keeps_4_of_5_uniformly():
    k = kept(30, 24, 300); assert len(k) == 240 and max(b - a for a, b in zip(k, k[1:])) == 2
def test_target_above_source_is_capped():
    d = FrameDecimator(30, 60); assert d.effective_fps == 30 and len(kept(30, 60, 90)) == 90
def test_25_to_10():
    k = kept(25, 10, 250); assert len(k) == 100 and max(b - a for a, b in zip(k, k[1:])) == 3
def test_none_target_keeps_all():
    assert FrameDecimator(25, None).effective_fps == 25 and len(kept(25, None, 50)) == 50
@pytest.mark.parametrize("bad", [0, float("nan")])
def test_invalid_source_fps_keeps_all(bad):
    d = FrameDecimator(bad, 20); assert d.effective_fps == 20 and len(kept(bad, 20, 40)) == 40
```

- [ ] **Step 2:** Run `pytest -q tests/test_decimator.py` → FAIL (`ModuleNotFoundError: playback`).
- [ ] **Step 3: Implementare `FrameDecimator`.** `should_keep` è puro in `source_index` (non a stato), così vale anche dopo il riavvio del loop:

```python
ratio = effective_fps / source_fps        # <= 1
should_keep(i) = floor(i * ratio) > floor((i - 1) * ratio)   # i = 0 sempre tenuto
```
Se `source_fps` non è un numero finito > 0: `effective_fps = target_fps or 30.0` e si tengono tutti i frame.

- [ ] **Step 4:** Run `pytest -q tests/test_decimator.py` → PASS.
- [ ] **Step 5: Applicare.**
  - `frame_reader_process` riceve `source_fps` e salta i frame con `should_keep` False (il contatore sorgente riparte da 0 al loop). Il reader non si autocadenza più: la cadenza la dà il renderer, e la backpressure della coda limita il reader.
  - Il renderer cadenza a `decimator.effective_fps`.
  - In `precomputer.py` sostituire `frame_step`/`frame_idx % frame_step < 1` con il decimatore; `metadata["target_fps"] = effective_fps`; `actual_frames` stimato con `ceil(frame_count * ratio)`.
  - `main.py`: `PrecomputedFramePlayer` riceve `metadata["target_fps"]` invece di `args.fps`.
- [ ] **Step 6:** Run `pytest -q` → PASS. Commit (`Fixed playback speed when --fps differs from source fps`).

### Task 3: `PlaybackClock` a scadenze assolute e overlay pulito (B3, B17)

**Files:**
- Modify: `playback.py`, `pipeline.py` (`_frame_renderer_thread`), `precomputer.py` (`_playback_thread_func`)
- Create: `tests/test_clock.py`

**Interfaces:**
- Produces: `playback.PlaybackClock(fps: float, now: Callable[[], float] = time.perf_counter, sleep: Callable[[float], None] = time.sleep)` con:
  - `start() -> None`: ancora `t0 = now()`; chiamato in automatico al primo `wait_for` se non fatto;
  - `deadline(n: int) -> float`: `t0 + n / fps`;
  - `wait_for(n: int) -> None`: dorme fino a `deadline(n)` se è nel futuro;
  - `lateness(n: int) -> float`: `now() - deadline(n)`;
  - `rebase(n: int) -> None`: `t0 = now() - n / fps`.
  - `n` è un contatore di riproduzione monotono, **non** l'indice del frame video (che si azzera al loop).

- [ ] **Step 1: Test** con orologio finto (`now` restituisce una variabile, `sleep` la incrementa):

```python
def test_wait_for_sleeps_until_absolute_deadline():
    t = FakeTime(); c = PlaybackClock(10, t.now, t.sleep); c.start()
    t.advance(0.03); c.wait_for(1); assert t.value == pytest.approx(0.1)
def test_no_drift_when_work_takes_time():
    t = FakeTime(); c = PlaybackClock(10, t.now, t.sleep); c.start()
    for n in range(1, 11): t.advance(0.05); c.wait_for(n)
    assert t.value == pytest.approx(1.0)
def test_lateness_and_rebase():
    t = FakeTime(); c = PlaybackClock(10, t.now, t.sleep); c.start()
    t.advance(0.35); assert c.lateness(2) == pytest.approx(0.15)
    c.rebase(2); assert c.lateness(2) == pytest.approx(0.0)
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implementare. **Step 4:** Run → PASS.
- [ ] **Step 5: Applicare** ai due renderer: un contatore `n` che parte da 0 e un `clock.wait_for(n)` prima di disegnare; rimuovere i calcoli `sleep_time` attuali; l'intervallo registrato nelle statistiche è `now - ultimo_disegno`. Ogni riga dell'overlay delle statistiche termina con `\033[K` (B17).
- [ ] **Step 6:** Run `pytest -q` → PASS. Commit (`Fixed frame pacing drift with absolute deadlines`).

### Task 4: Indice frame nella pipeline e code senza perdite (B4, B7, B8)

**Files:**
- Modify: `pipeline.py`
- Create: `tests/test_queues.py`

**Interfaces:**
- Produces: `pipeline.put_until_stopped(q, item, should_stop, timeout: float = 0.1) -> bool` — ritenta `q.put(item, timeout=timeout)` finché riesce (True) o finché `should_stop` è impostato (False). Gli elementi di `frame_queue` diventano `list[tuple[int, np.ndarray]]` e quelli di `ascii_queue` `list[tuple[int, str]]`, dove `int` è l'indice del frame **decimato**, che riparte da 0 a ogni loop.

- [ ] **Step 1: Test** (`queue.Queue` e `threading.Event` bastano, duck typing):

```python
def test_put_waits_for_space():
    q = queue.Queue(maxsize=1); q.put("a"); stop = threading.Event()
    threading.Timer(0.2, q.get).start()
    assert put_until_stopped(q, "b", stop) is True and q.get() == "b"
def test_put_gives_up_when_stopped():
    q = queue.Queue(maxsize=1); q.put("a"); stop = threading.Event()
    threading.Timer(0.2, stop.set).start()
    assert put_until_stopped(q, "b", stop) is False
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implementare `put_until_stopped`. **Step 4:** Run → PASS.
- [ ] **Step 5: Applicare.**
  - Ogni `put` di reader e converter, compreso `END_OF_VIDEO_MARKER` e l'ultimo batch parziale, passa da `put_until_stopped`.
  - Rimuovere tutta la logica di scarto: batch dimezzati, `batch[1:]`, `queue_ratio`/`_maxsize`, il fallback a frame singolo.
  - Il renderer calcola `video_time = index / effective_fps` dall'indice ricevuto (B4) e lo passa a `audio_player.update_video_time`.
  - Rimuovere `total_frames`/`video_duration` e l'apertura di `VideoFileClip` in `start()`.
- [ ] **Step 6:** Run `pytest -q` → PASS. Commit (`Fixed dropped frames and lost end-of-video marker in pipeline`).

### Task 5: Audio – loop, risincronizzazione, risorse (B5, B6, B16, B19, §4.7)

**Files:**
- Modify: `audio_player.py`
- Create: `tests/test_audio.py`

**Interfaces:**
- Consumes: niente di nuovo; `update_video_time(t: float)` invariato.
- Produces: `AudioPlayer.load_array(samples: np.ndarray, samplerate: int) -> None` (usato da `initialize()` e dai test: converte in `float32` C-contiguo e fa il downmix di più di 2 canali alla media su 2 canali); `AudioPlayer._audio_callback(outdata, frames, time_info, status)` non restituisce mai `CallbackStop`.

- [ ] **Step 1: Test** (callback chiamato direttamente, `outdata = np.zeros((frames, ch), np.float32)`):

```python
def test_callback_wraps_at_end():
    p = AudioPlayer("x"); p.load_array(np.arange(10, dtype=np.float64).reshape(10, 1).repeat(2, 1), 10)
    p.audio_time = 0.8; out = np.zeros((4, 2), np.float32)
    assert p._audio_callback(out, 4, None, None) is None
    assert out[:, 0].tolist() == [8, 9, 0, 1] and not p.should_stop.is_set()
def test_resync_applied_before_copy():
    p = AudioPlayer("x"); p.load_array(np.arange(100.0).reshape(100, 1).repeat(2, 1), 10)
    p.update_video_time(5.0); out = np.zeros((3, 2), np.float32)
    p._audio_callback(out, 3, None, None)
    assert out[:, 0].tolist() == [50, 51, 52] and p.audio_time == pytest.approx(5.3)
def test_downmix_and_dtype():
    p = AudioPlayer("x"); p.load_array(np.ones((5, 6)), 10)
    assert p.audio_array.shape == (5, 2) and p.audio_array.dtype == np.float32
def test_video_without_audio_track(tiny_video):
    assert AudioPlayer(tiny_video).initialize() is False
```

- [ ] **Step 2:** Run → FAIL. **Step 3: Implementare.**
  - La risincronizzazione pendente diventa un singolo attributo `self._pending_seek: float | None` (scritto da `update_video_time` quando `abs(audio_time - t) > sync_tolerance`), al posto di `sync_queue`.
  - Il callback, nell'ordine: applica il seek pendente, copia con wrap-around (`np.take(..., mode='wrap')` o due slice), avanza `audio_time` modulo la durata.
  - Nessuna chiamata al logger nel callback: `status` non vuoto incrementa `self.underflows`, loggato in `stop()`.
  - `initialize()`: import espliciti (`from moviepy import VideoFileClip`) e chiusura del `VideoFileClip` in un `finally`.
- [ ] **Step 4:** Run `pytest -q tests/test_audio.py` → PASS. Commit (`Fixed audio stopping after first loop and resync ordering`).

### Task 6: Correttezza della cache legacy (B9, B12, B13)

**Files:**
- Modify: `precomputer.py` (`FramePrecomputer.__init__`, `precompute_frames`, `is_cache_valid`)
- Create: `tests/test_legacy_cache.py`

**Interfaces:**
- Produces: i metadati includono `"ascii_palette"`; il nome della directory include `_p<sha1(palette)[:8]>`.

- [ ] **Step 1: Test** (con `tiny_video`, `output_dir=tmp_path/"c"`, larghezza 32, fps 12):

```python
def test_all_frames_written_before_completed(tiny_video, tmp_path):
    p = FramePrecomputer(tiny_video, 32, 12, " .:@", output_dir=str(tmp_path / "c"))
    md = p.precompute_frames()
    assert md["completed"] and md["processed_frames"] == 24
    assert all((tmp_path / "c" / f"frame_{i:06d}.txt").exists() for i in range(24))
def test_stale_metadata_is_recomputed(tiny_video, tmp_path):
    out = str(tmp_path / "c"); FramePrecomputer(tiny_video, 32, 12, " .:@", output_dir=out).precompute_frames()
    os.remove(os.path.join(out, "frame_000005.txt"))
    p = FramePrecomputer(tiny_video, 32, 12, " .:@", output_dir=out)
    assert not p.is_cache_valid(); p.precompute_frames()
    assert os.path.exists(os.path.join(out, "frame_000005.txt"))
def test_palette_invalidates_cache(tiny_video, tmp_path):
    out = str(tmp_path / "c"); FramePrecomputer(tiny_video, 32, 12, " .:@", output_dir=out).precompute_frames()
    assert not FramePrecomputer(tiny_video, 32, 12, "box", output_dir=out).is_cache_valid()
```

- [ ] **Step 2:** Run → FAIL. **Step 3: Implementare.**
  - Rimuovere l'early return su `metadata.pkl` in `precompute_frames`.
  - Una sentinella `None` per ogni worker in coda, poi `join()` senza timeout.
  - La barra di avanzamento consuma `result_queue` fino a `processed_count`.
  - Confronto della palette in `is_cache_valid`.
- [ ] **Step 4:** Run → PASS. Commit (`Fixed incomplete and stale high-performance cache`).

### Task 7: Palette, chiusura, logging, codice morto (B10, B11, B14, B15, B18)

**Files:**
- Modify: `main.py`, `utils.py`, `pipeline.py` (`stop`), `precomputer.py` (`stop`)
- Create: `tests/test_main_utils.py`

**Interfaces:**
- Produces:
  - `main.PALETTES: dict[str, str]`, con `standard` = `' \'`",;l!~+-][{)(\\/frxuvzYUCL0Zmqdbha*MW8%@'` (literal Python, 42 caratteri: sottocampionamento uniforme di `extended`);
  - `main.load_custom_palette(path: str) -> str` — legge in `utf-8`, rimuove solo `\r` e `\n`, `ValueError` se il risultato è vuoto;
  - `VideoPipeline.stop()` e `PrecomputedFramePlayer.stop()` idempotenti (flag `self._stopped`).

- [ ] **Step 1: Test:**

```python
def test_standard_palette():
    assert len(main.PALETTES["standard"]) == 42 and main.PALETTES["standard"][0] == " "
def test_custom_palette_keeps_leading_space(tmp_path):
    f = tmp_path / "p.txt"; f.write_text(" .░▓█\r\n", encoding="utf-8")
    assert main.load_custom_palette(str(f)) == " .░▓█"
def test_custom_palette_empty(tmp_path):
    f = tmp_path / "p.txt"; f.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError): main.load_custom_palette(str(f))
def test_process_logging_no_duplicates():
    a = configure_process_logging("T"); b = configure_process_logging("T")
    assert a is b and len(a.handlers) == 2 and a.propagate is False
```

- [ ] **Step 2:** Run → FAIL. **Step 3: Implementare.**
  - `get_ascii_palette` usa `PALETTES` e `load_custom_palette`; in caso di errore: log e fallback a `standard`. Help di `--palette` aggiornato a `standard (42 chars), extended (67 chars)`.
  - `configure_process_logging`: `propagate=False`, handler aggiunti solo se il logger non ne ha.
  - Il signal handler in `run_standard_mode` imposta solo `pipeline.should_stop`; niente `sys.exit` né `stop()` (li esegue il `finally`).
  - Rimuovere il ramo `args.fps is None`, `video_finished` e `default=100` sul positional `width`.
- [ ] **Step 4:** Run `pytest -q` → PASS.
- [ ] **Step 5: Verifica di fine Fase 1.** Run: `printf '\n' | timeout 60 .venv/Scripts/python.exe main.py <tiny o videos/5.mp4> 80 --fps 30 --no-loop` in entrambe le modalità (con e senza `--high-performance`) e con `--palette box`. Atteso: il processo esce da solo (exit code 0, non 124) e non ci sono `ERROR` in `logs/`. Chiedere all'utente il controllo visivo e audio con il proprio `run.bat`.
- [ ] **Step 6:** Commit (`Fixed palettes, shutdown and duplicated logs`).

---

## Fase 2 – Performance

### Task 8: `AsciiConverter` vettorizzato con run-length del colore (§4.1)

**Files:**
- Create: `ascii_converter.py`, `tests/test_converter.py`

**Interfaces:**
- Produces:
  - `CHAR_ASPECT_CORRECTION = 2.25`, `CONVERTER_VERSION = 2`;
  - `compute_ascii_height(src_w: int, src_h: int, width: int) -> int` (minimo 1);
  - `AsciiConverter(width: int, height: int, palette: str)` (`'box'` è la sentinella blocchi) con `resize(frame) -> np.ndarray`, `convert(resized) -> bytes`, `convert_frame(frame) -> bytes`.

- [ ] **Step 1: Test:**

```python
@pytest.mark.parametrize("palette", [" .:-=+*#%@", EXTENDED, "@", "box", "░▒▓█"])
def test_matches_reference(random_frame, palette):
    h = compute_ascii_height(160, 90, 120)
    out = AsciiConverter(120, h, palette).convert_frame(random_frame)
    assert parse_ansi_grid(out.decode("utf-8")) == reference_grid(random_frame, 120, palette)
def test_height_one():
    assert compute_ascii_height(1000, 10, 50) == 1
    assert len(AsciiConverter(50, 1, "@").convert_frame(np.zeros((10, 1000, 3), np.uint8)).split(b"\n")) == 1
def test_color_emitted_only_on_change():
    out = AsciiConverter(40, 3, "@").convert(np.full((3, 40, 3), 200, np.uint8))
    assert out.count(b"\033[38;5;") == 3
def test_two_color_rows_emit_two_escapes():
    f = np.zeros((4, 40, 3), np.uint8); f[:, 20:] = 255
    assert AsciiConverter(40, 4, "@").convert(f).count(b"[38;5;") == 8
```

- [ ] **Step 2:** Run → FAIL. **Step 3: Implementare.** Tutto precalcolato nel costruttore:
  - `char_lut`: `uint8[256]`, con `char_lut[g] = int(g / 255.0 * (n - 1))`;
  - `full_tok[color - 16, k] = f"\033[38;5;{color}m{ch}".encode()` e `short_tok[k] = ch.encode()`, entrambe array `object`;
  - `row_end = b"\033[0m\n"`.

  In `convert`:

```python
color = 16 + 36*min(5,r//43) + 6*min(5,g//43) + min(5,b//43)        # int16, shape (H, W)
k = char_lut[gray]
change = np.ones_like(color, bool); change[:, 1:] = color[:, 1:] != color[:, :-1]
cells = np.where(change, full_tok[color - 16, k], short_tok[k])     # object (H, W)
grid = np.empty((H, W + 1), object); grid[:, :W] = cells; grid[:, W] = row_end
return b"".join(grid.ravel().tolist())[:-1]                          # senza l'ultimo \n
```

- [ ] **Step 4:** Run `pytest -q tests/test_converter.py` → PASS.
- [ ] **Step 5:** Run `bench.py videos/5.mp4 300 --palette extended` e annotare `convert ms/frame` nel messaggio di commit. Commit (`Added vectorized AsciiConverter with color run-length`).

### Task 9: `FpsStats` condiviso (§4.3)

**Files:**
- Create: `fps_stats.py`, `tests/test_fps_stats.py`

**Interfaces:**
- Produces: `FpsStats(target_fps: float | None, history: int = 100, graph_points: int = 50)` con:
  - `record(interval_s: float) -> None`;
  - `overlay_lines(frame_label: str) -> list[str]`: `["Calcolando FPS..."]` senza dati; altrimenti riga delle statistiche più le righe del grafico (intestazione, barre, scala, in quest'ordine);
  - `summary() -> str`: `"FPS medio: x, Min: y, Max: z"`.

- [ ] **Step 1: Test:**

```python
def test_empty(): assert FpsStats(30).overlay_lines("Frame 0") == ["Calcolando FPS..."]
def test_steady_30fps():
    s = FpsStats(30); [s.record(1 / 30) for _ in range(20)]
    lines = s.overlay_lines("Frame 20"); assert "30.0" in lines[0] and "Frame 20" in lines[0] and len(lines) == 4
    assert s.summary().startswith("FPS medio: 30.00")
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implementare portando `create_fps_stats` e `create_frame_time_graph` da `pipeline.py` (stessa logica MAD, stessi colori e testi), su `deque(maxlen=…)`. **Step 4:** Run → PASS. **Step 5:** Commit (`Added shared FpsStats`).

### Task 10: `TerminalDisplay` con delta per riga (§4.2)

**Files:**
- Create: `terminal_display.py`, `tests/vt.py`, `tests/test_display.py`

**Interfaces:**
- Produces:
  - `TerminalDisplay(out: BinaryIO | None = None)` (default `sys.stdout.buffer`) con `show(frame: bytes, overlay: list[str] | None = None) -> None`, `invalidate() -> None`, `restore() -> None`;
  - `tests.vt.VirtualTerminal(rows=200, cols=400)` con `feed(data: bytes)` e `screen() -> list[str]` (solo testo, senza colori), che interpreta `\033[2J`, `\033[H`, `\033[{r};{c}H`, `\033[K`, SGR `m` (ignorato) e `\n` come CR+LF.

- [ ] **Step 1: Test:**

```python
def test_delta_equals_full_redraw():
    frames = [make_frame(seed) for seed in range(5)]   # 20 righe, alcune identiche tra frame consecutivi
    buf = io.BytesIO(); d = TerminalDisplay(buf); vt = VirtualTerminal()
    for f in frames:
        d.show(f); vt.feed(buf.getvalue()); buf.seek(0); buf.truncate()
        full = VirtualTerminal(); full.feed(b"\033[2J\033[H" + f)
        assert vt.screen() == full.screen()
def test_identical_frame_writes_almost_nothing():
    buf = io.BytesIO(); d = TerminalDisplay(buf); f = make_frame(0); d.show(f); n = len(buf.getvalue()); d.show(f)
    assert len(buf.getvalue()) - n < 16
def test_overlay_lines_cleared():
    # show(f, ["AAAAAA"]) poi show(f, ["B"]): vt.screen()[n_righe + 1].rstrip() == "B"
def test_restore_shows_cursor():
    buf = io.BytesIO(); TerminalDisplay(buf).restore(); assert buf.getvalue().endswith(b"\033[0m\033[?25h")
```

- [ ] **Step 2:** Run → FAIL. **Step 3: Implementare.**
  - Le righe del frame sono `frame.split(b"\n")`.
  - Ridisegno completo (`\033[2J\033[H` + frame) al primo frame, dopo `invalidate()` o se cambia il numero di righe.
  - Altrimenti: per ogni sequenza di righe consecutive cambiate, `\033[{r};1H` più le righe unite da `b"\n"`.
  - Overlay a partire dalla riga `len(rows) + 2`, ogni riga seguita da `\033[K`, codificata UTF-8.
  - Una sola `out.write` + `out.flush()` per `show`.
- [ ] **Step 4:** Run → PASS. **Step 5:** Commit (`Added TerminalDisplay with per-row delta rendering`).

### Task 11: Loop di riproduzione comune `run_playback` (§4.4)

**Files:**
- Modify: `playback.py`
- Create: `tests/test_playback.py`

**Interfaces:**
- Consumes: `PlaybackClock` (Task 3), `TerminalDisplay.show` (Task 10), `FpsStats` (Task 9), `AudioPlayer.update_video_time` (Task 5).
- Produces:
  - protocollo `FrameSource` con `next_frame() -> tuple[int, bytes] | None` (None = fine video) e `restart() -> None`;
  - `run_playback(source, fps: float, display, should_stop: threading.Event, loop: bool, stats: FpsStats | None = None, audio=None, clock: PlaybackClock | None = None) -> int` (restituisce i frame disegnati).

  Regole:
  - il contatore `n` del clock avanza per ogni frame estratto, disegnato o saltato;
  - con `0 < clock.lateness(n) < 1.0` e almeno un frame di ritardo (`> 1/fps`) il frame si **salta**;
  - con `lateness >= 1.0` si fa `clock.rebase(n)` e il frame si disegna (stallo, per esempio all'avvio);
  - `audio.update_video_time(index / fps)` prima di ogni disegno;
  - overlay `stats.overlay_lines(f"Frame {index}")` ogni 5 frame disegnati;
  - a fine video con `loop` si chiama `source.restart()`; se dopo un `restart()` la sorgente restituisce subito `None`, si esce.

- [ ] **Step 1: Test** con `ListSource(frames)`, `RecordingDisplay` e `FakeTime` (Task 3):

```python
def test_plays_all_in_order_without_loop(): ...           # disegnati == [0..9], ritorno 10
def test_skips_late_frames():                             # display.show fa avanzare FakeTime di 0.25s a fps 10
    ...  # indici disegnati strettamente crescenti, len < 10, ultimo == 9
def test_loop_restarts_until_stopped(): ...               # loop=True, stop impostato dopo 25 disegni: indici 0..9,0..9,0..4
def test_empty_source_with_loop_terminates():             # Review Focus 2
    assert run_playback(ListSource([]), 10, RecordingDisplay(), threading.Event(), loop=True) == 0
def test_audio_receives_video_time(): ...                 # FakeAudio registra [i / 10 for i in range(10)]
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implementare. **Step 4:** Run → PASS. **Step 5:** Commit (`Added shared run_playback loop`).

### Task 12: Modalità standard sul nucleo condiviso (§4.5)

**Files:**
- Modify: `pipeline.py` (riscrittura), `main.py` (`run_standard_mode`)
- Delete: `terminal_output_buffer.py`, `renderer.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `FrameDecimator`, `put_until_stopped`, `AsciiConverter`, `run_playback`.
- Produces:
  - `VideoPipeline(video_path, width: int, height: int, source_fps: float, target_fps: float | None, batch_size: int, palette: str, loop_video: bool)` con `start()`, `stop()` (idempotente), `frame_source() -> FrameSource`, `effective_fps: float` e `should_stop`;
  - processi top-level `frame_reader_process(video_path, width, height, source_fps, target_fps, frame_queue, should_stop, batch_size, loop_video)` e `frame_converter_process(width, height, palette, frame_queue, ascii_queue, should_stop)`;
  - la `frame_source()` restituisce bytes; `restart()` è un no-op (il loop lo fa il reader); `next_frame()` restituisce `None` solo con `END_OF_VIDEO_MARKER` o con lo stop impostato.

- [ ] **Step 1: Test** d'integrazione (processi reali, `tiny_video`):

```python
def test_no_loop_plays_every_decimated_frame_and_exits(tiny_video):
    p = VideoPipeline(tiny_video, 32, 8, 24.0, 12, 4, " .:@", loop_video=False); p.start()
    try:
        drawn = run_playback(p.frame_source(), p.effective_fps, TerminalDisplay(io.BytesIO()), p.should_stop, loop=False)
    finally:
        p.stop()
    assert drawn == 24
```
(`drawn` è uguale a 24 solo se nessun frame viene saltato: per farlo valere, a 12 fps su CPU normale non c'è ritardo; se il test risulta instabile, passare un `clock` con fps 1000.)

- [ ] **Step 2:** Run → FAIL. **Step 3: Implementare.**
  - Reader: `grab()` per ogni frame, `retrieve()` solo se `should_keep`, poi `cv2.resize(..., (width, height), INTER_NEAREST)`; batch di `(indice_decimato, ndarray)`.
  - Code con `maxsize = max(2, ceil(2 * effective_fps / batch_size))`.
  - Converter: un `AsciiConverter` per processo, `convert(resized)`.
  - Lo spacchettamento dei batch avviene dentro la `frame_source`.
  - Nessun thread di rendering: `run_standard_mode` esegue `run_playback` nel main thread con `TerminalDisplay()` e `FpsStats` se `--log_fps`; Ctrl+C → `KeyboardInterrupt` → `finally: pipeline.stop(); display.restore()`.
  - Rimuovere i signal handler (il default di SIGINT basta).
  - `--log_performance`: il converter logga a livello DEBUG la media dei ms/frame ogni 100 frame.
- [ ] **Step 4:** Run `pytest -q` → PASS. **Step 5:** Commit (`Rebuilt standard mode on shared converter and playback`).

### Task 13: Cache binaria e modalità high-performance (§4.6)

**Files:**
- Modify: `precomputer.py` (riscrittura completa: le vecchie classi `FramePrecomputer`/`PrecomputedFramePlayer` vengono rimosse), `main.py` (`run_high_performance_mode`), `bench.py` (precompute con la nuova API), `tests/test_reference.py` (rimuovere il test legacy, che non ha più oggetto)
- Delete: `tests/test_legacy_cache.py`
- Create: `tests/test_cache.py`

**Interfaces:**
- Consumes: `FrameDecimator`, `AsciiConverter`, `CONVERTER_VERSION`, `run_playback`/`FrameSource`.
- Produces:
  - `cache_dir_for(video_path, width, height, effective_fps, palette, cache_root="cached_frames") -> str`: `"<nome>_<sha1[:12]>"`, con hash calcolato su `os.path.abspath`, `st_size`, `st_mtime_ns`, width, height, `round(effective_fps, 3)`, palette, `CONVERTER_VERSION`;
  - `precompute(video_path, width, height, source_fps, target_fps, palette, cache_root="cached_frames", processes: int | None = None, max_frames: int | None = None) -> str | None` (directory della cache; `processes=0` converte nel processo corrente; `None` → `min(os.cpu_count(), 8)`);
  - `CachedFrames.open(cache_dir) -> CachedFrames | None` (None se manca `metadata.json`) con `__len__`, `frame(i) -> bytes`, `metadata: dict`, `close()`;
  - `CachedFrameSource(cached: CachedFrames)` che implementa `FrameSource`.

- [ ] **Step 1: Test:**

```python
# nth_kept_frame(video, n): helper locale del test, n-esimo frame tenuto da FrameDecimator(24.0, 12)
def test_roundtrip(tiny_video, tmp_path):
    d = precompute(tiny_video, 32, 8, 24.0, 12, " .:@", str(tmp_path), processes=0)
    c = CachedFrames.open(d); conv = AsciiConverter(32, 8, " .:@")
    assert len(c) == 24 and c.frame(3) == conv.convert_frame(nth_kept_frame(tiny_video, 3))
def test_key_changes(tiny_video, tmp_path):
    base = cache_dir_for(tiny_video, 32, 8, 12, " .:@", str(tmp_path))
    assert base != cache_dir_for(tiny_video, 32, 8, 12, "box", str(tmp_path))
    assert base != cache_dir_for(tiny_video, 32, 8, 6, " .:@", str(tmp_path))
    os.utime(tiny_video, ns=(1, 1)); assert base != cache_dir_for(tiny_video, 32, 8, 12, " .:@", str(tmp_path))
def test_interrupted_precompute_not_reused(tiny_video, tmp_path, monkeypatch):   # Review Focus 4
    calls = iter(range(100))
    def boom(self, r):
        if next(calls) == 5: raise KeyboardInterrupt
        return b"x"
    monkeypatch.setattr(AsciiConverter, "convert", boom)
    with pytest.raises(KeyboardInterrupt): precompute(tiny_video, 32, 8, 24.0, 12, " .:@", str(tmp_path), processes=0)
    d = cache_dir_for(tiny_video, 32, 8, 12, " .:@", str(tmp_path))
    assert CachedFrames.open(d) is None and not os.path.exists(os.path.join(d, "frames.bin.tmp"))
def test_multiprocess_matches_inline(tiny_video, tmp_path): ...  # processes=2 vs processes=0: stessi bytes per ogni frame
```

- [ ] **Step 2:** Run → FAIL. **Step 3: Implementare.**
  - Decodifica, decimazione e resize nel processo principale.
  - Con il pool: `Pool(processes, initializer=_init_worker, initargs=(width, height, palette))` e `imap(_convert_worker, frames_iter, chunksize=8)` (funzioni top-level).
  - Scrittura sequenziale su `frames.bin.tmp`, con gli offset raccolti in una lista; barra `tqdm` sui frame scritti.
  - A fine lavoro: `np.save(index.npy, np.array(offsets, int64))`, `os.replace(tmp, frames.bin)` e per ultimo `metadata.json` (`effective_fps`, `frames`, `width`, `height`, `palette`, `video_path`, `converter_version`).
  - In caso di qualunque eccezione (anche `KeyboardInterrupt`): rimozione di `.tmp`, `frames.bin` e `index.npy`, poi re-raise.
  - `CachedFrames` usa `mmap.mmap(fileno, 0, access=ACCESS_READ)`; `frame(i)` restituisce `mm[off[i]:off[i+1]]`.
  - `run_high_performance_mode`: `CachedFrames.open(cache_dir_for(...))` oppure `precompute`; se restituisce None, fallback a `run_standard_mode`. Poi `run_playback(CachedFrameSource, effective_fps, ...)` nel main thread, con lo stesso `finally` della Task 12.
- [ ] **Step 4:** Run `pytest -q` → PASS. **Step 5:** Commit (`Rebuilt high-performance mode on binary mmap cache`).

### Task 14: Integrazione finale, documentazione, benchmark (§4.8, §4.9, §6)

**Files:**
- Modify: `main.py`, `utils.py`, `readme.md`, `readme_it.md`, `CLAUDE.md`, `bench.py`
- Create: `docs/superpowers/bench-final.txt`

- [ ] **Step 1:** `main.py`:
  - l'altezza viene calcolata solo con `compute_ascii_height`;
  - `effective_fps` da `FrameDecimator(fps_originale, args.fps)`, passato a entrambe le modalità;
  - nessun riferimento residuo a `2.25` fuori da `ascii_converter.py`: verificare con `grep -rn "2.25" --include=*.py .`, che deve mostrare solo `ascii_converter.py` (e `tests/reference.py`).
- [ ] **Step 2:** Logging: nessun log per-frame sopra il livello DEBUG nei percorsi caldi. Verificare con `grep -n "logger\.\(info\|warning\)" playback.py terminal_display.py ascii_converter.py`, che non deve dare risultati dentro i loop.
- [ ] **Step 3:** Documentazione:
  - README (EN/IT): documentare `--high-performance`, la nuova palette `standard` e il fatto che le cache create da versioni precedenti vanno cancellate;
  - `CLAUDE.md`: aggiornare l'architettura (moduli condivisi, cache binaria con chiave completa, `--fps` limitato agli fps sorgente, comando `pytest`); rimuovere le note superate (duplicazione della conversione, cache senza palette, reader che non decima, `renderer.py`).
- [ ] **Step 4: Benchmark finale.**
  - Run: `bench.py videos/5.mp4 300 --palette extended` e `--palette box` → `docs/superpowers/bench-final.txt`.
  - Riportare all'utente la tabella baseline vs. finale.
  - Se `convert ms/frame` > 5.0 a larghezza 300, segnalarlo all'utente come candidato per `numba` (decisione dell'utente; non implementare).
- [ ] **Step 5:** Run `pytest -q` → PASS. Ripetere lo smoke test della Task 7 Step 5 su entrambe le modalità e le palette `extended` e `box` → uscita 0, nessun `ERROR` nei log. Chiedere all'utente la verifica visiva e audio con `run.bat` (loop e `--no-loop`).
- [ ] **Step 6:** Commit (`Updated docs and final benchmark`).
