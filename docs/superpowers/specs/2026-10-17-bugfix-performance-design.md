# Pixellator – Correzione bug e ottimizzazione performance

Data: 2026-10-17
Stato: design approvato in chat, in attesa di revisione dello spec

## 1. Obiettivo

Due parti, eseguite in sequenza:

1. **Fase 1 – Bug**: correggere tutti i bug noti nella struttura attuale, con interventi mirati, lasciando un player corretto e usabile.
2. **Fase 2 – Performance**: ristrutturare il codice per massimizzare le performance (libertà totale: nuovo formato cache, rimozione di codice morto, eventuali nuove dipendenze solo se giustificate dal benchmark).

### Caso d'uso di riferimento

`python main.py videos/5.mp4 300 --fps 60 --batch_size 100 --palette extended --high-performance --audio --log_fps`
su Windows Terminal.

### Criteri di successo

- Velocità di riproduzione corretta: il video dura quanto il file sorgente, indipendentemente da `--fps`.
- Audio sincronizzato e continuo anche dopo il loop.
- Nessun frame scartato silenziosamente dalle code; `--no-loop` termina sempre.
- Aspetto visivo invariato: stessi caratteri e stessi colori 256 per cella rispetto al converter attuale (palette `standard` esclusa, vedi B10).
- Misure di `bench.py` (ms/frame di conversione, byte/frame emessi, frame/s di precompute) nettamente migliori a fine Fase 2 rispetto alla baseline pre-Fase 1.

### Vincoli

- Commenti, docstring, log e stringhe utente in italiano.
- Funzioni dei processi top-level e picklable (start method `spawn` su Windows).
- Entrambe le modalità (standard e high-performance) restano supportate.
- Dipendenze runtime attuali (OpenCV, NumPy, tqdm, sounddevice, moviepy). `numba` è ammesso solo se il benchmark mostra che la costruzione dei bytes resta il collo di bottiglia dopo la vettorizzazione NumPy.

## 2. Fase 0 – Rete di sicurezza

Prima di qualunque modifica:

- `requirements-dev.txt` con `pytest`.
- `tests/test_converter.py`: test di caratterizzazione. Per frame sintetici casuali (seed fisso) già ridimensionati, decodifica l'output ANSI del converter attuale in una griglia `(carattere, codice colore)` per cella e la salva come riferimento atteso tramite una funzione di riferimento che replica la logica attuale (`gray/255*(n-1)` → carattere, `min(5, c//43)` → codice `16+36r+6g+b`). Il nuovo converter dovrà produrre la stessa griglia.
- `bench.py <video> <width> [--palette P] [--frames N]`: misura su N frame (default 300) ms/frame di conversione, byte/frame dell'output, frame/s del precompute. Baseline registrata su `videos/5.mp4` a larghezza 300 con palette `extended` e `box`.

## 3. Fase 1 – Correzione bug (struttura attuale)

| ID | File | Bug | Correzione |
|----|------|-----|------------|
| B1 | `pipeline.py` | Il reader non decima: con `--fps` ≠ fps sorgente il video va al rallentatore o accelerato | Decimazione con `fps_effettivi = min(fps_target, fps_sorgente)` e selezione uniforme ad accumulatore; il renderer cadenza a `fps_effettivi` |
| B2 | `precomputer.py` | Con `--fps` > fps sorgente estrae tutti i frame ma li riproduce a `--fps` (2x) | `target_fps` dei metadati = `fps_effettivi`; il player usa quello |
| B3 | `precomputer.py`, `pipeline.py` | Il ritmo dei frame deriva: il tempo di lettura/attesa non è contabilizzato | Scadenze assolute `t0 + n/fps` con `time.perf_counter()` |
| B4 | `pipeline.py` | Il contatore dei frame non si azzera al loop: il tempo video supera la durata, l'audio si ferma | Il tempo video deriva dall'indice del frame decimato, che riparte da 0 al loop |
| B5 | `audio_player.py` | A fine traccia il callback imposta `should_stop` e restituisce `CallbackStop`: l'audio muore dopo il primo loop | Lettura circolare del buffer (wrap-around); lo stream termina solo con `stop()` |
| B6 | `audio_player.py` | `start_idx` calcolato prima di applicare la risincronizzazione; `audio_time` resta indietro di un blocco | Elaborare la `sync_queue` prima di calcolare gli indici |
| B7 | `pipeline.py` | Reader e converter scartano frame a coda piena (batch dimezzati, `batch[1:]`, "metà batch") | `put` bloccante in un ciclo con timeout breve e controllo di `should_stop`; nessuno scarto |
| B8 | `pipeline.py` | `END_OF_VIDEO_MARKER` inviato con timeout 1s e perso se la coda è piena: `--no-loop` si blocca | Stesso `put` bloccante con controllo di stop di B7 |
| B9 | `precomputer.py` | `join(timeout=1.0)` sui worker, poi la cache viene marcata come completa mentre i frame si stanno ancora scrivendo; la barra di avanzamento non arriva al 100% | Sentinella `None` per ogni worker, `join()` senza timeout, aggiornamento della barra fino all'ultimo frame |
| B10 | `main.py` | Palette `standard` corrotta (concatenazione di due palette, luminosità non monotona) | Sostituita con la rampa classica a 42 caratteri ordinata per densità, coerente con l'help |
| B11 | `main.py` | Palette custom: `strip()` toglie lo spazio iniziale, `\n` finisce nei frame, encoding non esplicito | Lettura `utf-8`, rimozione dei soli `\r\n`, errore se la palette risulta vuota |
| B12 | `precomputer.py` | `precompute_frames` restituisce i metadati vecchi se `metadata.pkl` esiste, anche quando `is_cache_valid()` li ha scartati | Rimozione dell'early return: se si arriva a `precompute_frames` la cache viene rigenerata |
| B13 | `precomputer.py` | La chiave della cache ignora la palette | La palette entra nella chiave e nei metadati verificati (in Fase 2 sostituita dalla chiave completa, §4.6) |
| B14 | `utils.py` | I logger Renderer/processi propagano al root: messaggi doppi in console | `propagate=False`; handler aggiunti una sola volta per nome |
| B15 | `main.py`, `pipeline.py` | Con SIGINT `stop()` viene eseguito due volte (handler + `finally`) | L'handler imposta solo l'evento di stop; `stop()` idempotente con flag |
| B16 | `audio_player.py` | `VideoFileClip` mai chiuso; `from moviepy import *` | Import espliciti; clip chiuso dopo l'estrazione dell'array audio |
| B17 | `pipeline.py`, `precomputer.py` | Le righe delle statistiche non vengono pulite: restano residui di testo più lungo | `\033[K` a fine di ogni riga dell'overlay |
| B18 | `main.py`, `pipeline.py` | Codice morto: ramo `args.fps is None`, evento `video_finished`, default inutile sul positional `width` | Rimosso |
| B19 | `audio_player.py` | Audio con più di 2 canali: mismatch di shape con `channels=2` | Downmix a stereo in fase di inizializzazione |

Ogni bug corretto in un commit dedicato (o raggruppato per file quando banale). B1–B9 e B12 hanno test dove la logica è isolabile (decimatore, callback audio chiamato direttamente, invalidazione cache); il resto è verificato a mano.

## 4. Fase 2 – Architettura per le performance

### 4.1 `ascii_converter.py` – `AsciiConverter(width, height, palette)`

- Costruito una sola volta per processo; lookup table precalcolate nel costruttore.
- `convert(frame_bgr_resized) -> bytes`: output UTF-8, righe separate da `\n`, `\033[0m` a fine riga (come oggi).
- `resize(frame) -> ndarray`: `cv2.resize(..., INTER_NEAREST)` a `(width, height)`.
- Vettorizzazione:
  - `color = 16 + 36*(r//43) + 6*(g//43) + (b//43)` (con `min(5, ·)`), `char_idx` tramite una lookup table a 256 voci precalcolata con la formula attuale (`int(g/255.0*(n-1))` per g in 0..255), così l'equivalenza è esatta per costruzione.
  - **Run-length del colore**: la sequenza `\033[38;5;Nm` viene emessa solo nella prima cella di ogni riga e dove il colore cambia rispetto alla cella precedente. Le celle di continuazione emettono solo il carattere.
  - Costruzione: tabella NumPy di token bytes (`escape+char` e `solo char`), selezione con maschera di cambio colore, `b''.join` su riga intera.
  - Palette `box`: stesso percorso con un solo carattere `█`.
- L'altezza viene calcolata una volta in `main.py` (`int(aspect * width / 2.25)`, minimo 1) e passata a tutti i componenti. Il 2.25 resta definito in un solo punto (`CHAR_ASPECT_CORRECTION` in `ascii_converter.py`).
- Se dopo la vettorizzazione `bench.py` mostra che la conversione supera ~30% del budget di frame (16.7 ms a 60 fps) a larghezza 300, si valuta un kernel `numba` opzionale. Non aggiunto altrimenti.

### 4.2 `terminal_display.py` – `TerminalDisplay`

Sostituisce `terminal_output_buffer.py`. `renderer.py` viene eliminato.

- Scrive su `sys.stdout.buffer` (bytes), una sola `write` + `flush` per frame.
- **Delta per riga**: tiene i bytes delle righe del frame precedente; per ogni riga uguale non emette nulla, per una sequenza di righe cambiate emette `\033[{riga};1H` e le righe. Primo frame e frame dopo un salto: ridisegno completo.
- Overlay delle statistiche sotto il frame, ogni riga terminata con `\033[K`.
- `restore()`: reset colori e cursore visibile, chiamato in ogni percorso di uscita.

### 4.3 `fps_stats.py` – `FpsStats`

Unica implementazione di statistiche FPS, stabilità e grafico dei tempi di frame (oggi duplicata in `pipeline.py` e `precomputer.py`). `deque(maxlen=…)`, testo dell'overlay rigenerato ogni 5 frame come oggi. Riepilogo finale nei log.

### 4.4 `playback.py`

- `FrameDecimator(source_fps, target_fps)`: `effective_fps = min(target, source)`; `should_keep(i)` con accumulatore, selezione uniforme.
- `PlaybackClock(fps)`: scadenza del frame n = `t0 + n/fps` su `perf_counter()`; `wait_for(n)`; `is_late(n)` se in ritardo oltre un intervallo di frame → il frame viene saltato.
- `run_playback(source, clock, display, stats, audio, loop, should_stop)`: loop comune. `source` espone `next() -> (indice, bytes) | None` (fine video) e `restart()`; la modalità high-performance espone anche lo skip ad indice arbitrario. Il tempo video passato all'audio è `indice / effective_fps`.

### 4.5 Modalità standard (`pipeline.py`)

```
VideoCapture ─(Reader proc: decima + resize)─▶ frame_queue [lista di (indice, ndarray H×W×3)]
  ─(Converter proc: AsciiConverter)─▶ ascii_queue [lista di (indice, bytes)]
  ─(thread di riproduzione: run_playback)─▶ stdout.buffer
```

- Il reader usa `grab()` per i frame scartati e `retrieve()` solo per quelli tenuti; ridimensiona prima di mettere in coda (circa 120 KB per frame a 300×133 invece di ~6 MB a 1080p).
- Code bloccanti con ciclo timeout/stop (B7/B8). Dimensione delle code in batch: `max(2, ceil(2*fps_effettivi / batch_size))` per ~2 s di buffer.
- Se la riproduzione è in ritardo, i frame vengono saltati lato consumer (non disegnati) mantenendo l'avanzamento dell'indice.

### 4.6 Modalità high-performance (`precomputer.py`)

- Layout: `cached_frames/<nome>_<hash>/frames.bin`, `index.npy` (offset `int64`, lunghezza N+1), `metadata.json`.
- Hash della chiave su: path assoluto, dimensione e mtime del video, larghezza, altezza, fps effettivi, palette, `CONVERTER_VERSION`.
- Precompute: decodifica + decimazione + resize nel processo principale; conversione con `multiprocessing.Pool.imap` (ordinata, `chunksize` adattivo) con un `AsciiConverter` inizializzato per worker; scrittura sequenziale su `frames.bin.tmp`; a fine lavoro `index.npy`, rename atomico, `metadata.json` per ultimo (presenza = cache completa). Interruzione → file temporanei rimossi.
- Riproduzione: `mmap` su `frames.bin`, frame = slice `[off[i]:off[i+1]]` passato direttamente al display. Skip in caso di ritardo = salto di indice.
- Le vecchie cache (`frame_*.txt` + `metadata.pkl`) non vengono lette; una nota nel README spiega che vanno cancellate.

### 4.7 Audio (`audio_player.py`)

- Array `float32` contiguo, downmix a stereo, clip chiuso dopo l'estrazione.
- Callback senza logging né allocazioni evitabili: applica la risincronizzazione pendente, copia con wrap-around circolare, avanza la posizione.
- Il video resta il master: `update_video_time(t)` richiede una risincronizzazione solo oltre `sync_tolerance` (0.1 s); al loop la richiesta riporta l'audio a 0.

### 4.8 `main.py`

- Calcolo dell'altezza in un solo punto; risoluzione della palette con le correzioni B10/B11.
- Unico percorso di chiusura: SIGINT/SIGTERM impostano l'evento di stop; `finally` esegue `stop()` e `display.restore()`.

### 4.9 Logging

`propagate=False` per i logger di processo; nessun log nel percorso caldo per-frame sopra il livello DEBUG.

## 5. Gestione degli errori

- Video non apribile, primo frame illeggibile: errore e uscita con codice 1 (come oggi).
- Errore nel precompute: file temporanei rimossi, fallback alla modalità standard (come oggi).
- Audio non disponibile o in errore: warning, si continua senza audio.
- Eccezione nel loop di riproduzione: loggata, stop della pipeline, terminale ripristinato.

## 6. Test e verifica

- `tests/test_converter.py`: caratterizzazione cella per cella (palette `basic`, `extended`, palette a 1 carattere, `box`, altezza 1).
- `tests/test_decimator.py`: 30→24, 30→60 (limitato a 30), 25→10: numero di frame e distanza massima tra frame tenuti.
- `tests/test_cache.py`: roundtrip scrittura/lettura; invalidazione per palette, fps, mtime; cache interrotta mai riusata.
- `tests/test_display.py`: un terminale simulato (griglia che interpreta `\033[H`, `\033[r;1H`, `\n`, SGR) mostra lo stesso schermo con delta per riga e con ridisegno completo.
- `tests/test_audio.py`: callback chiamato direttamente su un array sintetico: wrap-around a fine buffer, risincronizzazione applicata prima della copia.
- `bench.py`: tabella baseline (pre-Fase 1) vs. finale (post-Fase 2).
- Verifica manuale (utente al terminale): entrambe le modalità × palette `extended` e `box` × `--no-loop` e loop × `--audio`.

## 7. Fuori scope

- Colore truecolor (24 bit) o nuove palette.
- Ridimensionamento dinamico al cambio di dimensione del terminale.
- Supporto a sorgenti diverse da file video locali.
