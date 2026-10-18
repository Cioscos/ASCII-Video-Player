# Rotazione durante la riproduzione — design

## Obiettivo

Durante la riproduzione l'utente ruota il video di 90° alla volta con un tasto,
in entrambe le modalità (standard e `--high-performance`), senza interrompere la
riproduzione oltre una breve attesa al cambio.

## Decisioni approvate

- Tasti: `r` ruota di 90° in senso orario, `R` (Maiusc+r) in senso antiorario;
  la rotazione cicla 0° → 90° → 180° → 270° → 0°.
- High-performance: a 0° si usa la cache; a 90°/180°/270° si passa alla
  conversione dal vivo (pipeline della modalità standard) dal frame corrente.
  Nessuna cache per le rotazioni.
- Approccio A: ogni cambio di rotazione **sostituisce** la sorgente (si ferma la
  pipeline corrente, se c'è, e se ne avvia una nuova con la rotazione richiesta),
  invece di comunicare la rotazione a processi già avviati.
- Il seek diventa disponibile anche in modalità standard (e da ruotato in
  high-performance) con lo stesso meccanismo: la pipeline riparte dal frame di
  destinazione. Il messaggio "Seek disponibile solo con --high-performance"
  sparisce.
- La rotazione non è persistente: ogni riproduzione parte da 0°. Nessuna opzione
  CLI né voce di menu.

## Comportamento

- **Attesa al cambio**: avviare una pipeline su Windows richiede ~1–2 s. In
  quell'intervallo resta a schermo l'ultimo frame con la riga di overlay
  `"Rotazione 90°…"` (angolo di destinazione). Lo stesso vale per il seek dal vivo
  con la riga `"Seek…"`.
- **Dimensioni**: a 90°/270° larghezza e altezza della sorgente si scambiano.
  - Larghezza automatica (`width` omesso): `auto_width` sulle dimensioni ruotate.
  - Larghezza esplicita: si mantiene a 0°/180°; a 90°/270° diventa
    `min(width, auto_width(ruotate))`, così il frame entra in altezza.
  - Altezza: `compute_ascii_height` sulle dimensioni ruotate.
- **Audio**: in pausa durante il cambio; riparte con `seek` al tempo del primo
  frame della nuova sorgente.
- **Display**: dopo il cambio, ridisegno completo (`TerminalDisplay.invalidate()`),
  anche quando il numero di righe non cambia (le righe possono essere più corte).
- **Statistiche FPS**: l'intervallo del cambio non entra nelle statistiche (come la
  pausa). Il clock viene riancorato.
- **Indici**: gli indici dei frame restano quelli decimati del video (gli stessi
  della cache), così audio, seek e ritorno alla cache restano allineati.
- **Ritorno a 0° in high-performance**: si ferma la pipeline e la cache riprende
  dal frame successivo all'ultimo mostrato.
- **Errore**: se la nuova pipeline non produce il primo frame entro 15 s o
  termina in errore, la si ferma, si ripristina la sorgente precedente (con la
  rotazione precedente, dal frame corrente) e compare per 2 s
  `"Rotazione non riuscita"`. Un errore della pipeline durante la riproduzione
  normale termina la riproduzione come oggi.
- **Seek dal vivo**: destinazione con `seek_target` sul numero totale di frame
  decimati stimato da `CAP_PROP_FRAME_COUNT`; se il totale non è noto (≤ 0) il
  seek in avanti non ha limite superiore (oltre la fine il reader termina o
  riparte col loop).

## Componenti

### `pipeline.py`

- `VideoPipeline(..., rotation=0, start_index=0)` e
  `frame_reader_process(..., rotation, start_index)`.
- Il reader si posiziona sul frame sorgente corrispondente a `start_index`
  (`CAP_PROP_POS_FRAMES`) e numera i frame da `start_index`. Al loop riparte da
  0 come oggi.
- Rotazione dopo il ridimensionamento: a 90°/270° si ridimensiona a
  `(resize_height, width)` e poi `cv2.rotate`, ottenendo `width × resize_height`
  (vale anche per `half`, dove `resize_height = pixel_rows(height, palette)`).

### `playback.py`

- `FrameDecimator.source_index(decimated_index) -> int`: inverso di
  `should_keep` (indice sorgente del frame decimato n-esimo).
- Nuovi tasti in `_KeyHandler`: `r` → `source.rotate(+1)`, `R` →
  `source.rotate(-1)`, solo se la sorgente ha `rotate`. Prima del cambio: overlay
  di attesa, audio in pausa. Dopo: `display.invalidate()`, audio `seek` + ripresa,
  `clock.rebase(n)`, statistiche senza l'intervallo del cambio.
- Il seek su una sorgente con `seek` resta com'è; se la sorgente dichiara un seek
  lento (`source.slow_seek`), prima mostra l'overlay `"Seek…"` e mette in pausa
  l'audio.

### Nuovo modulo `rotation.py`

- `ROTATIONS = (0, 90, 180, 270)`.
- `rotated_dimensions(src_w, src_h, rotation, width, term_cols, term_rows,
  reserved_rows) -> tuple[int, int]` (funzione pura; `width=None` = automatica).
- `RotatingSource`: FrameSource con `next_frame`, `restart`, `seek`, `__len__`,
  `rotate(step) -> bool`, `rotation`, `slow_seek`.
  - Costruita con una fabbrica di pipeline (`make_pipeline(rotation,
    start_index) -> VideoPipeline` già dimensionata), la cache opzionale
    (`CachedFrameSource` in high-performance) e il totale dei frame.
  - Con la cache e rotazione 0° delega alla cache; altrimenti alla pipeline
    corrente.
  - `rotate` avvia la nuova pipeline, attende il primo frame (lo tiene come
    prossimo frame) e restituisce False in caso di errore dopo aver ripristinato
    la sorgente precedente.
  - `close()` ferma la pipeline corrente.

### `main.py`

- Entrambe le modalità costruiscono una `RotatingSource` e la passano a
  `run_playback` con un `threading.Event` proprio (non più
  `pipeline.should_stop`); lo shutdown chiama `close()`.
- `CONTROLS_HINT` e `CONTROLS_HINT_HP` diventano uguali: pausa, uscita, seek,
  `r/R ruota`.
- README (en/it) e `CLAUDE.md` aggiornati.

## Test

- `rotated_dimensions`: 4 angoli, larghezza automatica ed esplicita (anche la
  riduzione a 90°/270°).
- `FrameDecimator.source_index` coerente con `should_keep` (vari rapporti di fps).
- `frame_reader_process` su clip sintetica: con `rotation=90` il frame è ruotato
  (pixel d'angolo noti) e con `start_index=k` il primo indice è `k` e il contenuto
  è quello del frame atteso.
- `RotatingSource` con pipeline finte: passaggio cache → pipeline → cache, seek da
  ruotato (nuova pipeline dal frame giusto), errore all'avvio con ripristino.
- `run_playback` con `r`/`R`: chiama `rotate`, invalida il display, l'audio va in
  pausa e riparte con seek, nessun intervallo del cambio nelle statistiche.

## Fuori ambito

- Cache precalcolata per le rotazioni.
- Rotazione iniziale da CLI/menu o persistente.
- Specchiatura (flip) del video.
