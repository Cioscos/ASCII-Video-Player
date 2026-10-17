# Pixellator – Nuove funzionalità: resa, controlli, menu, cache

Data: 2026-10-17
Stato: design approvato in chat, in attesa di revisione dello spec
Base: `master` dopo il lavoro di bugfix/performance (`docs/superpowers/specs/2026-10-17-bugfix-performance-design.md`)

## 1. Obiettivo

Aggiungere a Pixellator:

1. modalità di resa **mezzo blocco** (`▀`) a doppia risoluzione verticale;
2. **truecolor** (24 bit) opzionale;
3. **larghezza automatica** in base al terminale;
4. **controlli da tastiera** durante la riproduzione (pausa, seek, uscita);
5. **menu principale** interattivo navigabile con le frecce;
6. **gestione della cache** (elenco, cancellazione, limite di spazio con pulizia automatica).

Escluso: pacchetto installabile.

### Criteri di successo

- `python main.py` senza argomenti apre il menu; ogni comando da riga di comando esistente (es. `run.bat`) continua a funzionare identico.
- Con le nuove opzioni disattivate, output e performance restano quelli attuali (`bench.py` invariato per `extended` e `box` a 256 colori, a meno del rumore di misura).
- `half` e `truecolor` producono, cella per cella, la griglia attesa da un riferimento indipendente.
- Pausa, seek e uscita funzionano senza perdere la sincronizzazione audio.
- La cache non supera il limite configurato dopo un precompute, senza mai cancellare la cache appena creata o in uso.

### Vincoli

- Commenti, docstring, log e stringhe utente in italiano.
- Funzioni dei processi figli top-level e picklable (spawn su Windows).
- Nessuna nuova dipendenza runtime: menu e tastiera usano `msvcrt` (Windows) o `termios`/`select` (POSIX).
- Entrambe le modalità (standard e `--high-performance`) restano supportate.

## 2. Resa a schermo

### 2.1 Famiglia di converter

- `make_converter(width, height, palette, truecolor=False)` in `ascii_converter.py` restituisce un converter con l'interfaccia attuale: `resize(frame)`, `convert(resized) -> bytes`, `convert_frame(frame) -> bytes`, attributi `width`, `height`.
- `palette` può essere una stringa di caratteri, `'box'` o `'half'` (sentinelle).
- `AsciiConverter` (256 colori, caratteri e `box`) resta invariato: è il percorso veloce di default.
- Nuovi converter: truecolor per caratteri/`box`; `half` a 256 colori; `half` truecolor.
- Pipeline (`frame_converter_process`), precompute (`_init_worker`) e `bench.py` creano il converter solo tramite `make_converter`; i parametri `truecolor` e `palette` arrivano fino ai processi figli.
- Formato di output comune: UTF-8, righe separate da `\n`, ogni riga termina con `\033[0m`, nessun `\n` finale.

### 2.2 Modalità `half`

- `resize` produce `width × 2·height` pixel (`INTER_NEAREST`); la riga di terminale `y` usa i pixel `2y` (sopra) e `2y+1` (sotto).
- Ogni cella: carattere `▀`, colore del testo = pixel sopra, colore di sfondo = pixel sotto.
- 256 colori: stessa formula dei Global Constraints attuali (`16 + 36*min(5,r//43) + 6*min(5,g//43) + min(5,b//43)`), sequenza `\033[38;5;{F};48;5;{B}m▀`.
- Run-length sulla coppia (sopra, sotto): sequenza completa emessa nella prima cella di ogni riga e quando cambia almeno uno dei due colori; altrimenti solo `▀`.
- L'altezza in righe è calcolata come oggi (`compute_ascii_height`); la risoluzione verticale effettiva raddoppia.

### 2.3 Truecolor

- `--truecolor`: colore del testo `\033[38;2;R;G;Bm`; in `half` anche `48;2;R;G;B` per lo sfondo, nella stessa sequenza (`\033[38;2;R;G;B;48;2;R;G;Bm▀`).
- Carattere scelto come oggi (LUT sulla luminosità) per le palette di caratteri; `█` per `box`.
- Run-length sul colore RGB esatto (o sulla coppia in `half`).
- Costruzione vettorizzata: tabelle precalcolate dei byte dei numeri 0–255 e concatenazione su array `object`, senza cicli Python per cella.

### 2.4 Cache e versione

- `CONVERTER_VERSION = 3`.
- La chiave della cache (`cache_dir_for`) include `truecolor` oltre a palette (che già distingue `half`/`box`); `metadata.json` registra `truecolor`.

### 2.5 Larghezza automatica

- `width` diventa facoltativo da riga di comando (`nargs='?'`); nel menu il valore `auto`.
- `auto_width(src_w, src_h, term_cols, term_rows, reserved_rows) -> int`: la massima larghezza `w` tale che `w <= term_cols`, `compute_ascii_height(src_w, src_h, w) + reserved_rows <= term_rows`, `w <= src_w`; minimo 1.
- `reserved_rows` = 2 senza `--log_fps` (riga vuota + una riga di stato, es. `⏸ In pausa`); 6 con `--log_fps` (riga vuota + 4 righe di `FpsStats` + una riga di stato).
- Con larghezza esplicita resta l'avviso attuale se il frame è più alto del terminale.

### 2.6 Benchmark

`bench.py` accetta `--palette half` e `--truecolor`; i numeri per `extended`/`box` a 256 colori restano confrontabili con `docs/superpowers/bench-final.txt`.

## 3. Interazione

### 3.1 Lettore di tasti – `keyboard_input.py`

- `KeyReader` con `__enter__`/`__exit__`, `poll() -> str | None` (non bloccante) e `read(timeout) -> str | None`.
- Nomi dei tasti: `up`, `down`, `left`, `right`, `enter`, `esc`, `space`, `backspace`, singoli caratteri (`q`, cifre, lettere).
- Windows: `msvcrt.kbhit`/`getwch`, con i prefissi `\x00`/`\xe0` per le frecce.
- POSIX: modalità cbreak con `termios`, `select` per il non bloccante, sequenze `\033[A`…; `__exit__` ripristina sempre gli attributi del terminale.
- Se stdin non è un terminale interattivo, `poll()` restituisce sempre `None` senza errori.
- La decodifica delle sequenze è una funzione pura testabile (`decode_keys(chars) -> list[str]`).

### 3.2 Controlli durante la riproduzione

- `run_playback(..., controls=None)`: `controls` espone `poll() -> str | None`; controllato una volta per frame.
- Tasti: `space` pausa/ripresa; `q`/`esc` uscita; `left`/`right` −5 s/+5 s; `down`/`up` −30 s/+30 s.
- **Pausa**: il frame resta fermo, overlay con `⏸ In pausa` (anche senza `--log_fps`); audio in silenzio senza avanzare (`AudioPlayer.set_paused(bool)`); attesa dei tasti con `read(0.05)`; alla ripresa `clock.rebase(n)`, nessun frame saltato.
- **Seek**: solo se la sorgente ha `seek(index)` (`CachedFrameSource`). Indice di destinazione = indice corrente ± secondi·fps; indietro oltre l'inizio → 0; avanti oltre la fine → 0 con il loop attivo, `len-1` senza loop. Dopo il seek: `clock.rebase(n)`, `audio.update_video_time(nuovo_tempo)` forzato (seek immediato, anche sotto `sync_tolerance`), ridisegno.
- Sorgente senza `seek` (modalità standard): i tasti di seek mostrano per 2 s nell'overlay `Seek disponibile solo con --high-performance`.
- **Uscita** con `q`/`esc`: `run_playback` termina restituendo i frame disegnati; si percorre lo stesso `finally` di oggi (terminale ripristinato).
- La lettura dei tasti avviene nel thread principale; `KeyReader` è aperto in `run_standard_mode`/`run_high_performance_mode` attorno a `run_playback`.

### 3.3 Menu principale – `menu.py`

- Si apre con `python main.py` senza argomenti; con argomenti il comportamento è quello attuale.
- Voci: **Riproduci**, **Video**, **Impostazioni**, **Cache**, **Esci**.
  - **Video**: elenco dei file video in `videos/` (estensioni `.mp4 .mov .avi .mkv .webm`, case-insensitive) più "Inserisci percorso…".
  - **Impostazioni**: larghezza (`auto` o numero), fps, palette (basic/standard/extended/box/half/custom + percorso del file custom), truecolor, high-performance, audio, loop, statistiche FPS, limite cache.
  - **Cache**: vedi §3.4.
- Navigazione: `up`/`down` selezione, `left`/`right` cambio valore, `enter` conferma/modifica, `esc` indietro; i campi numerici e di testo si modificano digitando, `backspace` cancella, `enter` conferma.
- Struttura: `MenuState` (logica pura, riceve nomi di tasti, produce lo stato e un'eventuale azione `play`/`quit`) separato da `render_menu(state) -> str` (ridisegno completo dello schermo).
- `Settings` (dataclass) con caricamento/salvataggio in `settings.json` nella root (aggiunto a `.gitignore`) e conversione negli argomenti della CLI (`settings.to_argv() -> list[str]`), così la riproduzione usa un solo percorso di codice. File mancante o corrotto: valori di default e messaggio nel menu.
- Dopo la riproduzione si torna al menu. Dopo la scelta del video si mostra il frame di calibrazione come oggi.

### 3.4 Gestione della cache – `cache_manager.py`

- `CacheEntry`: `path`, `video_name`, `width`, `height`, `fps`, `palette`, `truecolor`, `size_bytes`, `last_used` (timestamp), `obsolete: bool`.
- `list_caches(root) -> list[CacheEntry]` ordinata per `last_used` decrescente; le directory del vecchio formato (`frame_*.txt` / `metadata.pkl`) sono `obsolete=True`; le directory senza `metadata.json` valido e non obsolete (precompute interrotto) sono anch'esse elencate come obsolete.
- `delete_cache(path)`: rimuove la directory; errori riportati come eccezione gestita dal chiamante.
- `enforce_limit(root, limit_bytes, keep: set[str]) -> list[str]`: cancella le cache non in `keep` in ordine di `last_used` crescente finché il totale ≤ limite; restituisce le directory cancellate; `limit_bytes == 0` disattiva.
- Ultimo utilizzo: `os.utime(metadata.json)` a ogni `CachedFrames.open` riuscito.
- Dopo ogni precompute riuscito: `enforce_limit(cache_root, limit, keep={cache_dir})`.
- Limite: default 2 GB; `--cache-limit` accetta byte o suffissi `K`/`M`/`G` (es. `5G`); `0` = nessun limite; impostabile dal menu.
- Schermata Cache: elenco (nome, dimensione, palette, data), totale occupato e limite; `enter` cancella la cache selezionata dopo conferma; `o` cancella tutte le obsolete dopo conferma.

## 4. Gestione degli errori

- Video inesistente o non apribile scelto dal menu: messaggio nel menu, si resta nel menu.
- `settings.json` corrotto: default, messaggio, il file viene riscritto al primo salvataggio.
- Errore di cancellazione di una cache: messaggio nel menu.
- Terminale senza supporto truecolor: nessun rilevamento automatico; l'opzione è a scelta dell'utente.
- Ogni uscita (fine video, `q`, Ctrl+C, errore) ripristina il terminale e, da menu, la modalità del terminale del `KeyReader`.

## 5. Test e verifica

- `tests/test_converter.py`: `half` e `truecolor` (anche combinati, palette di caratteri e `box`) confrontati cella per cella con un riferimento indipendente esteso (`tests/reference.py`: colore testo, colore sfondo, 256 e 24 bit); run-length.
- `tests/test_auto_width.py`: dimensioni finte del terminale, overlay, video stretti.
- `tests/test_keyboard.py`: `decode_keys` per Windows e POSIX.
- `tests/test_playback.py`: pausa (nessun frame avanzato, rebase), seek ±5/±30 con limiti, seek non supportato, uscita con `q`, audio in pausa/seek con orologio finto.
- `tests/test_audio.py`: `set_paused` emette silenzio senza avanzare.
- `tests/test_menu.py`: sequenze di tasti → stato, azioni, `to_argv`, salvataggio e caricamento (anche file corrotto).
- `tests/test_cache_manager.py`: elenco, obsolete, `enforce_limit` (ordine, `keep`, limite 0), parsing di `--cache-limit`.
- `bench.py`: tabella per `extended`, `box`, `half`, con e senza `--truecolor`.
- Verifica manuale dell'utente: menu, entrambe le modalità × `half`/truecolor, pausa/seek/uscita con audio.

## 6. Fuori scope

- Pacchetto installabile.
- Rilevamento automatico del supporto truecolor.
- Seek in modalità standard.
- Ridimensionamento dinamico durante la riproduzione.
