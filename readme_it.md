# ASCII Video Player

![ASCII Video Preview](https://github.com/Cioscos/Pixellator/blob/performance/doc/intro.png)

## 📝 Descrizione

**ASCII Video Player** è un convertitore video in ASCII art real-time ad alte prestazioni che visualizza i video nel terminale. Utilizza una pipeline parallela ottimizzata per convertire e visualizzare i frame, supportando la riproduzione audio sincronizzata e diverse palette di caratteri.

## ✨ Funzionalità

- **Riproduzione video in ASCII art** nel terminale con supporto a colori
- **Pipeline di elaborazione parallela** per prestazioni ottimali
- **Sincronizzazione audio-video**: l'audio segue l'orologio del video e si risincronizza sul tempo video corrente quando si discosta
- **Multiple palette di caratteri ASCII** (basic, standard, extended, box o custom)
- **Sistema di calibrazione automatico** per adattarsi alle dimensioni del terminale
- **Monitoraggio performance in tempo reale** con visualizzazione grafica degli FPS
- **Temporizzazione a scadenze assolute**: ogni frame ha una scadenza fissa; i frame vengono saltati solo quando il disegno resta indietro, così la riproduzione mantiene il ritmo anche su hardware meno potenti

## 🔧 Requisiti

- Python 3.12 (versioni fissate in `requirements.txt`)
- Terminale con supporto ai colori ANSI (la maggior parte dei terminali moderni)
- Dipendenze Python:
  - OpenCV (`opencv-python`)
  - NumPy
  - tqdm (barra di avanzamento di `--high-performance`)
  - Per l'audio (opzionale):
    - SoundDevice
    - MoviePy

## 📦 Installazione

1. Clona il repository:
   ```bash
   git clone https://github.com/utente/ascii-video-player.git
   cd ascii-video-player
   ```

2. Installa le dipendenze:
   ```bash
   # Installazione base
   pip install opencv-python numpy tqdm

   # Con supporto audio
   pip install opencv-python numpy tqdm sounddevice moviepy

   # Oppure, con le versioni fissate
   pip install -r requirements.txt
   ```

## 🚀 Utilizzo

La sintassi base è:

```bash
python main.py <video_path> <width> [opzioni]
```

Dove:
- `<video_path>`: percorso al file video da riprodurre
- `<width>`: larghezza dell'output ASCII (in caratteri)

### Esempi

Riproduzione semplice:
```bash
python main.py mio_video.mp4 100
```

Riproduzione con audio e frame rate specifico:
```bash
python main.py mio_video.mp4 120 --fps 24 --audio
```

Riproduzione con palette estesa e logging delle performance:
```bash
python main.py mio_video.mp4 80 --palette extended --log_fps --log_performance
```

### Opzioni disponibili

| Opzione | Descrizione |
|---------|-------------|
| `--fps N` | Imposta il frame rate target a N FPS (default: 10). È limitato agli FPS del video sorgente: la velocità di riproduzione è sempre corretta |
| `--audio` | Abilita la riproduzione audio sincronizzata |
| `--no-loop` | Disattiva il loop automatico del video |
| `--log_fps` | Abilita la visualizzazione delle statistiche FPS |
| `--log_performance` | Registra dati dettagliati sulle performance |
| `--verbose` | Mostra messaggi di log dettagliati nel terminale |
| `--batch_size N` | Dimensione del batch per l'elaborazione dei frame in modalità standard (default: 1) |
| `--high-performance` | Pre-converte l'intero video in una cache binaria su disco e lo riproduce da lì (vedi sotto) |
| `--palette TYPE` | Scelta della palette caratteri: "basic", "standard" (default), "extended", "box" o "custom" |
| `--custom-palette FILE` | File con caratteri personalizzati quando palette="custom" |

## 🏗️ Architettura

Entrambe le modalità condividono la stessa conversione (`ascii_converter.py`), lo stesso output su terminale (`terminal_display.py`) e lo stesso ciclo di riproduzione (`playback.py`), che gira sempre nel processo principale.

**Modalità standard** (default), pipeline multi-processo:

1. **Processo di lettura frame**: decodifica il video, scarta i frame per raggiungere gli FPS effettivi e li ridimensiona
2. **Processo di conversione ASCII**: trasforma i frame in stringhe di byte ASCII colorate
3. **Riproduzione (processo principale)**: cadenza i frame su scadenze assolute, ridisegna solo le righe cambiate e mantiene sincronizzato l'audio

**Modalità high-performance** (`--high-performance`): tutti i frame vengono convertiti in anticipo da un pool di processi in una cache binaria, poi riprodotti mappando in memoria il file della cache.

### Modalità high-performance

```bash
python main.py mio_video.mp4 300 --fps 60 --palette extended --high-performance
```

La cache si trova in `cached_frames/<video>_<hash>/` (`frames.bin`, `index.npy`, `metadata.json`). L'hash comprende percorso, dimensione e data di modifica del video, larghezza, altezza, FPS effettivi, palette e versione del convertitore: cambiare uno di questi valori crea una nuova cache. Se la pre-elaborazione fallisce, il player ripiega sulla modalità standard. Un Ctrl+C durante la pre-elaborazione non lascia cache parziali utilizzabili.

**Le cache create da versioni precedenti (`frame_*.txt` + `metadata.pkl`) vengono ignorate: cancella a mano le vecchie sottocartelle di `cached_frames/`.**

### Struttura del codice

- `main.py`: Punto di ingresso dell'applicazione, gestione degli argomenti CLI, palette
- `ascii_converter.py`: Conversione vettorizzata da frame ad ASCII colorato (run-length dei colori, output in byte)
- `terminal_display.py`: Output su terminale con aggiornamento per riga (delta)
- `playback.py`: `FrameDecimator`, `PlaybackClock` e ciclo `run_playback`
- `fps_stats.py`: Statistiche e grafico degli FPS
- `pipeline.py`: Processi di lettura/conversione della modalità standard
- `precomputer.py`: Cache binaria e pre-elaborazione della modalità high-performance
- `audio_player.py`: Gestione della riproduzione audio sincronizzata
- `calibration_frame.py`: Generazione del frame di calibrazione per il terminale
- `utils.py`: Funzioni di utilità e sistema di logging
- `bench.py`: Benchmark del convertitore

## 🔍 Note tecniche

### Performance

- La conversione dei frame è vettorizzata con NumPy e unisce le sequenze di colori uguali per ridurre i byte scritti
- Vengono riscritte solo le righe del terminale cambiate rispetto al frame precedente
- La modalità standard usa code bloccanti senza perdite tra i processi: nessun frame viene perso

### Controllo FPS

- Gli FPS effettivi sono `min(--fps, FPS sorgente)`, quindi la velocità di riproduzione è sempre corretta
- I frame sono cadenzati su scadenze assolute; i frame vengono saltati solo quando il disegno è in ritardo
- L'audio va in loop insieme al video e si risincronizza sul tempo del video

### Test

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## 📊 Visualizzazione delle performance

Quando l'opzione `--log_fps` è attivata, viene visualizzato un grafico in tempo reale che mostra:
- FPS corrente
- Stabilità del frame rate
- Grafico degli ultimi tempi di frame con rilevamento outlier

![Performance Graph](https://github.com/Cioscos/Pixellator/blob/performance/doc/stats.png)

## 🎨 Palette caratteri disponibili

- **Basic**: 10 caratteri (`" .:-=+*#%@"`)
- **Standard**: 42 caratteri, più dettagliata (default)
- **Extended**: 67 caratteri, massimo dettaglio
- **Box**: Utilizza blocchi Unicode colorati per la massima densità visiva
- **Custom**: Definita dall'utente tramite file di testo

## 📄 Licenza

Questo progetto è distribuito con licenza MIT. Consulta il file `LICENSE` per ulteriori dettagli.

## 👥 Contributi

Contributi, segnalazioni di bug e richieste di funzionalità sono benvenuti.
Per contribuire:

1. Forka il repository
2. Crea un nuovo branch (`git checkout -b feature/amazing-feature`)
3. Effettua le modifiche
4. Commit le modifiche (`git commit -m 'Add amazing feature'`)
5. Push sul branch (`git push origin feature/amazing-feature`)
6. Apri una Pull Request

---

*Creato con ❤️ per gli amanti dell'ASCII art e dell'ottimizzazione*