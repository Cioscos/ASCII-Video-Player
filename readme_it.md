# ASCII Video Player

![ASCII Video Preview](https://github.com/Cioscos/Pixellator/blob/performance/doc/intro.png)

## 📝 Descrizione

**ASCII Video Player** è un convertitore video in ASCII art real-time ad alte prestazioni che visualizza i video nel terminale. Utilizza una pipeline parallela ottimizzata per convertire e visualizzare i frame, supportando la riproduzione audio sincronizzata e diverse palette di caratteri.

## ✨ Funzionalità

- **Riproduzione video in ASCII art** nel terminale con supporto a colori
- **Pipeline di elaborazione parallela** per prestazioni ottimali
- **Sincronizzazione audio-video** con gestione adattiva dei ritardi
- **Multiple palette di caratteri ASCII** (basic, standard, extended, box o custom)
- **Sistema di calibrazione automatico** per adattarsi alle dimensioni del terminale
- **Monitoraggio performance in tempo reale** con visualizzazione grafica degli FPS
- **Gestione adattiva del carico** per mantenere la sincronizzazione anche su hardware meno potenti

## 🔧 Requisiti

- Python 3.6+
- Terminale con supporto ai colori ANSI (la maggior parte dei terminali moderni)
- Dipendenze Python:
  - OpenCV (`opencv-python`)
  - NumPy
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
   pip install opencv-python numpy

   # Con supporto audio
   pip install opencv-python numpy sounddevice moviepy
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
| `--fps N` | Imposta il frame rate target a N FPS (default: 10) |
| `--audio` | Abilita la riproduzione audio sincronizzata |
| `--no-loop` | Disattiva il loop automatico del video |
| `--log_fps` | Abilita la visualizzazione delle statistiche FPS |
| `--log_performance` | Registra dati dettagliati sulle performance |
| `--verbose` | Mostra messaggi di log dettagliati nel terminale |
| `--batch_size N` | Dimensione del batch per l'elaborazione dei frame (default: 1) |
| `--palette TYPE` | Scelta della palette caratteri: "basic", "standard", "extended", "box" o "custom" |
| `--custom-palette FILE` | File con caratteri personalizzati quando palette="custom" |

## 🏗️ Architettura

Il sistema utilizza una pipeline parallela multi-processo con:

1. **Processo di lettura frame**: estrae i frame dal video a frequenza controllata
2. **Processo di conversione ASCII**: trasforma i frame in rappresentazioni ASCII colorate
3. **Thread di rendering**: visualizza i frame ASCII sul terminale e gestisce la sincronizzazione

### Struttura del codice

- `main.py`: Punto di ingresso dell'applicazione, gestione degli argomenti CLI
- `pipeline.py`: Implementazione della pipeline parallela di elaborazione video
- `audio_player.py`: Gestione della riproduzione audio sincronizzata
- `calibration_frame.py`: Generazione del frame di calibrazione per il terminale
- `renderer.py`: Rendering ottimizzato dei frame ASCII
- `terminal_output_buffer.py`: Buffer ottimizzato per l'output del terminale
- `utils.py`: Funzioni di utilità e sistema di logging

## 🔍 Note tecniche

### Performance

- La conversione dei frame utilizza algoritmi vettorizzati NumPy per massimizzare l'efficienza
- Le ottimizzazioni includono pre-calcolo delle lookup tables, minimizzazione delle allocazioni di memoria e operazioni vettorizzate
- Il sistema di rendering utilizza un buffer ottimizzato per minimizzare le operazioni I/O sul terminale
- La gestione adattiva del carico rileva e si adatta automaticamente alla capacità di elaborazione della macchina host

### Controllo FPS

Il player tenta di mantenere il frame rate target specificato, con le seguenti strategie:
- Estrazione frame a velocità controllata
- Gestione del timing di rendering per mantenere FPS consistenti
- Monitoraggio della performance in tempo reale e adattamento dinamico

## 📊 Visualizzazione delle performance

Quando l'opzione `--log_fps` è attivata, viene visualizzato un grafico in tempo reale che mostra:
- FPS corrente
- Stabilità del frame rate
- Grafico degli ultimi tempi di frame con rilevamento outlier

![Performance Graph](https://github.com/Cioscos/Pixellator/blob/performance/doc/stats.png)

## 🎨 Palette caratteri disponibili

- **Basic**: 10 caratteri (`" .:-=+*#%@"`)
- **Standard**: 42 caratteri, più dettagliata
- **Extended**: 70 caratteri, massimo dettaglio
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