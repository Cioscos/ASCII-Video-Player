"""Statistiche FPS e grafico dei tempi di frame, condivisi dai loop di riproduzione."""
from collections import deque

BOLD = '\033[1m'  # Testo in grassetto
RESET = '\033[0m'  # Reset stile
GREEN = '\033[32m'  # Testo verde
YELLOW = '\033[33m'  # Testo giallo
RED = '\033[31m'  # Testo rosso
BLUE = '\033[34m'  # Testo blu

GRAPH_WIDTH = 30  # Larghezza massima del grafico in caratteri
BLOCK_CHARS = " ▁▂▃▄▅▆▇█"


class FpsStats:
    """Raccoglie gli intervalli tra frame e produce le righe di overlay."""

    def __init__(self, target_fps=None, history=100, graph_points=50):
        self.target_fps = target_fps
        self.graph_points = graph_points
        self._times = deque(maxlen=history)

    def record(self, interval_s):
        """Registra l'intervallo (in secondi) tra due frame mostrati."""
        if interval_s > 0:
            self._times.append(interval_s)

    def overlay_lines(self, frame_label):
        """Righe dell'overlay (senza newline né codici di cancellazione riga)."""
        if not self._times:
            return ["Calcolando FPS..."]
        graph = self._graph()
        return [self._stats_line(frame_label)] + graph.split("\n")

    def summary(self):
        """Riepilogo finale su tutta la cronologia disponibile."""
        if not self._times:
            return "FPS medio: 0.00, Min: 0.00, Max: 0.00"
        avg_time = sum(self._times) / len(self._times)
        avg_fps = 1.0 / avg_time
        min_fps = 1.0 / max(self._times)
        max_fps = 1.0 / min(self._times)
        return f"FPS medio: {avg_fps:.2f}, Min: {min_fps:.2f}, Max: {max_fps:.2f}"

    def _stats_line(self, frame_label):
        recent_times = list(self._times)[-10:]  # Ultimi 10 frame
        avg_time = sum(recent_times) / len(recent_times)
        current_fps = 1.0 / avg_time

        # Deviazione standard per la stabilità
        if len(recent_times) > 1:
            variance = sum((t - avg_time) ** 2 for t in recent_times) / len(recent_times)
            stability = 1 - min(1, (variance ** 0.5) / avg_time)  # 0=instabile, 1=stabile
        else:
            stability = 1

        target_fps = self.target_fps
        fps_ratio = None
        if target_fps:
            fps_ratio = current_fps / target_fps
            if 0.95 < fps_ratio < 1.05:  # Entro 5% del target
                fps_color = GREEN
            elif fps_ratio > 0.8:  # Almeno 80% del target
                fps_color = YELLOW
            else:
                fps_color = RED
        else:
            fps_color = GREEN

        bar_len = int(stability * 10)
        stability_bar = f"[{'|' * bar_len}{' ' * (10 - bar_len)}]"

        stats = (
            f"{BOLD}FPS: {fps_color}{current_fps:.1f}{RESET} "
            f"| Stabilità: {stability_bar} "
        )
        if target_fps:
            stats += f"| Target: {target_fps} FPS "
            stats += f"| Ratio: {fps_ratio:.2f} "
        stats += f"| {frame_label}"
        return stats

    def _graph(self):
        """Grafico ASCII dei tempi di frame con gestione degli outlier (MAD)."""
        recent = list(self._times)[-self.graph_points:]
        if len(recent) < 2:
            return "Raccolta dati per il grafico..."

        width = GRAPH_WIDTH
        times_ms = [t * 1000 for t in recent]
        target_time_ms = (1000 / self.target_fps) if self.target_fps else None

        original_min = min(times_ms)
        original_max = max(times_ms)

        # Rilevamento outlier con MAD
        times_sorted = sorted(times_ms)
        median = times_sorted[len(times_sorted) // 2]
        mad = sum(abs(x - median) for x in times_ms) / len(times_ms)
        threshold = 3.5 * mad
        has_outliers = any(abs(t - median) > threshold for t in times_ms)

        filtered_times = times_ms
        if has_outliers:
            filtered_times = [t for t in times_ms if abs(t - median) <= threshold]
            if not filtered_times:
                filtered_times = times_sorted[:int(len(times_ms) * 0.8)]

        min_time = min(filtered_times)
        max_time = max(filtered_times)

        # Intervallo minimo
        if max_time - min_time < 0.5:
            mid_value = (max_time + min_time) / 2
            min_time = mid_value - 0.5
            max_time = mid_value + 0.5

        # Intestazione
        if has_outliers:
            header = f"{BOLD}Frame Time (ms){RESET}: min={original_min:.1f} max={original_max:.1f}"
            if target_time_ms:
                header += f" target={target_time_ms:.1f}"
            header += f" {RED}[outliers rilevati]{RESET}"
        else:
            header = f"{BOLD}Frame Time (ms){RESET}: min={min_time:.1f} max={max_time:.1f}"
            if target_time_ms:
                header += f" target={target_time_ms:.1f}"

        # Limita i punti alla larghezza
        if len(times_ms) > width:
            stride = len(times_ms) // width
            samples = [times_ms[i] for i in range(0, len(times_ms), stride)][:width]
        else:
            samples = times_ms[-width:]

        graph_line = ""
        for time_ms in samples:
            if abs(time_ms - median) > threshold:
                graph_line += f"{RED}█{RESET}"  # Outlier: blocco rosso
                continue
            normalized = max(0, min(1, (time_ms - min_time) / (max_time - min_time)))
            block_index = min(int(normalized * (len(BLOCK_CHARS) - 1)), len(BLOCK_CHARS) - 1)
            if target_time_ms is not None:
                if time_ms < target_time_ms * 0.85:  # Molto veloce
                    color = BLUE
                elif time_ms < target_time_ms * 1.1:  # Vicino al target
                    color = GREEN
                elif time_ms < target_time_ms * 1.5:  # Leggermente lento
                    color = YELLOW
                else:  # Molto lento
                    color = RED
            else:
                color = GREEN
            graph_line += f"{color}{BLOCK_CHARS[block_index]}{RESET}"

        # Scala in basso
        min_label = f"{min_time:.1f}ms" if min_time < 10 else f"{min_time:.0f}ms"
        max_label = f"{max_time:.1f}ms" if max_time < 10 else f"{max_time:.0f}ms"
        spaces = max(1, width - len(min_label) - len(max_label))
        scale = f"{min_label}{' ' * spaces}{max_label}"
        if has_outliers and original_max > max_time:
            scale += f" [{RED}max: {original_max:.1f}ms{RESET}]"

        return f"{header}\n{graph_line}\n{scale}"
