import re
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime


def parse_log_file(file_path):
    """Analizza un file di log e estrae informazioni sui frame, tempi di conversione e rendering."""
    frame_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - Frame (\d+) - Conversion: (\d+\.\d+) ms, Total Rendering: (\d+\.\d+) ms')
    fps_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - \[LOG\] FPS display .+?: (\d+\.\d+)')

    data = []
    fps_data = []

    with open(file_path, 'r') as file:
        for line in file:
            frame_match = frame_pattern.match(line)
            fps_match = fps_pattern.match(line)

            if frame_match:
                timestamp_str, frame_num, conversion_time, rendering_time = frame_match.groups()
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                data.append({
                    'timestamp': timestamp,
                    'frame': int(frame_num),
                    'conversion_time': float(conversion_time),
                    'rendering_time': float(rendering_time),
                    'total_time': float(conversion_time) + float(rendering_time)
                })
            elif fps_match:
                timestamp_str, fps = fps_match.groups()
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                fps_data.append({
                    'timestamp': timestamp,
                    'fps': float(fps)
                })

    return pd.DataFrame(data), pd.DataFrame(fps_data)


def compute_statistics(df):
    """Calcola statistiche sui dati di rendering."""
    stats = {
        'conversion_time': {
            'mean': df['conversion_time'].mean(),
            'median': df['conversion_time'].median(),
            'min': df['conversion_time'].min(),
            'max': df['conversion_time'].max(),
            'std': df['conversion_time'].std()
        },
        'rendering_time': {
            'mean': df['rendering_time'].mean(),
            'median': df['rendering_time'].median(),
            'min': df['rendering_time'].min(),
            'max': df['rendering_time'].max(),
            'std': df['rendering_time'].std()
        },
        'total_time': {
            'mean': df['total_time'].mean(),
            'median': df['total_time'].median(),
            'min': df['total_time'].min(),
            'max': df['total_time'].max(),
            'std': df['total_time'].std()
        },
        'fps_calculated': 1000 / df['total_time'].mean() if df['total_time'].mean() > 0 else float('inf')
    }

    return stats


def identify_outliers(df, threshold=2.0):
    """Identifica i frame che sono outlier (hanno tempi di rendering molto alti)."""
    mean = df['rendering_time'].mean()
    std = df['rendering_time'].std()

    outliers = df[df['rendering_time'] > mean + threshold * std]
    return outliers


def detect_performance_issues(df):
    """Rileva potenziali problemi di performance."""
    issues = []

    # Cerca frame con tempo di rendering zero (potenziali errori)
    zero_rendering = df[df['rendering_time'] == 0]
    if not zero_rendering.empty:
        issues.append(f"Rilevati {len(zero_rendering)} frame con tempo di rendering zero.")

    # Cerca spikes nel tempo di rendering
    mean_rendering = df['rendering_time'].mean()
    spikes = df[df['rendering_time'] > 3 * mean_rendering]
    if not spikes.empty:
        issues.append(f"Rilevati {len(spikes)} frame con tempo di rendering 3x superiore alla media.")

    # Cerca pattern di rallentamento
    if len(df) > 10:
        first_half = df.iloc[:len(df) // 2]['rendering_time'].mean()
        second_half = df.iloc[len(df) // 2:]['rendering_time'].mean()
        if second_half > first_half * 1.5:
            issues.append(
                f"Rilevato rallentamento del rendering: la seconda metà dei frame è {second_half / first_half:.2f}x più lenta.")

    return issues


def visualize_data(df, fps_df, output_dir):
    """Crea visualizzazioni utili per analizzare le performance."""
    os.makedirs(output_dir, exist_ok=True)

    # Grafico 1: Tempo di conversione e rendering per frame
    plt.figure(figsize=(12, 6))
    plt.plot(df['frame'], df['conversion_time'], label='Tempo di Conversione (ms)')
    plt.plot(df['frame'], df['rendering_time'], label='Tempo di Rendering (ms)')
    plt.xlabel('Numero Frame')
    plt.ylabel('Tempo (ms)')
    plt.title('Tempi di Conversione e Rendering per Frame')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'conversion_rendering_times.png'))

    # Grafico 2: Distribuzione dei tempi di rendering
    plt.figure(figsize=(10, 6))
    plt.hist(df['rendering_time'], bins=20, alpha=0.7)
    plt.xlabel('Tempo di Rendering (ms)')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione dei Tempi di Rendering')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'rendering_time_distribution.png'))

    # Grafico 3: FPS nel tempo (se disponibile)
    if not fps_df.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(fps_df)), fps_df['fps'])
        plt.xlabel('Misurazione')
        plt.ylabel('FPS')
        plt.title('FPS nel Tempo')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'fps_over_time.png'))

    # Grafico 4: Boxplot dei tempi
    plt.figure(figsize=(10, 6))
    plt.boxplot([df['conversion_time'], df['rendering_time'], df['total_time']],
                labels=['Conversione', 'Rendering', 'Totale'])
    plt.ylabel('Tempo (ms)')
    plt.title('Distribuzione dei Tempi')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'time_distributions_boxplot.png'))


def compare_segments(df, segment_size=15):
    """Confronta segmenti di frame per identificare cambiamenti nelle performance."""
    if len(df) < segment_size * 2:
        return "Non ci sono abbastanza frame per confrontare i segmenti."

    num_segments = len(df) // segment_size
    segments = []

    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size
        segment_df = df.iloc[start_idx:end_idx]

        segment_stats = {
            'segment': i + 1,
            'frame_start': segment_df['frame'].min(),
            'frame_end': segment_df['frame'].max(),
            'avg_conversion': segment_df['conversion_time'].mean(),
            'avg_rendering': segment_df['rendering_time'].mean(),
            'avg_total': segment_df['total_time'].mean(),
            'fps_calculated': 1000 / segment_df['total_time'].mean() if segment_df['total_time'].mean() > 0 else float(
                'inf')
        }
        segments.append(segment_stats)

    return pd.DataFrame(segments)


def main():
    parser = argparse.ArgumentParser(description='Analizza log di rendering e fornisce metriche di performance.')
    parser.add_argument('log_file', help='Percorso del file di log da analizzare')
    parser.add_argument('--output', default='output', help='Directory dove salvare i risultati e i grafici')
    args = parser.parse_args()

    print(f"Analisi del file: {args.log_file}")
    df, fps_df = parse_log_file(args.log_file)

    if df.empty:
        print("Nessun dato di frame trovato nel file.")
        return

    # Calcola e mostra statistiche
    stats = compute_statistics(df)
    print("\n=== STATISTICHE GENERALI ===")
    print(f"Numero di frame analizzati: {len(df)}")
    print("\nTempo di Conversione (ms):")
    print(f"  Media: {stats['conversion_time']['mean']:.2f}")
    print(f"  Mediana: {stats['conversion_time']['median']:.2f}")
    print(f"  Min: {stats['conversion_time']['min']:.2f}")
    print(f"  Max: {stats['conversion_time']['max']:.2f}")
    print(f"  Deviazione Standard: {stats['conversion_time']['std']:.2f}")

    print("\nTempo di Rendering (ms):")
    print(f"  Media: {stats['rendering_time']['mean']:.2f}")
    print(f"  Mediana: {stats['rendering_time']['median']:.2f}")
    print(f"  Min: {stats['rendering_time']['min']:.2f}")
    print(f"  Max: {stats['rendering_time']['max']:.2f}")
    print(f"  Deviazione Standard: {stats['rendering_time']['std']:.2f}")

    print("\nTempo Totale (ms):")
    print(f"  Media: {stats['total_time']['mean']:.2f}")
    print(f"  Mediana: {stats['total_time']['median']:.2f}")
    print(f"  Min: {stats['total_time']['min']:.2f}")
    print(f"  Max: {stats['total_time']['max']:.2f}")
    print(f"  Deviazione Standard: {stats['total_time']['std']:.2f}")

    print(f"\nFPS Calcolati: {stats['fps_calculated']:.2f}")
    if not fps_df.empty:
        print(f"FPS Registrati: {fps_df['fps'].iloc[-1]:.2f}")

    # Identifica outlier
    outliers = identify_outliers(df)
    if not outliers.empty:
        print(f"\n=== OUTLIER RILEVATI ===")
        print(f"Numero di frame outlier: {len(outliers)}")
        print("Frame outlier:", outliers['frame'].tolist())

    # Rileva problemi di performance
    issues = detect_performance_issues(df)
    if issues:
        print("\n=== PROBLEMI DI PERFORMANCE RILEVATI ===")
        for issue in issues:
            print(f"- {issue}")

    # Confronta segmenti
    print("\n=== CONFRONTO SEGMENTI ===")
    segments_comparison = compare_segments(df)
    if isinstance(segments_comparison, str):
        print(segments_comparison)
    else:
        print(segments_comparison.to_string(index=False))

    # Crea visualizzazioni
    visualize_data(df, fps_df, args.output)
    print(f"\nGrafici salvati nella directory: {args.output}")

    # Salva i dati analizzati
    df.to_csv(os.path.join(args.output, 'analyzed_frames.csv'), index=False)
    if not fps_df.empty:
        fps_df.to_csv(os.path.join(args.output, 'fps_data.csv'), index=False)

    print("\nAnalisi completata!")


if __name__ == "__main__":
    main()
