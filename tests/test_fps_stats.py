from fps_stats import FpsStats


def test_empty():
    assert FpsStats(30).overlay_lines("Frame 0") == ["Calcolando FPS..."]


def test_steady_30fps():
    s = FpsStats(30)
    for _ in range(20):
        s.record(1 / 30)
    lines = s.overlay_lines("Frame 20")
    assert "30.0" in lines[0] and "Frame 20" in lines[0] and len(lines) == 4
    assert s.summary().startswith("FPS medio: 30.00")


def test_un_solo_campione_graph_in_raccolta():
    s = FpsStats(None)
    s.record(0.05)
    lines = s.overlay_lines("Frame 1")
    assert lines[1] == "Raccolta dati per il grafico..."


def test_outlier_evidenziato():
    s = FpsStats(30)
    for _ in range(19):
        s.record(1 / 30)
    s.record(0.5)
    lines = s.overlay_lines("Frame 20")
    assert "outliers rilevati" in lines[1]
    assert len(lines) == 4


def test_history_limitata():
    s = FpsStats(30, history=10)
    for _ in range(50):
        s.record(1 / 30)
    assert len(s._times) == 10
