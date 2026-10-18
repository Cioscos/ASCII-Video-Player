"""Funzioni pure per la rotazione del video in riproduzione."""

from ascii_converter import auto_width, compute_ascii_height

# Angoli (gradi, senso orario) in ordine di ciclo
ROTATIONS = (0, 90, 180, 270)


def rotated_dimensions(src_w, src_h, rotation, width, term_cols, term_rows, reserved_rows):
    """
    Dimensioni ASCII (larghezza, altezza in caratteri) del video ruotato.

    A 90/270 gradi larghezza e altezza sorgente si scambiano. Con `width=None`
    la larghezza è calcolata da `auto_width`; una larghezza esplicita a 90/270
    viene ridotta se non entra nel terminale.
    """
    if rotation in (90, 270):
        src_w, src_h = src_h, src_w
    fit = auto_width(src_w, src_h, term_cols, term_rows, reserved_rows)
    if width is None:
        w = fit
    elif rotation in (90, 270):
        w = min(width, fit)
    else:
        w = width
    return w, compute_ascii_height(src_w, src_h, w)
