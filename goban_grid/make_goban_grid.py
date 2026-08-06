#!/usr/bin/env python3
"""
Generate print-ready vector art for a traditional 19x19 goban grid.

All geometry is in millimetres. Lines are drawn as FILLED RECTANGLES (not
strokes) and star points as filled circles, so line weight can never be
altered by scaling or by a stroke-weight setting in the RIP.

Traditional Japanese proportions: the grid is deliberately NOT square.
"""

from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

# ---- Parameters (edit these to match the board) --------------------------
N = 19          # lines per side
SPACING_X = 22.0    # mm, line-to-line across the board's width (short axis)
SPACING_Y = 23.7    # mm, line-to-line along the board's depth (long axis)
LINE_W = 1.0        # mm, line width
HOSHI_D = 4.0       # mm, star point diameter
HOSHI_IDX = (3, 9, 15)   # 0-based: 4th, 10th, 16th lines
MARGIN = 30.0       # mm, blank artboard margin around the grid
OUT = "/mnt/user-data/outputs"
# --------------------------------------------------------------------------

GRID_W = (N - 1) * SPACING_X
GRID_H = (N - 1) * SPACING_Y
PAGE_W = GRID_W + LINE_W + 2 * MARGIN
PAGE_H = GRID_H + LINE_W + 2 * MARGIN

# centre coordinates of each line, measured from the artboard origin
X0 = MARGIN + LINE_W / 2
Y0 = MARGIN + LINE_W / 2
xs = [X0 + i * SPACING_X for i in range(N)]
ys = [Y0 + i * SPACING_Y for i in range(N)]


def f(v):
    """Trim float formatting."""
    return f"{v:.4f}".rstrip("0").rstrip(".")


def write_svg(path):
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        f'width="{f(PAGE_W)}mm" height="{f(PAGE_H)}mm" '
        f'viewBox="0 0 {f(PAGE_W)} {f(PAGE_H)}">',
        f'<title>Goban grid {N}x{N} — {f(SPACING_X)} x {f(SPACING_Y)} mm</title>',
        '<g id="grid" fill="#000000" stroke="none" shape-rendering="crispEdges">',
    ]
    # vertical lines
    parts.append('<g id="vertical-lines">')
    for x in xs:
        parts.append(
            f'<rect x="{f(x - LINE_W/2)}" y="{f(MARGIN)}" '
            f'width="{f(LINE_W)}" height="{f(GRID_H + LINE_W)}"/>'
        )
    parts.append("</g>")
    # horizontal lines
    parts.append('<g id="horizontal-lines">')
    for y in ys:
        parts.append(
            f'<rect x="{f(MARGIN)}" y="{f(y - LINE_W/2)}" '
            f'width="{f(GRID_W + LINE_W)}" height="{f(LINE_W)}"/>'
        )
    parts.append("</g>")
    # star points
    parts.append('<g id="star-points">')
    for i in HOSHI_IDX:
        for j in HOSHI_IDX:
            parts.append(
                f'<circle cx="{f(xs[i])}" cy="{f(ys[j])}" r="{f(HOSHI_D/2)}"/>'
            )
    parts.append("</g>")
    parts += ["</g>", "</svg>", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(parts))


def write_pdf(path):
    c = canvas.Canvas(path, pagesize=(PAGE_W * mm, PAGE_H * mm))
    c.setTitle(f"Goban grid {N}x{N}")
    c.setFillColorRGB(0, 0, 0)
    c.setStrokeAlpha(0)
    for x in xs:
        c.rect((x - LINE_W / 2) * mm, MARGIN * mm,
               LINE_W * mm, (GRID_H + LINE_W) * mm, stroke=0, fill=1)
    for y in ys:
        c.rect(MARGIN * mm, (y - LINE_W / 2) * mm,
               (GRID_W + LINE_W) * mm, LINE_W * mm, stroke=0, fill=1)
    for i in HOSHI_IDX:
        for j in HOSHI_IDX:
            c.circle(xs[i] * mm, ys[j] * mm, (HOSHI_D / 2) * mm,
                     stroke=0, fill=1)
    c.showPage()
    c.save()


if __name__ == "__main__":
    write_svg(f"{OUT}/goban-grid-19x19.svg")
    write_pdf(f"{OUT}/goban-grid-19x19.pdf")
    print(f"grid image area (outer edges of outer lines): "
          f"{GRID_W + LINE_W:.1f} x {GRID_H + LINE_W:.1f} mm")
    print(f"artboard: {PAGE_W:.1f} x {PAGE_H:.1f} mm")
