# Goban grid — artwork spec

Files: `goban-grid-19x19.pdf` (open in Illustrator, then Save As `.ai`) and
`goban-grid-19x19.svg` (identical geometry).

## Geometry — all dimensions final, do not scale

| Item | Value |
|---|---|
| Grid | 19 × 19 lines |
| Line spacing, short axis (board width) | 22.0 mm |
| Line spacing, long axis (board depth) | 23.7 mm |
| Line width | 0.8 mm |
| Star points (hoshi) | 4.0 mm diameter, 9 total, on the 4th/10th/16th lines |
| Image area (outer edge to outer edge of outer lines) | 397.0 × 427.6 mm |
| Artboard | 457.0 × 487.6 mm (30 mm blank margin all round) |

**The grid is intentionally not square.** The 22.0 × 23.7 mm rectangle is
traditional — it makes the stones read as evenly spaced when seen from a
player's angle. Do not "correct" it to a square grid, and do not scale the
file non-uniformly.

## Artwork construction

- All lines are **filled rectangles**, not stroked paths, and star points are
  filled circles. Nothing depends on a stroke-weight setting, so line width
  cannot drift during output.
- Single color, 100% K / solid black. No overprint, no trapping, no bleed.
- No text, no live effects, no clipping masks, no raster elements.
- Orient the long axis (23.7 mm spacing) front-to-back as the players sit.

## Printing notes

- Print on **bare, planed wood before any oil, wax, or topcoat** — finish
  applied first will cause the ink to sit on top or wick unevenly.
- Ask for an ink rated for wood with good adhesion under a later oil/wax
  topcoat; confirm the topcoat won't lift or bleed the ink.
- Request a test pull on an offcut from the same board before the real run —
  open-grain wood can wick ink along the grain and thicken lines.
- Registration is one-hit (single screen, single color), so squareness to the
  board edges is set by how the board is jigged, not by the art.

## Adjusting the file

`make_goban_grid.py` regenerates both files. Edit the parameters at the top
(spacing, line width, star point size, margin) and rerun — useful if the
board's finished size after planing calls for a slightly different grid.
