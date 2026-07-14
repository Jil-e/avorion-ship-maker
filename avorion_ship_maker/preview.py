"""Render a :class:`ShipModel` to a matplotlib 3D figure (voxel-box preview).

Each block is drawn as a shaded box (top faces brighter, undersides darker) so
the silhouette reads clearly. Ship axes are remapped so length runs along the
horizontal and "up" is vertical, matching how ships look in the game.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless-safe; Gradio embeds the Figure directly

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .model import ShipModel
from .orient import CORNER_TYPES, EDGE_TYPES, corner_faces, edge_faces

# the 6 faces of a unit box as vertex-index quads (ship-space corner order below)
_CUBE_FACES = [
    (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
    (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4),
]


def _rgb(color: str) -> tuple[float, float, float]:
    c = color[-6:] if len(color) >= 6 else "808080"
    return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _remap(p):
    """Ship (x,y,z) -> plot (x,z,y) so length is horizontal and +y is up."""
    return (p[0], p[2], p[1])


def _shade(base, verts):
    """Brightness from the face normal's ship-space +y component (top-lit)."""
    v = [np.asarray(w, float) for w in verts[:3]]
    n = np.cross(v[1] - v[0], v[2] - v[0])
    ny = n[1] / (np.linalg.norm(n) + 1e-9)
    b = 0.55 + 0.42 * max(0.0, ny) + 0.12 * max(0.0, -ny)
    return tuple(min(1.0, c * b) for c in base)


def _block_faces(b):
    """Return (polygons_in_plot_coords, per-face colors) for a block.

    Cubes render as boxes; Edge/Corner shape blocks render as real prisms so the
    preview matches the in-game silhouette.
    """
    lo = (b.lx, b.ly, b.lz)
    hi = (b.ux, b.uy, b.uz)
    base = _rgb(b.color)
    if b.index in EDGE_TYPES:
        faces_ship = edge_faces(lo, hi, b.look, b.up)
    elif b.index in CORNER_TYPES:
        faces_ship = corner_faces(lo, hi, b.look, b.up)
    else:
        x0, y0, z0, x1, y1, z1 = b.lx, b.ly, b.lz, b.ux, b.uy, b.uz
        v = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1),
             (x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)]
        faces_ship = [[v[i] for i in f] for f in _CUBE_FACES]
    polys, colors = [], []
    for face in faces_ship:
        colors.append(_shade(base, face))
        polys.append([_remap(p) for p in face])
    return polys, colors


def render_3d(ship: ShipModel, elev: float = 22, azim: float = -58,
              figsize: float = 6.0, bg: str = "#0d1117") -> "plt.Figure":
    """Return a matplotlib Figure with a shaded 3D voxel preview of ``ship``."""
    fig = plt.figure(figsize=(figsize, figsize * 0.75), facecolor=bg)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(bg)

    if ship.blocks:
        all_polys, all_colors = [], []
        for b in ship.blocks:
            q, c = _block_faces(b)
            all_polys.extend(q)
            all_colors.extend(c)
        coll = Poly3DCollection(all_polys, facecolors=all_colors,
                                edgecolors=(0, 0, 0, 0.18), linewidths=0.15)
        ax.add_collection3d(coll)

        (lx, ly, lz), (hx, hy, hz) = ship.bounds()
        xs, zs, ys = (lx, hx), (lz, hz), (ly, hy)
        ax.set_xlim(xs); ax.set_ylim(zs); ax.set_zlim(ys)
        # equal aspect from real extents
        ax.set_box_aspect((max(hx - lx, 1), max(hz - lz, 1), max(hy - ly, 1)))

    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_proj_type("persp")
    except Exception:
        pass
    fig.tight_layout(pad=0)
    return fig


def save_preview(ship: ShipModel, path: str, **kw) -> str:
    fig = render_3d(ship, **kw)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return path
