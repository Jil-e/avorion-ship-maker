"""Build the ``parent`` hierarchy Avorion stores for each block.

Every ``<item>`` in a ship plan records the index of the block it is attached to
(``parent``), forming a spanning tree rooted at a block with ``parent="-1"``.
We derive that tree from face-adjacency so the emitted ship is one connected
structure. Two boxes are considered connected when they share a real face:
they touch/overlap on all three axes and genuinely overlap (positive area) on
at least two of them.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .blocks import SHAPE_VARIANTS
from .model import Block

# shape-block type ids (wedges/corners) — legal but odd as a root block
_SHAPE_IDX = {i for pair in SHAPE_VARIANTS.values() for i in pair}


def build_parents(blocks: list[Block], eps: float = 1e-4) -> list[int]:
    """Return a list ``parents`` where ``parents[i]`` is the list-index of block
    ``i``'s parent, or ``-1`` for the single root. Length == ``len(blocks)``."""
    n = len(blocks)
    if n == 0:
        return []
    if n == 1:
        return [-1]

    lo = np.array([[b.lx, b.ly, b.lz] for b in blocks], dtype=float)
    hi = np.array([[b.ux, b.uy, b.uz] for b in blocks], dtype=float)
    centers = (lo + hi) / 2.0
    vol = np.prod(hi - lo, axis=1)

    # Root: prefer a solid full block CONTAINING the origin — the game anchors
    # a design at its root block (a founder ship's root is a cube at 0,0,0),
    # so a canonical centre root keeps "apply design" predictable. Fall back
    # to a central, sizeable block.
    contains0 = ((lo <= eps) & (hi >= -eps)).all(axis=1)
    is_shape = np.array([b.index in _SHAPE_IDX for b in blocks])
    score = (np.abs(centers[:, 0]) + np.abs(centers[:, 2]) - 0.001 * vol
             - 1000.0 * contains0 + 500.0 * is_shape)
    root = int(np.argmin(score))

    parents = [-1] * n
    visited = np.zeros(n, dtype=bool)
    visited[root] = True
    dq: deque[int] = deque([root])

    while dq:
        i = dq.popleft()
        ov = np.minimum(hi[i], hi) - np.maximum(lo[i], lo)  # per-axis overlap
        touch_all = (ov >= -eps).all(axis=1)                # boxes meet on every axis
        area_axes = (ov > eps).sum(axis=1)                  # axes with real overlap
        conn = touch_all & (area_axes >= 2)
        conn[i] = False
        for j in np.where(conn & ~visited)[0]:
            visited[j] = True
            parents[int(j)] = i
            dq.append(int(j))

    # Attach any stragglers (disconnected pieces) to the nearest visited block.
    if not visited.all():
        vis_idx = np.where(visited)[0]
        for j in np.where(~visited)[0]:
            d = np.abs(centers[vis_idx] - centers[j]).sum(axis=1)
            nearest = int(vis_idx[int(np.argmin(d))])
            parents[int(j)] = nearest
            visited[j] = True

    return parents
