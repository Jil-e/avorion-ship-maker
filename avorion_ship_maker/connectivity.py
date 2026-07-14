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

from .model import Block


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

    # Root: a central, sizeable block (near the x=0 / z=0 spine).
    score = np.abs(centers[:, 0]) + np.abs(centers[:, 2]) - 0.001 * vol
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
