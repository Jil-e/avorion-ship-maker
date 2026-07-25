"""Voxel scaffolding for the corpus-driven generator.

The generator shapes a ship on an integer voxel grid (occupancy + a per-voxel role
label), then :func:`greedy_merge` collapses runs of identically-attributed voxels
into as few axis-aligned boxes as possible so the exported ship has a sane block
count instead of one block per voxel.
"""
from __future__ import annotations

import numpy as np

EMPTY = -1


class VoxelGrid:
    """A ``(X, Y, Z)`` grid: X=port/starboard, Y=down/up, Z=aft/bow."""

    def __init__(self, x: int, y: int, z: int):
        self.X, self.Y, self.Z = int(x), int(y), int(z)
        self.occ = np.zeros((self.X, self.Y, self.Z), dtype=bool)
        self.role = np.full((self.X, self.Y, self.Z), EMPTY, dtype=np.int32)

    # -- construction -----------------------------------------------------
    def fill_profile(self, half_w: np.ndarray, half_h: np.ndarray,
                     boxiness: float, role: int) -> None:
        """Fill a solid hull whose cross-section at slice ``k`` is a superellipse
        of half-width ``half_w[k]`` and half-height ``half_h[k]`` (voxels).

        ``boxiness`` 0 -> elliptical, 1 -> nearly rectangular.
        """
        cx = (self.X - 1) / 2.0
        cy = (self.Y - 1) / 2.0
        p = 2.0 + float(np.clip(boxiness, 0, 1)) * 8.0
        ii = np.abs(np.arange(self.X) - cx)
        jj = np.abs(np.arange(self.Y) - cy)
        IX, JY = np.meshgrid(ii, jj, indexing="ij")  # (X, Y)
        for k in range(self.Z):
            hw = max(half_w[k], 0.001)
            hh = max(half_h[k], 0.001)
            metric = (IX / hw) ** p + (JY / hh) ** p
            mask = metric <= 1.0
            self.occ[:, :, k] |= mask
            self.role[:, :, k][mask] = role

    def fill_tube(self, cx, cy, hw, hh, p, role: int, z0: int = 0,
                  hh_dn=None, p_dn=None) -> None:
        """Fill a superellipse tube slice by slice, starting at slice ``z0``.

        All of ``cx, cy, hw, hh, p`` may be scalars or per-slice arrays; the
        tube length is the longest array. Unlike :meth:`fill_profile` the
        centre can move per slice, so offset hulls / nacelles / keels work.
        ``hh_dn``/``p_dn`` optionally give the BELOW-centreline half-height and
        exponent, for vertically asymmetric hulls (flat belly, shaped dorsal).
        """
        if hh_dn is None:
            hh_dn = hh
        if p_dn is None:
            p_dn = p
        args = [np.atleast_1d(np.asarray(a, float))
                for a in (cx, cy, hw, hh, p, hh_dn, p_dn)]
        n = max(len(a) for a in args)
        cx, cy, hw, hh, p, hh_dn, p_dn = (np.broadcast_to(a, (n,)) for a in args)
        ii = np.arange(self.X, dtype=float)
        jj = np.arange(self.Y, dtype=float)
        for idx in range(n):
            k = z0 + idx
            if k < 0 or k >= self.Z:
                continue
            w = max(float(hw[idx]), 0.001)
            dy = jj - float(cy[idx])
            up = dy >= 0
            IX = np.abs(ii - float(cx[idx]))[:, None] / w
            hu, pu = max(float(hh[idx]), 0.001), float(p[idx])
            hd, pd = max(float(hh_dn[idx]), 0.001), float(p_dn[idx])
            m_up = (IX ** pu + (np.abs(dy)[None, :] / hu) ** pu <= 1.0) & up[None, :]
            m_dn = (IX ** pd + (np.abs(dy)[None, :] / hd) ** pd <= 1.0) & ~up[None, :]
            mask = m_up | m_dn
            self.occ[:, :, k] |= mask
            self.role[:, :, k][mask] = role

    def paint_box(self, xr, yr, zr, role: int, only_if_empty: bool = False) -> None:
        """Set an inclusive voxel box ``[x0:x1, y0:y1, z0:z1]`` to ``role``.

        Ranges are ``(lo, hi)`` inclusive and clamped to the grid.
        """
        x0, x1 = self._clamp(xr, self.X)
        y0, y1 = self._clamp(yr, self.Y)
        z0, z1 = self._clamp(zr, self.Z)
        if x0 > x1 or y0 > y1 or z0 > z1:
            return
        sub_occ = self.occ[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]
        sub_role = self.role[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]
        if only_if_empty:
            mask = ~sub_occ
            sub_role[mask] = role
            sub_occ[mask] = True
        else:
            sub_occ[...] = True
            sub_role[...] = role

    def set_role_where(self, mask: np.ndarray, role: int) -> None:
        self.role[mask & self.occ] = role

    def _clamp(self, rng, n):
        lo, hi = rng
        return max(0, int(lo)), min(n - 1, int(hi))

    # -- queries ----------------------------------------------------------
    def shell_mask(self) -> np.ndarray:
        """Voxels that have at least one empty face-neighbour (the outer skin)."""
        occ = self.occ
        exposed = np.zeros_like(occ)
        # pad with False and compare shifts on all 6 faces
        for axis in (0, 1, 2):
            for shift in (1, -1):
                nb = np.roll(occ, shift, axis=axis)
                # rolled-in border counts as empty (outside the hull)
                idx = [slice(None)] * 3
                idx[axis] = 0 if shift == 1 else -1
                nb[tuple(idx)] = False
                exposed |= occ & ~nb
        return exposed

    def interior_mask(self) -> np.ndarray:
        return self.occ & ~self.shell_mask()


def greedy_merge(attr: np.ndarray):
    """Merge a ``(X,Y,Z)`` int array (``EMPTY``=-1 for holes) into boxes.

    Returns a list of ``(i0, i1, j0, j1, k0, k1, attr_id)`` inclusive boxes that
    exactly tile the non-empty voxels, greedily grown x -> y -> z.
    """
    X, Y, Z = attr.shape
    used = np.zeros((X, Y, Z), dtype=bool)
    boxes = []
    for k in range(Z):
        for j in range(Y):
            for i in range(X):
                a = attr[i, j, k]
                if a == EMPTY or used[i, j, k]:
                    continue
                # grow +x
                i1 = i
                while i1 + 1 < X and attr[i1 + 1, j, k] == a and not used[i1 + 1, j, k]:
                    i1 += 1
                # grow +y (whole x-span must match & be free)
                j1 = j
                while j1 + 1 < Y and np.all(attr[i:i1 + 1, j1 + 1, k] == a) \
                        and not np.any(used[i:i1 + 1, j1 + 1, k]):
                    j1 += 1
                # grow +z (whole x/y face must match & be free)
                k1 = k
                while k1 + 1 < Z and np.all(attr[i:i1 + 1, j:j1 + 1, k1 + 1] == a) \
                        and not np.any(used[i:i1 + 1, j:j1 + 1, k1 + 1]):
                    k1 += 1
                used[i:i1 + 1, j:j1 + 1, k:k1 + 1] = True
                boxes.append((i, i1, j, j1, k, k1, int(a)))
    return boxes
