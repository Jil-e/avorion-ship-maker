"""Turn a :class:`ShipSpec` into a :class:`ShipModel` of Avorion blocks.

Pipeline:  profile the hull -> add parts (engines, bridge, wings, fins) ->
label every voxel with a role -> resolve roles to armour skin / inner hull /
glow / tech -> mirror for symmetry -> merge voxels into boxes -> emit blocks.
"""
from __future__ import annotations

import random

import numpy as np

from .. import orient
from ..blocks import Role, SHAPE_VARIANTS, index_for
from ..model import Block, ShipModel
from .palette import contrast, mix, shade, to_argb
from .spec import LAYOUTS, ShipSpec
from .voxel import EMPTY, VoxelGrid, greedy_merge

# ---- voxel role codes (internal) ---------------------------------------
(R_HULL, R_ARMOR, R_ENGINE, R_ENGINE_GLOW, R_BRIDGE, R_WING, R_FIN,
 R_GLOW, R_ACCENT, R_GEN, R_GYRO, R_THRUST, R_CARGO, R_CREW,
 R_BATTERY, R_INTEGRITY, R_SHIELD, R_DIRTHRUST, R_TURRET,
 R_PANEL) = range(20)


# ---- hull silhouette ----------------------------------------------------

# bow shapes: (name, w_end, h_end, w_exp, h_exp) — how width/height die out
_NOSE_KINDS = [
    ("needle", 0.10, 0.32, 0.85, 1.20),   # classic sharp cone
    ("chisel", 0.16, 0.80, 0.90, 1.60),   # narrows, keeps height (vertical blade)
    ("shovel", 0.62, 0.16, 1.50, 0.80),   # flattens, keeps width (flat scoop)
    ("blunt",  0.50, 0.55, 1.70, 1.70),   # short rounded ram
]

# layout -> weight, per hull class (picked from the seed when spec.layout is auto)
_LAYOUT_WEIGHTS = {
    "fighter":    (("mono", 4), ("pods", 3), ("twin", 2), ("fork", 3)),
    "corvette":   (("mono", 4), ("pods", 3), ("twin", 2), ("keel", 1), ("fork", 2)),
    "frigate":    (("mono", 3), ("pods", 3), ("keel", 2), ("twin", 1), ("fork", 2), ("hammer", 1)),
    "cruiser":    (("mono", 3), ("keel", 3), ("pods", 2), ("twin", 1), ("hammer", 2), ("fork", 1)),
    "battleship": (("mono", 3), ("keel", 4), ("pods", 1), ("twin", 1), ("hammer", 3)),
    "freighter":  (("mono", 3), ("twin", 3), ("keel", 2), ("pods", 1), ("hammer", 2), ("crab", 2)),
    "miner":      (("mono", 2), ("twin", 2), ("keel", 2), ("pods", 2), ("hammer", 1), ("fork", 3), ("crab", 4)),
    "carrier":    (("mono", 2), ("pods", 4), ("twin", 2), ("keel", 1), ("hammer", 3)),
    "station":    (("mono", 1),),
}


def _choose_layout(spec: ShipSpec, rng: random.Random) -> str:
    if spec.layout in LAYOUTS:
        return spec.layout
    opts = _LAYOUT_WEIGHTS.get(spec.hull_class, _LAYOUT_WEIGHTS["frigate"])
    return rng.choices([n for n, _ in opts], weights=[w for _, w in opts])[0]


def _piecewise(vals, cuts, Z, hard, blend):
    """Per-slice array from per-segment values; soft joints get a cosine blend,
    hard joints stay as steps (terraced hulls)."""
    raw = np.empty(Z)
    edges = [0] + list(cuts) + [Z]
    for i, v in enumerate(vals):
        raw[edges[i]:edges[i + 1]] = v
    for i, c in enumerate(cuts):
        if hard[i]:
            continue
        lo, hi = max(0, c - blend), min(Z, c + blend)
        if hi - lo < 2:
            continue
        s = (1 - np.cos(np.linspace(0, np.pi, hi - lo))) / 2
        raw[lo:hi] = vals[i] * (1 - s) + vals[i + 1] * s
    return raw


def _hull_arrays(spec: ShipSpec, rng: random.Random):
    """Per-slice half-width/height, vertical offset and superellipse exponent.

    The hull is a chain of 2-5 random segments, each with its own cross-section
    size, vertical offset and boxiness, joined by steps or blends, finished with
    a randomly picked bow shape — so seeds give genuinely different silhouettes,
    not one jittered tube.
    """
    Z = spec.length
    base_hw = max((spec.width - 1) / 2.0, 1.0)
    base_hh = max((spec.height - 1) / 2.0, 1.0)
    cyber = spec.style == "cyberpunk"
    box = float(np.clip(spec.boxiness, 0, 1))
    if cyber:                            # CP2077: slab-sided, no soft curves
        box = max(box, 0.85)

    # longitudinal segments
    nseg = int(np.clip(2 + Z // 16 + rng.randint(0, 1), 2, 5))
    w = [0.5 + rng.random() for _ in range(nseg)]
    cum = np.cumsum(w)
    cuts = sorted({int(np.clip(Z * c / cum[-1], 2, Z - 3)) for c in cum[:-1]})
    nseg = len(cuts) + 1
    wm = [0.72 + 0.45 * rng.random() for _ in range(nseg)]
    hm = [0.68 + 0.50 * rng.random() for _ in range(nseg)]
    wm = [v / max(wm) for v in wm]          # widest segment uses the full width
    hm = [v / max(hm) for v in hm]
    yo = [(rng.random() - 0.45) * 0.30 * base_hh for _ in range(nseg)]
    pd = [(rng.random() - 0.5) * 0.30 for _ in range(nseg)]
    hard = [rng.random() < 0.25 + 0.55 * box for _ in cuts]  # boxy => terraces
    blend = max(2, int(Z * (0.06 + 0.05 * rng.random())))

    hw = base_hw * _piecewise(wm, cuts, Z, hard, blend)
    hh = base_hh * _piecewise(hm, cuts, Z, hard, blend)
    yoff = _piecewise(yo, cuts, Z, hard, blend)
    pexp = 2.0 + np.clip(box + _piecewise(pd, cuts, Z, hard, blend), 0.05, 1.0) * 8.0

    zc = np.arange(Z) / max(Z - 1, 1)        # 0 = stern, 1 = bow
    # bow taper with a per-seed nose style (boxy hulls avoid needle noses)
    kinds = _NOSE_KINDS if box < 0.75 else _NOSE_KINDS[1::2]
    if spec.hull_class == "miner":       # flat working noses (shovel / ram)
        kinds = _NOSE_KINDS[2:]
    _, we, he, pw, ph = kinds[rng.randrange(len(kinds))]
    nose = float(np.clip(spec.nose * (0.8 + 0.5 * rng.random()), 0.05, 0.9))
    t = np.clip((zc - (1 - nose)) / nose, 0, 1)
    hw = hw * (1.0 - (1.0 - we) * t ** pw)
    hh = hh * (1.0 - (1.0 - he) * t ** ph)
    # stern taper
    tail = float(np.clip(spec.taper_tail * (0.7 + 0.7 * rng.random()), 0.0, 0.5))
    if tail > 0:
        tt = np.clip(zc / tail, 0, 1)
        hw = hw * (0.80 + 0.20 * tt)
        hh = hh * (0.86 + 0.14 * tt)
    # mid-body bulge (smooth; NO high-frequency waviness — at voxel resolution
    # it rounds to lone one-voxel bumps that read as glitches)
    bulge = 0.03 + 0.05 * rng.random()
    hw = hw * ((1 - bulge) + bulge * np.sin(np.pi * zc))
    if cyber:
        # facet the profiles into wide flat bands with hard steps — the
        # smooth taper otherwise reads as a 1-voxel staircase, not CP2077
        qw = max(1.5, base_hw * 0.16)
        qh = max(1.5, base_hh * 0.16)
        # rounding may bump a band above the nominal size — clip so the
        # hull never outgrows the grid the caller sized from spec.width
        hw = np.clip(np.round(hw / qw) * qw, 1.0, base_hw)
        hh = np.clip(np.round(hh / qh) * qh, 1.0, base_hh)
        yoff = np.round(yoff)            # flat decks, no vertical drift
        pexp = np.full_like(pexp, 9.0)   # hard rectangular cross-section
    return hw, hh, yoff, pexp


def _paint_hull(g: VoxelGrid, spec: ShipSpec, layout: str, rng: random.Random) -> None:
    """Fill the hull occupancy for the chosen layout archetype."""
    Z = g.Z
    cx = (g.X - 1) / 2.0
    cy = (g.Y - 1) / 2.0
    base_hw = max((spec.width - 1) / 2.0, 1.0)
    base_hh = max((spec.height - 1) / 2.0, 1.0)
    hw, hh, yoff, pexp = _hull_arrays(spec, rng)

    # most real ships are not vertically symmetric: flat-ish belly, shaped top
    belly_f = (0.82 + 0.15 * rng.random()) if rng.random() < 0.6 else None

    def belly(up_hh, p_arr):
        """kwargs for fill_tube: a flatter, slightly shallower lower half."""
        if belly_f is None:
            return {}
        return dict(hh_dn=up_hh * belly_f, p_dn=p_arr + 2.5)

    def ramp(n):
        return (1 - np.cos(np.linspace(0, np.pi, max(n, 2)))) / 2

    if layout == "twin":
        sub = hw * 0.42
        off = hw * 0.58
        g.fill_tube(cx - off, cy + yoff, sub, hh * 0.9, pexp, R_HULL)
        g.fill_tube(cx + off, cy + yoff, sub, hh * 0.9, pexp, R_HULL)
        # central connector deck bridging the two hulls
        z0 = int(Z * (0.28 + 0.08 * rng.random()))
        z1 = min(Z - 1, int(Z * (0.62 + 0.12 * rng.random())))
        sl = slice(z0, z1 + 1)
        n = z1 + 1 - z0
        e = np.minimum(np.minimum(ramp(n) * 3, ramp(n)[::-1] * 3), 1.0)
        g.fill_tube(cx, cy + yoff[sl] + hh[sl] * 0.15,
                    (off[sl] + sub[sl] * 0.6) * e,
                    hh[sl] * 0.38 * (0.4 + 0.6 * e), 6.0, R_HULL, z0=z0)
    elif layout == "pods":
        g.fill_tube(cx, cy + yoff, hw * 0.74, hh, pexp, R_HULL)
        # engine nacelles hugging the flanks, spindle-shaped
        n_hw = max(1.0, base_hw * (0.15 + 0.05 * rng.random()))
        z1 = min(Z - 1, int(Z * (0.45 + 0.25 * rng.random())))
        n = z1 + 1
        t = np.linspace(0, 1, n)
        prof = np.clip(np.minimum(t * 6, (1 - t) * 3), 0, 1) ** 0.7
        n_off = base_hw - n_hw
        ncy = cy + (rng.random() - 0.6) * 0.3 * base_hh
        g.fill_tube(cx - n_off, ncy, n_hw * prof, n_hw * 1.1 * prof, 2.4, R_HULL)
        g.fill_tube(cx + n_off, ncy, n_hw * prof, n_hw * 1.1 * prof, 2.4, R_HULL)
        # pylons tying the nacelles to the hull
        for f in (0.30, 0.70):
            k = int(z1 * f)
            g.paint_box((int(cx - n_off), int(cx + n_off) + 1),
                        (int(ncy), int(ncy) + 1),
                        (k, k + max(1, Z // 30)), R_HULL, only_if_empty=True)
    elif layout == "hammer":
        # slimmer body + a wide transverse bow section (Hammerhead style)
        g.fill_tube(cx, cy + yoff, hw * 0.72, hh, pexp, R_HULL, **belly(hh, pexp))
        z0h = int(Z * (0.72 + 0.08 * rng.random()))
        z1h = min(Z - 1, int(Z * 0.96))
        sl = slice(z0h, z1h + 1)
        n = z1h + 1 - z0h
        t = np.linspace(0.0, 1.0, max(n, 2))
        cap = np.clip(np.minimum(t * 4, (1 - t) * 2.5), 0, 1) ** 0.5
        head_w = base_hw * (0.9 + 0.1 * rng.random())
        g.fill_tube(cx, cy + yoff[sl], head_w * (0.72 + 0.28 * cap),
                    np.maximum(hh[sl] * 0.8, 1.2), 6.0, R_HULL, z0=z0h)
    elif layout == "fork":
        # blunt main body + two prongs converging toward the bow
        zc = int(Z * (0.52 + 0.12 * rng.random()))
        body = slice(0, zc + 1)
        g.fill_tube(cx, cy + yoff[body], np.maximum(hw[body], base_hw * 0.55),
                    hh[body], pexp[body], R_HULL, **belly(hh[body], pexp[body]))
        sl = slice(max(0, zc - 2), Z)
        p_hw = np.maximum(hw[sl] * 0.30, 1.0)
        # keep a visible slot between the prongs even on narrow rolled hulls
        off = np.maximum(np.maximum(hw[sl] * 0.62, base_hw * 0.30), p_hw + 1.6)
        p_hh = np.maximum(hh[sl] * 0.45, 1.0)
        ncy = cy + yoff[sl]
        g.fill_tube(cx - off, ncy, p_hw, p_hh, 2.6, R_HULL, z0=sl.start)
        g.fill_tube(cx + off, ncy, p_hw, p_hh, 2.6, R_HULL, z0=sl.start)
        # palm: a solid web across the prong roots — on thin hulls the slot
        # floor (p_hw + 1.6) pushes the prongs clear off the body and the
        # ship split into three pieces
        for i in range(min(3, zc + 1 - sl.start)):
            k = sl.start + i
            g.paint_box((int(round(cx - off[i])), int(round(cx + off[i]))),
                        (int(round(ncy[i] - p_hh[i])), int(round(ncy[i] + p_hh[i]))),
                        (k, k), R_HULL, only_if_empty=True)
    elif layout == "crab":
        # industrial crab: a broad flat carapace aft + two claw arms that
        # converge toward the bow like pincers (mining-rig look)
        zc = int(Z * (0.52 + 0.10 * rng.random()))     # carapace front
        body = slice(0, zc + 1)
        g.fill_tube(cx, cy + yoff[body],
                    np.maximum(hw[body], base_hw * 0.80),
                    np.maximum(hh[body] * 0.82, 1.2), pexp[body] + 2.0,
                    R_HULL, **belly(np.maximum(hh[body] * 0.82, 1.2),
                                    pexp[body] + 2.0))
        sl = slice(max(0, int(zc - Z * 0.08)), Z)
        n = Z - sl.start
        t = np.linspace(0.0, 1.0, max(n, 2))
        c_hw = np.maximum(hw[sl] * 0.26, 1.2)          # claw thickness
        # shoulders wide at the root, pincer tips converging near the bow
        conv = 0.30 + 0.12 * rng.random()
        off = base_hw * (0.80 - conv * t ** 0.8)
        c_hh = np.maximum(hh[sl] * 0.50, 1.0)
        ccy = cy + yoff[sl] - hh[sl] * (0.10 + 0.15 * t)   # claws sag forward
        # freeze the sag over the pincer tip: the slit-cut jaws must line
        # up slice to slice or the lower jaw ends up floating
        jaw = max(2, int(Z * 0.07))
        ccy[max(0, n - jaw - 1):] = ccy[max(0, n - jaw - 1)]
        g.fill_tube(cx - off, ccy, c_hw, c_hh, 2.4, R_HULL, z0=sl.start)
        g.fill_tube(cx + off, ccy, c_hw, c_hh, 2.4, R_HULL, z0=sl.start)
        # pincer tips: a horizontal slit splits each claw into open jaws
        for i in range(max(0, n - jaw), n):
            k = sl.start + i
            for sgn in (-1, 1):
                xc = cx + sgn * off[i]
                x0 = max(0, int(round(xc - c_hw[i])))
                x1 = min(g.X - 1, int(round(xc + c_hw[i])))
                yj = int(round(ccy[i]))
                if 0 <= yj < g.Y:
                    g.occ[x0:x1 + 1, yj, k] = False
    elif layout == "keel":
        cym = cy + yoff - base_hh * 0.12
        g.fill_tube(cx, cym, hw, hh * 0.78, pexp, R_HULL, **belly(hh * 0.78, pexp))
        # dorsal superstructure straddling the hull top, sloped at both ends
        z0 = int(Z * (0.12 + 0.08 * rng.random()))
        z1 = min(Z - 1, int(Z * (0.60 + 0.18 * rng.random())))
        sl = slice(z0, z1 + 1)
        n = z1 + 1 - z0
        r = ramp(n)
        prof = np.minimum(np.minimum(r * 4, r[::-1] * 2.5), 1.0)
        rv = base_hh * 0.30 * prof + 0.8
        kcy = (cym[sl] + hh[sl] * 0.78) + 0.6 * rv   # bottom sinks 0.4*rv into hull
        g.fill_tube(cx, kcy, hw[sl] * (0.30 + 0.10 * rng.random()),
                    rv, 4.0, R_HULL, z0=z0)
    else:  # mono
        g.fill_tube(cx, cy + yoff, hw, hh, pexp, R_HULL, **belly(hh, pexp))


def _carve(g: VoxelGrid, xr, yr, zr, role: int) -> None:
    """Recolour an existing (occupied) box region without adding volume."""
    x0, x1 = g._clamp(xr, g.X)
    y0, y1 = g._clamp(yr, g.Y)
    z0, z1 = g._clamp(zr, g.Z)
    if x0 > x1 or y0 > y1 or z0 > z1:
        return
    m = g.occ[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]
    g.role[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1][m] = role


def _add_engines(g: VoxelGrid, spec: ShipSpec, hull: tuple[int, int],
                 rng: random.Random, wall: bool = False) -> None:
    """Carve engine bays into the stern. Engines are distributed over the
    occupied x-clusters there, so twin hulls / nacelles each get their own.

    ``wall=True`` (industrial hulls) trades the small round nozzles for a
    monolithic engine wall: each stern cluster becomes a few BIG engine
    slabs spanning its full cross-section, glow face across the whole stern
    — the way players build miners/freighters (CAT-style, few large blocks).
    """
    n = max(int(spec.engines), 0)
    if n == 0:
        return
    hX, hY = hull
    Z = g.Z
    elen = max(2, int(round(Z * (0.10 + 0.06 * rng.random()))))
    cy = (g.Y - 1) // 2
    band = g.occ[:, max(0, cy - max(1, hY // 4)):cy + max(1, hY // 4) + 1, 1:3]
    xs = np.where(band.any(axis=(1, 2)))[0]
    if not len(xs):
        return
    runs, s, p = [], int(xs[0]), int(xs[0])
    for x in xs[1:]:
        if x == p + 1:
            p = int(x)
            continue
        runs.append((s, p))
        s = p = int(x)
    runs.append((s, p))
    if wall:
        elen = max(3, int(round(Z * 0.15)))
        for a, b in runs:
            # full occupied y-extent of this cluster near the stern
            ys = np.where(g.occ[a:b + 1, :, 1:elen + 1].any(axis=(0, 2)))[0]
            if not len(ys):
                continue
            _carve(g, (a, b), (int(ys[0]), int(ys[-1])), (0, elen), R_ENGINE)
            _carve(g, (a, b), (int(ys[0]), int(ys[-1])), (0, 0), R_ENGINE_GLOW)
        return
    widths = [b - a + 1 for a, b in runs]
    total = sum(widths)
    counts = ([max(1, round(n * w / total)) for w in widths]
              if len(runs) > 1 else [n])
    yr = (cy - max(1, hY // 4), cy + max(1, hY // 4))
    for (a, b), cnt in zip(runs, counts):
        w = b - a + 1
        mid = (a + b) / 2.0
        xcs = [mid] if cnt == 1 else list(np.linspace(mid - w * 0.33, mid + w * 0.33, cnt))
        rad = max(1, int(round(w / (cnt * 4))))
        for xc in xcs:
            xi = int(round(xc))
            _carve(g, (xi - rad, xi + rad), yr, (0, elen), R_ENGINE)
            # glowing exhaust face at the very stern
            _carve(g, (xi - rad, xi + rad), yr, (0, 0), R_ENGINE_GLOW)


def _add_bridge(g: VoxelGrid, spec: ShipSpec, hull: tuple[int, int],
                cab_h: int, rng: random.Random) -> None:
    """A raised cabin sitting on top of the hull (grid is padded above it)."""
    if not spec.bridge or cab_h <= 0:
        return
    hX, hY = hull
    cx = (g.X - 1) // 2
    zc = int(g.Z * (0.55 + 0.10 * rng.random()))
    zl = max(2, int(g.Z * 0.12))
    w = max(1, hX // 6)
    z1 = min(g.Z - 1, zc + zl - 1)
    # flat roof: every column fills from its own hull top up to a common height
    tops = _hull_tops(g, (cx - w, cx + w), (zc, z1))
    if not tops:
        return
    roof = min(tops.values()) + cab_h
    for (x, k), top in tops.items():
        if top + 1 <= roof:
            g.paint_box((x, x), (top + 1, roof), (k, k), R_BRIDGE, only_if_empty=True)
    # narrower, lower cockpit step in front (the bevel pass slopes it)
    step = _hull_tops(g, (cx - max(1, w // 2), cx + max(1, w // 2)),
                      (z1 + 1, min(g.Z - 1, z1 + max(1, zl // 2))))
    roof2 = min(roof, (min(step.values()) if step else roof) + max(1, cab_h // 2))
    for (x, k), top in step.items():
        if top + 1 <= roof2:
            g.paint_box((x, x), (top + 1, roof2), (k, k), R_BRIDGE, only_if_empty=True)


def _hull_tops(g: VoxelGrid, xr, zr) -> dict[tuple[int, int], int]:
    """Topmost occupied voxel per (x, z) column inside the given footprint."""
    tops = {}
    for k in range(max(0, int(zr[0])), min(g.Z, int(zr[1]) + 1)):
        for x in range(max(0, int(xr[0])), min(g.X, int(xr[1]) + 1)):
            col = np.where(g.occ[x, :, k])[0]
            if len(col):
                tops[(x, k)] = int(col[-1])
    return tops


def _add_turret_mounts(g: VoxelGrid, spec: ShipSpec, hull: tuple[int, int],
                       rng: random.Random) -> None:
    """Turret-base pads (block type 20) on flat dorsal hull spots.

    Pads are elongated fore-aft (the way players build them: ~1x2 plates
    along the mount axis) and each greedy-merges into a single block the
    game treats as a turret socket. Odd counts get a centreline pad; the
    rest go on the port flank — centred on the OUTERMOST occupied hull
    cluster, so on claw layouts (crab/fork) they land right on the claw
    tops — and the symmetry pass mirrors them into port/starboard pairs.
    """
    n = max(int(spec.turrets), 0)
    if n == 0:
        return
    hX, hY = hull
    cx = (g.X - 1) // 2
    p = max(1, min(3, hX // 9))          # pad width, voxels
    pl = max(2, 2 * p)                   # pad length (fore-aft), voxels
    ok_roles = (R_HULL, R_ARMOR)

    def place(x0, x1, z0, z1):
        """Put a pad on the footprint if it is on-hull and near-flat."""
        tops = _hull_tops(g, (x0, x1), (z0, z1))
        if len(tops) < (x1 - x0 + 1) * (z1 - z0 + 1):
            return False
        if max(tops.values()) - min(tops.values()) > 1:
            return False
        if any(g.role[x, t, k] not in ok_roles for (x, k), t in tops.items()):
            return False
        y = max(tops.values()) + 1
        if y >= g.Y:
            return False
        g.paint_box((x0, x1), (y, y), (z0, z1), R_TURRET, only_if_empty=True)
        return True

    want_pairs = n // 2
    want_center = n % 2
    w2 = p // 2                          # centre pads span 2*w2+1 columns
    for z0 in range(int(g.Z * 0.88) - pl, int(g.Z * 0.22), -(pl + 2)):
        if want_center + want_pairs == 0:
            break
        z1 = z0 + pl - 1
        if want_center and place(cx - w2, cx + w2, z0, z1):
            want_center = 0
            continue
        if want_pairs:
            row = np.where(g.occ[:, :, z0].any(axis=1))[0]
            left = row[row <= cx]
            if not len(left):
                continue
            # occupied x-clusters on the port side; the outermost one is the
            # claw / flank ridge — that's where players put miner pads
            runs, st, pr = [], int(left[0]), int(left[0])
            for x in left[1:]:
                if x == pr + 1:
                    pr = int(x)
                    continue
                runs.append((st, pr))
                st = pr = int(x)
            runs.append((st, pr))
            a, b = runs[0]
            xm = (a + b) // 2
            placed = place(max(a, xm - w2), min(b, max(a, xm - w2) + p - 1), z0, z1)
            if not placed:                       # fall back to the inner flank
                half_w = min(cx - int(row[0]), hX * 0.5)
                x1 = cx - max(2, int(half_w * 0.5))
                placed = place(x1 - p + 1, x1, z0, z1)
            if placed:
                want_pairs -= 1


# wing archetypes, modelled on classic sci-fi silhouettes:
#   swept   — planar swept-back wings, sometimes slight di-/anhedral (SC Sabre/Gladius)
#   forward — forward-swept raider look
#   delta   — wide triangular plates (Elite Cobra/Viper)
#   gull    — steep inner section, flat past the knee (Mustang / Lambda shuttle)
#   xfoil   — two angled blades per side, X from the front (X-wing / Buccaneer)
#   tipfin  — swept wing ending in a vertical winglet (Gladius / Cutlass)
#   tippod  — wing carrying an engine/weapon pod on the tip (Hornet / Firefly)
#   vpods   — not wings at all: stubby angled pylons carrying functional
#             module pods, engines first (Serenity / podracer outriggers)
_WING_KINDS = {"swept": 4, "forward": 2, "delta": 3, "gull": 3,
               "xfoil": 2, "tipfin": 3, "tippod": 2, "vpods": 3}
_WING_KINDS_UTILITY = {"swept": 3, "delta": 2, "tipfin": 2, "tippod": 3,
                       "vpods": 5}


def _add_pylon_pods(g: VoxelGrid, spec: ShipSpec, span: int,
                    hull: tuple[int, int], rng: random.Random) -> None:
    """Engine outriggers (the Serenity / podracer reference): not lifting
    wings but stubby pylons — planar, V or Λ — carrying an axis-aligned
    functional module pod. Combat hulls get engine nacelles with a glowing
    exhaust; utility hulls may roll a generator or cargo pod instead."""
    hX, hY = hull
    cy = (g.Y - 1) // 2
    Z = g.Z

    p_len = max(5, int(Z * (0.24 + 0.12 * rng.random())))   # pod length
    z_mid = int(Z * (0.28 + 0.20 * rng.random()))           # pylon station
    chord = max(3, int(Z * 0.10))
    z0 = max(0, z_mid - chord // 2)
    z1 = min(Z - 1, z0 + chord - 1)

    # attach to the widest flank over the pylon chord
    rootL = rootR = None
    for k in range(z0, z1 + 1):
        row = np.where(g.occ[:, cy, k])[0]
        if len(row):
            rootL = int(row[0]) if rootL is None else min(rootL, int(row[0]))
            rootR = int(row[-1]) if rootR is None else max(rootR, int(row[-1]))
    if rootL is None:
        return

    reach = max(3, int(span * (0.45 + 0.30 * rng.random())))   # strut length
    y_cap = max(2.0, hY * 0.55)
    slope = rng.choice((-1, -1, 1, 0)) * (0.35 + 0.50 * rng.random())
    slope = float(np.clip(slope, -y_cap / reach, y_cap / reach))
    th = max(2, hY // 6)                                       # strut thickness

    dy = 0
    for s in range(reach):                                     # the pylons
        # dy starts at ZERO: the flank was probed at cy, and the outermost
        # hull column often exists only there — starting one row off left
        # the whole strut+pod assembly floating
        dy = int(round(slope * s))
        for side, root in ((-1, rootL), (1, rootR)):
            x = root + side * (s + 1)
            g.paint_box((x, x), (cy + dy - th // 2, cy + dy + (th - 1) - th // 2),
                        (z0, z1), R_WING, only_if_empty=True)

    # the module pod, flush against the pylon tip, aligned with the hull
    pw = max(1, int(round(hY * 0.14 + 0.6)))                   # pod half-size
    pod_z0 = max(0, z_mid - int(p_len * (0.40 + 0.25 * rng.random())))
    pod_z1 = min(Z - 1, pod_z0 + p_len - 1)
    pcy = cy + dy
    content = "engine"
    if spec.hull_class in ("freighter", "miner", "carrier"):
        content = rng.choice(("engine", "engine", "gen", "cargo"))
    core = {"engine": R_ENGINE, "gen": R_GEN, "cargo": R_CARGO}[content]
    for side, root in ((-1, rootL), (1, rootR)):
        px0 = root + side * (reach + 1)                        # pod inner face
        pxa, pxb = sorted((px0, px0 + side * 2 * pw))
        # hull-coloured cowling with the functional module as its core
        g.paint_box((pxa, pxb), (pcy - pw, pcy + pw), (pod_z0, pod_z1),
                    R_HULL, only_if_empty=True)
        if content == "engine":                                # glowing exhaust
            g.paint_box((pxa + 1, pxb - 1), (pcy - pw + 1, pcy + pw - 1),
                        (pod_z0, pod_z1 - 1), core)
            g.paint_box((pxa, pxb), (pcy - pw, pcy + pw), (pod_z0, pod_z0),
                        R_ENGINE_GLOW)
        else:
            g.paint_box((pxa + 1, pxb - 1), (pcy - pw + 1, pcy + pw - 1),
                        (pod_z0 + 1, pod_z1 - 1), core)
            g.paint_box((pxa, pxb), (pcy - pw, pcy + pw),      # accent collar
                        (pod_z1 - 1, pod_z1 - 1), R_ACCENT)


def _add_wings(g: VoxelGrid, spec: ShipSpec, span: int, hull: tuple[int, int],
               rng: random.Random) -> None:
    """Wings protruding from the flanks; a per-seed archetype (see _WING_KINDS)."""
    if not spec.wings or span <= 0:
        return
    hX, hY = hull
    cy = (g.Y - 1) // 2
    if spec.wing_kind in _WING_KINDS:      # user forced a specific archetype
        kind = spec.wing_kind
    else:
        table = (_WING_KINDS_UTILITY
                 if spec.hull_class in ("freighter", "miner", "carrier", "station")
                 else _WING_KINDS)
        kind = rng.choices(list(table), weights=list(table.values()))[0]

    if kind == "vpods":                    # module outriggers, not wings
        _add_pylon_pods(g, spec, span, hull, rng)
        return

    z0 = int(g.Z * (0.20 + 0.16 * rng.random()))       # root trailing edge
    chord = max(4, int(g.Z * (0.22 + 0.18 * rng.random())))
    sweep = (0.5 + 0.8 * rng.random()) * chord         # tip shift sternward
    taper = 0.3 + 0.3 * rng.random()                   # tip chord / root chord
    if kind == "forward":
        sweep = -sweep * 0.8
    elif kind == "delta":
        chord = max(5, int(g.Z * (0.32 + 0.16 * rng.random())))
        sweep, taper = chord * 0.15, 0.15

    if kind == "xfoil":
        a = 0.35 + 0.35 * rng.random()
        blades = (a, -a)
        span = max(3, int(span * 0.85))
    elif kind == "gull":
        blades = (rng.choice((-1, 1)) * (0.45 + 0.35 * rng.random()),)
    else:                                              # mostly planar, slight tilt
        blades = (rng.choice((-1, 0, 0, 1)) * 0.25 * rng.random(),)
    knee = 0.35 + 0.25 * rng.random()                  # gull: kink along the span

    # vertical reach follows the HULL height, not the span: a flattened ship
    # must not sprout wings taller than itself (the height slider otherwise
    # looks dead — the ship's bbox stayed pinned by the tilted wings)
    y_cap = max(2.0, hY * 0.55)
    blades = tuple(float(np.clip(b, -y_cap / span, y_cap / span))
                   for b in blades)

    # solid plane: adjacent columns must overlap well, so cap the sweep rate
    sweep = float(np.clip(sweep, -span * 0.7, span * 0.7))

    # snapshot the hull flank BEFORE painting (wings must not grow off themselves)
    edges = {}
    for k in range(g.Z):
        row = np.where(g.occ[:, cy, k])[0]
        if len(row):
            edges[k] = (int(row[0]), int(row[-1]))
    root_ks = [k for k in range(max(0, z0), min(g.Z, z0 + chord)) if k in edges]
    if not root_ks:
        return
    # one straight root per side (outermost flank over the root chord)
    rootL = min(edges[k][0] for k in root_ks)
    rootR = max(edges[k][1] for k in root_ks)

    thick_root = 1 if hY < 6 else (2 if hY < 14 else 3)
    tips = []                                          # (dy, zs, ch) at the tip
    for slope in blades:
        prev_dy, prev_zs, prev_ch = 0, z0, chord
        for s in range(span):
            t = s / max(span - 1, 1)
            ch = max(2, int(round(chord * (1 - (1 - taper) * t))))
            zs = z0 - int(round(sweep * t))
            # keep consecutive columns z-overlapping (face-connected)
            zs = int(np.clip(zs, prev_zs - (ch - 1), prev_zs + prev_ch - 1))
            if kind == "gull":
                dy = int(round(slope * min(t, knee) * span))
            else:
                dy = int(round(slope * s))
            ylo, yhi = min(dy, prev_dy), max(dy, prev_dy)
            th = thick_root if t < 0.7 else 1
            k0, k1 = max(0, zs), min(g.Z - 1, zs + ch - 1)
            for side, root in ((-1, rootL), (1, rootR)):
                x = root + side * (s + 1)
                g.paint_box((x, x), (cy + ylo, cy + yhi + th - 1), (k0, k1),
                            R_WING, only_if_empty=True)
            prev_dy, prev_zs, prev_ch = dy, zs, ch
        tips.append((prev_dy, prev_zs, prev_ch))

    # root fairing: bridge any gap between the curved flank and the straight root
    for k in root_ks:
        eL, eR = edges[k]
        if eL > rootL:
            g.paint_box((rootL, eL - 1), (cy, cy + thick_root - 1), (k, k),
                        R_WING, only_if_empty=True)
        if eR < rootR:
            g.paint_box((eR + 1, rootR), (cy, cy + thick_root - 1), (k, k),
                        R_WING, only_if_empty=True)

    # wingtip features
    for dy, zs, ch in tips:
        for side, root in ((-1, rootL), (1, rootR)):
            x = root + side * span
            if kind == "tipfin":                       # vertical winglet
                fh = max(2, int(round(hY * 0.5)))
                g.paint_box((x, x), (cy + dy - 1, cy + dy + fh),
                            (max(0, zs), min(g.Z - 1, zs + max(2, ch))),
                            R_FIN, only_if_empty=True)
            elif kind == "tippod":                     # engine pod on the tip
                xin = x - side                         # one voxel back toward hull
                k0, k1 = max(0, zs - 2), min(g.Z - 1, zs + ch + 1)
                g.paint_box((min(x, xin), max(x, xin)),
                            (cy + dy - 1, cy + dy + 1), (k0, k1), R_ENGINE)
                g.paint_box((min(x, xin), max(x, xin)),
                            (cy + dy - 1, cy + dy + 1), (k0, k0), R_ENGINE_GLOW)


def _add_fins(g: VoxelGrid, spec: ShipSpec, fin_h: int, hull: tuple[int, int],
              rng: random.Random) -> None:
    """A dorsal tail fin: tall at the stern, sloping down toward the bow."""
    if not spec.fins or fin_h <= 0:
        return
    hX, hY = hull
    cx = (g.X - 1) // 2
    z0 = int(g.Z * (0.06 + 0.06 * rng.random()))
    zl = max(3, int(g.Z * (0.14 + 0.10 * rng.random())))
    w = max(1, hX // 10)
    for k in range(z0, min(g.Z, z0 + zl)):
        t = (k - z0) / max(zl - 1, 1)
        h = int(round(fin_h * (1.0 - 0.85 * t)))
        if h <= 0:
            continue
        for x in range(cx - w + 1, cx + w):
            col = np.where(g.occ[x, :, k])[0]
            if not len(col):
                continue
            top = int(col[-1])
            g.paint_box((x, x), (top + 1, top + h), (k, k), R_FIN, only_if_empty=True)


def _add_mining_rig(g: VoxelGrid, spec: ShipSpec, hull: tuple[int, int],
                    layout: str, rng: random.Random) -> None:
    """Miner identity: heavy-machinery language (CAT vibes, per the user's
    reference) — reinforced shoulder pedestals with turret pads where the
    manipulator arms plug in, a ventral dozer blade with a hazard-striped
    cutting lip, and low armour skirts. No cranes, no silo tubes."""
    if spec.hull_class != "miner":
        return
    hX, hY = hull
    X, Y, Z = g.X, g.Y, g.Z
    cx = (X - 1) // 2

    # manipulator shoulders: a chunky pedestal on the forward port flank
    # (mirrored by symmetry), crowned with a turret-base pad for the arm.
    # Try several stations — the first one clear of the bridge/engines wins.
    p = max(1, min(3, hX // 9))
    ph = max(2, hY // 6)
    for fz in (0.62 + 0.08 * rng.random(), 0.50, 0.40):
        kz = int(Z * fz)
        raw = _hull_tops(g, (0, cx), (kz - p, kz + p))
        # hull columns only: a pedestal must not stand on the cab or a wing
        tops = {q: j for q, j in raw.items()
                if g.role[q[0], j, q[1]] == R_HULL}
        # innermost port edge across the whole footprint span, so the
        # pedestal never hangs off a tapering flank (hammer bows narrow)
        edges = []
        for k in range(kz - p, kz + p + 1):
            cols = [x for x in range(cx + 1) if (x, k) in tops]
            if not cols:
                edges = None
                break
            edges.append(cols[0])
        if edges is None:
            continue
        x_lo = min(max(edges) + 1, cx - p - 1)
        foot = [(x, k) for x in range(x_lo, x_lo + p + 1)
                for k in range(kz - p, kz + p + 1)]
        if not all(q in tops for q in foot):
            continue
        deck = max(tops[q] for q in foot)
        if deck + ph + 1 > Y - 1:        # no headroom at this station
            continue
        for x, k in foot:                # solid riser up to a common top
            g.paint_box((x, x), (tops[(x, k)] + 1, deck + ph), (k, k),
                        R_HULL, only_if_empty=True)
        g.paint_box((x_lo, x_lo + p), (deck + ph, deck + ph),
                    (kz - p, kz + p), R_ACCENT)          # collar ring
        g.paint_box((x_lo, x_lo + p), (deck + ph + 1, deck + ph + 1),
                    (kz - p, kz + p), R_TURRET)          # the arm's pad
        break

    if rng.random() < 0.75 and layout != "fork":
        # ventral dozer blade, two plates thick, with an alternating hazard
        # lip like a CAT bucket edge. It runs ONLY along the flat wide
        # stretch of the belly ahead of midships — following a rising bow
        # made it climb the nose like a jagged staircase, and under a fork
        # it hung across the slot between the prongs.
        w = max(2, int(hX * 0.40))
        jmid = (Y - 1) // 2
        belly = {}
        for k in range(int(Z * 0.45), Z):
            col = np.where(g.occ[cx, :, k])[0]
            row = np.where(g.occ[:, jmid, k])[0]
            if len(col) and len(row) and (cx - row[0]) >= max(2, int(w * 0.8)):
                belly[k] = int(col[0])
        run = {k for k, j in belly.items()
               if belly and j <= min(belly.values()) + 1}
        if run:
            end = max(run)               # forward-most contiguous stretch
            start = end
            while start - 1 in run:
                start -= 1
            start = max(start, end - max(5, int(Z * 0.22)))
            if end - start >= 3:
                for k in range(start, end + 1):
                    jb = belly[k]
                    g.paint_box((cx - w, cx + w), (jb - 2, jb - 1), (k, k),
                                R_FIN, only_if_empty=True)
                jb = belly[end]          # striped cutting lip at the front
                for i, x in enumerate(range(cx - w, cx + w + 1)):
                    role = R_ACCENT if (i // 2) % 2 == 0 else R_FIN
                    g.paint_box((x, x), (jb - 3, jb - 3), (end - 1, end),
                                role, only_if_empty=True)

    if rng.random() < 0.6:
        # low armour skirts hugging the lower flanks (bumper plates), with
        # a horizontal accent stripe through the middle
        z0 = int(Z * (0.22 + 0.06 * rng.random()))
        z1 = int(Z * (0.58 + 0.08 * rng.random()))
        cyv = (Y - 1) // 2
        j0 = cyv - max(2, int(hY * 0.28))
        j1 = cyv - 1
        jm = (j0 + j1) // 2
        for k in range(z0, z1 + 1):
            for j in range(j0, j1 + 1):
                row = np.where(g.occ[:, j, k])[0]
                if len(row):
                    g.paint_box((row[0] - 1, row[0] - 1), (j, j), (k, k),
                                R_ACCENT if j == jm else R_FIN,
                                only_if_empty=True)


def _keep_main_component(g: VoxelGrid) -> None:
    """Keep only the voxel component containing the hull core.

    Carving and part placement on extreme rolls can strand fragments (a
    pincer jaw, a clipped pod); a multi-piece ship is invalid in game, so
    anything not attached to the main hull is dropped — the same guarantee
    _drop_floaters gives turret sections."""
    occ = g.occ
    if not occ.any():
        return
    idx = np.argwhere(occ)
    ctr = np.array([g.X / 2.0, g.Y / 2.0, g.Z / 2.0])
    seed = idx[int(np.argmin(np.abs(idx - ctr).sum(axis=1)))]
    region = np.zeros_like(occ)
    region[tuple(seed)] = True
    while True:
        grown = region
        for axis in (0, 1, 2):
            for shift in (1, -1):
                r = np.roll(region, shift, axis=axis)
                sl = [slice(None)] * 3
                sl[axis] = 0 if shift == 1 else -1
                r[tuple(sl)] = False
                grown = grown | r
        grown &= occ
        if grown.sum() == region.sum():
            break
        region = grown
    g.occ &= region


def _enforce_symmetry(g: VoxelGrid) -> None:
    half = g.X // 2
    if half == 0:
        return
    g.occ[g.X - half:] = g.occ[:half][::-1]
    g.role[g.X - half:] = g.role[:half][::-1]


def _resolve_roles(g: VoxelGrid, spec: ShipSpec, rng: random.Random) -> None:
    """Split generic hull voxels into outer armour vs inner hull, add glow lines
    and a few embedded tech blocks so the type mix resembles a real ship."""
    shell = g.shell_mask()
    hull = g.occ & (g.role == R_HULL)

    # outer skin -> armour, interior -> hull
    g.role[hull & shell] = R_ARMOR
    g.role[hull & ~shell] = R_HULL

    # surface detailing (glow lines + accent seam), density scaled by spec.detail
    cy = int((g.Y - 1) / 2.0)
    cx = int((g.X - 1) / 2.0)
    d = float(spec.detail)
    cyber = spec.style == "cyberpunk"
    if d > 0.15:
        spacing = max(2, int(round(4 - 2.0 * d)))   # denser dashes at high detail
        # cyberpunk runs its neon in unbroken full-length strips
        keep = np.ones(g.Z, bool) if cyber else \
            (np.arange(g.Z) % spacing == rng.randrange(spacing))
        # glow line along both flanks at mid height
        band = np.zeros_like(g.occ)
        band[:, cy, :] = True
        flank = band & shell & (g.role == R_ARMOR)
        flank[:, :, ~keep] = False
        g.role[flank] = R_GLOW
        # accent seam along the dorsal centreline
        if d > 0.4:
            top = np.zeros_like(g.occ)
            for k in range(g.Z):
                col = np.where(g.occ[cx, :, k])[0]
                if len(col):
                    top[cx, col[-1], k] = True
            seam = top & shell & (g.role == R_ARMOR)
            seam[:, :, ~keep] = False
            g.role[seam] = R_ACCENT
    if cyber:
        _cyber_trim(g, shell, rng)


def _cyber_trim(g: VoxelGrid, shell: np.ndarray, rng: random.Random) -> None:
    """Cyberpunk-2077 vehicle language: hard right angles dressed in neon.

    Convex top chines get an unbroken accent trim line, the belly edges get
    neon underglow, and the flat flanks get recessed panel seams every few
    plates — straight lines only, matching the style's near-zero bevel."""
    exp = _exposed_faces(g.occ)
    side = exp[4] | exp[5]
    armr = shell & (g.role == R_ARMOR)
    zmask = np.zeros(g.Z, bool)
    zmask[int(g.Z * 0.05):int(g.Z * 0.97)] = True

    def lines(m):
        """Keep only voxels inside a z-run of >= 3 — long straight strips,
        no lone studs on every little step of the hull."""
        r_prev = np.roll(m, 1, axis=2)
        r_prev[:, :, 0] = False          # no bow<->stern wraparound
        r_next = np.roll(m, -1, axis=2)
        r_next[:, :, -1] = False
        return m & r_prev & r_next

    chine = lines(exp[3] & side & armr)           # top edge -> accent trim
    under = lines(exp[2] & side & armr) & ~chine  # belly edge -> underglow
    chine[:, :, ~zmask] = False
    under[:, :, ~zmask] = False
    g.role[chine] = R_ACCENT
    g.role[under] = R_GLOW
    seam = (np.arange(g.Z) % 7 == rng.randrange(7))
    pan = side & armr & ~chine & ~under    # dark seam rings on the flanks
    pan[:, :, ~seam] = False
    g.role[pan] = R_PANEL



# role -> (block-type index, colour, look) resolved against the spec palette
def _attr_table(spec: ShipSpec) -> dict[int, tuple[int, str, int, int]]:
    prim, sec, glow = spec.primary, spec.secondary, spec.glow
    acc = contrast(prim, spec.accent)   # keep accents legible against the hull
    m = spec.material
    A = to_argb
    # a bright accent tints the whole wing olive — cyberpunk keeps them graphite
    wing_col = shade(prim, 1.2) if spec.style == "cyberpunk" \
        else mix(prim, acc, 0.25)
    inner = max(0, m - 1)   # interior structure is a tier below the outer armour
    return {
        R_HULL:        (index_for(Role.HULL),   A(shade(sec, 0.75)), 1, inner),
        R_ARMOR:       (index_for(Role.ARMOR),  A(prim),             1, m),
        R_ENGINE:      (index_for(Role.ENGINE), A(shade(sec, 0.55)), 0, m),
        R_ENGINE_GLOW: (index_for(Role.GLOW),   A(glow),             0, m),
        R_BRIDGE:      (index_for(Role.ARMOR),  A(acc),              1, m),
        R_WING:        (index_for(Role.ARMOR),  A(wing_col),         1, m),
        R_FIN:         (index_for(Role.ARMOR),  A(shade(sec, 0.9)),  1, m),
        R_GLOW:        (index_for(Role.GLOW),   A(glow),             1, m),
        R_ACCENT:      (index_for(Role.ARMOR),  A(acc),              1, m),
        R_GEN:         (index_for(Role.GENERATOR), A(shade(sec, 0.5)), 0, m),
        R_GYRO:        (index_for(Role.GYRO),   A(shade(sec, 0.5)),  0, m),
        R_THRUST:      (index_for(Role.THRUSTER), A(shade(sec, 0.45)), 0, m),
        R_CARGO:       (index_for(Role.CARGO),  A(shade(sec, 0.6)),  0, max(0, m - 1)),
        R_CREW:        (index_for(Role.CREW),   A(shade(sec, 0.6)),  0, 0),
        R_BATTERY:     (index_for(Role.BATTERY), A(shade(sec, 0.55)), 0, m),
        R_INTEGRITY:   (index_for(Role.INTEGRITY), A(shade(sec, 0.5)), 0, m),
        R_SHIELD:      (index_for(Role.SHIELD), A(mix(sec, glow, 0.3)), 0, max(2, m)),
        R_DIRTHRUST:   (index_for(Role.DIR_THRUSTER), A(shade(sec, 0.45)), 0, m),
        R_TURRET:      (index_for(Role.TURRET_BASE), A(shade(prim, 1.45)), 1, m),
        R_PANEL:       (index_for(Role.ARMOR),  A(shade(prim, 0.5)),  1, m),
    }


# profile -> (power, gyro, cargo, thruster) zone multipliers
_FUNC_PROFILES = {
    "balanced": (1.0, 1.0, 1.0, 1),
    "agile":    (1.0, 2.1, 0.0, 2),   # gyro arrays + double thruster bands
    "cargo":    (0.9, 0.8, 1.9, 1),
    "power":    (1.8, 0.9, 0.7, 1),
}


def _functional(g: VoxelGrid, spec: ShipSpec) -> None:
    """Fill interior hull with operable tech blocks by zone, scaled by ``functional``.

    Vanilla thrust/energy ratios live in the engine (not recoverable), so this uses
    sensible proportions: power (generators/batteries) aft, gyros/integrity/shield
    amidships, crew (and cargo for haulers) forward, thrusters on the flanks.
    ``func_profile`` reweights the zones — "agile" trades cargo for big gyro
    arrays and doubled thruster bands so the ship actually turns.
    """
    f = float(np.clip(spec.functional, 0.0, 1.0))
    if f <= 0.001:
        return
    X, Y, Z = g.X, g.Y, g.Z
    interior = g.occ & ~g.shell_mask() & (g.role == R_HULL)
    if not interior.any():
        return
    cx, cy = (X - 1) // 2, (Y - 1) // 2
    w = 0.25 + 0.55 * f

    hauler = spec.hull_class in ("freighter", "miner", "carrier")
    prof = spec.func_profile
    if prof not in _FUNC_PROFILES:
        prof = ("cargo" if hauler else
                "agile" if spec.hull_class in ("fighter", "corvette") else
                "balanced")
    p_m, g_m, c_m, t_m = _FUNC_PROFILES[prof]
    if hauler:
        # lesson from the user's hand edits: a working hauler wants more
        # power and fewer crew quarters than the balanced fill gives it
        p_m *= 1.3
    sq = np.sqrt

    def zone(z0, z1, role, wf, hf):
        hw = max(0, int((X * 0.5 - 1) * min(1.0, wf)))
        hh = max(0, int((Y * 0.5 - 1) * min(1.0, hf)))
        a, b = max(0, int(z0)), min(Z - 1, int(z1))
        xs, xe, ys, ye = cx - hw, cx + hw + 1, cy - hh, cy + hh + 1
        sub = interior[xs:xe, ys:ye, a:b + 1]
        g.role[xs:xe, ys:ye, a:b + 1][sub] = role

    zone(Z * 0.12, Z * 0.34, R_GEN, w * sq(p_m), w * sq(p_m))   # aft: power
    zone(Z * 0.30, Z * 0.42, R_BATTERY, w * 0.85 * sq(p_m), w * 0.85 * sq(p_m))
    zone(Z * 0.34, Z * 0.46, R_INTEGRITY, w * 0.55, w * 0.55)
    gz1 = Z * (0.56 if g_m > 1.5 else 0.52)                     # amidships: gyros
    zone(Z * 0.42, gz1, R_GYRO, w * 0.7 * sq(g_m), w * 0.7 * sq(g_m))
    if spec.material >= 2:
        zone(Z * 0.50, Z * 0.60, R_SHIELD, w * 0.5, w * 0.5)
    if (hauler or c_m > 1.0) and c_m > 0:
        zone(Z * 0.46, Z * 0.62, R_CARGO, w * sq(c_m), w * sq(c_m))
    crew_w = 0.7 if prof == "agile" else 0.6 if hauler else 1.0
    zone(Z * 0.60, Z * 0.82, R_CREW, w * crew_w, w)

    # maneuvering thrusters on the port/starboard *surface* (outermost voxel)
    occ = g.occ
    # heavy hulls: tall thruster columns at the stern quarters — braking and
    # turning authority for a big slow ship (from the user's hand edits)
    if hauler:
        ys_occ = np.where(occ.any(axis=(0, 2)))[0]
        jr = max(2, (int(ys_occ[-1]) - int(ys_occ[0])) // 3) if len(ys_occ) else 2
        for k in range(int(Z * 0.28), int(Z * 0.42)):
            for j in range(max(0, cy - jr), min(Y, cy + jr + 1)):
                row = np.where(occ[:, j, k])[0]
                if len(row):
                    g.role[row[0], j, k] = R_THRUST
                    g.role[row[-1], j, k] = R_THRUST
    rows = 1 + t_m                       # band half-height in voxel rows
    for k in range(int(Z * 0.24), int(Z * (0.46 if t_m == 1 else 0.54))):
        for j in range(max(0, cy - rows), min(Y, cy + rows + 1)):
            row = np.where(occ[:, j, k])[0]
            if len(row):
                g.role[row[0], j, k] = R_THRUST
                g.role[row[-1], j, k] = R_THRUST
    if t_m > 1:                          # agile: a second band near the bow
        for k in range(int(Z * 0.60), int(Z * 0.68)):
            for j in range(max(0, cy - 1), min(Y, cy + 2)):
                row = np.where(occ[:, j, k])[0]
                if len(row):
                    g.role[row[0], j, k] = R_THRUST
                    g.role[row[-1], j, k] = R_THRUST
    # directional thrusters near the bow flanks
    for k in range(int(Z * 0.70), int(Z * 0.84)):
        for j in (range(max(0, cy - 1), min(Y, cy + 2)) if t_m > 1 else (cy,)):
            row = np.where(occ[:, j, k])[0]
            if len(row):
                g.role[row[0], j, k] = R_DIRTHRUST
                g.role[row[-1], j, k] = R_DIRTHRUST


_FACE_CODES = [4, 5, 3, 2, 1, 0]  # +x -x +y -y +z -z  (see orient.INT2VEC)


def _exposed_faces(occ: np.ndarray) -> dict[int, np.ndarray]:
    """For each face direction code, a bool array: is that face of the voxel exposed?"""
    out = {}
    for code in _FACE_CODES:
        vec = orient.INT2VEC[code]
        axis = 0 if vec[0] else (1 if vec[1] else 2)
        shift = vec[axis]
        nb = np.roll(occ, -shift, axis=axis)
        idx = [slice(None)] * 3
        idx[axis] = -1 if shift > 0 else 0
        nb[tuple(idx)] = False
        out[code] = occ & ~nb
    return out


def _bevel(g: VoxelGrid, spec: ShipSpec, table):
    """Decide per-voxel shape: 0=cube, 1=edge wedge, 2=corner, with (look,up).

    Chamfers convex edges/corners of bevelable roles (those with hull/armour/glow
    shape variants). Deterministic + x-symmetric so the ship stays symmetric.
    """
    X, Y, Z = g.X, g.Y, g.Z
    kind = np.zeros((X, Y, Z), np.int8)
    look = np.full((X, Y, Z), 1, np.int8)
    up = np.full((X, Y, Z), 3, np.int8)
    if spec.bevel <= 0.01:
        return kind, look, up

    bevelable_roles = {r for r, (idx, *_) in table.items() if idx in SHAPE_VARIANTS}
    bevelable = g.occ & np.isin(g.role, list(bevelable_roles))
    exposed = _exposed_faces(g.occ)
    expcount = sum(exposed[c].astype(np.int8) for c in _FACE_CODES)
    cand = np.argwhere(bevelable & (expcount >= 2) & (expcount <= 3))

    # pass 1: collect entries so run lengths are known before deciding
    entries = []
    run_len: dict[tuple, int] = {}
    edge_pair: dict[tuple, tuple] = {}   # (i,j,k) -> its exposed face pair
    corners = []
    for (i, j, k) in cand:
        codes = [c for c in _FACE_CODES if exposed[c][i, j, k]]
        mi = min(int(i), X - 1 - int(i))
        if len(codes) == 2:
            d1, d2 = codes
            if orient.AXIS_OF[d1] == orient.AXIS_OF[d2]:
                continue  # opposite faces (a slab), not a convex edge
            # one identity per whole straight edge run: collapse the run axis
            run_axis = ({0, 1, 2} - {orient.AXIS_OF[d1], orient.AXIS_OF[d2]}).pop()
            key = [mi, int(j), int(k)]
            key[run_axis] = -1
            pair = min(d1, d2) * 7 + max(d1, d2)
            lk, u = orient.bevel_edge_orient(d1, d2)
            rk = (key[0], key[1], key[2], pair)
            run_len[rk] = run_len.get(rk, 0) + 1
            edge_pair[(int(i), int(j), int(k))] = (d1, d2)
            entries.append((int(i), int(j), int(k), 1, lk, u, rk))
        elif len({orient.AXIS_OF[c] for c in codes}) == 3:
            corners.append((int(i), int(j), int(k), codes, mi))

    # corner voxels (3 exposed faces) are ALWAYS closed with a Corner
    # tetrahedron whose cut octant equals the exposed octant — its mating
    # triangle is exactly the adjacent wedge run's cross-section, so it
    # SEALS the run end. (Round 10e tried "continuation wedges" here: a
    # wedge only covers two faces, so half of the third face stayed open
    # and every chamfer run became a see-through triangular tunnel in
    # game — models/img_2.png.) When the corner sits at the end of a
    # straight run it ADOPTS that run's key, sharing the run's on/off
    # decision — a chamfered run always gets its sealing cap and a
    # skipped (square) run keeps a square corner.
    for i, j, k, codes, mi in corners:
        lu = orient.bevel_corner_orient(*codes)
        if lu is None:
            continue                     # unknown octant: leave the cube
        best = None
        for a in range(3):
            d1, d2 = [c for c in codes if orient.AXIS_OF[c] != a]
            if orient.AXIS_OF[d1] == orient.AXIS_OF[d2]:
                continue
            run_axis = ({0, 1, 2} - {orient.AXIS_OF[d1], orient.AXIS_OF[d2]}).pop()
            score = 0
            for s in (1, -1):
                step = [0, 0, 0]
                step[run_axis] = s
                nb = (i + step[0], j + step[1], k + step[2])
                if edge_pair.get(nb) in ((d1, d2), (d2, d1)):
                    score += 1
            if score and (best is None or score > best[0]):
                best = (score, d1, d2, run_axis)
        if best is not None:             # cap of a straight run: share its fate
            _, d1, d2, run_axis = best
            key = [mi, j, k]
            key[run_axis] = -1
            pair = min(d1, d2) * 7 + max(d1, d2)
            rk = (key[0], key[1], key[2], pair)
        else:                            # isolated corner: its own identity
            rk = (mi, j, k, sum(codes) * 7)
        entries.append((i, j, k, 2, lu[0], lu[1], rk))

    # pass 2: the stochastic skip applies ONLY to long straight runs. On a
    # curved/terraced chine every voxel is its own 1-cell "run", and a
    # per-voxel roll peppered those edges with square teeth (seen in game).
    # Corner caps carry their run's rk, so a run and its sealing tetrahedra
    # always decide together — never a chamfered run with a square cap or
    # a lone tet on a square run.
    for i, j, k, shape, lk, u, rk in entries:
        if (run_len.get(rk, 0) >= 3
                and g.role[i, j, k] not in (R_WING, R_FIN)):
            h = ((rk[0] * 73856093) ^ (rk[1] * 19349663) ^ (rk[2] * 83492791)
                 ^ (rk[3] * 0x9e3779b1) ^ (int(spec.seed) * 2654435761)) & 0x7fffffff
            if (h % 1000) / 1000.0 >= spec.bevel:
                continue
        kind[i, j, k] = shape
        look[i, j, k], up[i, j, k] = lk, u
    return kind, look, up


def build_ship(spec: ShipSpec) -> ShipModel:
    """Build a complete ship from a spec."""
    spec = spec.resolved()

    # one rng per subsystem, so toggling e.g. wings doesn't reshuffle the hull
    def _rng(tag: str) -> random.Random:
        return random.Random(f"{spec.seed}:{tag}")

    hX = max(3, int(spec.width))
    hY = max(3, int(spec.height))
    Z = max(4, int(spec.length))
    layout = _choose_layout(spec, _rng("layout"))

    # pad the grid so wings / fins / bridge / keel can protrude beyond the hull
    combat = spec.hull_class not in ("freighter", "miner", "carrier", "station")
    wing_scale = (0.7, 0.5) if combat else (0.5, 0.3)   # fighters: span ~ hull width
    wing_span = max(5 if combat else 4, int(round(
        hX * (wing_scale[0] + wing_scale[1] * _rng("wingspan").random()))))
    fin_h = max(2, int(round(hY * (0.40 + 0.35 * _rng("finh").random()))))
    cab_h = max(2, hY // 3)
    # miners carry external rig gear (shoulders/blade/skirts) — reserve room
    rig_pad = (2, max(4, hY // 6 + 2)) if spec.hull_class == "miner" else (0, 0)
    # vpods module pods stick past the strut tip — reserve extra width
    pad_x = max((wing_span + max(3, int(hY * 0.3) + 2)) if spec.wings else 0,
                rig_pad[0])
    # wings may tilt (di-/anhedral, x-foils) or end in winglets — reserve height
    wing_pad_y = (int(round(wing_span * 0.8)) + max(2, hY // 2)) if spec.wings else 0
    pad_y = max(fin_h if spec.fins else 0, cab_h if spec.bridge else 0,
                max(2, hY // 3) if layout == "keel" else 0, wing_pad_y,
                rig_pad[1],
                1 if spec.turrets else 0)   # turret pads sit 1 voxel above the hull
    X = hX + 2 * pad_x
    Y = hY + 2 * pad_y   # symmetric so the hull stays centred in the grid
    g = VoxelGrid(X, Y, Z)

    _paint_hull(g, spec, layout, _rng("hull"))

    _add_engines(g, spec, (hX, hY), _rng("engines"),
                 wall=spec.hull_class in ("miner", "freighter") or layout == "crab")
    _add_bridge(g, spec, (hX, hY), cab_h, _rng("bridge"))
    _add_wings(g, spec, wing_span, (hX, hY), _rng("wings"))
    _add_fins(g, spec, fin_h, (hX, hY), _rng("fins"))
    _add_mining_rig(g, spec, (hX, hY), layout, _rng("rig"))
    _add_turret_mounts(g, spec, (hX, hY), _rng("turrets"))

    _keep_main_component(g)
    _enforce_symmetry(g)
    _resolve_roles(g, spec, _rng("detail"))
    _functional(g, spec)
    _enforce_symmetry(g)  # keep glow/tech placement symmetric too

    table = _attr_table(spec)
    shape_kind, look_arr, up_arr = _bevel(g, spec, table)

    # the game's build grid is 0.25: quantize the voxel pitch to it and anchor
    # the lattice so EVERY block face lands on a 0.25 multiple — hand-placed
    # blocks in game then butt flush against generated ones (grid snapping)
    GRID = 0.25
    s = max(GRID, round(spec.block_size * spec.scale / GRID) * GRID)
    sh_x = round(X / 2 * s / GRID) * GRID   # ~centred, but on the 0.25 lattice
    sh_y = round(Y / 2 * s / GRID) * GRID
    sh_z = round(Z / 2 * s / GRID) * GRID
    ship = ShipModel(name=spec.name)
    ship.layout = layout   # remembered for the UI stats line
    ship.pitch = s         # effective block pitch, for the UI stats line

    def emit(i0, i1, j0, j1, k0, k1, idx, color, mat, look, up):
        ship.add(Block(
            lx=i0 * s - sh_x, ly=j0 * s - sh_y, lz=k0 * s - sh_z,
            ux=(i1 + 1) * s - sh_x, uy=(j1 + 1) * s - sh_y, uz=(k1 + 1) * s - sh_z,
            index=idx, material=mat, look=look, up=up, color=color, secondary="00000000",
        ))

    # cubes (non-beveled voxels) merge into big boxes
    cube_attr = np.where(g.occ & (shape_kind == 0), g.role, EMPTY).astype(np.int32)
    for (i0, i1, j0, j1, k0, k1, rid) in greedy_merge(cube_attr):
        idx, color, look, mat = table.get(rid, table[R_HULL])
        if rid == R_TURRET:
            # turret sockets are thin plates hugging the hull top (the way
            # players build them), not full-voxel cubes towering over it
            ship.add(Block(
                lx=i0 * s - sh_x, ly=j0 * s - sh_y, lz=k0 * s - sh_z,
                ux=(i1 + 1) * s - sh_x, uy=j0 * s - sh_y + GRID,
                uz=(k1 + 1) * s - sh_z,
                index=idx, material=mat, look=look, up=3,
                color=color, secondary="00000000"))
            continue
        emit(i0, i1, j0, j1, k0, k1, idx, color, mat, look, 3)

    # beveled voxels emit individually as edge/corner shape blocks
    for (i, j, k) in np.argwhere(shape_kind > 0):
        rid = int(g.role[i, j, k])
        cube_idx, color, _, mat = table.get(rid, table[R_HULL])
        edge_idx, corner_idx = SHAPE_VARIANTS.get(cube_idx, (cube_idx, cube_idx))
        idx = edge_idx if shape_kind[i, j, k] == 1 else corner_idx
        emit(i, i, j, j, k, k, idx, color, mat, int(look_arr[i, j, k]), int(up_arr[i, j, k]))

    ship.drop_degenerate()
    return ship
