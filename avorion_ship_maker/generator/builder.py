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
from .spec import ShipSpec
from .voxel import EMPTY, VoxelGrid, greedy_merge

# ---- voxel role codes (internal) ---------------------------------------
(R_HULL, R_ARMOR, R_ENGINE, R_ENGINE_GLOW, R_BRIDGE, R_WING, R_FIN,
 R_GLOW, R_ACCENT, R_GEN, R_GYRO, R_THRUST, R_CARGO, R_CREW,
 R_BATTERY, R_INTEGRITY, R_SHIELD, R_DIRTHRUST) = range(18)


def _profile(spec: ShipSpec, rng: random.Random):
    """Per-slice half-width and half-height arrays (in voxels), stern -> bow.

    The rng jitters nose/tail/bulge and adds a gentle waviness so every seed
    yields a visibly different silhouette from the same spec.
    """
    Z = spec.length
    base_hw = max((spec.width - 1) / 2.0, 1.0)
    base_hh = max((spec.height - 1) / 2.0, 1.0)
    nose = float(np.clip(spec.nose * (0.8 + 0.5 * rng.random()), 0.02, 0.9))
    tail = float(np.clip(spec.taper_tail * (0.7 + 0.7 * rng.random()), 0.0, 0.5))
    bulge = 0.04 + 0.06 * rng.random()
    wamp = 0.05 * rng.random()
    wfreq = rng.randint(2, 4)
    wphase = rng.random() * 2 * np.pi
    hw = np.empty(Z)
    hh = np.empty(Z)
    for k in range(Z):
        zc = k / max(Z - 1, 1)               # 0 = stern, 1 = bow
        ws = hs = 1.0
        if zc < tail and tail > 0:            # gentle stern narrowing
            t = zc / tail
            ws = 0.82 + 0.18 * t
            hs = 0.88 + 0.12 * t
        if zc > 1 - nose:                     # bow taper (pointier in width)
            t = (zc - (1 - nose)) / nose      # 0..1 into the nose
            ws = 1.0 - (1.0 - 0.12) * (t ** 0.85)
            hs = 1.0 - (1.0 - 0.35) * (t ** 1.2)
        # mid-body bulge + low-frequency waviness (seed flavour)
        ws *= (1.0 - bulge) + bulge * np.sin(np.pi * min(max(zc, 0), 1))
        ws *= 1.0 + wamp * np.sin(wfreq * np.pi * zc + wphase)
        hw[k] = base_hw * ws
        hh[k] = base_hh * hs
    return hw, hh


def _add_engines(g: VoxelGrid, spec: ShipSpec, hull: tuple[int, int],
                 rng: random.Random) -> None:
    n = max(int(spec.engines), 0)
    if n == 0:
        return
    hX, hY = hull
    Z = g.Z
    elen = max(2, int(round(Z * (0.11 + 0.06 * rng.random()))))
    cx = (g.X - 1) / 2.0
    cy = (g.Y - 1) / 2.0
    # symmetric x-offsets for the pods, kept inside the hull footprint
    if n == 1:
        offsets = [0.0]
    else:
        spread = (hX - 1) / 2.0 * (0.55 + 0.2 * rng.random())
        offsets = list(np.linspace(-spread, spread, n))
    rad = max(1, int(round(hX / (max(n, 1) * 4))))
    for off in offsets:
        xc = int(round(cx + off))
        yr = (int(cy - max(1, hY // 4)), int(cy + max(1, hY // 4)))
        g.paint_box((xc - rad, xc + rad), yr, (0, elen), R_ENGINE)
        # glowing exhaust face at the very stern
        g.paint_box((xc - rad, xc + rad), yr, (0, 0), R_ENGINE_GLOW)


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


def _add_wings(g: VoxelGrid, spec: ShipSpec, span: int, hull: tuple[int, int],
               rng: random.Random) -> None:
    """Swept-back wings protruding from the flanks (grid is padded sideways)."""
    if not spec.wings or span <= 0:
        return
    hX, hY = hull
    cy = (g.Y - 1) // 2
    z0 = int(g.Z * (0.26 + 0.14 * rng.random()))
    zl = max(3, int(g.Z * (0.20 + 0.12 * rng.random())))
    sweep = 0.55 + 0.4 * rng.random()
    thick = 1 if hY < 8 else 2
    for k in range(z0, min(g.Z, z0 + zl)):
        t = (k - z0) / max(zl - 1, 1)
        ext = int(round(span * (1.0 - sweep * t)))   # long at the root, swept tip
        if ext <= 0:
            continue
        for j in range(cy, cy + thick):
            row = np.where(g.occ[:, j, k])[0]
            if not len(row):
                continue
            lo, hi = int(row[0]), int(row[-1])
            g.paint_box((lo - ext, lo - 1), (j, j), (k, k), R_WING, only_if_empty=True)
            g.paint_box((hi + 1, hi + ext), (j, j), (k, k), R_WING, only_if_empty=True)


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
    if d > 0.15:
        spacing = max(2, int(round(4 - 2.0 * d)))   # denser dashes at high detail
        keep = (np.arange(g.Z) % spacing == rng.randrange(spacing))
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



# role -> (block-type index, colour, look) resolved against the spec palette
def _attr_table(spec: ShipSpec) -> dict[int, tuple[int, str, int, int]]:
    prim, sec, glow = spec.primary, spec.secondary, spec.glow
    acc = contrast(prim, spec.accent)   # keep accents legible against the hull
    m = spec.material
    A = to_argb
    inner = max(0, m - 1)   # interior structure is a tier below the outer armour
    return {
        R_HULL:        (index_for(Role.HULL),   A(shade(sec, 0.75)), 1, inner),
        R_ARMOR:       (index_for(Role.ARMOR),  A(prim),             1, m),
        R_ENGINE:      (index_for(Role.ENGINE), A(shade(sec, 0.55)), 0, m),
        R_ENGINE_GLOW: (index_for(Role.GLOW),   A(glow),             0, m),
        R_BRIDGE:      (index_for(Role.ARMOR),  A(acc),              1, m),
        R_WING:        (index_for(Role.ARMOR),  A(mix(prim, acc, 0.25)), 1, m),
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
    }


def _functional(g: VoxelGrid, spec: ShipSpec) -> None:
    """Fill interior hull with operable tech blocks by zone, scaled by ``functional``.

    Vanilla thrust/energy ratios live in the engine (not recoverable), so this uses
    sensible proportions: power (generators/batteries) aft, gyros/integrity/shield
    amidships, crew (and cargo for haulers) forward, thrusters on the flanks.
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

    def zone(z0, z1, role, wf, hf):
        hw = max(0, int((X * 0.5 - 1) * min(1.0, wf)))
        hh = max(0, int((Y * 0.5 - 1) * min(1.0, hf)))
        a, b = max(0, int(z0)), min(Z - 1, int(z1))
        xs, xe, ys, ye = cx - hw, cx + hw + 1, cy - hh, cy + hh + 1
        sub = interior[xs:xe, ys:ye, a:b + 1]
        g.role[xs:xe, ys:ye, a:b + 1][sub] = role

    hauler = spec.hull_class in ("freighter", "miner", "carrier")
    zone(Z * 0.12, Z * 0.34, R_GEN, w, w)             # aft: generators
    zone(Z * 0.30, Z * 0.42, R_BATTERY, w * 0.85, w * 0.85)
    zone(Z * 0.34, Z * 0.46, R_INTEGRITY, w * 0.55, w * 0.55)
    zone(Z * 0.42, Z * 0.52, R_GYRO, w * 0.7, w * 0.7)  # amidships
    if spec.material >= 2:
        zone(Z * 0.50, Z * 0.60, R_SHIELD, w * 0.5, w * 0.5)
    if hauler:
        zone(Z * 0.46, Z * 0.62, R_CARGO, w, w)
    zone(Z * 0.60, Z * 0.82, R_CREW, w, w)              # forward: crew

    # maneuvering thrusters on the port/starboard *surface* (outermost voxel)
    occ = g.occ
    for k in range(int(Z * 0.24), int(Z * 0.46)):
        for j in range(max(0, cy - 1), min(Y, cy + 2)):
            row = np.where(occ[:, j, k])[0]
            if len(row):
                g.role[row[0], j, k] = R_THRUST
                g.role[row[-1], j, k] = R_THRUST
    # directional thrusters near the bow flanks
    for k in range(int(Z * 0.70), int(Z * 0.84)):
        row = np.where(occ[:, cy, k])[0]
        if len(row):
            g.role[row[0], cy, k] = R_DIRTHRUST
            g.role[row[-1], cy, k] = R_DIRTHRUST


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

    for (i, j, k) in cand:
        codes = [c for c in _FACE_CODES if exposed[c][i, j, k]]
        mi = min(int(i), X - 1 - int(i))
        h = ((mi * 73856093) ^ (int(j) * 19349663) ^ (int(k) * 83492791)
             ^ (int(spec.seed) * 2654435761)) & 0x7fffffff
        if (h % 1000) / 1000.0 >= spec.bevel:
            continue
        if len(codes) == 2:
            d1, d2 = codes
            if orient.AXIS_OF[d1] == orient.AXIS_OF[d2]:
                continue  # opposite faces (a slab), not a convex edge
            lk, u = orient.bevel_edge_orient(d1, d2)
            kind[i, j, k] = 1
        else:  # 3 exposed faces
            if len({orient.AXIS_OF[c] for c in codes}) < 3:
                continue
            lu = orient.bevel_corner_orient(*codes)
            if lu is None:
                continue
            lk, u = lu
            kind[i, j, k] = 2
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

    # pad the grid so wings / fins / bridge can protrude beyond the hull
    wing_span = max(3, int(round(hX * (0.45 + 0.35 * _rng("wingspan").random()))))
    fin_h = max(2, int(round(hY * (0.40 + 0.35 * _rng("finh").random()))))
    cab_h = max(2, hY // 3)
    pad_x = wing_span if spec.wings else 0
    pad_y = max(fin_h if spec.fins else 0, cab_h if spec.bridge else 0)
    X = hX + 2 * pad_x
    Y = hY + 2 * pad_y   # symmetric so the hull stays centred in the grid
    g = VoxelGrid(X, Y, Z)

    hw, hh = _profile(spec, _rng("hull"))
    g.fill_profile(hw, hh, spec.boxiness, R_HULL)

    _add_engines(g, spec, (hX, hY), _rng("engines"))
    _add_bridge(g, spec, (hX, hY), cab_h, _rng("bridge"))
    _add_wings(g, spec, wing_span, (hX, hY), _rng("wings"))
    _add_fins(g, spec, fin_h, (hX, hY), _rng("fins"))

    _enforce_symmetry(g)
    _resolve_roles(g, spec, _rng("detail"))
    _functional(g, spec)
    _enforce_symmetry(g)  # keep glow/tech placement symmetric too

    table = _attr_table(spec)
    shape_kind, look_arr, up_arr = _bevel(g, spec, table)

    s = spec.block_size * spec.scale
    ship = ShipModel(name=spec.name)

    def emit(i0, i1, j0, j1, k0, k1, idx, color, mat, look, up):
        ship.add(Block(
            lx=(i0 - X / 2) * s, ly=(j0 - Y / 2) * s, lz=(k0 - Z / 2) * s,
            ux=(i1 + 1 - X / 2) * s, uy=(j1 + 1 - Y / 2) * s, uz=(k1 + 1 - Z / 2) * s,
            index=idx, material=mat, look=look, up=up, color=color, secondary="00000000",
        ))

    # cubes (non-beveled voxels) merge into big boxes
    cube_attr = np.where(g.occ & (shape_kind == 0), g.role, EMPTY).astype(np.int32)
    for (i0, i1, j0, j1, k0, k1, rid) in greedy_merge(cube_attr):
        idx, color, look, mat = table.get(rid, table[R_HULL])
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
