"""Procedural Avorion turret designs (the ``<turret_design>`` XML format).

A turret is three block plans — static ``base``, yawing ``body`` and the
elevating ``barrel`` — plus one muzzle position per barrel. The schema and
units come from the game's own saved designs (``ships/auto-turrets``) and
the reference turrets in ``models/Lady_turret*.xml``.

What the references taught us (they are ~240 blocks, half of them wedges):
cylinders are faked with OCTAGONAL prisms built from boxes + edge wedges
(gun drums, mantlets, muzzle rings), bases wear hazard-striped rims, the
housing reads as an asymmetric operator cab with a glowing window, and a
multi-barrel gun emits several ``muzzlePosition`` entries. This builder
speaks that vocabulary; coordinates snap to the spec's ``step`` grid
(0.01 by default) and every subsystem rolls its own look from the seed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .. import orient
from ..blocks import Role, index_for
from ..model import Block, ShipModel
from .palette import mix, shade, to_argb
from .spec import DEFAULT_STYLE, STYLES

# visual archetypes; RU names live in the UI
TURRET_KINDS: tuple[str, ...] = ("cannon", "chaingun", "laser", "railgun", "launcher")

DEFAULT_STEP = 0.01                    # default build-grid step for turrets
_HULL = index_for(Role.BLANK_HULL)     # 2 — what the game's own turrets use
_GLOW = index_for(Role.GLOW)           # 150 — emissive tips / stripes
_EDGE = 100                            # hull edge wedge (needs look/up)
_CORNER = 101                          # hull corner (needs look/up)

# face codes as in builder._FACE_CODES: +x -x +y -y +z -z
_XP, _XN, _YP, _YN, _ZP, _ZN = 4, 5, 3, 2, 1, 0

_HAZARD_A, _HAZARD_B = "c23b22", "e2e2e2"   # warning-stripe pair
_DARK = "141414"


@dataclass
class TurretSpec:
    kind: str = "cannon"
    size: float = 1.0                   # Avorion turret slot size (0.5 .. 3+)
    barrels: int | None = None          # None => archetype default (per seed)
    style: str = DEFAULT_STYLE          # colour preset, shared with ships
    step: float = DEFAULT_STEP          # build-grid the coordinates snap to
    seed: int = 0


@dataclass
class TurretModel:
    """The three plans + metadata needed to serialize a <turret_design>."""
    size: float = 1.0
    coaxial: bool = False
    shot_color: int = 0                 # signed int32 AARRGGBB
    base: list[Block] = field(default_factory=list)
    body: list[Block] = field(default_factory=list)
    barrel: list[Block] = field(default_factory=list)
    body_pivot: tuple = (0.0, 0.0, 0.0)
    barrel_pivot: tuple = (0.0, 0.0, 0.0)
    muzzles: list = field(default_factory=list)   # one (x, y, z) per barrel

    def assembled(self) -> ShipModel:
        """One ShipModel with all sections placed — for previews only."""
        m = ShipModel(name="turret")
        for b in self.base:
            m.add(b)
        bx, by, bz = self.body_pivot
        for b in self.body:
            m.add(_shift(b, bx, by, bz))
        px, py, pz = self.barrel_pivot
        for b in self.barrel:
            m.add(_shift(b, px, py, pz))
        return m


def _shift(b: Block, dx: float, dy: float, dz: float) -> Block:
    return Block(lx=b.lx + dx, ly=b.ly + dy, lz=b.lz + dz,
                 ux=b.ux + dx, uy=b.uy + dy, uz=b.uz + dz,
                 index=b.index, material=b.material, look=b.look, up=b.up,
                 color=b.color, secondary=b.secondary)


def _shot_color(glow_hex: str) -> int:
    v = int("ff" + glow_hex[-6:], 16)
    return v - 2 ** 32 if v >= 2 ** 31 else v


def build_turret(spec: TurretSpec) -> TurretModel:
    """Build a full turret design from the spec (deterministic per seed)."""
    st = STYLES.get(spec.style, STYLES[DEFAULT_STYLE])
    prim, sec, acc, glow = st["primary"], st["secondary"], st["accent"], st["glow"]
    mat = int(st.get("material", 1))
    S = max(0.25, float(spec.size))
    W = 0.5 * S                          # base half-extent (matches game designs)
    kind = spec.kind if spec.kind in TURRET_KINDS else "cannon"
    step = max(0.001, float(spec.step or DEFAULT_STEP))
    ring_col = mix(prim, "f0f8ff", 0.55)          # bright machined metal
    barrel_metal = shade(sec, 0.75)

    def _q(v: float) -> float:
        """Snap to the spec's build grid."""
        return round(round(v / step) * step, 6)

    # one rng per subsystem: overriding barrels must not reshuffle the body
    def _rng(tag: str) -> random.Random:
        return random.Random(f"{spec.seed}:t:{tag}")

    t = TurretModel(size=spec.size, shot_color=_shot_color(glow))

    def box(sect, x0, y0, z0, x1, y1, z1, color, idx=_HULL, look=1, up=3):
        lo = [_q(min(x0, x1)), _q(min(y0, y1)), _q(min(z0, z1))]
        hi = [_q(max(x0, x1)), _q(max(y0, y1)), _q(max(z0, z1))]
        for a in range(3):               # a coarse grid must not flatten a block
            if hi[a] - lo[a] < step / 2:
                hi[a] = _q(lo[a] + step)
        sect.append(Block(lx=lo[0], ly=lo[1], lz=lo[2],
                          ux=hi[0], uy=hi[1], uz=hi[2],
                          index=idx, material=mat, look=look, up=up,
                          color=to_argb(color), secondary="00000000"))

    def wedge(sect, x0, y0, z0, x1, y1, z1, color, d1, d2, idx=_EDGE):
        lk, up = orient.bevel_edge_orient(d1, d2)
        box(sect, x0, y0, z0, x1, y1, z1, color, idx=idx, look=lk, up=up)

    def octo_z(sect, cx, cy, z0, z1, w, color, cut=0.34, idx=_HULL):
        """Octagonal prism along z — the wedge-built stand-in for a cylinder."""
        widx = 151 if idx == _GLOW else _EDGE     # glow prisms use glow wedges
        c = _q(max(w * cut, step))
        if w - c < step:                 # too small to cut -> plain box
            box(sect, cx - w, cy - w, z0, cx + w, cy + w, z1, color, idx=idx)
            return
        box(sect, cx - w, cy - (w - c), z0, cx + w, cy + (w - c), z1, color, idx=idx)
        box(sect, cx - (w - c), cy + (w - c), z0, cx + (w - c), cy + w, z1, color, idx=idx)
        box(sect, cx - (w - c), cy - w, z0, cx + (w - c), cy - (w - c), z1, color, idx=idx)
        for sx, dx in ((1, _XP), (-1, _XN)):
            for sy, dy in ((1, _YP), (-1, _YN)):
                wedge(sect, cx + sx * (w - c), cy + sy * (w - c), z0,
                      cx + sx * w, cy + sy * w, z1, color, dx, dy, idx=widx)

    def octo_x(sect, x0, x1, cy, cz, w, color, cut=0.34):
        """Octagonal prism along x — the hinge drums of a robo-arm."""
        c = _q(max(w * cut, step))
        if w - c < step:
            box(sect, x0, cy - w, cz - w, x1, cy + w, cz + w, color)
            return
        box(sect, x0, cy - (w - c), cz - w, x1, cy + (w - c), cz + w, color)
        box(sect, x0, cy + (w - c), cz - (w - c), x1, cy + w, cz + (w - c), color)
        box(sect, x0, cy - w, cz - (w - c), x1, cy - (w - c), cz + (w - c), color)
        for sy, dy in ((1, _YP), (-1, _YN)):
            for sz, dz in ((1, _ZP), (-1, _ZN)):
                wedge(sect, x0, cy + sy * (w - c), cz + sz * (w - c),
                      x1, cy + sy * w, cz + sz * w, color, dy, dz)

    def octo_y(sect, cx, cz, y0, y1, w, color, cut=0.34):
        """Octagonal prism along y — collars, masts, pedestal drums."""
        c = _q(max(w * cut, step))
        if w - c < step:
            box(sect, cx - w, y0, cz - w, cx + w, y1, cz + w, color)
            return
        box(sect, cx - w, y0, cz - (w - c), cx + w, y1, cz + (w - c), color)
        box(sect, cx - (w - c), y0, cz + (w - c), cx + (w - c), y1, cz + w, color)
        box(sect, cx - (w - c), y0, cz - w, cx + (w - c), y1, cz - (w - c), color)
        for sx, dx in ((1, _XP), (-1, _XN)):
            for sz, dz in ((1, _ZP), (-1, _ZN)):
                wedge(sect, cx + sx * (w - c), y0, cz + sz * (w - c),
                      cx + sx * w, y1, cz + sz * w, color, dx, dz)

    def octo_taper(sect, cx, cy, z0, z1, w_big, w_small, color):
        """Machined cone: sloped wedges wrap the thin segment where it
        leaves a thicker one, instead of a bare step."""
        c = _q(max(w_small * 0.34, step))
        f = _q(w_small - c)              # flat-band half-width of the core
        if w_big - w_small < step or f < step:
            return
        wedge(sect, cx - f, cy + w_small, z0, cx + f, cy + w_big, z1,
              color, _YP, _ZP)
        wedge(sect, cx - f, cy - w_big, z0, cx + f, cy - w_small, z1,
              color, _YN, _ZP)
        wedge(sect, cx + w_small, cy - f, z0, cx + w_big, cy + f, z1,
              color, _XP, _ZP)
        wedge(sect, cx - w_big, cy - f, z0, cx - w_small, cy + f, z1,
              color, _XN, _ZP)

    def drum_face(sect, cx, cy, z_at, w, depth):
        """Muzzle drum: bright ring face with a dark protruding core."""
        octo_z(sect, cx, cy, z_at, z_at + depth, w, ring_col, cut=0.3)
        octo_z(sect, cx, cy, z_at + depth, z_at + depth + 0.02 * S,
               w * 0.55, _DARK, cut=0.3)
        return _q(z_at + depth + 0.02 * S)

    # =============================================================== base
    rb = _rng("base")
    metal = shade(sec, 0.7)
    ph = _q((0.05 + 0.03 * rb.random()) * S)          # plate height
    pw = _q(W * (0.9 + 0.14 * rb.random()))           # plate half-width
    base_kind = rb.choice(("square", "octo", "tiered"))
    if base_kind == "octo":
        c8 = _q(pw * 0.35)
        box(t.base, -pw + c8, 0, -pw, pw - c8, ph, -pw + c8, metal)
        box(t.base, -pw, 0, -pw + c8, pw, ph, pw - c8, metal)
        box(t.base, -pw + c8, 0, pw - c8, pw - c8, ph, pw, metal)
        for sx, sz in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            wedge(t.base, sx * (pw - c8), 0, sz * (pw - c8), sx * pw, ph, sz * pw,
                  metal, _XP if sx > 0 else _XN, _ZP if sz > 0 else _ZN)
    else:
        box(t.base, -pw, 0, -pw, pw, ph, pw, metal)
    plate_top = ph
    seat = ph
    if base_kind == "tiered":
        box(t.base, -pw * 0.74, ph, -pw * 0.74, pw * 0.74, ph + 0.04 * S,
            pw * 0.74, shade(sec, 0.85))
        seat = _q(ph + 0.04 * S)
    # hazard stripes / plain trim around the plate rim (reference look)
    haz_p = 0.8 if spec.style in ("industrial", "hazard") else 0.3
    if base_kind != "octo" and rb.random() < haz_p:
        bt = _q(max(0.03 * S, step))
        nseg = 6
        seg = 2 * pw / nseg
        for i in range(nseg):
            col = _HAZARD_A if i % 2 == 0 else _HAZARD_B
            a0, a1 = -pw + i * seg, -pw + (i + 1) * seg
            box(t.base, -pw - bt, 0, a0, -pw, plate_top, a1, col)
            box(t.base, pw, 0, a0, pw + bt, plate_top, a1,
                _HAZARD_B if i % 2 == 0 else _HAZARD_A)
            box(t.base, a0, 0, -pw - bt, a1, plate_top, -pw, col)
            box(t.base, a0, 0, pw, a1, plate_top, pw + bt,
                _HAZARD_B if i % 2 == 0 else _HAZARD_A)
    elif base_kind != "octo" and rb.random() < 0.5:    # rim bolts instead
        bs = _q(0.035 * S)
        bx = _q(pw - 2 * bs)
        for sxx in (1, -1):
            for fz in rb.choice(((-0.7, 0.7), (-0.7, 0.0, 0.7))):
                bz = _q(fz * bx)
                box(t.base, sxx * bx - bs, plate_top, bz - bs,
                    sxx * bx + bs, plate_top + bs, bz + bs, shade(sec, 0.55))
    # collar: an octagonal pedestal drum the body sits in
    cw = _q(pw * (0.45 + 0.1 * rb.random()))
    collar_h = _q(seat + (0.05 + 0.03 * rb.random()) * S)
    octo_y(t.base, 0, 0, seat, collar_h, cw, prim)

    # ================================================== robo-arm (laser)
    # mining-manipulator look from the reference: shoulder tower with a
    # clevis, an elbow hinge drum, and the whole forearm as the elevating
    # barrel section — in game the elbow really nods
    arm_ok = kind == "laser" and (spec.barrels is None or int(spec.barrels or 0) <= 1)
    if arm_ok and _rng("armroll").random() < 0.65:
        ra = _rng("arm")
        hcol = ra.choice((prim, shade(prim, 0.88), mix(prim, sec, 0.35)))
        aw = _q(0.16 * S * (0.85 + 0.3 * ra.random()))     # shoulder half-width
        y0a = collar_h
        tow_h = _q((0.30 + 0.12 * ra.random()) * S)
        ys1 = _q(y0a + tow_h)
        tz0, tzf = _q(-0.22 * S), _q(0.12 * S)             # tower z extent
        box(t.body, -aw, y0a, tz0, aw, ys1, tzf, hcol)
        wedge(t.body, -aw, ys1, tzf - 0.06 * S, aw, ys1 + 0.06 * S, tzf,
              hcol, _YP, _ZP)
        # rear equipment pack + cooling vents on it
        box(t.body, -aw * 0.8, y0a + tow_h * 0.15, tz0 - 0.1 * S,
            aw * 0.8, y0a + tow_h * 0.7, tz0, shade(sec, 0.8))
        for i in range(2):
            vy = y0a + tow_h * (0.25 + 0.2 * i)
            box(t.body, -aw * 0.55, vy, tz0 - 0.12 * S, aw * 0.55,
                vy + tow_h * 0.09, tz0 - 0.1 * S, _DARK)
        # glow status strip on the tower face
        box(t.body, -0.04 * S, y0a + tow_h * 0.12, tzf, 0.04 * S,
            y0a + tow_h * 0.5, tzf + 0.015 * S, glow, idx=_GLOW)
        # clevis cheeks that hold the elbow, with accent hinge caps
        ct = _q(max(0.035 * S, 2 * step))
        ye = _q(ys1 + 0.10 * S)
        for s in (1, -1):
            box(t.body, s * aw, ys1 - 0.12 * S, -0.13 * S,
                s * (aw + ct), ye + 0.11 * S, 0.13 * S, shade(hcol, 0.85))
            box(t.body, s * (aw + ct), ye - 0.035 * S, -0.035 * S,
                s * (aw + ct + 0.012 * S), ye + 0.035 * S, 0.035 * S, acc)

        # ---- forearm = the elevating section, pivot at the elbow --------
        # its own rolls: boom build, head mount, counterweight
        t.barrel_pivot = (0.0, ye, 0.0)
        boom_kind = ra.choice(("solid", "twin"))
        head_kind = ra.choice(("inline", "under"))
        wh = _q(0.11 * S)
        octo_x(t.barrel, -aw, aw, 0, 0, wh, shade(prim, 0.85))   # hinge hub
        if ra.random() < 0.7:                                    # counterweight
            box(t.barrel, -aw * 0.7, -0.08 * S, -(0.16 + 0.08 * ra.random()) * S,
                aw * 0.7, 0.08 * S, -wh, shade(sec, 0.8))
        bh = _q(wh * 0.6)
        bw = _q(aw * 0.75)
        alen = _q((0.45 + 0.55 * ra.random()) * S)
        z_b1 = _q(wh + alen)
        if boom_kind == "twin":                                  # two rails
            rt = _q(max(bw * 0.42, 2 * step))
            for s in (1, -1):
                box(t.barrel, s * (bw - rt) if s > 0 else -bw, -bh, wh,
                    s * bw if s > 0 else -(bw - rt), bh, z_b1, hcol)
            box(t.barrel, -bw * 0.9, bh, wh + 0.06 * S, bw * 0.9,
                bh + 0.015 * S, z_b1 - 0.06 * S, glow, idx=_GLOW)
        else:                                                    # solid boom
            box(t.barrel, -bw, -bh, wh, bw, bh, z_b1, hcol)
            box(t.barrel, -0.03 * S, bh, wh + 0.06 * S, 0.03 * S,
                bh + 0.015 * S, z_b1 - 0.06 * S, glow, idx=_GLOW)
            for s in (1, -1):                                    # side plates
                box(t.barrel, s * bw, -bh * 0.6, wh + 0.08 * S,
                    s * (bw + 0.015 * S), bh * 0.6, z_b1 - 0.08 * S,
                    shade(hcol, 0.8))
            box(t.barrel, -0.025 * S, -bh - 0.05 * S, wh + 0.12 * S, 0.025 * S,
                -bh, wh + 0.22 * S, shade(sec, 0.6))             # actuator block
        wh2 = _q(max(bh * 1.4, 0.08 * S))                        # wrist drum
        zw = _q(z_b1 + wh2)
        octo_x(t.barrel, -bw * 0.9, bw * 0.9, 0, zw, wh2, shade(prim, 0.85))
        er = _q(wh2 * 0.7)
        if head_kind == "under":         # emitter slung under the wrist
            bh2 = _q(max(0.06 * S, er + step))   # keep the head clear of it
            hy = _q(-wh2 - bh2)
            box(t.barrel, -bw * 0.5, hy, zw - 0.05 * S, bw * 0.5, -wh2,
                zw + 0.05 * S, shade(hcol, 0.85))
            e0 = _q(zw + 0.05 * S)
            octo_z(t.barrel, 0, hy, e0, e0 + 0.07 * S, er, ring_col, cut=0.3)
            octo_z(t.barrel, 0, hy, e0 + 0.07 * S, e0 + 0.12 * S, er * 0.55,
                   glow, idx=_GLOW)
            t.muzzles = [(0.0, hy, _q(e0 + 0.12 * S))]
        else:                            # emitter straight off the wrist
            e_end = _q(zw + wh2)
            octo_z(t.barrel, 0, 0, e_end, e_end + 0.07 * S, er, ring_col, cut=0.3)
            octo_z(t.barrel, 0, 0, e_end + 0.07 * S, e_end + 0.12 * S,
                   er * 0.55, glow, idx=_GLOW)
            t.muzzles = [(0.0, 0.0, _q(e_end + 0.12 * S))]
        return t

    # =============================================================== body
    rh = _rng("body")
    hcol = rh.choice((prim, shade(prim, 0.88), mix(prim, sec, 0.35)))
    hw = _q({"cannon": 0.68, "chaingun": 0.75, "laser": 0.5,
             "railgun": 0.6, "launcher": 0.85}[kind] * W * (0.85 + 0.3 * rh.random()))
    hh = _q({"cannon": 0.30, "chaingun": 0.26, "laser": 0.36,
             "railgun": 0.22, "launcher": 0.38}[kind] * S * (0.85 + 0.3 * rh.random()))
    hz0 = _q({"cannon": -0.72, "chaingun": -0.80, "laser": -0.60,
              "railgun": -0.90, "launcher": -0.60}[kind] * W * (0.85 + 0.3 * rh.random()))
    hz1 = _q({"cannon": 0.55, "chaingun": 0.50, "laser": 0.60,
              "railgun": 0.70, "launcher": 0.50}[kind] * W * (0.85 + 0.3 * rh.random()))
    y0 = collar_h
    y1 = _q(y0 + hh)
    arch = rh.choice(("slab", "wedge", "stepped", "turtle", "barbette"))
    ch = _q(min(0.09 * S, hh * 0.4))
    roof_y = y1                          # where roof features stand

    pk = 0.0                             # how far the cheek plates stick out
    if arch == "wedge":                  # low, strongly sloped glacis nose
        y1 = _q(y0 + hh * 0.85)
        roof_y = y1
        box(t.body, -hw, y0, hz0, hw, y1, hz1, hcol)
        glac = _q(min(0.3 * S, (hz1 - hz0) * 0.5))
        wedge(t.body, -hw, y0, hz1, hw, y1, hz1 + glac, hcol, _YP, _ZP)
        wedge(t.body, -hw, y1, hz0, hw, y1 + ch, hz0 + ch, hcol, _YP, _ZN)
    elif arch == "stepped":              # two tiers, upper set back
        ym = _q(y0 + hh * 0.55)
        box(t.body, -hw, y0, hz0, hw, ym, hz1, hcol)
        uw, uz0, uz1 = _q(hw * 0.68), _q(hz0 + 0.08 * S), _q(hz1 - 0.06 * S)
        box(t.body, -uw, ym, uz0, uw, y1, uz1, shade(hcol, 0.92))
        wedge(t.body, -uw, y1, uz1 - ch, uw, y1 + ch, uz1, hcol, _YP, _ZP)
        hz0, hz1 = uz0, uz1              # dorsal features sit on the upper tier
    elif arch == "turtle":               # chamfered on every top edge
        box(t.body, -hw, y0, hz0, hw, y1, hz1, hcol)
        wedge(t.body, -hw + ch, y1, hz1 - ch, hw - ch, y1 + ch, hz1, hcol, _YP, _ZP)
        wedge(t.body, -hw + ch, y1, hz0, hw - ch, y1 + ch, hz0 + ch, hcol, _YP, _ZN)
        wedge(t.body, hw - ch, y1, hz0 + ch, hw, y1 + ch, hz1 - ch, hcol, _YP, _XP)
        wedge(t.body, -hw, y1, hz0 + ch, -hw + ch, y1 + ch, hz1 - ch, hcol, _YP, _XN)
        for sx, dx in ((1, _XP), (-1, _XN)):
            for sz, dz in ((1, _ZP), (-1, _ZN)):
                lu = orient.bevel_corner_orient(_YP, dx, dz)
                if lu:
                    zz1 = hz1 if sz > 0 else hz0 + ch
                    zz0 = hz1 - ch if sz > 0 else hz0
                    box(t.body, sx * (hw - ch), y1, zz0, sx * hw, y1 + ch, zz1,
                        hcol, idx=_CORNER, look=lu[0], up=lu[1])
    elif arch == "barbette":             # octagonal drum housing + flat cap
        cb = _q(min(hw, (hz1 - hz0) / 2) * 0.32)
        box(t.body, -hw, y0, hz0 + cb, hw, y1, hz1 - cb, hcol)
        box(t.body, -(hw - cb), y0, hz1 - cb, hw - cb, y1, hz1, hcol)
        box(t.body, -(hw - cb), y0, hz0, hw - cb, y1, hz0 + cb, hcol)
        for sx, dx in ((1, _XP), (-1, _XN)):
            for sz, dz in ((1, _ZP), (-1, _ZN)):
                wedge(t.body, sx * (hw - cb), y0, sz * ((hz1 if sz > 0 else -hz0) - cb),
                      sx * hw, y1, sz * (hz1 if sz > 0 else -hz0), hcol, dx, dz)
        box(t.body, -(hw - cb), y1, hz0 + cb, hw - cb, y1 + 0.03 * S,
            hz1 - cb, shade(hcol, 0.9))
        roof_y = _q(y1 + 0.03 * S)
        ch = cb                          # keeps roof features off the rim
    else:                                # slab: cheek armour + glacis chamfer
        box(t.body, -hw, y0, hz0, hw, y1, hz1, hcol)
        wedge(t.body, -hw, y1, hz1 - ch, hw, y1 + ch, hz1, hcol, _YP, _ZP)
        pk = _q(0.03 * S)
        for s in (1, -1):
            box(t.body, s * hw, y0 + hh * 0.15, hz0 + 0.06 * S,
                s * (hw + pk), y0 + hh * 0.8, hz1 - 0.08 * S, shade(hcol, 0.8))

    # ---- bolt-on features, rolled independently -------------------------
    rf = _rng("feat")
    zc = _q((hz0 + hz1) / 2)
    fy0, fy1 = (ym, y1) if arch == "stepped" else (y0, y1)
    fw = uw if arch == "stepped" else hw
    fh = fy1 - fy0
    has_ammo = rf.random() < 0.5 or kind == "launcher"
    if has_ammo:                         # rear ammo/equipment box
        box(t.body, -fw * 0.7, fy0 + fh * 0.1, hz0 - (0.1 + 0.08 * rf.random()) * S,
            fw * 0.7, fy0 + fh * (0.75 + 0.2 * rf.random()), hz0, shade(sec, 0.8))
    else:                                # cooling vents on the rear face
        for i in range(3):
            vy = fy0 + fh * (0.2 + 0.22 * i)
            box(t.body, -fw * 0.55, vy, hz0 - 0.02 * S, fw * 0.55,
                vy + fh * 0.1, hz0, _DARK)
    drum_side = 0
    if kind == "chaingun" or rf.random() < 0.18:   # ammo drum on a flank
        drum_side = rf.choice((1, -1))
        box(t.body, drum_side * (hw + pk), y0 + hh * 0.08, -0.3 * W,
            drum_side * (hw + pk + 0.2 * S), y0 + hh * 0.55, 0.2 * W,
            shade(sec, 0.85))
    if rf.random() < 0.55:               # operator window: dark frame + glow pane
        ws = -drum_side if drum_side else rf.choice((1, -1))
        wy0, wy1 = _q(fy0 + fh * 0.45), _q(fy0 + fh * 0.8)
        wz0, wz1 = _q(hz0 + (hz1 - hz0) * 0.55), _q(hz0 + (hz1 - hz0) * 0.85)
        wx = _q(fw + (pk if arch == "slab" else 0))
        box(t.body, ws * wx, wy0, wz0, ws * (wx + 0.012 * S), wy1, wz1, _DARK)
        box(t.body, ws * (wx + 0.012 * S), wy0 + fh * 0.05, wz0 + 0.02 * S,
            ws * (wx + 0.02 * S), wy1 - fh * 0.05, wz1 - 0.02 * S, glow, idx=_GLOW)
    # top features stay inside the chamfer-free zone of the roof
    tz0, tz1 = _q(hz0 + ch), _q(hz1 - ch)
    if rf.random() < 0.4:                # sensor mast + emissive tip
        ms = _q(max(0.035 * S, 2 * step))
        mz = _q(max(hz0 + (hz1 - hz0) * 0.3, tz0 + ms))
        mh = _q((0.12 + 0.1 * rf.random()) * S)
        octo_y(t.body, 0, mz, roof_y, roof_y + mh, ms, shade(sec, 0.6))
        box(t.body, -ms, roof_y + mh, mz - ms, ms, roof_y + mh + 0.04 * S, mz + ms,
            glow, idx=_GLOW)
        tz0 = _q(mz + ms)                # later roof features start behind it
    if rf.random() < 0.35:               # rangefinder stubs, rear flank quarter
        ry = _q(fy0 + fh * 0.6)
        sx = _q(fw + (pk if arch == "slab" else 0))
        rz = _q(hz0 + (hz1 - hz0) * 0.25)
        for s in (1, -1):
            box(t.body, s * sx, ry, rz - 0.05 * S, s * (sx + 0.1 * S),
                ry + fh * 0.22, rz + 0.05 * S, shade(sec, 0.65))
    if rf.random() < 0.45:               # cable conduit, below the window band
        cs = -drum_side if drum_side else rf.choice((1, -1))
        box(t.body, cs * (hw + pk), y0 + hh * 0.24, hz0 + (hz1 - hz0) * 0.2,
            cs * (hw + pk + 0.02 * S), y0 + hh * 0.38, hz0 + (hz1 - hz0) * 0.85,
            shade(sec, 0.5))
    # one accent element, placement rolled ("front" clashes with a wedge glacis)
    place = rf.choice(("sides", "top") if arch == "wedge" else ("sides", "top", "front"))
    # flank bands are height-separated: accent 8-22%, conduit 24-38%,
    # window 45-80%, rangefinder 60-82% — they may share a flank safely
    if place == "sides":
        for s in ((-drum_side,) if drum_side else (1, -1)):
            box(t.body, s * (hw + pk), y0 + hh * 0.08, zc - 0.12 * S,
                s * (hw + pk + 0.015 * S), y0 + hh * 0.22, zc + 0.12 * S, acc)
    elif place == "top":
        az0 = max(tz0, _q(hz0 + (hz1 - hz0) * 0.5))
        az1 = min(tz1, _q(hz0 + (hz1 - hz0) * 0.9))
        if az1 > az0:
            box(t.body, -hw * 0.25, roof_y, az0, hw * 0.25, roof_y + 0.02 * S,
                az1, acc)
    else:
        box(t.body, -fw * 0.6, fy0 + fh * 0.32, hz1, fw * 0.6,
            fy0 + fh * 0.68, hz1 + 0.02 * S, acc)
    if place != "top" and rf.random() < 0.6:   # dorsal glow strip
        gz = max(tz0, _q(hz0 + (hz1 - hz0) * (0.45 + 0.25 * rf.random())))
        gz1 = min(tz1, _q(gz + 0.14 * S))
        if gz1 > gz:
            box(t.body, -0.05 * S, roof_y, gz, 0.05 * S, roof_y + 0.02 * S, gz1,
                glow, idx=_GLOW)

    # ============================================================== barrel
    rr = _rng("barrel")
    nb = spec.barrels if spec.barrels and spec.barrels > 0 else \
        {"cannon": rr.choice((1, 1, 2)), "chaingun": rr.choice((2, 3, 4)),
         "laser": 1, "railgun": 2, "launcher": rr.choice((2, 3, 4))}[kind]
    nb = max(1, min(4, int(nb)))
    cal = _q({"cannon": 0.15, "chaingun": 0.085, "laser": 0.11,
              "railgun": 0.07, "launcher": 0.11}[kind] * S * (0.85 + 0.3 * rr.random()))
    cal = max(cal, 2 * step)
    L = _q({"cannon": 1.05, "chaingun": 0.85, "laser": 0.90,
            "railgun": 1.45, "launcher": 0.55}[kind] * S * (0.75 + 0.5 * rr.random()))
    gap = {"railgun": 2.6, "launcher": 1.3}.get(kind, 2.4)
    spread = _q((nb - 1) * gap * cal)
    z0t = _q(0.06 * S)

    # pivot: mid-housing, but raised so the fattest barrel part clears the
    # base plate (the gatling drum used to sink into it)
    d = _q(max(0.8 * cal, 0.03 * S))               # gatling cluster radius
    dw = min(_q(1.9 * (d + cal) + 0.02 * S), _q(0.3 * S))   # gatling drum
    m_w = min(1.9 * cal if nb == 1 else 1.1 * cal, 0.2 * S)  # cannon mantlet
    cols = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2)}[nb]
    pitch = _q(2.0 * cal)
    clear = {"chaingun": dw, "cannon": m_w, "laser": 1.7 * cal,
             "railgun": 2 * cal,
             "launcher": cols[1] * pitch / 2 + 0.06 * S}[kind]
    piv_y = _q(max((y0 + y1) / 2, plate_top + clear + 0.02 * S))
    piv_z = _q(0.15 * W)
    t.barrel_pivot = (0.0, piv_y, piv_z)

    # yoke spans all tubes and reaches back over the pivot
    yk = _q(max(cal * 1.1, 0.09 * S))
    yx = _q(max(spread / 2 + cal, 0.14 * S))
    box(t.barrel, -yx, -yk, -0.22 * S, yx, yk, z0t, shade(prim, 0.85))
    for s in (1, -1):                    # trunnion caps
        box(t.barrel, s * yx, -yk * 0.6, -0.16 * S, s * (yx + 0.04 * S),
            yk * 0.6, -0.02 * S, shade(sec, 0.6))

    if kind == "launcher":
        bx = _q(cols[0] * pitch / 2 + 0.02 * S)
        by = _q(cols[1] * pitch / 2 + 0.02 * S)
        z_box = z0t
        if rr.random() < 0.5:            # blast shield plate ahead of the yoke
            box(t.barrel, -bx - 0.04 * S, -by - 0.04 * S, z0t,
                bx + 0.04 * S, by + 0.04 * S, z0t + 0.03 * S, shade(sec, 0.6))
            z_box = _q(z0t + 0.03 * S)
        bcol = mix(sec, prim, 0.45)
        zL = _q(z_box + L)
        box(t.barrel, -bx, -by, z_box, bx, by, zL, bcol)
        chb = _q(min(0.05 * S, by * 0.5))
        wedge(t.barrel, -bx + chb, by, zL - chb, bx - chb, by + chb, zL,
              bcol, _YP, _ZP)            # chamfered pod roof, front and rear
        wedge(t.barrel, -bx + chb, by, z_box, bx - chb, by + chb, z_box + chb,
              bcol, _YP, _ZN)
        box(t.barrel, -bx * 0.8, by, zL - 0.14 * S, bx * 0.8, by + 0.015 * S,
            zL - 0.09 * S, acc)          # accent band near the muzzle end
        for s in (1, -1):
            box(t.barrel, s * bx, -by * 0.6, zL - 0.14 * S,
                s * (bx + 0.012 * S), by * 0.6, zL - 0.09 * S, acc)
        for i in range(cols[0]):         # raised tube collars + dark caps
            for j in range(cols[1]):
                cxx = _q((i - (cols[0] - 1) / 2) * pitch)
                cyy = _q((j - (cols[1] - 1) / 2) * pitch)
                octo_z(t.barrel, cxx, cyy, zL, zL + 0.03 * S, cal * 0.62,
                       shade(sec, 0.6))
                box(t.barrel, cxx - cal * 0.3, cyy - cal * 0.3, zL + 0.03 * S,
                    cxx + cal * 0.3, cyy + cal * 0.3, zL + 0.045 * S, _DARK)
                t.muzzles.append((cxx, cyy, _q(zL + 0.045 * S)))
    elif kind == "chaingun":
        # reference look: a fat octagonal gun drum with a bright ring face
        # and the barrel cluster poking out of its dark core
        pat = {1: ((0, 0),), 2: ((-1, 0), (1, 0)),
               3: ((-1, -0.6), (1, -0.6), (0, 0.8)),
               4: ((-1, -1), (1, -1), (-1, 1), (1, 1))}[nb]
        d_len = _q(0.45 * L)
        octo_z(t.barrel, 0, 0, z0t, z0t + d_len, dw, barrel_metal)
        face_z = drum_face(t.barrel, 0, 0, _q(z0t + d_len), dw, 0.04 * S)
        # accent ring stripe across the drum's flat top band
        box(t.barrel, -dw * 0.6, dw, z0t + d_len * 0.35, dw * 0.6,
            dw + 0.015 * S, z0t + d_len * 0.65, acc)
        for fx, fy in pat:
            x, yy = _q(fx * d), _q(fy * d)
            box(t.barrel, x - cal / 2, yy - cal / 2, face_z,
                x + cal / 2, yy + cal / 2, z0t + L, barrel_metal)
            box(t.barrel, x - cal * 0.6, yy - cal * 0.6, z0t + L,
                x + cal * 0.6, yy + cal * 0.6, z0t + L + 0.03 * S, shade(sec, 0.55))
            t.muzzles.append((x, yy, _q(z0t + L + 0.03 * S)))
    else:
        brake = rr.choice(("ring", "baffle", "none")) if kind == "cannon" else None
        for k in range(nb):
            x = _q((k - (nb - 1) / 2) * gap * cal)
            z_end = _q(z0t + L)
            if kind == "cannon":
                # mantlet drum at the root, then octagonal barrel segments
                # with machined cone tapers at each width step
                m_end = _q(z0t + 0.14 * S)
                octo_z(t.barrel, x, 0, z0t, m_end, m_w, ring_col)
                zs = _q(m_end + (z_end - m_end) * (0.3 + 0.12 * rr.random()))
                octo_z(t.barrel, x, 0, m_end, zs, cal * 1.0, barrel_metal)
                octo_taper(t.barrel, x, 0, m_end, _q(m_end + 0.05 * S),
                           m_w, cal * 1.0, ring_col)
                zb = _q(z_end - (0.10 * S if brake != "none" else 0.0))
                octo_z(t.barrel, x, 0, zs, zb, cal * 0.62, barrel_metal)
                octo_taper(t.barrel, x, 0, zs, _q(zs + 0.04 * S),
                           cal * 1.0, cal * 0.62, barrel_metal)
                if brake == "ring":       # bright muzzle drum
                    drum_face(t.barrel, x, 0, zb, cal * 1.05, 0.08 * S)
                elif brake == "baffle":   # stepped double baffle
                    zm = _q(zb + 0.05 * S)
                    octo_z(t.barrel, x, 0, zb, zm, cal * 0.95, shade(sec, 0.55))
                    octo_z(t.barrel, x, 0, zm, z_end, cal * 0.7, shade(sec, 0.55))
                else:
                    octo_z(t.barrel, x, 0, zb, z_end, cal * 0.5, _DARK)
                if rr.random() < 0.55 and nb == 1:  # recoil cylinder above
                    box(t.barrel, x - cal * 0.3, cal * 1.0, _q(m_end + 0.06 * S),
                        x + cal * 0.3, cal * 1.5, _q(m_end + L * 0.4), shade(sec, 0.6))
                t.muzzles.append((x, 0.0, _q(z_end + 0.02 * S)))
            elif kind == "laser":
                # emitter dish, then tube segments between wider focus rings
                m_end = _q(z0t + 0.12 * S)
                e_w = min(1.7 * cal if nb == 1 else 1.05 * cal, 0.18 * S)
                octo_z(t.barrel, x, 0, z0t, m_end, e_w, prim)
                octo_taper(t.barrel, x, 0, m_end, _q(m_end + 0.04 * S),
                           e_w, cal * 0.45, prim)
                n_rings = rr.choice((2, 3))
                zprev = m_end
                for i in range(n_rings):
                    zr = _q(z0t + L * (0.35 + 0.5 * (i + 1) / (n_rings + 0.5)))
                    box(t.barrel, x - cal * 0.45, -cal * 0.45, zprev,
                        x + cal * 0.45, cal * 0.45, zr, barrel_metal)
                    octo_z(t.barrel, x, 0, zr, _q(zr + 0.04 * S), cal * 0.8,
                           ring_col)
                    zprev = _q(zr + 0.04 * S)
                box(t.barrel, x - cal * 0.45, -cal * 0.45, zprev,
                    x + cal * 0.45, cal * 0.45, z_end, barrel_metal)
                octo_z(t.barrel, x, 0, z_end, _q(z_end + 0.07 * S), cal * 0.6,
                       glow, idx=_GLOW)
                # glow rail over the first tube segment only (rings are
                # wider, and it starts past the emitter taper)
                first_ring = _q(z0t + L * (0.35 + 0.5 / (n_rings + 0.5)))
                box(t.barrel, x - cal * 0.15, cal * 0.45, _q(m_end + 0.05 * S),
                    x + cal * 0.15, cal * 0.45 + 0.02 * S, first_ring,
                    glow, idx=_GLOW)
                t.muzzles.append((x, 0.0, _q(z_end + 0.07 * S)))
            elif kind == "railgun":
                box(t.barrel, x - cal / 2, -cal / 2, z0t, x + cal / 2, cal / 2,
                    z_end, barrel_metal)
                t.muzzles.append((x, 0.0, z_end))
        if kind == "railgun" and nb >= 2:
            inx = _q(spread / 2 - cal / 2)
            for fz in (0.3, 0.62, 0.9):  # spacer bars between the rails
                box(t.barrel, -inx, -cal * 0.4, _q(z0t + L * fz), inx, cal * 0.4,
                    _q(z0t + L * fz + 0.04 * S), mix(sec, acc, 0.5))
            box(t.barrel, -inx, -cal * 0.25, z0t, inx, cal * 0.25,
                _q(z0t + L * 0.14), glow, idx=_GLOW)   # arc glow near the breech
            for s in (1, -1):            # capacitor drums outside the rails
                octo_z(t.barrel, s * (spread / 2 + cal / 2 + 0.8 * cal), 0,
                       z0t, _q(z0t + L * 0.35), cal * 0.8, shade(sec, 0.8))
            t.muzzles = [(0.0, 0.0, _q(z_end))]        # one shot between rails

    if not t.muzzles:
        t.muzzles = [(0.0, 0.0, _q(z0t + L))]
    return t
