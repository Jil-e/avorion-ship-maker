"""Procedural Avorion turret designs (the ``<turret_design>`` XML format).

A turret is three block plans — static ``base``, yawing ``body`` and the
elevating ``barrel`` — plus a muzzle position. The schema and units were
taken from the game's own saved designs (``ships/auto-turrets/*.xml``):
coordinates are in build units around the mount point, +z is forward,
+y is up, and the barrel plan is local to its elevation pivot.
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

_HULL = index_for(Role.BLANK_HULL)     # 2 — what the game's own turrets use
_GLOW = index_for(Role.GLOW)           # 150 — emissive tips / stripes
_EDGE = 100                            # hull edge wedge (needs look/up)

# face codes as in builder._FACE_CODES: +x -x +y -y +z -z
_XP, _XN, _YP, _YN, _ZP, _ZN = 4, 5, 3, 2, 1, 0


@dataclass
class TurretSpec:
    kind: str = "cannon"
    size: float = 1.0                   # Avorion turret slot size (0.5 .. 3+)
    barrels: int | None = None          # None => archetype default (per seed)
    style: str = DEFAULT_STYLE          # colour preset, shared with ships
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
    muzzle: tuple = (0.0, 0.0, 0.0)

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
    rng = random.Random(f"{spec.seed}:turret")
    st = STYLES.get(spec.style, STYLES[DEFAULT_STYLE])
    prim, sec, acc, glow = st["primary"], st["secondary"], st["accent"], st["glow"]
    mat = int(st.get("material", 1))
    S = max(0.25, float(spec.size))
    W = 0.5 * S                          # base half-extent (matches game designs)
    kind = spec.kind if spec.kind in TURRET_KINDS else "cannon"

    t = TurretModel(size=spec.size, shot_color=_shot_color(glow))

    def box(sect, x0, y0, z0, x1, y1, z1, color, idx=_HULL, look=1, up=3):
        sect.append(Block(lx=min(x0, x1), ly=min(y0, y1), lz=min(z0, z1),
                          ux=max(x0, x1), uy=max(y0, y1), uz=max(z0, z1),
                          index=idx, material=mat, look=look, up=up,
                          color=to_argb(color), secondary="00000000"))

    # ---- base: mounting plate + collar --------------------------------
    ph = 0.07 * S
    box(t.base, -W, 0, -W, W, ph, W, shade(sec, 0.7))
    box(t.base, -0.62 * W, ph, -0.62 * W, 0.62 * W, ph + 0.06 * S, 0.62 * W, prim)

    # ---- body: yaw housing, proportions per archetype ------------------
    hw = {"cannon": 0.68, "chaingun": 0.75, "laser": 0.5,
          "railgun": 0.6, "launcher": 0.85}[kind] * W
    hh = {"cannon": 0.30, "chaingun": 0.26, "laser": 0.36,
          "railgun": 0.22, "launcher": 0.38}[kind] * S * (0.9 + 0.2 * rng.random())
    hz0 = {"cannon": -0.72, "chaingun": -0.80, "laser": -0.60,
           "railgun": -0.90, "launcher": -0.60}[kind] * W
    hz1 = {"cannon": 0.55, "chaingun": 0.50, "laser": 0.60,
           "railgun": 0.70, "launcher": 0.50}[kind] * W
    y0 = ph + 0.06 * S                   # housing bottom = collar top
    y1 = y0 + hh
    box(t.body, -hw, y0, hz0, hw, y1, hz1, prim)
    # chamfered top edges front/back (real wedge blocks, oriented)
    ch = min(0.08 * S, hh * 0.45)
    lk, up = orient.bevel_edge_orient(_YP, _ZP)
    box(t.body, -hw, y1, hz1 - ch, hw, y1 + ch, hz1, prim, idx=_EDGE, look=lk, up=up)
    lk, up = orient.bevel_edge_orient(_YP, _ZN)
    box(t.body, -hw, y1, hz0, hw, y1 + ch, hz0 + ch, prim, idx=_EDGE, look=lk, up=up)
    # rear accent stripe + sensor glow (both sit flush on the housing)
    box(t.body, -hw * 0.8, y0 + hh * 0.35, hz0 - 0.04 * S,
        hw * 0.8, y0 + hh * 0.65, hz0, acc)
    box(t.body, -0.08 * S, y1, hz0 + ch, 0.08 * S, y1 + 0.05 * S,
        hz0 + ch + 0.16 * S, glow, idx=_GLOW)
    if kind == "chaingun":               # ammo drum on the flank
        box(t.body, hw, y0 + hh * 0.1, -0.3 * W, hw + 0.22 * S, y0 + hh * 0.9,
            0.2 * W, shade(sec, 0.8))
    if kind == "laser":                  # dorsal glow spine on the housing top
        box(t.body, -0.05 * S, y1, hz0 + ch + 0.2 * S, 0.05 * S, y1 + 0.04 * S,
            hz1 - ch, glow, idx=_GLOW)

    # ---- barrel: yoke + tubes, local to the elevation pivot ------------
    piv_y = (y0 + y1) / 2
    piv_z = 0.15 * W
    t.barrel_pivot = (0.0, piv_y, piv_z)

    nb = spec.barrels if spec.barrels and spec.barrels > 0 else \
        {"cannon": rng.choice((1, 1, 2)), "chaingun": rng.choice((2, 3, 3)),
         "laser": 1, "railgun": 2, "launcher": 4}[kind]
    nb = max(1, min(4, int(nb)))
    cal = {"cannon": 0.15, "chaingun": 0.085, "laser": 0.11,
           "railgun": 0.07, "launcher": 0.16}[kind] * S
    L = {"cannon": 1.05, "chaingun": 0.80, "laser": 0.90,
         "railgun": 1.45, "launcher": 0.55}[kind] * S * (0.85 + 0.3 * rng.random())
    gap = {"railgun": 2.6, "launcher": 1.35}.get(kind, 2.4)
    spread = (nb - 1) * gap * cal
    z0t = 0.06 * S                       # tubes start just ahead of the yoke

    # yoke spans all tubes and reaches back over the pivot
    yk = max(cal * 1.1, 0.09 * S)
    box(t.barrel, -(spread / 2 + cal), -yk, -0.22 * S,
        spread / 2 + cal, yk, z0t, shade(prim, 0.85))

    if kind == "launcher":               # 2x2 grid of short fat tubes
        rows = ((-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5))[:nb]
        pitch = 2.3 * cal
        for fx, fy in rows:
            x, y = fx * pitch, fy * pitch
            box(t.barrel, x - cal, y - cal, z0t, x + cal, y + cal, z0t + L,
                shade(sec, 0.8))
            box(t.barrel, x - cal * 0.55, y - cal * 0.55, z0t + L,
                x + cal * 0.55, y + cal * 0.55, z0t + L + 0.02 * S, "1c1c1c")
        t.muzzle = (0.0, 0.0, z0t + L)
    else:
        for k in range(nb):
            x = (k - (nb - 1) / 2) * gap * cal
            tube_end = z0t + L - (0.10 * S if kind == "cannon" else 0.0)
            box(t.barrel, x - cal / 2, -cal / 2, z0t, x + cal / 2, cal / 2,
                tube_end, shade(sec, 0.75))
            if kind == "cannon":         # muzzle brake ahead of the tube
                box(t.barrel, x - cal * 0.8, -cal * 0.8, tube_end,
                    x + cal * 0.8, cal * 0.8, z0t + L, shade(sec, 0.55))
            elif kind == "laser":        # emissive emitter tip
                box(t.barrel, x - cal * 0.7, -cal * 0.7, z0t + L,
                    x + cal * 0.7, cal * 0.7, z0t + L + 0.07 * S, glow, idx=_GLOW)
            elif kind == "railgun" and k == 0 and nb == 2:
                # spacer bars between the rails' inner faces
                for fz in (0.35, 0.75):
                    box(t.barrel, -spread / 2 + cal / 2, -cal * 0.4, z0t + L * fz,
                        spread / 2 - cal / 2, cal * 0.4, z0t + L * fz + 0.05 * S,
                        mix(sec, acc, 0.5))
        tip = z0t + L + (0.07 * S if kind == "laser" else 0.0)
        t.muzzle = (0.0, 0.0, tip)

    return t
