"""The :class:`ShipSpec` — every knob the procedural builder reads.

A spec is produced either by the text parser (from a free-form description) or
directly from the UI controls. ``hull_class`` and ``style`` supply presets that
individual fields override when set explicitly.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, replace


# hull_class -> randomization RULES, not fixed values. A (lo, hi) tuple is the
# range each variant rolls its own value from (per-seed, so a gallery click
# reproduces the exact ship); a bare number is fixed; for the part toggles
# (wings/fins/bridge) the number is the probability of "on". Explicit spec
# fields always override the roll.
# length = Z (fore/aft), width = X (full, port-starboard), height = Y (full).
HULL_CLASSES: dict[str, dict] = {
    "fighter":    dict(length=(16, 26),  width=(10, 16), height=(6, 9),   engines=(1, 2), turrets=(0, 1), wings=0.95, fins=0.70, bridge=0.90, boxiness=(0.20, 0.55), nose=(0.35, 0.60), taper_tail=(0.08, 0.20)),
    "corvette":   dict(length=(24, 38),  width=(12, 18), height=(7, 11),  engines=(1, 3), turrets=(1, 2), wings=0.85, fins=0.60, bridge=0.95, boxiness=(0.35, 0.65), nose=(0.30, 0.50), taper_tail=(0.08, 0.18)),
    "frigate":    dict(length=(38, 56),  width=(13, 20), height=(10, 15), engines=(2, 3), turrets=(2, 4), wings=0.75, fins=0.35, bridge=0.95, boxiness=(0.45, 0.72), nose=(0.28, 0.48), taper_tail=(0.06, 0.14)),
    "cruiser":    dict(length=(54, 80),  width=(17, 26), height=(12, 19), engines=(2, 4), turrets=(3, 6), wings=0.25, fins=0.60, bridge=0.95, boxiness=(0.55, 0.80), nose=(0.24, 0.42), taper_tail=(0.06, 0.14)),
    "battleship": dict(length=(76, 110), width=(23, 34), height=(16, 25), engines=(3, 6), turrets=(5, 9), wings=0.10, fins=0.55, bridge=0.95, boxiness=(0.65, 0.88), nose=(0.20, 0.36), taper_tail=(0.05, 0.12)),
    "freighter":  dict(length=(48, 76),  width=(18, 29), height=(16, 25), engines=(2, 4), turrets=(1, 3), wings=0.10, fins=0.15, bridge=0.90, boxiness=(0.78, 0.97), nose=(0.10, 0.28), taper_tail=(0.04, 0.10)),
    "miner":      dict(length=(32, 52),  width=(14, 23), height=(12, 19), engines=(1, 3), turrets=(1, 2), wings=0.15, fins=0.20, bridge=0.90, boxiness=(0.60, 0.95), nose=(0.12, 0.35), taper_tail=(0.05, 0.14)),
    "carrier":    dict(length=(68, 100), width=(25, 38), height=(15, 23), engines=(2, 4), turrets=(3, 6), wings=0.60, fins=0.20, bridge=0.90, boxiness=(0.70, 0.92), nose=(0.18, 0.34), taper_tail=(0.05, 0.12)),
    "station":    dict(length=(30, 48),  width=(30, 48), height=(30, 48), engines=0,      turrets=(2, 6), wings=0.0,  fins=0.0,  bridge=0.0,  boxiness=(0.85, 1.0),  nose=(0.03, 0.10), taper_tail=(0.03, 0.10)),
}

DEFAULT_CLASS = "frigate"


def _mid(v):
    return (v[0] + v[1]) / 2.0 if isinstance(v, tuple) else v


def class_baseline(name: str) -> dict:
    """Midpoint of every class rule — the 'typical' ship of that class.

    Used by the text parser as the base that size words ("огромный", "плоский")
    multiply; the builder itself rolls inside the full ranges instead.
    """
    hc = HULL_CLASSES.get(name, HULL_CLASSES[DEFAULT_CLASS])
    return {k: _mid(v) for k, v in hc.items()}

# style -> aesthetic preset (colours + armour density + detail).
STYLES: dict[str, dict] = {
    "military":   dict(primary="4a4f52", secondary="2b2e30", accent="c0392b", glow="ff5a2a", armor=0.7, detail=0.75, material=1),
    "civilian":   dict(primary="d8dde0", secondary="9aa0a4", accent="2e86c1", glow="7fd8ff", armor=0.35, detail=0.65, material=1),
    "stealth":    dict(primary="1a1c1e", secondary="0d0e10", accent="6c3483", glow="9b59b6", armor=0.8, detail=0.55, material=1),
    "industrial": dict(primary="6e5b3a", secondary="3d3527", accent="e39a1c", glow="ffb020", armor=0.55, detail=0.8, material=0),
    "sleek":      dict(primary="ecf0f1", secondary="34495e", accent="1abc9c", glow="6ff3d6", armor=0.4, detail=0.7, material=3),
    "hazard":     dict(primary="e0c020", secondary="1c1c1c", accent="e0c020", glow="fff040", armor=0.6, detail=0.75, material=1),
}

DEFAULT_STYLE = "military"

# hull layout archetypes (None/auto => picked from the seed, weighted by class)
LAYOUTS: tuple[str, ...] = ("mono", "twin", "pods", "keel", "hammer", "fork")
LAYOUT_NAMES = {"mono": "монокорпус", "twin": "катамаран",
                "pods": "гондолы", "keel": "надстройка",
                "hammer": "молот (широкий нос)", "fork": "вилка (раздвоенный нос)"}

# wing archetypes (None/auto => picked from the seed; see builder._WING_KINDS)
WING_KINDS: tuple[str, ...] = ("swept", "forward", "delta", "gull",
                               "xfoil", "tipfin", "tippod")


@dataclass
class ShipSpec:
    """Full specification for one generated ship.

    Dimensions (``length``/``width``/``height``) are in *voxels*; the real Avorion
    size is ``voxels * block_size * scale``. Fields left as ``None`` inherit from
    the ``hull_class`` / ``style`` presets when :meth:`resolved` is called.
    """

    name: str = "Generated Ship"
    hull_class: str = DEFAULT_CLASS
    style: str = DEFAULT_STYLE
    layout: str | None = None           # mono/twin/pods/keel; None => from seed

    # geometry (voxels) — None => take from hull_class preset
    length: int | None = None
    width: int | None = None
    height: int | None = None

    # shaping 0..1
    boxiness: float | None = None       # 0 = smooth/elliptical cross-section, 1 = rectangular
    nose: float | None = None           # fraction of length that tapers at the bow
    taper_tail: float | None = None     # fraction of length that tapers at the stern
    armor: float | None = None          # thickness/probability of outer armour skin
    detail: float | None = None         # amount of surface greebles + glow accents
    bevel: float | None = None          # 0=blocky cubes, 1=chamfer all convex edges/corners
    functional: float | None = None     # 0=pure aesthetics, 1=max operable-block content

    # parts
    engines: int | None = None
    turrets: int | None = None          # dorsal turret-base pads (block 25)
    wings: bool | None = None
    wing_kind: str | None = None        # one of WING_KINDS; None => from seed
    fins: bool | None = None
    bridge: bool | None = None

    # scale
    block_size: float = 1.0             # Avorion units per voxel ("step")
    scale: float = 1.0                  # overall multiplier

    # appearance (hex RGB, no alpha) — None => from style
    primary: str | None = None
    secondary: str | None = None
    accent: str | None = None
    glow: str | None = None
    material: int | None = None

    symmetry: bool = True               # bilateral symmetry across x=0
    seed: int = 0

    def resolved(self) -> "ShipSpec":
        """Return a copy with every ``None`` filled from the class/style rules.

        Ranged class rules are *rolled* with a per-field RNG derived from the
        seed — every ``None`` field gets its own value per variant, explicit
        fields stay untouched, and re-resolving the same spec is idempotent.
        """
        hc = HULL_CLASSES.get(self.hull_class, HULL_CLASSES[DEFAULT_CLASS])
        st = STYLES.get(self.style, STYLES[DEFAULT_STYLE])

        def roll(cur, key, as_int=False):
            if cur is not None:
                return cur
            rule = hc[key]
            if isinstance(rule, tuple):
                rng = random.Random(f"{self.seed}:cls:{key}")
                return rng.randint(int(rule[0]), int(rule[1])) if as_int \
                    else rng.uniform(rule[0], rule[1])
            return rule

        def roll_part(cur, key):
            if cur is not None:
                return bool(cur)
            return random.Random(f"{self.seed}:cls:{key}").random() < float(hc[key])

        pick = lambda cur, src, key: cur if cur is not None else src[key]
        return replace(
            self,
            length=int(roll(self.length, "length", as_int=True)),
            width=int(roll(self.width, "width", as_int=True)),
            height=int(roll(self.height, "height", as_int=True)),
            boxiness=float(roll(self.boxiness, "boxiness")),
            nose=float(roll(self.nose, "nose")),
            taper_tail=float(roll(self.taper_tail, "taper_tail")),
            armor=float(pick(self.armor, st, "armor")),
            detail=float(pick(self.detail, st, "detail")),
            bevel=float(self.bevel if self.bevel is not None else st.get("bevel", 0.85)),
            functional=float(self.functional if self.functional is not None else 0.5),
            engines=int(roll(self.engines, "engines", as_int=True)),
            turrets=int(roll(self.turrets, "turrets", as_int=True)),
            wings=roll_part(self.wings, "wings"),
            fins=roll_part(self.fins, "fins"),
            bridge=roll_part(self.bridge, "bridge"),
            primary=pick(self.primary, st, "primary"),
            secondary=pick(self.secondary, st, "secondary"),
            accent=pick(self.accent, st, "accent"),
            glow=pick(self.glow, st, "glow"),
            material=int(pick(self.material, st, "material")),
        )
