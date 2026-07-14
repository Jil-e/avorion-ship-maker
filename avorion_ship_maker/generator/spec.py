"""The :class:`ShipSpec` — every knob the procedural builder reads.

A spec is produced either by the text parser (from a free-form description) or
directly from the UI controls. ``hull_class`` and ``style`` supply presets that
individual fields override when set explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace


# hull_class -> geometric preset (grid dimensions in voxels + part defaults).
# length = Z (fore/aft), width = X (full, port-starboard), height = Y (full).
HULL_CLASSES: dict[str, dict] = {
    "fighter":    dict(length=14, width=10, height=5,  engines=1, wings=True,  fins=True,  bridge=True,  boxiness=0.35, nose=0.45, taper_tail=0.15),
    "corvette":   dict(length=22, width=10, height=7,  engines=2, wings=True,  fins=True,  bridge=True,  boxiness=0.5,  nose=0.4,  taper_tail=0.12),
    "frigate":    dict(length=34, width=12, height=9,  engines=2, wings=True,  fins=False, bridge=True,  boxiness=0.6,  nose=0.38, taper_tail=0.1),
    "cruiser":    dict(length=50, width=16, height=12, engines=3, wings=False, fins=True,  bridge=True,  boxiness=0.68, nose=0.32, taper_tail=0.1),
    "battleship": dict(length=72, width=22, height=16, engines=4, wings=False, fins=True,  bridge=True,  boxiness=0.78, nose=0.28, taper_tail=0.08),
    "freighter":  dict(length=46, width=18, height=16, engines=2, wings=False, fins=False, bridge=True,  boxiness=0.9,  nose=0.18, taper_tail=0.06),
    "miner":      dict(length=30, width=14, height=12, engines=2, wings=False, fins=False, bridge=True,  boxiness=0.85, nose=0.22, taper_tail=0.08),
    "carrier":    dict(length=64, width=24, height=14, engines=3, wings=True,  fins=False, bridge=True,  boxiness=0.82, nose=0.25, taper_tail=0.08),
    "station":    dict(length=30, width=30, height=30, engines=0, wings=False, fins=False, bridge=False, boxiness=0.95, nose=0.05, taper_tail=0.05),
}

DEFAULT_CLASS = "frigate"

# style -> aesthetic preset (colours + armour density + detail).
STYLES: dict[str, dict] = {
    "military":   dict(primary="4a4f52", secondary="2b2e30", accent="c0392b", glow="ff5a2a", armor=0.7, detail=0.6, material=1),
    "civilian":   dict(primary="d8dde0", secondary="9aa0a4", accent="2e86c1", glow="7fd8ff", armor=0.35, detail=0.5, material=1),
    "stealth":    dict(primary="1a1c1e", secondary="0d0e10", accent="6c3483", glow="9b59b6", armor=0.8, detail=0.4, material=1),
    "industrial": dict(primary="6e5b3a", secondary="3d3527", accent="e39a1c", glow="ffb020", armor=0.55, detail=0.7, material=0),
    "sleek":      dict(primary="ecf0f1", secondary="34495e", accent="1abc9c", glow="6ff3d6", armor=0.4, detail=0.55, material=3),
    "hazard":     dict(primary="e0c020", secondary="1c1c1c", accent="e0c020", glow="fff040", armor=0.6, detail=0.65, material=1),
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
        """Return a copy with every ``None`` filled from the class/style presets."""
        hc = HULL_CLASSES.get(self.hull_class, HULL_CLASSES[DEFAULT_CLASS])
        st = STYLES.get(self.style, STYLES[DEFAULT_STYLE])
        pick = lambda cur, src, key: cur if cur is not None else src[key]
        return replace(
            self,
            length=int(pick(self.length, hc, "length")),
            width=int(pick(self.width, hc, "width")),
            height=int(pick(self.height, hc, "height")),
            boxiness=float(pick(self.boxiness, hc, "boxiness")),
            nose=float(pick(self.nose, hc, "nose")),
            taper_tail=float(pick(self.taper_tail, hc, "taper_tail")),
            armor=float(pick(self.armor, st, "armor")),
            detail=float(pick(self.detail, st, "detail")),
            bevel=float(self.bevel if self.bevel is not None else st.get("bevel", 0.7)),
            functional=float(self.functional if self.functional is not None else 0.5),
            engines=int(pick(self.engines, hc, "engines")),
            wings=bool(pick(self.wings, hc, "wings")),
            fins=bool(pick(self.fins, hc, "fins")),
            bridge=bool(pick(self.bridge, hc, "bridge")),
            primary=pick(self.primary, st, "primary"),
            secondary=pick(self.secondary, st, "secondary"),
            accent=pick(self.accent, st, "accent"),
            glow=pick(self.glow, st, "glow"),
            material=int(pick(self.material, st, "material")),
        )
