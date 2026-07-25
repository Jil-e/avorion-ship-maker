"""Avorion block-type registry.

Every ``index`` value used here is taken from the *real* sample ships in
``models/`` (Kruto-1.xml, Shustik.xml), so anything this generator emits
references a block type Avorion already knows and will load.

The *roles* (semantic names) are inferred from where each type sits in the
sample ships (position, volume, colour, material) plus Avorion domain
knowledge. They are used by generators and previews to describe what is placed
where; the
exact functional balance of the resulting ship is not guaranteed (this is a
*design* tool), but the geometry is valid and importable.
"""
from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    HULL = "hull"                 # index 1  — light structural filler (full cube)
    BLANK_HULL = "blank_hull"     # index 2  — untextured hull
    ARMOR = "armor"               # index 8  — heavy plating (full cube)
    FRAMEWORK = "framework"       # index 9  — very light structural
    STONE = "stone"               # index 4
    ENGINE = "engine"             # index 3  — main thrust (stern), glows
    THRUSTER = "thruster"         # index 7  — braking / lateral thrust
    DIR_THRUSTER = "dir_thruster" # index 13 — directional thruster
    GYRO = "gyro"                 # index 14 — gyro array (turning)
    DAMPENER = "dampener"         # index 15 — inertia dampener
    GENERATOR = "generator"       # index 52 — energy generation
    BATTERY = "battery"           # index 51 — energy container
    INTEGRITY = "integrity"       # index 53 — integrity field generator
    SHIELD = "shield"             # index 50 — shield generator
    COMPUTER = "computer"         # index 54 — computer core
    CARGO = "cargo"               # index 5  — cargo bay
    CREW = "crew"                 # index 6  — crew quarters
    HANGAR = "hangar"             # index 10
    ASSEMBLY = "assembly"         # index 17
    GLOW = "glow"                 # index 150 — glow (full cube, emissive)
    LIGHT = "light"               # index 61
    TURRET_BASE = "turret_base"   # index 20 — turret mounting socket


# Role -> Avorion block *type* index. Verified against Greatheart's community
# "Block IDs (Index) Mapped" sheet + the Boxelware forum + OBJ-converter source.
# NOTE: every index below is a FULL-CUBE block. Shape variants (edges/corners/
# wedges: Hull 100-103, Armour 104-107, Glow 151-154, ...) are intentionally NOT
# used here — the generator emits axis-aligned boxes, and a shape block placed in a
# full box would render as a wedge, not a cube. Add them only with orientation.
ROLE_INDEX: dict[Role, int] = {
    Role.HULL: 1,
    Role.BLANK_HULL: 2,
    Role.ARMOR: 8,
    Role.FRAMEWORK: 9,
    Role.STONE: 4,
    Role.ENGINE: 3,
    Role.THRUSTER: 7,
    Role.DIR_THRUSTER: 13,
    Role.GYRO: 14,
    Role.DAMPENER: 15,
    Role.GENERATOR: 52,
    Role.BATTERY: 51,
    Role.INTEGRITY: 53,
    Role.SHIELD: 50,
    Role.COMPUTER: 54,
    Role.CARGO: 5,
    Role.CREW: 6,
    Role.HANGAR: 10,
    Role.ASSEMBLY: 17,
    Role.GLOW: 150,
    Role.LIGHT: 61,
    # GROUND TRUTH: the game's own plans (data/plans/*) and pads the game
    # binds real turret designs to all use 20; none use 25.
    Role.TURRET_BASE: 20,
}

# index -> human name, for the full known catalog (full-cube blocks + note on shapes).
BLOCK_NAMES: dict[int, str] = {
    1: "Hull", 2: "Blank Hull", 3: "Engine", 4: "Stone", 5: "Cargo Bay",
    6: "Crew Quarters", 7: "Thruster", 8: "Armour", 9: "Framework", 10: "Hangar",
    20: "Turret Base", 25: "Armored Turret Base",
    13: "Directional Thruster", 14: "Gyro Array", 15: "Inertia Dampener", 17: "Assembly",
    50: "Shield Generator", 51: "Battery", 52: "Generator", 53: "Integrity Field Generator",
    54: "Computer Core", 55: "Hyperspace Core", 61: "Light",
    100: "Hull Edge", 101: "Hull Corner", 102: "Hull Outer Corner", 103: "Hull Inner Corner",
    104: "Armour Edge", 105: "Armour Corner", 106: "Armour Outer Corner",
    107: "Armour Inner Corner", 150: "Glow", 151: "Glow Edge", 152: "Glow Corner",
    153: "Glow Outer Corner", 154: "Glow Inner Corner",
}

# All block-type indices seen in the reference ships — the "known-valid" set.
KNOWN_INDICES: set[int] = {
    1, 2, 3, 5, 6, 7, 8, 9, 13, 14, 15, 19, 20, 22, 24, 25,
    51, 52, 53, 61, 100, 101, 102, 103, 104, 105, 106, 107, 113, 123, 124,
    150, 151, 152, 153, 154,
}

# Material tiers (index -> name); higher = stronger/lighter in-game.
MATERIALS: dict[int, str] = {
    0: "Iron",
    1: "Titanium",
    2: "Naonite",
    3: "Trinium",
    4: "Xanion",
    5: "Ogonite",
    6: "Avorion",
}

MATERIAL_BY_NAME: dict[str, int] = {v.lower(): k for k, v in MATERIALS.items()}


# base full-cube index -> (edge-shape index, corner-shape index).
# Used by the bevel pass to chamfer convex hull edges/corners.
SHAPE_VARIANTS: dict[int, tuple[int, int]] = {
    1: (100, 101),    # Hull -> Hull Edge, Hull Corner
    8: (104, 105),    # Armour -> Armour Edge, Armour Corner
    150: (151, 152),  # Glow -> Glow Edge, Glow Corner
}

# Real auto-ships use all three corner solids in each material family.
CORNER_VARIANTS: dict[int, tuple[int, int, int]] = {
    1: (101, 102, 103),
    8: (105, 106, 107),
    150: (152, 153, 154),
}


def index_for(role: Role) -> int:
    """Block-type index for a role (falls back to Hull)."""
    return ROLE_INDEX.get(role, ROLE_INDEX[Role.HULL])


def material_id(name_or_id) -> int:
    """Resolve a material given a name (case-insensitive) or an integer id."""
    if isinstance(name_or_id, int):
        return name_or_id if name_or_id in MATERIALS else 1
    return MATERIAL_BY_NAME.get(str(name_or_id).strip().lower(), 1)


def block_registry() -> dict[int, str]:
    """Return an index -> role-name map (for UI / debugging)."""
    reg = {idx: role.value for role, idx in ROLE_INDEX.items()}
    for idx in KNOWN_INDICES:
        reg.setdefault(idx, f"type_{idx}")
    return dict(sorted(reg.items()))
