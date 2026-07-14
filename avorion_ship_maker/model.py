"""Core data model: a single :class:`Block` and a :class:`ShipModel` collection.

A block is an axis-aligned box (Avorion's fundamental building unit) defined by a
lower corner ``(lx, ly, lz)`` and an upper corner ``(ux, uy, uz)`` plus type,
material, orientation and colour metadata that mirror the Avorion ``<block>`` tag.

Coordinate convention (matches the sample ships in ``models/``):
    x — right (+) / left (-)     ship is mirror-symmetric across x=0
    y — up (+)  / down (-)
    z — forward/bow (+) / aft/stern (-)
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable


def fmt_num(v: float) -> str:
    """Format a coordinate the way Avorion does: no trailing ``.0`` for integers,
    trimmed decimals otherwise. Also scrubs binary-float noise (``0.30000004``)."""
    r = round(float(v), 6)
    if r == int(r):
        return str(int(r))
    return f"{r:.6f}".rstrip("0").rstrip(".")


@dataclass
class Block:
    """One Avorion block (an axis-aligned box)."""

    lx: float
    ly: float
    lz: float
    ux: float
    uy: float
    uz: float
    index: int = 1          # block *type* id (see blocks.ROLE_INDEX)
    material: int = 1       # 0=Iron 1=Titanium 2=Naonite 3=Trinium 4=Xanion 5=Ogonite 6=Avorion
    look: int = 1           # texture variant
    up: int = 3             # orientation code 0..5
    color: str = "ff1e1e1e"       # ARGB hex, 8 chars
    secondary: str = "00000000"   # secondaryColor ARGB hex

    def __post_init__(self) -> None:
        # Guarantee lower < upper on every axis (Avorion requires it).
        if self.lx > self.ux:
            self.lx, self.ux = self.ux, self.lx
        if self.ly > self.uy:
            self.ly, self.uy = self.uy, self.ly
        if self.lz > self.uz:
            self.lz, self.uz = self.uz, self.lz

    # -- derived geometry -------------------------------------------------
    @property
    def size(self) -> tuple[float, float, float]:
        return (self.ux - self.lx, self.uy - self.ly, self.uz - self.lz)

    @property
    def center(self) -> tuple[float, float, float]:
        return ((self.lx + self.ux) / 2, (self.ly + self.uy) / 2, (self.lz + self.uz) / 2)

    @property
    def volume(self) -> float:
        dx, dy, dz = self.size
        return dx * dy * dz

    def is_degenerate(self, eps: float = 1e-6) -> bool:
        dx, dy, dz = self.size
        return dx <= eps or dy <= eps or dz <= eps

    def mirrored_x(self) -> "Block":
        """Return a copy mirrored across the x=0 plane (for bilateral symmetry).

        The ``up`` orientation and ``look`` are preserved; that is visually fine
        for the box-based shapes this generator emits.
        """
        return replace(self, lx=-self.ux, ux=-self.lx)

    def scaled(self, factor: float) -> "Block":
        return replace(
            self,
            lx=self.lx * factor, ly=self.ly * factor, lz=self.lz * factor,
            ux=self.ux * factor, uy=self.uy * factor, uz=self.uz * factor,
        )

    def translated(self, dx: float = 0, dy: float = 0, dz: float = 0) -> "Block":
        return replace(
            self,
            lx=self.lx + dx, ly=self.ly + dy, lz=self.lz + dz,
            ux=self.ux + dx, uy=self.uy + dy, uz=self.uz + dz,
        )


@dataclass
class ShipModel:
    """An ordered collection of blocks plus a display name."""

    blocks: list[Block] = field(default_factory=list)
    name: str = "Ship"

    def add(self, block: Block) -> Block:
        self.blocks.append(block)
        return block

    def extend(self, blocks: Iterable[Block]) -> None:
        self.blocks.extend(blocks)

    def __len__(self) -> int:
        return len(self.blocks)

    def drop_degenerate(self, eps: float = 1e-6) -> None:
        self.blocks = [b for b in self.blocks if not b.is_degenerate(eps)]

    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return ``((minx,miny,minz),(maxx,maxy,maxz))`` over all blocks."""
        if not self.blocks:
            return ((0, 0, 0), (0, 0, 0))
        lo = [min(b.lx for b in self.blocks),
              min(b.ly for b in self.blocks),
              min(b.lz for b in self.blocks)]
        hi = [max(b.ux for b in self.blocks),
              max(b.uy for b in self.blocks),
              max(b.uz for b in self.blocks)]
        return (tuple(lo), tuple(hi))

    def dimensions(self) -> tuple[float, float, float]:
        (lx, ly, lz), (hx, hy, hz) = self.bounds()
        return (hx - lx, hy - ly, hz - lz)

    def total_volume(self) -> float:
        return sum(b.volume for b in self.blocks)

    def type_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for b in self.blocks:
            counts[b.index] = counts.get(b.index, 0) + 1
        return counts
