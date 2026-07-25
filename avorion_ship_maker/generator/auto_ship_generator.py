"""Generate new ships from measured geometry, not copied ship blocks.

The files in Avorion's ``auto-ships`` directory are treated as a geometry
corpus.  Each plan is normalised into a small occupancy field, its longitudinal
profiles are measured, and duplicate plans are removed.  Generation blends
several measured shapes, applies a low-frequency deformation, rasterises the
result on a fresh grid, and only then creates new armour, hull, system, and
turret-pad blocks.

No source block coordinates or source block metadata are emitted by this
module.  The corpus contributes shape statistics and functional-density
statistics only.
"""
from __future__ import annotations

import hashlib
import os
import random
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is a project requirement
    Image = None

from ..model import Block, ShipModel
from ..xml_io import parse_xml
from .voxel import EMPTY, VoxelGrid, greedy_merge


TEMPLATE_FILES = {
    "main": "Main.xml",
    "carry": "Carry4.xml",
    "an_perdole": "An Perdole.xml",
    "crab": "Crab-0.xml",
}

LAYOUT_GROUPS = {
    "falcon": ("main", "carry", "an_perdole"),
    "hammerhead": ("main", "carry"),
    "main": ("main",),
    "carry": ("carry", "an_perdole"),
    "an_perdole": ("an_perdole", "carry"),
    "crab": ("crab", "main"),
}

# These are omitted while measuring the external form.  They are placed again
# as synthetic systems after the new hull occupancy has been created.
FUNCTIONAL_BLOCKS = {
    3, 5, 6, 7, 10, 13, 14, 15, 17, 19, 20, 25,
    50, 51, 52, 53, 54, 55, 60, 61,
}
SYSTEM_BLOCKS = (3, 5, 6, 7, 10, 13, 14, 15, 17, 19, 50, 51, 52, 53, 54, 55)

CANONICAL_X = 48
CANONICAL_Y = 24
CANONICAL_Z = 96
GRID_PITCH = 0.25
FALCON_REFERENCE = Path(__file__).resolve().parents[1] / "Falcon_base.png"


@dataclass(frozen=True)
class AutoShipSpec:
    """Small, user-facing input surface for corpus-driven generation."""

    length: float = 64.0
    width: float = 24.0
    height: float = 12.0
    subsystem_cells: int = 15
    layout: str = "falcon"
    materials: tuple[int, ...] = (1,)
    seed: int = 0
    reference_image: str | None = None


@dataclass(frozen=True)
class _ShapeDNA:
    """Measured shape data for one unique source plan."""

    name: str
    group: str
    occupancy: np.ndarray
    half_width: np.ndarray
    half_height: np.ndarray
    y_offset: np.ndarray
    density: np.ndarray
    functional_ratio: float
    system_mix: tuple[tuple[int, float], ...]
    turret_rate: float


def _auto_ship_root() -> Path:
    return Path(os.path.expandvars(r"%APPDATA%\Avorion\ships\auto-ships"))


def _template_path(name: str) -> Path:
    filename = TEMPLATE_FILES[name]
    path = _auto_ship_root() / filename
    if path.exists():
        return path
    fallback = Path(__file__).resolve().parents[2] / "models" / filename
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Не найден auto-ship шаблон {filename}: {_auto_ship_root()}"
    )


def _sample_group(path: Path) -> str:
    name = path.stem.lower().replace("-", " ")
    if "crab" in name:
        return "crab"
    if "perdole" in name:
        return "an_perdole"
    if "carry" in name or "torgash" in name:
        return "carry"
    if "main" in name:
        return "main"
    # Autosaves are still useful measurements, but are deliberately neutral:
    # they cannot outweigh an explicitly requested family.
    return "unknown"


def _rasterize_source(ship: ShipModel) -> np.ndarray:
    """Rasterise structural boxes into a canonical, translation-free field."""
    structural = [block for block in ship.blocks if block.index not in FUNCTIONAL_BLOCKS]
    if not structural:
        structural = list(ship.blocks)
    # Functional blocks can protrude outside the armour silhouette (engines and
    # pads are common examples), so the geometry coordinate frame is measured
    # from the structural envelope only.
    (lx, ly, lz) = (
        min(block.lx for block in structural),
        min(block.ly for block in structural),
        min(block.lz for block in structural),
    )
    (ux, uy, uz) = (
        max(block.ux for block in structural),
        max(block.uy for block in structural),
        max(block.uz for block in structural),
    )
    size = (max(ux - lx, 0.001), max(uy - ly, 0.001), max(uz - lz, 0.001))
    field = np.zeros((CANONICAL_X, CANONICAL_Y, CANONICAL_Z), dtype=bool)

    def span(lo: float, hi: float, origin: float, extent: float, count: int):
        first = max(0, int(np.floor((lo - origin) / extent * count)))
        last = min(count - 1, int(np.ceil((hi - origin) / extent * count) - 1))
        return first, max(first, last)

    for block in structural:
        x0, x1 = span(block.lx, block.ux, lx, size[0], CANONICAL_X)
        y0, y1 = span(block.ly, block.uy, ly, size[1], CANONICAL_Y)
        z0, z1 = span(block.lz, block.uz, lz, size[2], CANONICAL_Z)
        field[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1] = True
    return field


def _profiles(field: np.ndarray):
    """Measure half extents, vertical centre, and fill density per z slice."""
    x_count, y_count, z_count = field.shape
    cx, cy = (x_count - 1) / 2.0, (y_count - 1) / 2.0
    half_width = np.zeros(z_count, dtype=float)
    half_height = np.zeros(z_count, dtype=float)
    y_offset = np.zeros(z_count, dtype=float)
    density = np.zeros(z_count, dtype=float)
    for z in range(z_count):
        xs, ys = np.where(field[:, :, z])
        if not len(xs):
            continue
        width = max(int(xs.max() - xs.min() + 1), 1)
        height = max(int(ys.max() - ys.min() + 1), 1)
        half_width[z] = width / (2.0 * x_count)
        half_height[z] = height / (2.0 * y_count)
        y_offset[z] = (float(ys.mean()) - cy) / y_count
        density[z] = len(xs) / (width * height)
    # A source may have a sparse bow slice. Interpolation makes that a measured
    # taper rather than an accidental zero in the blended profile.
    positions = np.arange(z_count)
    valid = half_width > 0
    if valid.any():
        first, last = positions[valid][0], positions[valid][-1]
        for values in (half_width, half_height, y_offset, density):
            values[:first] = values[first]
            values[last + 1:] = values[last]
            values[:] = np.interp(positions, positions[valid], values[valid])
    return half_width, half_height, y_offset, density


def _system_mix(ship: ShipModel) -> tuple[tuple[int, float], ...]:
    volumes = Counter()
    total = 0.0
    for block in ship.blocks:
        if block.index in SYSTEM_BLOCKS:
            volumes[block.index] += block.volume
            total += block.volume
    if not total:
        return ((3, 0.35), (14, 0.20), (52, 0.20), (51, 0.15), (5, 0.10))
    return tuple((index, volume / total) for index, volume in volumes.items())


@lru_cache(maxsize=1)
def _load_corpus() -> tuple[_ShapeDNA, ...]:
    """Load, deduplicate, and measure every useful plan in auto-ships."""
    root = _auto_ship_root()
    paths = sorted(root.glob("*.xml")) if root.exists() else []
    if not paths:
        paths = [_template_path(name) for name in TEMPLATE_FILES]

    seen: set[bytes] = set()
    result: list[_ShapeDNA] = []
    for path in paths:
        try:
            source = parse_xml(str(path))
        except (OSError, ValueError):
            continue
        if len(source.blocks) < 100:
            continue
        field = _rasterize_source(source)
        # Deduplicate by measured geometry, not by block count or system mix.
        # Main2 and autosaves often have different internals but are the same
        # hull, and must not get extra statistical weight for that reason.
        fingerprint = hashlib.sha1(field.tobytes()).digest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        half_width, half_height, y_offset, density = _profiles(field)
        total_volume = max(source.total_volume(), 0.001)
        functional_volume = sum(
            block.volume for block in source.blocks if block.index in FUNCTIONAL_BLOCKS
        )
        dimensions = source.dimensions()
        pads = sum(block.index in (20, 25) for block in source.blocks)
        result.append(_ShapeDNA(
            name=path.stem,
            group=_sample_group(path),
            occupancy=field,
            half_width=half_width,
            half_height=half_height,
            y_offset=y_offset,
            density=density,
            functional_ratio=float(functional_volume / total_volume),
            system_mix=_system_mix(source),
            turret_rate=float(pads / max(dimensions[2], 1.0)),
        ))
    if not result:
        raise FileNotFoundError(f"В corpus нет пригодных кораблей: {root}")
    return tuple(result)


@lru_cache(maxsize=8)
def _load_mask(path: str):
    if Image is None:
        return None
    source = Path(path)
    if not source.exists():
        return None
    with Image.open(source).convert("RGBA") as image:
        rgba = np.asarray(image)
    alpha = rgba[..., 3]
    if alpha.min() == 255:
        rgb = rgba[..., :3].astype(float)
        corners = np.array([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]])
        background = np.median(corners, axis=0)
        mask = np.linalg.norm(rgb - background, axis=2) > 24
    else:
        mask = alpha > 24
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _reference_for_spec(spec: AutoShipSpec) -> str | None:
    """Return an explicit reference or the built-in Falcon top silhouette."""
    if spec.reference_image:
        return spec.reference_image
    if spec.layout == "falcon" and FALCON_REFERENCE.exists():
        return str(FALCON_REFERENCE)
    return None


def _select_shapes(spec: AutoShipSpec, rng: random.Random):
    """Return the complete measured corpus with a layout preference.

    A layout is a bias, not a hard source filter.  Every unique geometry in the
    folder contributes to the blend; the requested family only receives extra
    weight.  This is important for Carry and An Perdole, whose XML exports often
    share exactly the same structural field and are already deduplicated.
    """
    corpus = list(_load_corpus())
    preferred = set(LAYOUT_GROUPS.get(spec.layout, ("main",)))
    selected = corpus
    weights = np.asarray([
        (2.5 if shape.group in preferred else 1.0) * (0.85 + rng.random() * 0.3)
        for shape in selected
    ], dtype=float)
    weights /= max(weights.sum(), 0.001)
    return selected, weights


def _blend_shapes(selected, weights):
    probability = np.zeros_like(selected[0].occupancy, dtype=float)
    half_width = np.zeros(CANONICAL_Z, dtype=float)
    half_height = np.zeros(CANONICAL_Z, dtype=float)
    y_offset = np.zeros(CANONICAL_Z, dtype=float)
    density = np.zeros(CANONICAL_Z, dtype=float)
    functional_ratio = 0.0
    turret_rate = 0.0
    system_mix = Counter()
    for shape, weight in zip(selected, weights):
        probability += shape.occupancy * weight
        half_width += shape.half_width * weight
        half_height += shape.half_height * weight
        y_offset += shape.y_offset * weight
        density += shape.density * weight
        functional_ratio += shape.functional_ratio * weight
        turret_rate += shape.turret_rate * weight
        for index, part in shape.system_mix:
            system_mix[index] += float(weight) * part
    total_mix = sum(system_mix.values()) or 1.0
    return (
        probability, half_width, half_height, y_offset, density,
        functional_ratio, tuple((i, v / total_mix) for i, v in system_mix.items()),
        turret_rate,
    )


def _deform_probability(probability: np.ndarray, spec: AutoShipSpec,
                        rng: random.Random) -> np.ndarray:
    """Apply a smooth new longitudinal deformation to the measured blend."""
    x_count, _y_count, z_count = probability.shape
    centre = (x_count - 1) / 2.0
    result = np.empty_like(probability)
    phase = rng.random() * np.pi * 2.0
    amplitude = 0.04 + rng.random() * 0.07
    for z in range(z_count):
        t = z / max(z_count - 1, 1)
        scale = 1.0 + amplitude * np.sin(np.pi * t + phase)
        # Layout changes are shape operations over the blend, not template
        # selection: the same measured corpus can produce a different hull.
        if spec.layout == "crab":
            scale *= 1.0 + 0.10 * (1.0 - t)
        elif spec.layout == "hammerhead":
            scale *= 1.0 + 0.12 * max(0.0, t - 0.72) / 0.28
        source_x = np.clip(np.rint((np.arange(x_count) - centre) / scale + centre),
                           0, x_count - 1).astype(int)
        result[:, :, z] = probability[source_x, :, z]
    return result


def _make_occupancy(spec: AutoShipSpec, profile, rng: random.Random):
    probability, half_width, half_height, y_offset, density, *_ = profile
    probability = _deform_probability(probability, spec, rng)
    width = max(1, int(round(float(spec.width) / GRID_PITCH)))
    height = max(1, int(round(float(spec.height) / GRID_PITCH)))
    length = max(1, int(round(float(spec.length) / GRID_PITCH)))
    xmap = np.minimum(CANONICAL_X - 1,
                      (np.arange(width) + 0.5) / width * CANONICAL_X).astype(int)
    ymap = np.minimum(CANONICAL_Y - 1,
                      (np.arange(height) + 0.5) / height * CANONICAL_Y).astype(int)
    zmap = np.minimum(CANONICAL_Z - 1,
                      (np.arange(length) + 0.5) / length * CANONICAL_Z).astype(int)
    sampled = probability[np.ix_(xmap, ymap, zmap)]

    # Enforce the blended measured envelope after resampling. This prevents a
    # consensus hull from growing fat merely because one source was oversized.
    zpos = np.linspace(0.0, 1.0, length)
    hw = np.interp(zpos, np.linspace(0.0, 1.0, CANONICAL_Z), half_width)
    hh = np.interp(zpos, np.linspace(0.0, 1.0, CANONICAL_Z), half_height)
    yo = np.interp(zpos, np.linspace(0.0, 1.0, CANONICAL_Z), y_offset)
    xnorm = np.abs((np.arange(width) - (width - 1) / 2.0) / width)[:, None, None]
    ynorm = np.abs((np.arange(height)[None, :, None] - (height - 1) / 2.0
                    - yo[None, None, :] * height) / height)
    envelope = (xnorm <= hw[None, None, :] * 1.04) & \
               (ynorm <= hh[None, None, :] * 1.04)
    measured_density = np.interp(
        zpos, np.linspace(0.0, 1.0, CANONICAL_Z), density,
    )
    # Dense source sections are allowed to survive a slightly higher consensus
    # threshold; sparse sections retain their measured ribs and cavities.
    threshold_wave = (0.46 + 0.055 * np.sin(np.pi * zpos + rng.random() * 6.28)
                      + (0.52 - measured_density) * 0.06)
    occupancy = (sampled >= threshold_wave[None, None, :]) & envelope

    # An uploaded silhouette is a top-view constraint over the newly generated
    # field. It never brings source blocks back into the result.
    reference_path = _reference_for_spec(spec)
    mask = _load_mask(reference_path) if reference_path else None
    if mask is not None:
        mask_x = np.minimum(mask.shape[1] - 1,
                            ((np.arange(width) + 0.5) / width * mask.shape[1]).astype(int))
        mask_z = np.minimum(mask.shape[0] - 1,
                            ((1.0 - (np.arange(length) + 0.5) / length)
                             * mask.shape[0]).astype(int))
        top_view = mask[np.ix_(mask_z, mask_x)]
        occupancy &= top_view.T[:, None, :]

    grid = VoxelGrid(width, height, length)
    grid.occ[:] = occupancy
    return grid


def _keep_main_component(grid: VoxelGrid) -> None:
    """Drop detached islands created by the probabilistic shape blend."""
    if not grid.occ.any():
        return
    points = np.argwhere(grid.occ)
    centre = np.asarray([grid.X / 2.0, grid.Y / 2.0, grid.Z / 2.0])
    seed = points[int(np.argmin(np.abs(points - centre).sum(axis=1)))]
    reached = np.zeros_like(grid.occ)
    frontier = np.zeros_like(grid.occ)
    frontier[tuple(seed)] = True
    while frontier.any():
        reached |= frontier
        grown = frontier.copy()
        for axis in (0, 1, 2):
            for shift in (-1, 1):
                shifted = np.roll(frontier, shift, axis=axis)
                edge = [slice(None)] * 3
                edge[axis] = 0 if shift == 1 else -1
                shifted[tuple(edge)] = False
                grown |= shifted
        frontier = grown & grid.occ & ~reached
    grid.occ[:] = reached


def _fit_to_requested_frame(grid: VoxelGrid) -> None:
    """Resample the measured proportions into the requested bounding frame."""
    points = np.argwhere(grid.occ)
    if not len(points):
        return
    low = points.min(axis=0)
    high = points.max(axis=0)
    xmap = np.rint(np.linspace(low[0], high[0], grid.X)).astype(int)
    ymap = np.rint(np.linspace(low[1], high[1], grid.Y)).astype(int)
    zmap = np.rint(np.linspace(low[2], high[2], grid.Z)).astype(int)
    grid.occ[:] = grid.occ[np.ix_(xmap, ymap, zmap)]


def _paint_systems(grid: VoxelGrid, spec: AutoShipSpec, profile) -> None:
    """Place new functional blocks using measured system-density statistics."""
    _probability, _hw, _hh, _yo, _density, functional_ratio, system_mix, _rate = profile
    shell = grid.shell_mask()
    attr = np.full(grid.occ.shape, EMPTY, dtype=np.int32)
    attr[grid.occ & ~shell] = 1
    attr[grid.occ & shell] = 8
    available = grid.occ & ~shell
    slots = min(15, max(0, int(spec.subsystem_cells)))
    target = int(round(grid.occ.sum() * functional_ratio * slots / 15.0))
    flight_minimum = max(4, int(round(grid.occ.sum() * 0.002)))
    target = max(target, flight_minimum)

    def take(mask, amount, index):
        nonlocal available
        coords = np.argwhere(mask & available)
        if not len(coords) or amount <= 0:
            return 0
        order = np.argsort(coords[:, 2])
        chosen = coords[order[:amount]]
        attr[tuple(chosen.T)] = index
        available[tuple(chosen.T)] = False
        return len(chosen)

    # Engines and directional thrusters are kept at the stern even at zero
    # subsystem slots; otherwise the generated ship is only a static shell.
    z_indices = np.indices(grid.occ.shape)[2]
    stern = z_indices < max(2, int(grid.Z * 0.16))
    engine_count = max(1, int(round(target * 0.10)))
    used = take(stern, engine_count, 3)
    used += take(stern, max(1, engine_count // 3), 7)
    remaining = max(0, target - used)

    # The measured functional mix controls *which* systems occupy the new
    # interior, while their coordinates are generated from the new occupancy.
    for index, ratio in sorted(system_mix, key=lambda item: item[1], reverse=True):
        if index not in SYSTEM_BLOCKS or remaining <= 0:
            continue
        amount = max(1, int(round(remaining * ratio)))
        central_mask = ((z_indices > grid.Z * 0.12) &
                        (z_indices < grid.Z * 0.90))
        used_now = take(central_mask, min(amount, remaining), index)
        remaining -= used_now

    if remaining:
        take(available, remaining, 14)
    grid.role[:] = attr


def _add_turret_pads(grid: VoxelGrid, profile, rng: random.Random) -> None:
    """Put fresh turret-base cells on measured-style dorsal stations."""
    turret_rate = profile[-1]
    count = int(np.clip(round(grid.Z * turret_rate), 4, 12))
    attr = grid.role
    cx = (grid.X - 1) / 2.0
    stations = np.linspace(grid.Z * 0.18, grid.Z * 0.82, count)
    placed = 0
    for station in stations:
        z = int(round(station))
        side = -1 if rng.random() < 0.5 else 1
        offsets = (0.0, side * grid.X * 0.18, -side * grid.X * 0.18)
        for offset in offsets:
            x = int(round(cx + offset))
            if not (0 <= x < grid.X):
                continue
            ys = np.where(grid.occ[x, :, z])[0]
            if not len(ys):
                continue
            y = int(ys[-1])
            if attr[x, y, z] in (1, 8):
                attr[x, y, z] = 20
                placed += 1
                break
    if placed == 0:
        return


def _to_ship(grid: VoxelGrid, spec: AutoShipSpec, rng: random.Random) -> ShipModel:
    materials = tuple(sorted(set(spec.materials))) or (1,)
    boxes = greedy_merge(grid.role)
    width = grid.X * GRID_PITCH
    height = grid.Y * GRID_PITCH
    length = grid.Z * GRID_PITCH
    result = ShipModel(
        name=f"AutoShip_{spec.layout}_{spec.seed}_geometry_dna",
        layout=spec.layout,
        pitch=GRID_PITCH,
    )
    for x0, x1, y0, y1, z0, z1, index in boxes:
        if index == EMPTY:
            continue
        material = rng.choice(materials)
        result.add(Block(
            lx=-width / 2 + x0 * GRID_PITCH,
            ly=-height / 2 + y0 * GRID_PITCH,
            lz=-length / 2 + z0 * GRID_PITCH,
            ux=-width / 2 + (x1 + 1) * GRID_PITCH,
            uy=-height / 2 + (y1 + 1) * GRID_PITCH,
            uz=-length / 2 + (z1 + 1) * GRID_PITCH,
            index=int(index),
            material=material,
            look=1,
            up=3,
        ))
    result.drop_degenerate()
    return result


def generate_auto_ship(spec: AutoShipSpec) -> ShipModel:
    """Build a new ship from the measured geometry distribution."""
    if spec.layout not in LAYOUT_GROUPS:
        raise ValueError(f"Неизвестная компоновка auto-ship: {spec.layout}")
    rng = random.Random(f"{spec.seed}:geometry-dna")
    selected, weights = _select_shapes(spec, rng)
    profile = _blend_shapes(selected, weights)
    grid = _make_occupancy(spec, profile, rng)
    _keep_main_component(grid)
    if not _reference_for_spec(spec):
        _fit_to_requested_frame(grid)
    _paint_systems(grid, spec, profile)
    _add_turret_pads(grid, profile, rng)
    ship = _to_ship(grid, spec, rng)
    # Keep the source selection visible for diagnostics without serialising it.
    source_names = ",".join(shape.name for shape in selected)
    digest = hashlib.sha1(source_names.encode("utf-8")).hexdigest()[:6]
    ship.name = f"AutoShip_{spec.layout}_{spec.seed}_dna_{digest}"
    return ship
