"""Serialize a :class:`ShipModel` to Avorion ship-plan XML, and parse it back.

The output byte format matches ships exported by the game itself:
UTF-8, CRLF line endings, tab indentation, the exact ``<block>`` attribute
order, and ``<item parent=.. index=..>`` wrappers carrying the connectivity
tree. Turrets are omitted (a bare hull loads fine in the shipyard).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

from .connectivity import build_parents
from .model import Block, ShipModel, fmt_num

_HEADER = '<?xml version="1.0" encoding="utf-8"?>'
_NL = "\r\n"


def _block_tag(b: Block) -> str:
    return (
        f'<block lx="{fmt_num(b.lx)}" ly="{fmt_num(b.ly)}" lz="{fmt_num(b.lz)}"'
        f' ux="{fmt_num(b.ux)}" uy="{fmt_num(b.uy)}" uz="{fmt_num(b.uz)}"'
        f' index="{b.index}" material="{b.material}" look="{b.look}" up="{b.up}"'
        f' color="{b.color}" secondaryColor="{b.secondary}"/>'
    )


def to_xml(ship: ShipModel) -> str:
    """Return the full ship-plan XML document as a string (CRLF line endings)."""
    parents = build_parents(ship.blocks)
    lines = [_HEADER, "<ship_design>", '\t<plan accumulateHealth="true" convex="false">']
    for i, b in enumerate(ship.blocks):
        lines.append(f'\t\t<item parent="{parents[i]}" index="{i}">')
        lines.append(f"\t\t\t{_block_tag(b)}")
        lines.append("\t\t</item>")
    lines.append("\t</plan>")
    lines.append("</ship_design>")
    return _NL.join(lines) + _NL


def save_xml(ship: ShipModel, path: str) -> str:
    """Write the ship to ``path`` with exact Avorion byte formatting. Returns path."""
    data = to_xml(ship).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _plan_lines(blocks, indent: str) -> list[str]:
    parents = build_parents(blocks)
    lines = [f'{indent}<plan accumulateHealth="true" convex="false">']
    for i, b in enumerate(blocks):
        lines.append(f'{indent}\t<item parent="{parents[i]}" index="{i}">')
        lines.append(f"{indent}\t\t{_block_tag(b)}")
        lines.append(f"{indent}\t</item>")
    lines.append(f"{indent}</plan>")
    return lines


def turret_to_xml(t) -> str:
    """Serialize a :class:`TurretModel` to the game's <turret_design> format.

    Matches the byte format of designs the game itself saves under
    ``ships/auto-turrets`` (CRLF, tabs, section pivots, muzzle, version).
    """
    lines = [_HEADER,
             f'<turret_design size="{fmt_num(t.size)}"'
             f' coaxial="{"true" if t.coaxial else "false"}"'
             f' shot_color="{t.shot_color}">']
    sections = (("base", t.base, (0.0, 0.0, 0.0)),
                ("body", t.body, t.body_pivot),
                ("barrel", t.barrel, t.barrel_pivot))
    for tag, blocks, (px, py, pz) in sections:
        lines.append(f'\t<{tag} px="{fmt_num(px)}" py="{fmt_num(py)}" pz="{fmt_num(pz)}">')
        lines.extend(_plan_lines(blocks, "\t\t"))
        lines.append(f"\t</{tag}>")
    mx, my, mz = t.muzzle
    lines.append(f'\t<muzzlePosition x="{fmt_num(mx)}" y="{fmt_num(my)}" z="{fmt_num(mz)}"/>')
    lines.append('\t<version major="2" minor="0" patch="0"/>')
    lines.append("</turret_design>")
    return _NL.join(lines) + _NL


def save_turret_xml(t, path: str) -> str:
    """Write a turret design to ``path`` (exact Avorion byte format)."""
    with open(path, "wb") as fh:
        fh.write(turret_to_xml(t).encode("utf-8"))
    return path


def parse_xml(path: str) -> ShipModel:
    """Parse an Avorion ship file into a :class:`ShipModel` (main hull plan only).

    Works on both files this tool writes and real game exports (whose turret
    designs are nested under ``<turretDesign>`` and are skipped — only the
    top-level ``<plan>`` child of ``<ship_design>`` is read).
    """
    tree = ET.parse(path)
    root = tree.getroot()
    # The main hull is the <plan> that is a direct child of <ship_design>.
    plan = None
    for child in root:
        if child.tag == "plan":
            plan = child
            break
    if plan is None:  # fall back to the first plan anywhere
        plan = root.find(".//plan")

    ship = ShipModel(name="Imported")
    if plan is None:
        return ship
    for item in plan.findall("item"):
        el = item.find("block")
        if el is None:
            continue
        a = el.attrib
        ship.add(Block(
            lx=float(a["lx"]), ly=float(a["ly"]), lz=float(a["lz"]),
            ux=float(a["ux"]), uy=float(a["uy"]), uz=float(a["uz"]),
            index=int(a.get("index", 1)), material=int(a.get("material", 1)),
            look=int(a.get("look", 1)), up=int(a.get("up", 3)),
            color=a.get("color", "ff1e1e1e"), secondary=a.get("secondaryColor", "00000000"),
        ))
    return ship
