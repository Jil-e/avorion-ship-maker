"""Avorion Ship Maker — procedural generator of Avorion ship blueprints (.xml).

Public API:
    from avorion_ship_maker import ShipModel, Block, generate, save_xml
"""
from .model import Block, ShipModel
from .blocks import Role, ROLE_INDEX, MATERIALS, block_registry
from .xml_io import to_xml, save_xml, parse_xml
from .generator.spec import ShipSpec
from .generator.builder import build_ship
from .generator.text_parser import parse_description

__all__ = [
    "Block", "ShipModel", "Role", "ROLE_INDEX", "MATERIALS", "block_registry",
    "to_xml", "save_xml", "parse_xml", "ShipSpec", "build_ship",
    "parse_description", "generate",
]


def generate(description: str = "", spec: "ShipSpec | None" = None, **overrides) -> ShipModel:
    """High-level one-shot: text description (+optional overrides) -> ShipModel.

    If ``spec`` is given it is used directly; otherwise the description is parsed
    into a :class:`ShipSpec` and any keyword ``overrides`` are applied on top.
    """
    if spec is None:
        spec = parse_description(description)
        for key, value in overrides.items():
            if value is not None and hasattr(spec, key):
                setattr(spec, key, value)
    return build_ship(spec)
