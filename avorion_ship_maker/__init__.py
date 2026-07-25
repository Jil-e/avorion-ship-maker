"""Avorion Ship Maker — corpus-driven Avorion ship blueprints."""
from dataclasses import replace

from .model import Block, ShipModel
from .blocks import Role, ROLE_INDEX, MATERIALS, block_registry
from .xml_io import to_xml, save_xml, parse_xml
from .generator.auto_ship_generator import AutoShipSpec, generate_auto_ship

__all__ = [
    "Block", "ShipModel", "Role", "ROLE_INDEX", "MATERIALS", "block_registry",
    "to_xml", "save_xml", "parse_xml", "AutoShipSpec", "generate_auto_ship",
    "generate",
]


def generate(description: str = "", spec: "AutoShipSpec | None" = None,
             **overrides) -> ShipModel:
    """High-level one-shot entry point for the corpus-driven generator."""
    if spec is None:
        text = (description or "").lower()
        layout = "crab" if "crab" in text else \
            "hammerhead" if "hammer" in text else \
            "carry" if "carry" in text or "груз" in text else \
            "an_perdole" if "perdole" in text else \
            "main" if "main" in text else "falcon"
        spec = AutoShipSpec(layout=layout)
    values = {
        key: value for key, value in overrides.items()
        if value is not None and key in spec.__dataclass_fields__
    }
    if values:
        spec = replace(spec, **values)
    return generate_auto_ship(spec)
