"""Procedural ship-generation subpackage."""
from .spec import ShipSpec, HULL_CLASSES, STYLES
from .builder import build_ship
from .text_parser import parse_description

__all__ = ["ShipSpec", "HULL_CLASSES", "STYLES", "build_ship", "parse_description"]
