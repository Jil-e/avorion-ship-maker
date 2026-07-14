"""Colour helpers. Avorion stores colours as ``AARRGGBB`` hex strings."""
from __future__ import annotations


def _clean(rgb: str) -> str:
    rgb = rgb.strip().lstrip("#")
    if len(rgb) == 8:      # already AARRGGBB -> drop alpha
        rgb = rgb[2:]
    if len(rgb) == 3:      # shorthand rgb
        rgb = "".join(c * 2 for c in rgb)
    if len(rgb) != 6:
        rgb = "1e1e1e"
    return rgb.lower()


def to_argb(rgb: str, alpha: str = "ff") -> str:
    """``'c0392b'`` -> ``'ffc0392b'`` (opaque). Accepts ``#rgb``/``rgb``/``AARRGGBB``."""
    return f"{alpha}{_clean(rgb)}"


def shade(rgb: str, factor: float) -> str:
    """Lighten (factor>1) or darken (factor<1) an RGB hex string."""
    rgb = _clean(rgb)
    ch = [int(rgb[i:i + 2], 16) for i in (0, 2, 4)]
    ch = [max(0, min(255, int(round(c * factor)))) for c in ch]
    return "".join(f"{c:02x}" for c in ch)


def _dist(a: str, b: str) -> int:
    a, b = _clean(a), _clean(b)
    return sum(abs(int(a[i:i + 2], 16) - int(b[i:i + 2], 16)) for i in (0, 2, 4))


def contrast(base: str, accent: str, threshold: int = 70) -> str:
    """Return ``accent`` unless it is too close to ``base``; then a legible,
    lighter/darker derivative of ``base`` so accents never vanish."""
    if _dist(base, accent) >= threshold:
        return _clean(accent)
    # base is dark -> lighten; base is light -> darken
    b = _clean(base)
    lum = sum(int(b[i:i + 2], 16) for i in (0, 2, 4)) / 3.0
    return shade(base, 2.1) if lum < 128 else shade(base, 0.45)


def mix(a: str, b: str, t: float) -> str:
    """Linear blend of two RGB hex strings (t=0 -> a, t=1 -> b)."""
    a, b = _clean(a), _clean(b)
    out = []
    for i in (0, 2, 4):
        ca, cb = int(a[i:i + 2], 16), int(b[i:i + 2], 16)
        out.append(f"{int(round(ca + (cb - ca) * t)):02x}")
    return "".join(out)


# Named colours understood by the text parser (English + Russian).
NAMED_COLORS: dict[str, str] = {
    "red": "c0392b", "красный": "c0392b",
    "blue": "2e86c1", "синий": "2e86c1", "голубой": "5dade2",
    "green": "27ae60", "зелёный": "27ae60", "зеленый": "27ae60",
    "yellow": "f1c40f", "жёлтый": "f1c40f", "желтый": "f1c40f",
    "orange": "e67e22", "оранжевый": "e67e22",
    "purple": "8e44ad", "фиолетовый": "8e44ad",
    "cyan": "1abc9c", "бирюзовый": "1abc9c",
    "white": "ecf0f1", "белый": "ecf0f1",
    "black": "17202a", "чёрный": "17202a", "черный": "17202a",
    "grey": "7f8c8d", "gray": "7f8c8d", "серый": "7f8c8d",
    "gold": "d4ac0d", "золотой": "d4ac0d",
    "silver": "bdc3c7", "серебристый": "bdc3c7",
    "pink": "e91e63", "розовый": "e91e63",
    "teal": "16a085", "navy": "1b2631", "тёмно-синий": "1b2631",
    "crimson": "922b21", "малиновый": "922b21",
}
