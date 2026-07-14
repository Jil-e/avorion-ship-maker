"""Heuristic natural-language parser: a free-form description -> :class:`ShipSpec`.

Bilingual (English + Russian). It recognises a hull class, a visual style,
colours, size/shape modifiers, part toggles, material and explicit dimensions.
Anything not mentioned stays ``None`` and inherits from the class/style presets.
"""
from __future__ import annotations

import re

from ..blocks import material_id
from .palette import NAMED_COLORS
from .spec import HULL_CLASSES, ShipSpec

CLASS_KEYWORDS: dict[str, list[str]] = {
    "fighter":    ["fighter", "interceptor", "scout", "истребитель", "перехватчик", "разведчик"],
    "corvette":   ["corvette", "gunship", "корвет"],
    "frigate":    ["frigate", "destroyer", "фрегат", "эсминец"],
    "cruiser":    ["cruiser", "крейсер"],
    "battleship": ["battleship", "dreadnought", "capital", "battlecruiser", "линкор", "дредноут", "линейный"],
    "freighter":  ["freighter", "hauler", "cargo", "transport", "trader", "грузовик", "грузовоз", "транспорт", "торговец"],
    "miner":      ["miner", "mining", "шахтёр", "шахтер", "добыт", "майнер"],
    "carrier":    ["carrier", "авианосец", "носитель"],
    "station":    ["station", "outpost", "starbase", "станция", "база"],
}

LAYOUT_KEYWORDS: dict[str, list[str]] = {
    "twin": ["катамаран", "двухкорпус", "catamaran", "twin hull", "twin-hull", "двойной корпус"],
    "pods": ["гондол", "nacelle", "пилон"],
    "keel": ["надстройк", "гребен", "гребн", "superstructure"],
    "hammer": ["молот", "hammerhead", "широкий нос", "т-обра"],
    "fork": ["вилк", "раздвоен", "клешн", "forked", "prong"],
    "mono": ["монокорпус", "monohull", "один корпус"],
}

WING_KIND_KEYWORDS: dict[str, list[str]] = {
    "forward": ["обратная стреловидн", "обратной стреловидн", "forward-swept", "forward swept"],
    "delta": ["дельта", "delta"],
    "gull": ["чайк", "gull"],
    "xfoil": ["x-wing", "xfoil", "икс-крыл", "x-обра"],
    "tipfin": ["винглет", "winglet"],
    "tippod": ["гондолы на крыл", "на концах крыл"],
}

STYLE_KEYWORDS: dict[str, list[str]] = {
    "military":   ["military", "warship", "combat", "battle", "военн", "боев", "воен"],
    "civilian":   ["civilian", "passenger", "yacht", "гражданск", "пассажир", "яхта"],
    "stealth":    ["stealth", "stealthy", "cloak", "стелс", "скрытн", "невидим"],
    "industrial": ["industrial", "utility", "worker", "промышленн", "рабоч", "утилитарн"],
    "sleek":      ["sleek", "elegant", "luxury", "обтекаемый", "элегантн", "гладк", "люкс"],
    "hazard":     ["hazard", "construction", "rescue", "аварийн", "спасательн"],
}

# stems (substring match) so any word ending is caught: "больш" -> большой/ое/ая/ие/им...
_SMALL = ["small", "tiny", "little", "compact", "маленьк", "небольш", "компактн", "малют", "крошечн"]
_BIG = ["big", "large", "heavy", "больш", "крупн", "тяжёл", "тяжел"]
_HUGE = ["huge", "massive", "giant", "enormous", "colossal", "огромн", "гигантск", "массивн", "колоссальн"]

_LONG = ["long", "elongated", "needle", "длинн", "вытянут", "удлин", "игл", "иглоподобн"]
_WIDE = ["wide", "broad", "широк", "широч", "пузат", "толст"]
_NARROW = ["narrow", "slim", "узк", "тонк", "стройн", "худ"]
_FLAT = ["flat", "плоск", "приплюснут", "низк", "лепёшк"]
_TALL = ["tall", "высок", "башнеподобн"]
_BOXY = ["boxy", "blocky", "rectangular", "brick", "угловат", "блочн", "прямоугольн", "кирпич", "коробч", "гранён"]
_SMOOTH = ["smooth", "rounded", "curved", "streamlined", "обтекаем", "гладк", "округл", "плавн", "сглаж", "каплевидн"]

_WORD_NUM = {"single": 1, "one": 1, "twin": 2, "dual": 2, "double": 2, "two": 2,
             "triple": 3, "three": 3, "quad": 4, "four": 4, "six": 6,
             "один": 1, "одним": 1, "два": 2, "двумя": 2, "три": 3, "тремя": 3,
             "четыре": 4, "четырьмя": 4, "шесть": 6}


def _has(text: str, words) -> bool:
    return any(w in text for w in words)


def _word(text: str, word: str) -> bool:
    """Whole-word-ish match that also fires on Cyrillic stems."""
    return re.search(r"(?<![0-9a-zа-яё])" + re.escape(word), text) is not None


def parse_description(text: str) -> ShipSpec:
    t = (text or "").lower().replace("ё", "е")
    spec = ShipSpec()
    if text and text.strip():
        first = text.strip().splitlines()[0][:48].strip()
        spec.name = first or "Generated Ship"

    # hull class
    for cls, keys in CLASS_KEYWORDS.items():
        if _has(t, [k.replace("ё", "е") for k in keys]):
            spec.hull_class = cls
            break
    # style
    for st, keys in STYLE_KEYWORDS.items():
        if _has(t, [k.replace("ё", "е") for k in keys]):
            spec.style = st
            break
    # hull layout
    for lay, keys in LAYOUT_KEYWORDS.items():
        if _has(t, [k.replace("ё", "е") for k in keys]):
            spec.layout = lay
            break
    # wing archetype (implies wings on)
    for wk, keys in WING_KIND_KEYWORDS.items():
        if _has(t, [k.replace("ё", "е") for k in keys]):
            spec.wing_kind = wk
            spec.wings = True
            break

    # colours (first -> primary, second -> accent)
    found: list[str] = []
    for word, hexv in NAMED_COLORS.items():
        w = word.replace("ё", "е")
        if _word(t, w) and hexv not in found:
            found.append(hexv)
    if found:
        spec.primary = found[0]
        if len(found) > 1:
            spec.accent = found[1]

    # material (English stem or Russian stem -> canonical English name)
    material_stems = {
        "iron": "iron", "titan": "titanium", "naonit": "naonite", "trini": "trinium",
        "xanion": "xanion", "ogonit": "ogonite", "avorion": "avorion",
        "желез": "iron", "титан": "titanium", "наонит": "naonite", "трини": "trinium",
        "ксанион": "xanion", "огонит": "ogonite", "аворион": "avorion",
    }
    for stem, canonical in material_stems.items():
        if stem in t:
            spec.material = material_id(canonical)
            break

    # start from class preset dimensions, then apply modifiers
    base = HULL_CLASSES.get(spec.hull_class, HULL_CLASSES["frigate"])
    L, W, H = float(base["length"]), float(base["width"]), float(base["height"])

    mult = 1.0
    if _has(t, _SMALL):
        mult *= 0.6
    if _has(t, _BIG):
        mult *= 1.5
    if _has(t, _HUGE):
        mult *= 2.2
    L *= mult; W *= mult; H *= mult

    if _has(t, _LONG):
        L *= 1.3; W *= 0.85
    if _has(t, _WIDE):
        W *= 1.4
    if _has(t, _NARROW):
        W *= 0.65
    if _has(t, _FLAT):
        H *= 0.6
    if _has(t, _TALL):
        H *= 1.5

    if _has(t, _BOXY):
        spec.boxiness = 0.95
    elif _has(t, _SMOOTH):
        spec.boxiness = 0.2

    # explicit dimensions, e.g. "length 40", "40 long", "width 12"
    for key, setter in (("length", "L"), ("width", "W"), ("height", "H"),
                        ("длин", "L"), ("ширин", "W"), ("высот", "H")):
        m = re.search(key + r"\D{0,6}(\d{1,3})", t)
        if m:
            val = float(m.group(1))
            if setter == "L":
                L = val
            elif setter == "W":
                W = val
            else:
                H = val

    spec.length = int(max(4, round(L)))
    spec.width = int(max(3, round(W)))
    spec.height = int(max(3, round(H)))

    # part toggles
    if _has(t, ["no wings", "wingless", "без крыл", "бескрыл"]):
        spec.wings = False
    elif _has(t, ["wing", "крыл"]):
        spec.wings = True
    if _has(t, ["no fins", "без килей", "без плавник"]):
        spec.fins = False
    elif _has(t, ["fin", "плавник", "киль", "стабилизатор"]):
        spec.fins = True
    if _has(t, ["no bridge", "без мостик", "без кабин"]):
        spec.bridge = False
    elif _has(t, ["bridge", "cockpit", "мостик", "кабин", "кокпит", "рубк"]):
        spec.bridge = True

    # engine count
    em = re.search(r"(\d+)\s*(?:engine|двигател|мотор)", t)
    if em:
        spec.engines = int(em.group(1))
    else:
        for word, num in _WORD_NUM.items():
            if re.search(re.escape(word) + r"\s+(?:engine|двигател|мотор)", t):
                spec.engines = num
                break

    return spec
