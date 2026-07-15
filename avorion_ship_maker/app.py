"""Gradio UI for the Avorion Ship Maker.

Workflow (preview-first): describe the ship, press Enter / 🎲 to get a strip of
variants, click one to open it in the big WebGL view, fine-tune with the
controls on the right, download the Avorion-ready ``.xml``. Run with::

    python -m avorion_ship_maker.app
"""
from __future__ import annotations

import os
import random
import re
import tempfile

import gradio as gr

from . import webgl
from .blocks import BLOCK_NAMES, MATERIALS
from .generator.builder import build_ship
from .generator.spec import (HULL_CLASSES, LAYOUT_NAMES, LAYOUTS, STYLES,
                             WING_KINDS, ShipSpec)
from .generator.text_parser import parse_description
from .generator.turret_builder import TURRET_KINDS, TurretSpec, build_turret
from .preview import save_preview
from .xml_io import save_turret_xml

AUTO = "auto"
N_VARIANTS = 6
_EXPORT_DIR = os.path.join(tempfile.gettempdir(), "avorion_ship_maker")
os.makedirs(_EXPORT_DIR, exist_ok=True)

CLASS_NAMES = {"fighter": "истребитель", "corvette": "корвет", "frigate": "фрегат",
               "cruiser": "крейсер", "battleship": "линкор", "freighter": "грузовоз",
               "miner": "шахтёр", "carrier": "авианосец", "station": "станция"}
STYLE_NAMES = {"military": "военный", "civilian": "гражданский", "stealth": "стелс",
               "industrial": "промышленный", "sleek": "обтекаемый", "hazard": "аварийный"}
MATERIAL_NAMES = {0: "железо", 1: "титан", 2: "наонит", 3: "триний",
                  4: "ксанион", 5: "огонит", 6: "аворион"}
WING_KIND_NAMES = {"swept": "стреловидные", "forward": "обратная стреловидность",
                   "delta": "дельта", "gull": "«чайка»", "xfoil": "X-плоскости",
                   "tipfin": "с винглетами", "tippod": "гондолы на концах"}


def _safe_name(name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", (name or "ship").strip(), flags=re.U).strip("_")
    return (slug or "ship")[:40]


def build_spec(description, hull_class, style, layout, length, width, height,
               engines, wings, wing_kind, fins, bridge, boxiness, armor, detail,
               bevel, functional, material, block_size, scale, symmetry, seed,
               use_colors, primary, secondary, accent, glow, turrets=-1) -> ShipSpec:
    """Compose a spec: description supplies defaults, UI controls override them.

    Anything left on "авто" is rolled per-seed inside the class rules by
    ``ShipSpec.resolved()`` — that's what makes variants of one class differ.
    """
    spec = parse_description(description or "")
    spec.seed = int(seed)   # before height: the relative delta needs the roll

    if hull_class != AUTO:
        spec.hull_class = hull_class
    if style != AUTO:
        spec.style = style
    if layout != AUTO:
        spec.layout = layout
    if length and length > 0:
        spec.length = int(length)
    if width and width > 0:
        spec.width = int(width)
    if height:  # relative: 0 = auto, +N taller, -N flatter (on top of the roll)
        base_h = spec.height if spec.height is not None else spec.resolved().height
        spec.height = max(3, int(base_h) + int(height))
    if engines is not None and engines >= 0:
        spec.engines = int(engines)
    if turrets is not None and turrets >= 0:
        spec.turrets = int(turrets)
    for val, attr in ((wings, "wings"), (fins, "fins"), (bridge, "bridge")):
        if val != AUTO:
            setattr(spec, attr, val == "yes")
    if wing_kind != AUTO:
        spec.wing_kind = wing_kind
        if wings == AUTO:          # picking a wing style implies wings on
            spec.wings = True
    if boxiness is not None and boxiness >= 0:
        spec.boxiness = float(boxiness)
    if armor is not None and armor >= 0:
        spec.armor = float(armor)
    if detail is not None and detail >= 0:
        spec.detail = float(detail)
    if bevel is not None and bevel >= 0:
        spec.bevel = float(bevel)
    if functional is not None and functional >= 0:
        spec.functional = float(functional)
    if material != AUTO:
        spec.material = int(material)

    spec.block_size = float(block_size)
    spec.scale = float(scale)
    spec.symmetry = bool(symmetry)

    if use_colors:
        spec.primary, spec.secondary = primary, secondary
        spec.accent, spec.glow = accent, glow
    return spec


def _understood_md(spec: ShipSpec) -> str:
    """A short line showing what the description/controls resolved to."""
    r = spec.resolved()
    bits = [f"класс **{CLASS_NAMES.get(r.hull_class, r.hull_class)}**",
            f"стиль **{STYLE_NAMES.get(r.style, r.style)}**",
            f"размер **{r.length}×{r.width}×{r.height}**",
            f"материал **{MATERIAL_NAMES.get(r.material, r.material)}**",
            f"двигателей **{r.engines}**",
            f"турелей **{r.turrets}**"]
    if r.layout:
        bits.insert(2, f"компоновка **{LAYOUT_NAMES.get(r.layout, r.layout)}**")
    extras = [n for n, on in (("крылья", r.wings), ("кили", r.fins), ("мостик", r.bridge)) if on]
    if extras:
        bits.append(" + ".join(extras))
    return "🔎 Понято: " + ", ".join(bits)


def _stats_md(ship, spec) -> str:
    (lx, ly, lz), (hx, hy, hz) = ship.bounds()
    dx, dy, dz = round(hx - lx, 1), round(hy - ly, 1), round(hz - lz, 1)
    counts = ship.type_counts()
    rows = "\n".join(
        f"| {BLOCK_NAMES.get(idx, f'type {idx}')} | `{idx}` | {n} |"
        for idx, n in sorted(counts.items(), key=lambda kv: -kv[1])
    )
    lay = LAYOUT_NAMES.get(ship.layout, ship.layout or "—")
    return (f"**Блоков:** {len(ship)} · **Компоновка:** {lay} · "
            f"**Габариты (Д×Ш×В):** {dz} × {dx} × {dy}\n\n"
            f"| Блок | index | шт. |\n|---|---|---|\n{rows}")


_PLACEHOLDER = (
    '<div style="height:600px;display:flex;flex-direction:column;gap:10px;align-items:center;'
    'justify-content:center;color:#93a0b4;background:#0d1117;border-radius:12px;'
    'font-family:system-ui,sans-serif;text-align:center;padding:0 40px">'
    '<div style="font-size:40px">🛠️</div>'
    '<div style="font-size:15px;max-width:48ch">Выберите справа <b style="color:#ff8a5c">класс'
    '</b> (и, если хочется, стиль/компоновку) — варианты построятся сами.<br>'
    'Клик по варианту открывает его здесь, ползунки докручивают вживую. '
    'Описание текстом — опция, не обязанность.</div></div>')


def generate(description, hull_class, style, layout, length, width, height,
             engines, wings, wing_kind, fins, bridge, boxiness, armor, detail,
             bevel, functional, material, block_size, scale, symmetry, seed,
             use_colors, primary, secondary, accent, glow, turrets=-1):
    """Main handler: build the ship, render WebGL preview, write the .xml."""
    try:
        spec = build_spec(description, hull_class, style, layout, length, width, height,
                          engines, wings, wing_kind, fins, bridge, boxiness, armor, detail,
                          bevel, functional, material, block_size, scale, symmetry, seed,
                          use_colors, primary, secondary, accent, glow, turrets)
        ship = build_ship(spec)
        if not ship.blocks:
            raise ValueError("Пустой корпус — увеличьте размеры.")
        html = webgl.ship_to_iframe(ship, height=600)
        base = os.path.join(_EXPORT_DIR, _safe_name(spec.name))
        from .xml_io import save_xml
        xml_path = save_xml(ship, base + ".xml")
        html_path = webgl.save_html(ship, base + "_preview.html")
        return html, _stats_md(ship, spec), xml_path, _understood_md(spec), "", html_path
    except Exception as exc:
        return _PLACEHOLDER, "", None, "", f"⚠️ Ошибка: {exc}", None


_SEED_ARG = 21   # position of `seed` in the handler argument list


def make_variants(*vals, progress=gr.Progress()):
    """🎲 / Enter: build N variants with fresh seeds, open the first one."""
    vals = list(vals)
    thumbs, seeds = [], []
    for i in range(N_VARIANTS):
        progress(i / N_VARIANTS, desc=f"Строю вариант {i + 1}/{N_VARIANTS}…")
        vseed = random.randint(1, 999_999)
        vals[_SEED_ARG] = vseed
        try:
            spec = build_spec(*vals)
            ship = build_ship(spec)
            if not ship.blocks:
                continue
            png = os.path.join(_EXPORT_DIR, f"variant_{vseed}.png")
            save_preview(ship, png, figsize=3.2)
            rs = spec.resolved()
            cap = f"{rs.length}×{rs.width}×{rs.height} · дв. {rs.engines}"
            if ship.layout:
                cap += f" · {LAYOUT_NAMES.get(ship.layout, ship.layout)}"
            thumbs.append((png, cap))
            seeds.append(vseed)
        except Exception:
            continue
    progress(1.0, desc="Открываю первый вариант…")
    if not seeds:
        return ([], [], _PLACEHOLDER, "", None, "",
                "⚠️ Не удалось построить ни одного варианта — проверьте размеры.", None, vals[_SEED_ARG])
    vals[_SEED_ARG] = seeds[0]
    main = generate(*vals)
    return (thumbs, seeds, *main, seeds[0])


def pick_variant(evt: gr.SelectData, seeds, *vals):
    """Click on a gallery thumbnail: open that variant in the main preview."""
    vals = list(vals)
    if seeds and evt.index is not None and int(evt.index) < len(seeds):
        vals[_SEED_ARG] = int(seeds[int(evt.index)])
    return (*generate(*vals), vals[_SEED_ARG])


EXAMPLES = [
    ["маленький красный истребитель с крыльями"],
    ["большой военный крейсер, чёрный с оранжевым, 3 двигателя"],
    ["long sleek white civilian freighter"],
    ["угловатый промышленный шахтёр из железа"],
    ["stealth destroyer, purple, no wings"],
    ["огромный линкор, синий с золотым, 4 двигателя"],
    ["грузовой катамаран, жёлтый с чёрным"],
    ["крейсер с гондолами, зелёный"],
]


# ------------------------------------------------------------------ turrets

TURRET_KIND_NAMES = {"cannon": "пушка", "chaingun": "автопушка", "laser": "лазер",
                     "railgun": "рельсотрон", "launcher": "ракетная установка"}
_GAME_TURRET_DIR = os.path.expandvars(r"%APPDATA%\Avorion\ships\auto-turrets")

_T_PLACEHOLDER = (
    '<div style="height:460px;display:flex;flex-direction:column;gap:10px;align-items:center;'
    'justify-content:center;color:#93a0b4;background:#0d1117;border-radius:12px;'
    'font-family:system-ui,sans-serif;text-align:center;padding:0 40px">'
    '<div style="font-size:40px">🎯</div>'
    '<div style="font-size:15px;max-width:44ch">Выберите <b style="color:#ff8a5c">тип орудия'
    '</b> — варианты построятся сами.<br>Готовый .xml можно скачать или положить '
    'сразу в папку игры.</div></div>')


def _turret_spec(kind, size, barrels, style, seed) -> TurretSpec:
    return TurretSpec(kind=kind, size=float(size),
                      barrels=None if barrels is None or int(barrels) <= 0 else int(barrels),
                      style=style, seed=int(seed))


def turret_generate(kind, size, barrels, style, seed):
    """Build one turret design: WebGL preview + game-ready .xml."""
    try:
        t = build_turret(_turret_spec(kind, size, barrels, style, seed))
        model = t.assembled()
        html = webgl.ship_to_iframe(model, height=460)
        path = os.path.join(_EXPORT_DIR, f"turret_{kind}_{int(seed)}.xml")
        save_turret_xml(t, path)
        nb = len(t.base) + len(t.body) + len(t.barrel)
        info = (f"🔎 **{TURRET_KIND_NAMES.get(kind, kind)}** · размер **{size}** · "
                f"блоков **{nb}** · стиль **{STYLE_NAMES.get(style, style)}**")
        return html, path, info, ""
    except Exception as exc:
        return _T_PLACEHOLDER, None, "", f"⚠️ Ошибка: {exc}"


_T_SEED_ARG = 4   # position of `seed` in the turret handler argument list


def turret_variants(*vals, progress=gr.Progress()):
    """🎲: build N turret variants with fresh seeds, open the first one."""
    vals = list(vals)
    thumbs, seeds = [], []
    for i in range(N_VARIANTS):
        progress(i / N_VARIANTS, desc=f"Турель {i + 1}/{N_VARIANTS}…")
        vseed = random.randint(1, 999_999)
        vals[_T_SEED_ARG] = vseed
        try:
            model = build_turret(_turret_spec(*vals)).assembled()
            png = os.path.join(_EXPORT_DIR, f"turret_{vseed}.png")
            save_preview(model, png, figsize=2.6)
            thumbs.append((png, f"seed {vseed}"))
            seeds.append(vseed)
        except Exception:
            continue
    if not seeds:
        return ([], [], _T_PLACEHOLDER, None, "",
                "⚠️ Не удалось построить ни одной турели.", vals[_T_SEED_ARG])
    vals[_T_SEED_ARG] = seeds[0]
    return (thumbs, seeds, *turret_generate(*vals), seeds[0])


def turret_pick(evt: gr.SelectData, seeds, *vals):
    """Click on a turret thumbnail: open that variant in the main preview."""
    vals = list(vals)
    if seeds and evt.index is not None and int(evt.index) < len(seeds):
        vals[_T_SEED_ARG] = int(seeds[int(evt.index)])
    return (*turret_generate(*vals), vals[_T_SEED_ARG])


def turret_save_to_game(kind, size, barrels, style, seed):
    """Write the design straight into the game's auto-turrets folder."""
    if not os.path.isdir(_GAME_TURRET_DIR):
        return f"⚠️ Папка игры не найдена: `{_GAME_TURRET_DIR}`"
    t = build_turret(_turret_spec(kind, size, barrels, style, seed))
    name = f"generated_{kind}_{int(seed)}.xml"
    save_turret_xml(t, os.path.join(_GAME_TURRET_DIR, name))
    return f"✅ Сохранено в игру: `{name}` (редактор дизайна турели → загрузить)"

_CSS = """
.gradio-container { max-width: 1520px !important; margin: 0 auto; }
#preview-col { position: sticky; top: 10px; align-self: flex-start; }
#variants-gallery figure { border-radius: 8px; overflow: hidden; }
footer { display: none !important; }
/* deep-space palette (dark theme is forced on load) */
.dark {
  --body-background-fill: #080c12;
  --background-fill-primary: #0e131b;
  --background-fill-secondary: #141b26;
  --block-background-fill: #0e131b;
  --input-background-fill: #141b26;
  --border-color-primary: #202a3a;
  --block-border-color: #1b2331;
  --body-text-color: #dde4ee;
  --block-info-text-color: #93a0b4;
  --color-accent: #ff5a2a;
  /* labels are quiet text, not orange pills — orange is for actions only */
  --block-title-background-fill: transparent;
  --block-title-text-color: #c3cddc;
  --block-label-background-fill: transparent;
  --block-label-text-color: #93a0b4;
  --checkbox-label-background-fill: transparent;
  --checkbox-label-background-fill-hover: transparent;
}
"""

# reload once with the dark theme (matches the space scene in the preview)
_FORCE_DARK = """
() => {
  const u = new URL(window.location.href);
  if (u.searchParams.get('__theme') !== 'dark') {
    u.searchParams.set('__theme', 'dark');
    window.location.replace(u.href);
  }
}
"""


def build_ui() -> gr.Blocks:
    yn = [("авто", AUTO), ("да", "yes"), ("нет", "no")]
    cls_choices = [("авто", AUTO)] + [(CLASS_NAMES[k], k) for k in HULL_CLASSES]
    style_choices = [("авто", AUTO)] + [(STYLE_NAMES[k], k) for k in STYLES]
    layout_choices = [("авто", AUTO)] + [(LAYOUT_NAMES[k], k) for k in LAYOUTS]
    mat_choices = [("авто", AUTO)] + \
                  [(f"{MATERIAL_NAMES[k]} ({v})", str(k)) for k, v in MATERIALS.items()]

    with gr.Blocks(title="Avorion Ship Maker") as demo:
        with gr.Tabs():
            with gr.Tab("🚀 Корабль"):
                seeds_state = gr.State([])

                with gr.Row(equal_height=False):
                    # ------------------------------------------ the ship (left)
                    with gr.Column(scale=8, elem_id="preview-col"):
                        err = gr.Markdown("")
                        preview = gr.HTML(_PLACEHOLDER)
                        variants_gal = gr.Gallery(
                            label="Варианты — клик открывает в большом окне",
                            columns=N_VARIANTS, rows=1, height=140,
                            allow_preview=False, elem_id="variants-gallery")
                        with gr.Row():
                            download = gr.DownloadButton("⬇️ Скачать чертёж .xml", variant="primary")
                            preview_file = gr.DownloadButton("🌐 3D-превью отдельным .html")
                        with gr.Accordion("📊 Состав корабля", open=False):
                            stats = gr.Markdown("")

                    # ------------------------------------------ controls (right)
                    with gr.Column(scale=4):
                        gr.Markdown("## 🚀 Avorion Ship Maker")
                        with gr.Row():
                            hull_class = gr.Dropdown(cls_choices, value=AUTO, label="Класс",
                                                     info="выберите — варианты построятся сами")
                            style = gr.Dropdown(style_choices, value=AUTO, label="Стиль")
                        with gr.Row():
                            layout = gr.Dropdown(layout_choices, value=AUTO, label="Компоновка")
                            wing_kind = gr.Dropdown(
                                [("авто — от seed", AUTO)] +
                                [(WING_KIND_NAMES[k], k) for k in WING_KINDS],
                                value=AUTO, label="Вид крыльев")
                        variants_btn = gr.Button(f"🎲 Подобрать {N_VARIANTS} вариантов",
                                                 variant="primary", size="lg")
                        understood = gr.Markdown("")
                        bevel = gr.Slider(0, 1, 0.85, step=0.05, label="Скос граней",
                                          info="0 — кубы · 1 — гладкий силуэт из клиньев")
                        functional = gr.Slider(0, 1, 0.5, step=0.05, label="Начинка",
                                               info="0 — только внешний вид · 1 — максимум рабочих блоков")
                        with gr.Accordion("✍️ Описание текстом (опционально)", open=False):
                            desc = gr.Textbox(
                                label="Опишите корабль",
                                placeholder="напр.: большой военный крейсер, чёрный с оранжевым",
                                lines=2, info="Enter — построить варианты по описанию")
                            gr.Examples(EXAMPLES, inputs=[desc],
                                        label="Примеры (клик — подставить, Enter — сгенерировать)")

                        with gr.Accordion("📐 Размеры и масштаб", open=False):
                            length = gr.Slider(0, 120, 0, step=1, label="Длина, вокселей",
                                               info="0 — авто: у каждого варианта своя, по правилам класса")
                            width = gr.Slider(0, 60, 0, step=1, label="Ширина, вокселей",
                                              info="0 — авто: у каждого варианта своя, по правилам класса")
                            height = gr.Slider(-20, 20, 0, step=1, label="Высота, поправка",
                                               info="0 — авто · −N ниже · +N выше")
                            with gr.Row():
                                block_size = gr.Slider(0.25, 4.0, 1.0, step=0.05, label="Шаг сетки",
                                                       info="размер одного блока в метрах игры")
                                scale = gr.Slider(0.25, 5.0, 1.0, step=0.05, label="Масштаб",
                                                  info="умножает весь корабль целиком")

                        with gr.Accordion("🔧 Форма и части", open=False):
                            boxiness = gr.Slider(-1, 1, -1, step=0.05, label="Сечение корпуса",
                                                 info="−1 — авто · 0 — обтекаемое · 1 — коробка")
                            armor = gr.Slider(-1, 1, -1, step=0.05, label="Броня",
                                              info="−1 — авто · доля бронированной обшивки")
                            detail = gr.Slider(-1, 1, -1, step=0.05, label="Детализация",
                                               info="−1 — авто · полосы свечения и акценты на обшивке")
                            engines = gr.Slider(-1, 6, -1, step=1, label="Двигатели",
                                                info="−1 — авто: случайное число по правилам класса")
                            turrets = gr.Slider(-1, 12, -1, step=1, label="Турели",
                                                info="площадки под турели на корпусе · −1 — авто по классу")
                            with gr.Row():
                                wings = gr.Radio(yn, value=AUTO, label="Крылья")
                                fins = gr.Radio(yn, value=AUTO, label="Кили")
                                bridge = gr.Radio(yn, value=AUTO, label="Мостик")

                        with gr.Accordion("🎨 Материал и цвета", open=False):
                            material = gr.Dropdown(mat_choices, value=AUTO, label="Материал",
                                                   info="тир блоков: железо — 1-й, аворион — 7-й")
                            use_colors = gr.Checkbox(False, label="Задать цвета вручную (иначе — из стиля)")
                            with gr.Row():
                                primary = gr.ColorPicker("#4a4f52", label="Основной")
                                secondary = gr.ColorPicker("#2b2e30", label="Вторичный")
                                accent = gr.ColorPicker("#c0392b", label="Акцент")
                                glow = gr.ColorPicker("#ff5a2a", label="Свечение")

                        with gr.Row():
                            symmetry = gr.Checkbox(True, label="Симметрия")
                            seed = gr.Number(0, label="Seed", precision=0,
                                             info="тот же seed — тот же корабль")

                inputs = [desc, hull_class, style, layout, length, width, height, engines, wings,
                          wing_kind, fins, bridge, boxiness, armor, detail, bevel, functional,
                          material, block_size, scale, symmetry, seed, use_colors, primary,
                          secondary, accent, glow, turrets]
                outputs = [preview, stats, download, understood, err, preview_file]
                v_outputs = [variants_gal, seeds_state] + outputs + [seed]

                # main cycle: Enter / 🎲 -> variants; click a thumbnail -> open it.
                # Lazy mode: every "what ship do I want" control (class, style,
                # layout, parts, module counts) rebuilds the batch of variants.
                wanted = (hull_class, style, layout, wing_kind, wings, fins, bridge,
                          engines, turrets)
                variants_btn.click(make_variants, inputs=inputs, outputs=v_outputs)
                desc.submit(make_variants, inputs=inputs, outputs=v_outputs)
                for c in wanted:
                    if isinstance(c, gr.Slider):
                        c.release(make_variants, inputs=inputs, outputs=v_outputs)
                    else:
                        c.input(make_variants, inputs=inputs, outputs=v_outputs)
                variants_gal.select(pick_variant, inputs=[seeds_state] + inputs,
                                    outputs=outputs + [seed])
                # fine-tuning: the remaining controls rebuild the CURRENT ship live
                # (.input, not .change: seed write-back must not re-trigger)
                for c in inputs:
                    if c in wanted:
                        continue  # lazy mode above
                    if isinstance(c, gr.Slider):
                        c.release(generate, inputs=inputs, outputs=outputs)
                    elif isinstance(c, gr.Textbox):
                        pass  # Enter is handled by .submit above
                    else:
                        c.input(generate, inputs=inputs, outputs=outputs)

            # ------------------------------------------------- turret designer
            with gr.Tab("🎯 Турель"):
                t_seeds_state = gr.State([])

                with gr.Row(equal_height=False):
                    with gr.Column(scale=8):
                        t_err = gr.Markdown("")
                        t_preview = gr.HTML(_T_PLACEHOLDER)
                        t_gal = gr.Gallery(
                            label="Варианты — клик открывает в большом окне",
                            columns=N_VARIANTS, rows=1, height=140,
                            allow_preview=False)
                        with gr.Row():
                            t_download = gr.DownloadButton("⬇️ Скачать дизайн .xml",
                                                           variant="primary")
                            t_to_game = gr.Button("🎮 Положить в папку игры")
                        t_saved = gr.Markdown("")

                    with gr.Column(scale=4):
                        gr.Markdown("## 🎯 Генератор турелей")
                        t_kind = gr.Dropdown(
                            [(TURRET_KIND_NAMES[k], k) for k in TURRET_KINDS],
                            value="cannon", label="Тип орудия",
                            info="выберите — варианты построятся сами")
                        with gr.Row():
                            t_size = gr.Dropdown(["0.5", "1", "1.5", "2", "3"],
                                                 value="1", label="Размер слота")
                            t_style = gr.Dropdown(
                                [(STYLE_NAMES[k], k) for k in STYLES],
                                value="military", label="Стиль")
                        t_barrels = gr.Slider(0, 4, 0, step=1, label="Стволы",
                                              info="0 — авто по типу орудия")
                        t_btn = gr.Button(f"🎲 Подобрать {N_VARIANTS} вариантов",
                                          variant="primary", size="lg")
                        t_info = gr.Markdown("")
                        t_seed = gr.Number(0, label="Seed", precision=0,
                                           info="тот же seed — та же турель")
                        gr.Markdown(
                            "Файл кладётся в `ships/auto-turrets` — в игре откройте "
                            "**дизайн турели** (в режиме строительства) и загрузите его.")

                t_inputs = [t_kind, t_size, t_barrels, t_style, t_seed]
                t_outputs = [t_preview, t_download, t_info, t_err]
                tv_outputs = [t_gal, t_seeds_state] + t_outputs + [t_seed]

                t_btn.click(turret_variants, inputs=t_inputs, outputs=tv_outputs)
                for c in (t_kind, t_size, t_style):
                    c.input(turret_variants, inputs=t_inputs, outputs=tv_outputs)
                t_barrels.release(turret_variants, inputs=t_inputs, outputs=tv_outputs)
                t_seed.input(turret_generate, inputs=t_inputs, outputs=t_outputs)
                t_gal.select(turret_pick, inputs=[t_seeds_state] + t_inputs,
                             outputs=t_outputs + [t_seed])
                t_to_game.click(turret_save_to_game, inputs=t_inputs, outputs=[t_saved])

        # a default ship right away, so the page never opens empty
        demo.load(generate, inputs=inputs, outputs=outputs)
    return demo


def main():
    demo = build_ui()
    demo.launch(inbrowser=True, css=_CSS, js=_FORCE_DARK,
                theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"))


if __name__ == "__main__":
    main()
