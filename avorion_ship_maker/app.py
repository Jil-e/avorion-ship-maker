"""Gradio UI for the Avorion Ship Maker.

Type a description (Russian or English), tweak the parameters, generate — see a
3D preview and download an Avorion-ready ``.xml`` blueprint. Run with::

    python -m avorion_ship_maker.app
"""
from __future__ import annotations

import os
import re
import tempfile

import gradio as gr

from .blocks import BLOCK_NAMES, MATERIALS
from .generator.builder import build_ship
from .generator.spec import HULL_CLASSES, STYLES, ShipSpec
from .generator.text_parser import parse_description
from .preview import render_3d
from .xml_io import save_xml

AUTO = "auto"
_EXPORT_DIR = os.path.join(tempfile.gettempdir(), "avorion_ship_maker")
os.makedirs(_EXPORT_DIR, exist_ok=True)


def _safe_name(name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", (name or "ship").strip(), flags=re.U).strip("_")
    return (slug or "ship")[:40]


def build_spec(description, hull_class, style, length, width, height,
               engines, wings, fins, bridge, boxiness, armor, detail,
               bevel, functional, material, block_size, scale, symmetry, seed,
               use_colors, primary, secondary, accent, glow) -> ShipSpec:
    """Compose a spec: description supplies defaults, UI controls override them."""
    spec = parse_description(description or "")

    if hull_class != AUTO:
        spec.hull_class = hull_class
    if style != AUTO:
        spec.style = style
    if length and length > 0:
        spec.length = int(length)
    if width and width > 0:
        spec.width = int(width)
    if height and height > 0:
        spec.height = int(height)
    if engines is not None and engines >= 0:
        spec.engines = int(engines)
    for val, attr in ((wings, "wings"), (fins, "fins"), (bridge, "bridge")):
        if val != AUTO:
            setattr(spec, attr, val == "да / yes")
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
        spec.material = int(material.split(" ")[0])

    spec.block_size = float(block_size)
    spec.scale = float(scale)
    spec.symmetry = bool(symmetry)
    spec.seed = int(seed)

    if use_colors:
        spec.primary, spec.secondary = primary, secondary
        spec.accent, spec.glow = accent, glow
    return spec


def _stats_md(ship, spec) -> str:
    (lx, ly, lz), (hx, hy, hz) = ship.bounds()
    dx, dy, dz = round(hx - lx, 1), round(hy - ly, 1), round(hz - lz, 1)
    counts = ship.type_counts()
    rows = "\n".join(
        f"| {BLOCK_NAMES.get(idx, f'type {idx}')} | `{idx}` | {n} |"
        for idx, n in sorted(counts.items(), key=lambda kv: -kv[1])
    )
    return (
        f"### 🛰️ {spec.name}\n"
        f"**Класс:** {spec.hull_class} · **Стиль:** {spec.style} · "
        f"**Материал:** {MATERIALS.get(spec.material, spec.material)}\n\n"
        f"**Блоков:** {len(ship)} · **Габариты (Д×Ш×В):** {dz} × {dx} × {dy}\n\n"
        f"| Блок | index | шт. |\n|---|---|---|\n{rows}"
    )


def generate(description, hull_class, style, length, width, height,
             engines, wings, fins, bridge, boxiness, armor, detail,
             bevel, functional, material, block_size, scale, symmetry, seed,
             use_colors, primary, secondary, accent, glow, elev, azim):
    """Main handler: build the ship, render it, write the .xml."""
    try:
        spec = build_spec(description, hull_class, style, length, width, height,
                          engines, wings, fins, bridge, boxiness, armor, detail,
                          bevel, functional, material, block_size, scale, symmetry, seed,
                          use_colors, primary, secondary, accent, glow)
        ship = build_ship(spec)
        if not ship.blocks:
            raise ValueError("Сгенерирован пустой корпус — увеличьте размеры.")
        fig = render_3d(ship, elev=elev, azim=azim)
        path = os.path.join(_EXPORT_DIR, f"{_safe_name(spec.name)}.xml")
        save_xml(ship, path)
        return ship, fig, _stats_md(ship, spec), path, ""
    except Exception as exc:  # surface the error in the UI instead of a stack trace
        return None, None, "", None, f"⚠️ Ошибка: {exc}"


def rerender(ship, elev, azim):
    """Re-render the stored ship from a new camera angle (no regeneration)."""
    if ship is None or not getattr(ship, "blocks", None):
        return None
    return render_3d(ship, elev=elev, azim=azim)


EXAMPLES = [
    ["маленький красный истребитель с крыльями"],
    ["большой военный крейсер, чёрный с оранжевым, 3 двигателя"],
    ["long sleek white civilian freighter"],
    ["угловатый промышленный шахтёр из железа"],
    ["stealth destroyer, purple, no wings"],
    ["огромный линкор, синий с золотым, 4 двигателя"],
]


def build_ui() -> gr.Blocks:
    yn = [AUTO, "да / yes", "нет / no"]
    mats = [AUTO] + [f"{k} {v}" for k, v in MATERIALS.items()]

    with gr.Blocks(title="Avorion Ship Maker") as demo:
        state = gr.State(None)
        gr.Markdown(
            "# 🚀 Avorion Ship Maker\n"
            "Опишите корабль словами (можно по-русски) и/или задайте параметры — "
            "получите 3D-превью и готовый чертёж `.xml` для Avorion."
        )
        with gr.Row():
            # ------------------------------------------------ controls
            with gr.Column(scale=5):
                desc = gr.Textbox(
                    label="Описание корабля",
                    placeholder="напр.: большой военный крейсер, чёрный с оранжевым, 3 двигателя",
                    lines=2,
                )
                with gr.Row():
                    hull_class = gr.Dropdown([AUTO] + list(HULL_CLASSES), value=AUTO, label="Класс корпуса")
                    style = gr.Dropdown([AUTO] + list(STYLES), value=AUTO, label="Стиль")
                with gr.Accordion("📐 Размеры и масштаб", open=True):
                    with gr.Row():
                        length = gr.Slider(0, 120, 0, step=1, label="Длина (0=авто)")
                        width = gr.Slider(0, 60, 0, step=1, label="Ширина (0=авто)")
                        height = gr.Slider(0, 60, 0, step=1, label="Высота (0=авто)")
                    with gr.Row():
                        block_size = gr.Slider(0.25, 4.0, 1.0, step=0.05, label="Шаг (размер блока)")
                        scale = gr.Slider(0.25, 5.0, 1.0, step=0.05, label="Масштаб")
                with gr.Accordion("🔧 Форма и части", open=False):
                    with gr.Row():
                        boxiness = gr.Slider(-1, 1, -1, step=0.05, label="Угловатость (-1=авто)")
                        armor = gr.Slider(-1, 1, -1, step=0.05, label="Броня (-1=авто)")
                        detail = gr.Slider(-1, 1, -1, step=0.05, label="Детализация (-1=авто)")
                    with gr.Row():
                        bevel = gr.Slider(-1, 1, -1, step=0.05,
                                          label="Скос рёбов/клинья (-1=авто, 0=кубы, 1=гладко)")
                        functional = gr.Slider(0, 1, 0.5, step=0.05,
                                               label="Функциональность (0=красота … 1=начинка)")
                    with gr.Row():
                        engines = gr.Slider(-1, 6, -1, step=1, label="Двигатели (-1=авто)")
                        wings = gr.Radio(yn, value=AUTO, label="Крылья")
                        fins = gr.Radio(yn, value=AUTO, label="Кили")
                        bridge = gr.Radio(yn, value=AUTO, label="Мостик")
                with gr.Accordion("🎨 Материал и цвета", open=False):
                    material = gr.Dropdown(mats, value=AUTO, label="Материал")
                    use_colors = gr.Checkbox(False, label="Переопределить цвета вручную")
                    with gr.Row():
                        primary = gr.ColorPicker("#4a4f52", label="Основной")
                        secondary = gr.ColorPicker("#2b2e30", label="Вторичный")
                        accent = gr.ColorPicker("#c0392b", label="Акцент")
                        glow = gr.ColorPicker("#ff5a2a", label="Свечение")
                with gr.Row():
                    symmetry = gr.Checkbox(True, label="Симметрия")
                    seed = gr.Number(0, label="Seed", precision=0)
                gen_btn = gr.Button("🛠️ Сгенерировать корабль", variant="primary", size="lg")
                gr.Examples(EXAMPLES, inputs=[desc], label="Примеры")

            # ------------------------------------------------ output
            with gr.Column(scale=6):
                err = gr.Markdown("")
                plot = gr.Plot(label="3D-превью")
                with gr.Row():
                    elev = gr.Slider(-90, 90, 22, step=2, label="Наклон камеры")
                    azim = gr.Slider(-180, 180, -58, step=2, label="Поворот камеры")
                download = gr.File(label="Скачать чертёж (.xml)")
                stats = gr.Markdown("")

        inputs = [desc, hull_class, style, length, width, height, engines, wings,
                  fins, bridge, boxiness, armor, detail, bevel, functional, material,
                  block_size, scale, symmetry, seed, use_colors, primary, secondary,
                  accent, glow, elev, azim]
        gen_btn.click(generate, inputs=inputs,
                      outputs=[state, plot, stats, download, err])
        for ctrl in (elev, azim):
            ctrl.release(rerender, inputs=[state, elev, azim], outputs=[plot])
    return demo


def main():
    demo = build_ui()
    demo.launch(inbrowser=True,
                theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"))


if __name__ == "__main__":
    main()
