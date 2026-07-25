"""Gradio UI for the measured-geometry Avorion ship generator."""
from __future__ import annotations

import os
import re
import tempfile

import gradio as gr

from . import webgl
from .blocks import MATERIALS
from .generator.auto_ship_generator import AutoShipSpec, generate_auto_ship
from .model import ShipModel
from .preview import save_preview
from .xml_io import save_xml


_EXPORT_DIR = os.path.join(tempfile.gettempdir(), "avorion_ship_maker")
os.makedirs(_EXPORT_DIR, exist_ok=True)

AUTO_LAYOUT_NAMES = {
    "falcon": "Falcon · разнесённые корпуса",
    "hammerhead": "Hammerhead · широкий нос",
    "main": "Main · длинный корпус",
    "carry": "Carry · грузовой корпус",
    "an_perdole": "An Perdole · промышленный midline",
    "crab": "Crab · широкий боковой корпус",
}
MATERIAL_NAMES = {
    0: "железо", 1: "титан", 2: "наонит", 3: "триний",
    4: "ксанион", 5: "огонит", 6: "аворион",
}
_FUNCTIONAL_BLOCKS = {
    3, 5, 6, 7, 10, 13, 14, 15, 17, 19, 20, 25,
    50, 51, 52, 53, 54, 55, 60, 61,
}

_PLACEHOLDER = (
    '<div style="height:500px;display:flex;flex-direction:column;gap:10px;'
    'align-items:center;justify-content:center;color:#93a0b4;background:#0d1117;'
    'border-radius:12px;font-family:system-ui,sans-serif;text-align:center;'
    'padding:0 40px"><div style="font-size:40px">🧬</div>'
    '<div style="font-size:15px;max-width:48ch">Настройте геометрию справа и '
    'запустите генератор.</div></div>'
)
_CSS = """
.gradio-container { max-width: 1500px !important; background: #090d12; }
#preview-col { min-height: 520px; }
footer { display: none !important; }
"""


def _safe_name(name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", (name or "ship").strip(), flags=re.U).strip("_")
    return (slug or "ship")[:60]


def _stats_md(ship: ShipModel, spec: AutoShipSpec) -> str:
    counts = ship.type_counts()
    functional_volume = sum(
        block.volume for block in ship.blocks if block.index in _FUNCTIONAL_BLOCKS
    )
    pads = counts.get(20, 0) + counts.get(25, 0)
    dims = tuple(round(float(value), 2) for value in ship.dimensions())
    return (
        f"**Блоков:** {len(ship)} · **Рабочий объём:** {functional_volume:g} · "
        f"**Площадок:** {pads} · **Габариты W×H×L:** {dims} · "
        f"**Subsystem slots:** {spec.subsystem_cells}"
    )


def auto_ship_generate(layout, length, width, height, subsystem_cells,
                       materials, seed, reference_image=None):
    """Generate a full ship using only the measured-geometry pipeline."""
    try:
        mats = tuple(int(value) for value in (materials or ["1"]))
        spec = AutoShipSpec(
            length=max(8.0, float(length)),
            width=max(4.0, float(width)),
            height=max(3.0, float(height)),
            subsystem_cells=min(15, max(0, int(subsystem_cells))),
            layout=layout,
            materials=mats,
            seed=int(seed),
            reference_image=str(reference_image) if reference_image else None,
        )
        ship = generate_auto_ship(spec)
        html = webgl.ship_to_iframe(ship, height=500)
        path = os.path.join(_EXPORT_DIR, f"auto_{layout}_{int(seed)}.xml")
        save_xml(ship, path)
        stats = _stats_md(ship, spec)
        info = (
            f"**{AUTO_LAYOUT_NAMES.get(layout, layout)}** · "
            f"материалы: {', '.join(MATERIAL_NAMES.get(m, str(m)) for m in mats)} · "
            "все уникальные auto-ship геометрии участвуют в blend"
        )
        return html, stats, path, info, ""
    except Exception as exc:
        return _PLACEHOLDER, "", None, "", f"⚠️ Ошибка: {exc}"


def auto_ship_save_to_game(layout, length, width, height, subsystem_cells,
                           materials, seed, reference_image, ship_name):
    game_dir = os.path.expandvars(r"%APPDATA%\Avorion\ships")
    if not os.path.isdir(game_dir):
        return f"⚠️ Папка игры не найдена: `{game_dir}`"
    try:
        mats = tuple(int(value) for value in (materials or ["1"]))
        spec = AutoShipSpec(
            length=max(8.0, float(length)), width=max(4.0, float(width)),
            height=max(3.0, float(height)),
            subsystem_cells=min(15, max(0, int(subsystem_cells))),
            layout=layout, materials=mats, seed=int(seed),
            reference_image=str(reference_image) if reference_image else None,
        )
        ship = generate_auto_ship(spec)
        stem = _safe_name(ship_name) if str(ship_name or "").strip() else ship.name
        path = os.path.join(game_dir, f"{stem}.xml")
        save_xml(ship, path)
        save_preview(ship, path + ".png", figsize=4.0, azim=122)
        return f"✅ Сохранено в игру: `{os.path.basename(path)}`"
    except Exception as exc:
        return f"⚠️ Ошибка: {exc}"


def build_ui() -> gr.Blocks:
    layout_choices = [(label, key) for key, label in AUTO_LAYOUT_NAMES.items()]
    material_choices = [(MATERIAL_NAMES[key], str(key)) for key in MATERIALS]

    with gr.Blocks(title="Avorion Auto-Ship Maker") as demo:
        with gr.Row(equal_height=False):
            with gr.Column(scale=8, elem_id="preview-col"):
                a_error = gr.Markdown("")
                a_preview = gr.HTML(_PLACEHOLDER)
                a_info = gr.Markdown("")
                a_stats = gr.Markdown("")
                a_download = gr.DownloadButton(
                    "⬇️ Скачать Auto-Ship .xml", variant="primary"
                )
                with gr.Accordion("🎮 Сохранить в Avorion", open=False):
                    with gr.Row():
                        a_game_name = gr.Textbox(
                            label="Имя корабля", placeholder="автоматически", scale=2
                        )
                        a_game_button = gr.Button("Сохранить")
                    a_game_status = gr.Markdown("")

            with gr.Column(scale=4):
                gr.Markdown("# 🧬 Auto-Ship Forge")
                gr.Markdown(
                    "Генератор анализирует все уникальные корабли из `auto-ships`, "
                    "смешивает их геометрические поля и строит новый корпус. "
                    "Исходные XML-блоки не копируются."
                )
                a_layout = gr.Dropdown(
                    layout_choices, value="falcon", label="Компоновка"
                )
                with gr.Row():
                    a_length = gr.Number(64, label="Длина", precision=1)
                    a_width = gr.Number(24, label="Ширина", precision=1)
                    a_height = gr.Number(12, label="Высота", precision=1)
                a_cells = gr.Slider(
                    0, 15, 15, step=1, label="Subsystem slots",
                    info="Функциональный объём рассчитывается из corpus"
                )
                a_materials = gr.CheckboxGroup(
                    material_choices, value=["1"], label="Разрешённые материалы"
                )
                a_reference = gr.Image(
                    type="filepath", label="PNG/JPG референс силуэта",
                    height=140
                )
                a_seed = gr.Number(17, label="Seed", precision=0)
                a_button = gr.Button(
                    "🧬 Сгенерировать новый корабль", variant="primary", size="lg"
                )

        a_inputs = [a_layout, a_length, a_width, a_height, a_cells,
                    a_materials, a_seed, a_reference]
        a_outputs = [a_preview, a_stats, a_download, a_info, a_error]
        a_button.click(auto_ship_generate, inputs=a_inputs, outputs=a_outputs)
        a_game_button.click(
            auto_ship_save_to_game,
            inputs=a_inputs + [a_game_name],
            outputs=[a_game_status],
        )
        demo.load(auto_ship_generate, inputs=a_inputs, outputs=a_outputs)
    return demo


def main():
    build_ui().launch(
        inbrowser=True,
        css=_CSS,
        theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"),
    )


if __name__ == "__main__":
    main()
