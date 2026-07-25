# Avorion Auto-Ship Maker

Генератор кораблей Avorion, который строит новые планы по геометрическому
анализу коллекции `auto-ships`.

## Что делает генератор

- Загружает все пригодные XML из `%APPDATA%/Avorion/ships/auto-ships`.
- Удаляет дубликаты по нормализованной геометрии корпуса.
- Измеряет occupancy-поле, продольные профили ширины и высоты, вертикальное
  смещение, плотность сечений и функциональную нагрузку.
- Смешивает все уникальные формы corpus с bias выбранной компоновки.
- Применяет seed-зависимую низкочастотную деформацию.
- Строит новый корпус на сетке `0.25`, а не переносит блоки исходных XML.
- Синтезирует броню, внутренние системы и турельные площадки отдельно.
- Не генерирует `<turretDesign>`.
- Для Falcon без загруженного изображения использует встроенный
  `avorion_ship_maker/Falcon_base.png`.

## Компоновки

- `falcon` — разнесённые корпуса и центральный канал по Falcon silhouette.
- `hammerhead` — расширение носовой части.
- `main` — длинный midline-корпус.
- `carry` — плотная промышленная геометрия.
- `an_perdole` — промышленная вариация carry-family.
- `crab` — широкий корпус с боковой асимметрией профиля.

## Запуск UI

```bash
pip install -r requirements.txt
python main.py
```

Интерфейс содержит один генератор Auto-Ship: layout, размеры, subsystem slots,
материалы, PNG/JPG-силуэт и seed. Falcon автоматически получает встроенный
референс, если пользователь не загрузил другой.

## CLI

```bash
python -m avorion_ship_maker.cli falcon -o falcon.xml --seed 17
python -m avorion_ship_maker.cli carry -o carry.xml --length 64 --width 24 --height 12
python -m avorion_ship_maker.cli crab -o crab.xml --slots 8 --material 1 --material 2
```

Доступные параметры: `--length`, `--width`, `--height`, `--slots`, `--material`,
`--reference`, `--seed` и `--png`.

## Python API

```python
from avorion_ship_maker import AutoShipSpec, generate_auto_ship, save_xml

spec = AutoShipSpec(
    layout="falcon",
    length=64,
    width=24,
    height=12,
    subsystem_cells=15,
    materials=(1, 2),
    seed=17,
)
ship = generate_auto_ship(spec)
save_xml(ship, "falcon.xml")
```

## Структура проекта

```text
avorion_ship_maker/
├── model.py                         # Block / ShipModel
├── blocks.py                        # реестр типов Avorion
├── connectivity.py                  # дерево связности XML
├── xml_io.py                        # сериализация и парсинг XML
├── preview.py                       # PNG preview
├── webgl.py                         # интерактивное 3D-превью
├── app.py                           # единственный ship-generator UI
├── cli.py                           # командная строка
└── generator/
    ├── auto_ship_generator.py       # анализ corpus и новая сборка корабля
    └── voxel.py                     # occupancy grid и greedy merge
```

## Загрузка в Avorion

Скопируйте XML в `%APPDATA%/Avorion/ships/` или используйте в игре
`Build Mode -> Load Ship -> from file`.
