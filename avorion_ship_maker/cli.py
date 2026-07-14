"""Command-line generation, for scripting without the UI::

    python -m avorion_ship_maker.cli "большой военный крейсер, синий" -o cruiser.xml
    python -m avorion_ship_maker.cli "long white freighter" --scale 1.5 --png out.png
"""
from __future__ import annotations

import argparse

from . import generate, save_xml
from .preview import save_preview


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate an Avorion ship blueprint from text.")
    p.add_argument("description", help="ship description (Russian or English)")
    p.add_argument("-o", "--out", default="ship.xml", help="output .xml path")
    p.add_argument("--png", default=None, help="also save a preview PNG to this path")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--step", type=float, default=1.0, help="block size (grid step)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    ship = generate(args.description, scale=args.scale, block_size=args.step, seed=args.seed)
    save_xml(ship, args.out)
    dims = tuple(round(d, 1) for d in ship.dimensions())
    print(f"Wrote {args.out}: {len(ship)} blocks, dimensions (W,H,L)={dims}")
    if args.png:
        save_preview(ship, args.png)
        print(f"Wrote preview {args.png}")


if __name__ == "__main__":
    main()
