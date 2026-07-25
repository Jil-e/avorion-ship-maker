"""Command-line access to the corpus-driven ship generator."""
from __future__ import annotations

import argparse

from . import AutoShipSpec, generate_auto_ship, save_xml
from .preview import save_preview


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate an Avorion Auto-Ship blueprint.")
    p.add_argument("description", nargs="?", default="falcon",
                   help="layout name: falcon, hammerhead, main, carry, an_perdole, crab")
    p.add_argument("-o", "--out", default="ship.xml", help="output .xml path")
    p.add_argument("--png", default=None, help="also save a preview PNG to this path")
    p.add_argument("--length", type=float, default=64.0)
    p.add_argument("--width", type=float, default=24.0)
    p.add_argument("--height", type=float, default=12.0)
    p.add_argument("--slots", type=int, default=15, help="subsystem slots, 0..15")
    p.add_argument("--material", type=int, action="append", default=None,
                   help="allowed material id; repeat for a palette")
    p.add_argument("--reference", default=None, help="PNG/JPG silhouette reference")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    layout = args.description.lower().replace("-", "_")
    if layout not in {"falcon", "hammerhead", "main", "carry", "an_perdole", "crab"}:
        layout = "falcon"
    spec = AutoShipSpec(
        layout=layout, length=args.length, width=args.width, height=args.height,
        subsystem_cells=min(15, max(0, args.slots)),
        materials=tuple(args.material or [1]), seed=args.seed,
        reference_image=args.reference,
    )
    ship = generate_auto_ship(spec)
    save_xml(ship, args.out)
    dims = tuple(round(d, 1) for d in ship.dimensions())
    print(f"Wrote {args.out}: {len(ship)} blocks, dimensions (W,H,L)={dims}")
    if args.png:
        save_preview(ship, args.png)
        print(f"Wrote preview {args.png}")


if __name__ == "__main__":
    main()
