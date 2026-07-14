"""Interactive WebGL preview of a :class:`ShipModel` (Three.js in an iframe).

Builds a compact face mesh (cubes + real wedge/corner prisms) from the ship and
embeds a self-contained Three.js scene with mouse orbit/zoom. Used by the Gradio
UI via ``ship_to_iframe`` and available as a standalone file via ``save_html``.

The payload is deliberately small (vertices + per-face palette indices, no
normals — flat shading derives them on the GPU): big ships used to blow past
the browser's data-URI limits and the preview silently failed to load.
"""
from __future__ import annotations

import html as _html
import json

import numpy as np

from .model import ShipModel
from .orient import CORNER_TYPES, EDGE_TYPES, corner_faces, edge_faces

_CUBE_FACES = [
    (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
    (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4),
]


def _rgb(color: str):
    c = color[-6:] if len(color) >= 6 else "808080"
    # sRGB -> linear: three.js expects linear vertex colours with sRGB output
    return [(int(c[i:i + 2], 16) / 255.0) ** 2.2 for i in (0, 2, 4)]


def _cube_faces(b):
    x0, y0, z0, x1, y1, z1 = b.lx, b.ly, b.lz, b.ux, b.uy, b.uz
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1),
         (x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)]
    return [[v[i] for i in f] for f in _CUBE_FACES]


def _mesh_payload(ship: ShipModel):
    """Compact mesh: flat vertex list, per-face (nverts, colorIdx), palette."""
    verts: list[float] = []
    faces: list[int] = []
    palette: list[list[float]] = []
    pal_idx: dict[tuple, int] = {}
    for b in ship.blocks:
        rgb = tuple(round(c, 4) for c in _rgb(b.color))
        ci = pal_idx.get(rgb)
        if ci is None:
            ci = pal_idx[rgb] = len(palette)
            palette.append(list(rgb))
        lo, hi = (b.lx, b.ly, b.lz), (b.ux, b.uy, b.uz)
        if b.index in EDGE_TYPES:
            flist = edge_faces(lo, hi, b.look, b.up)
        elif b.index in CORNER_TYPES:
            flist = corner_faces(lo, hi, b.look, b.up)
        else:
            flist = _cube_faces(b)
        for face in flist:
            faces.extend((len(face), ci))
            for p in face:
                verts.extend((round(float(p[0]), 3), round(float(p[1]), 3),
                              round(float(p[2]), 3)))
    return verts, faces, palette


def build_html(ship: ShipModel, bg: str = "#0d1117") -> str:
    """Return a full self-contained HTML document rendering the ship in WebGL."""
    verts, faces, palette = _mesh_payload(ship)
    (lx, ly, lz), (hx, hy, hz) = ship.bounds()
    data = json.dumps({
        "v": verts, "f": faces, "p": palette,
        "center": [(lx + hx) / 2, (ly + hy) / 2, (lz + hz) / 2],
        "size": max(hx - lx, hy - ly, hz - lz, 1.0),
        "bg": bg,
    }, separators=(",", ":"))
    return _TEMPLATE.replace("/*DATA*/null", data).replace("__BG__", bg)


def ship_to_iframe(ship: ShipModel, height: int = 520, bg: str = "#0d1117") -> str:
    """Return an <iframe srcdoc=...> embedding the WebGL scene, for gr.HTML.

    ``srcdoc`` (not a data: URI): data URIs get truncated by the browser on
    big ships and their opaque origin defeats the HTTP cache for three.js.
    """
    if not ship.blocks:
        return f'<div style="height:{height}px"></div>'
    doc = _html.escape(build_html(ship, bg), quote=True)
    return (f'<iframe title="ship-preview" '
            f'style="width:100%;height:{height}px;border:0;border-radius:12px;'
            f'display:block;background:{bg}" '
            f'sandbox="allow-scripts allow-same-origin" '
            f'srcdoc="{doc}"></iframe>')


def save_html(ship: ShipModel, path: str, bg: str = "#0d1117") -> str:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(build_html(ship, bg))
    return path


_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>html,body{margin:0;height:100%;overflow:hidden;background:__BG__;font-family:sans-serif}
#hint{position:fixed;left:10px;bottom:8px;color:#8b949e;font-size:12px;user-select:none}
#load,#err{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;
color:#c9d1d9;padding:20px;text-align:center;font-size:14px}</style></head>
<body>
<div id="hint">ЛКМ — вращать · колесо — зум · ПКМ — сдвиг</div>
<div id="load">⏳ Загрузка 3D…</div>
<div id="err" style="display:none">Не удалось загрузить 3D-движок (нет доступа к сети?).</div>
<script type="importmap">
{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js",
"three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}
</script>
<script type="module">
const D = /*DATA*/null;
try {
  const THREE = await import('three');
  const { OrbitControls } = await import('three/addons/controls/OrbitControls.js');
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(D.bg);
  const cam = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 0.1, D.size*40);
  const R = D.size;
  cam.position.set(D.center[0]+R*1.1, D.center[1]+R*0.8, D.center[2]+R*1.5);
  const renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setPixelRatio(devicePixelRatio); renderer.setSize(innerWidth, innerHeight);
  document.body.appendChild(renderer.domElement);
  // expand the compact payload: faces -> indexed geometry + vertex colors
  // (no normal attribute: flatShading derives face normals on the GPU)
  const nv = D.v.length / 3;
  const col = new Float32Array(nv * 3);
  const idx = [];
  let vi = 0;
  for (let i = 0; i < D.f.length; i += 2) {
    const n = D.f[i], c = D.p[D.f[i+1]];
    for (let k = 0; k < n; k++) {
      col[(vi+k)*3] = c[0]; col[(vi+k)*3+1] = c[1]; col[(vi+k)*3+2] = c[2];
    }
    for (let k = 1; k < n-1; k++) idx.push(vi, vi+k, vi+k+1);
    vi += n;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(D.v, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  g.setIndex(idx);
  const mat = new THREE.MeshStandardMaterial({vertexColors:true, metalness:0.15, roughness:0.72,
    flatShading:true, side:THREE.DoubleSide});
  scene.add(new THREE.Mesh(g, mat));
  scene.add(new THREE.HemisphereLight(0xffffff, 0x223044, 0.75));
  const key = new THREE.DirectionalLight(0xffffff, 1.15); key.position.set(1,1.6,1.2); scene.add(key);
  const rim = new THREE.DirectionalLight(0x88aaff, 0.5); rim.position.set(-1,0.4,-1); scene.add(rim);
  const ctr = new OrbitControls(cam, renderer.domElement);
  ctr.target.set(D.center[0], D.center[1], D.center[2]);
  ctr.enableDamping = true; ctr.autoRotate = true; ctr.autoRotateSpeed = 0.6; ctr.update();
  addEventListener('resize', () => {
    cam.aspect = innerWidth/innerHeight; cam.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });
  renderer.domElement.addEventListener('pointerdown', () => ctr.autoRotate = false);
  document.getElementById('load').style.display = 'none';
  (function loop(){ requestAnimationFrame(loop); ctr.update(); renderer.render(scene, cam); })();
} catch (e) {
  document.getElementById('load').style.display = 'none';
  document.getElementById('err').style.display = 'flex';
  console.error(e);
}
</script>
</body></html>"""
