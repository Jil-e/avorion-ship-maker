"""Interactive WebGL preview of a :class:`ShipModel` (Three.js in an iframe).

Builds a flat triangle mesh (cubes + real wedge/corner prisms) from the ship and
embeds a self-contained Three.js scene with mouse orbit/zoom. Used by the Gradio
UI via ``ship_to_iframe`` and available as a standalone file via ``save_html``.
"""
from __future__ import annotations

import base64
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
    return [int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def _cube_faces(b):
    x0, y0, z0, x1, y1, z1 = b.lx, b.ly, b.lz, b.ux, b.uy, b.uz
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1),
         (x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)]
    return [[v[i] for i in f] for f in _CUBE_FACES]


def _mesh_arrays(ship: ShipModel):
    """Flatten the ship into position/normal/color arrays (triangle soup)."""
    pos, nrm, col = [], [], []
    for b in ship.blocks:
        rgb = _rgb(b.color)
        lo, hi = (b.lx, b.ly, b.lz), (b.ux, b.uy, b.uz)
        if b.index in EDGE_TYPES:
            faces = edge_faces(lo, hi, b.look, b.up)
        elif b.index in CORNER_TYPES:
            faces = corner_faces(lo, hi, b.look, b.up)
        else:
            faces = _cube_faces(b)
        for face in faces:
            pts = [np.asarray(p, float) for p in face]
            n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            ln = float(np.linalg.norm(n))
            n = (n / ln) if ln > 1e-9 else np.array([0.0, 1.0, 0.0])
            for i in range(1, len(pts) - 1):          # fan-triangulate
                for p in (pts[0], pts[i], pts[i + 1]):
                    pos.extend((float(p[0]), float(p[1]), float(p[2])))
                    nrm.extend((float(n[0]), float(n[1]), float(n[2])))
                    col.extend(rgb)
    return pos, nrm, col


def build_html(ship: ShipModel, bg: str = "#0d1117") -> str:
    """Return a full self-contained HTML document rendering the ship in WebGL."""
    pos, nrm, col = _mesh_arrays(ship)
    (lx, ly, lz), (hx, hy, hz) = ship.bounds()
    data = json.dumps({
        "pos": pos, "nrm": nrm, "col": col,
        "center": [(lx + hx) / 2, (ly + hy) / 2, (lz + hz) / 2],
        "size": max(hx - lx, hy - ly, hz - lz, 1.0),
        "bg": bg,
    })
    return _TEMPLATE.replace("/*DATA*/null", data).replace("__BG__", bg)


def ship_to_iframe(ship: ShipModel, height: int = 520, bg: str = "#0d1117") -> str:
    """Return an <iframe> (data-URI) embedding the WebGL scene, for gr.HTML."""
    if not ship.blocks:
        return f'<div style="height:{height}px"></div>'
    doc = build_html(ship, bg)
    b64 = base64.b64encode(doc.encode("utf-8")).decode("ascii")
    return (f'<iframe title="ship-preview" '
            f'style="width:100%;height:{height}px;border:0;border-radius:12px;'
            f'display:block;background:{bg}" '
            f'sandbox="allow-scripts allow-same-origin" '
            f'src="data:text/html;base64,{b64}"></iframe>')


def save_html(ship: ShipModel, path: str, bg: str = "#0d1117") -> str:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(build_html(ship, bg))
    return path


_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>html,body{margin:0;height:100%;overflow:hidden;background:__BG__;font-family:sans-serif}
#hint{position:fixed;left:10px;bottom:8px;color:#8b949e;font-size:12px;user-select:none}
#err{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;color:#c9d1d9;
padding:20px;text-align:center;font-size:14px}</style></head>
<body>
<div id="hint">ЛКМ — вращать · колесо — зум · ПКМ — сдвиг</div>
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
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(D.pos, 3));
  g.setAttribute('normal',   new THREE.Float32BufferAttribute(D.nrm, 3));
  g.setAttribute('color',    new THREE.Float32BufferAttribute(D.col, 3));
  const mat = new THREE.MeshStandardMaterial({vertexColors:true, metalness:0.15, roughness:0.72, flatShading:true});
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
  (function loop(){ requestAnimationFrame(loop); ctr.update(); renderer.render(scene, cam); })();
} catch (e) { document.getElementById('err').style.display='flex'; console.error(e); }
</script>
</body></html>"""
