"""Block orientation + shape geometry for Avorion shape blocks (edges/corners).

Avorion orients a block with two integer codes ``look`` and ``up`` (each 0-5,
selecting one of six axis directions). For full cubes the codes are always the
default ``look=1, up=3``; for shape blocks (wedges/corners) they aim the slope.

The integer->axis mapping was derived from the reference ships (default cube =
look1/up3 => +z / +y). The slope semantics are EMPIRICAL, taken from the
game's own plans (ground truth, not scripts):

* Edges — data/plans/arrow.xml: its arrowhead diagonals prove a wedge with
  (look, up) chamfers the corner between its +look and +up faces, and the
  encoding is symmetric under a look/up swap (the arrow uses both orders).
* Corners — adjacency analysis over fighter/tutorialship/localship plans
  (faces without neighbours = the cut octant) gives a UNANIMOUS
  (look, up) -> cut-octant table; it is NOT expressible as fixed signs in
  the (look, up, cross) frame, so it lives here as literal tables.

Emission (bevel_*_orient) and preview decoding (edge_faces/corner_faces)
both live here so they can never drift apart.
"""
from __future__ import annotations

import numpy as np

# int code (0-5) -> unit direction. Anchors: look=1 -> +z (forward), up=3 -> +y.
# lf=+x, rg=-x, fw=+z, bw=-z, up=+y, dn=-y (from generator.lua).
INT2VEC: dict[int, tuple[int, int, int]] = {
    0: (0, 0, -1),   # -z  (bw)
    1: (0, 0, 1),    # +z  (fw)   <- default look
    2: (0, -1, 0),   # -y  (dn)
    3: (0, 1, 0),    # +y  (up)   <- default up
    4: (1, 0, 0),    # +x  (lf)
    5: (-1, 0, 0),   # -x  (rg)
}
VEC2INT: dict[tuple[int, int, int], int] = {v: k for k, v in INT2VEC.items()}

# Named directions matching generator.lua (lf=+x, rg=-x). Calibrated on arrow.xml.
FW, BW, UP, DN, LF, RG = 1, 0, 3, 2, 4, 5


def frame(look: int, up: int):
    """Return orthonormal (look, up, right) unit vectors for a (look, up) code."""
    L = np.array(INT2VEC[look], dtype=float)
    U = np.array(INT2VEC[up], dtype=float)
    R = np.cross(U, L)      # right-handed; calibrated against arrow.xml
    return L, U, R


# ---- edge/corner orientation tables ------------------------------------------
# EDGES, verified by re-rendering data/plans/arrow.xml (only cut(+look,+up)
# reproduces the arrow; the lua-derived −look reading produced sawteeth
# in game): the wedge chamfers the corner between its +look and +up faces,
# order-insensitive. Keys use the two outward faces.
EDGE_LOOKUP = {
    ("+x", "+y"): (LF, UP), ("-x", "+y"): (RG, UP),
    ("+x", "-y"): (LF, DN), ("-x", "-y"): (RG, DN),
    ("+z", "+y"): (FW, UP), ("-z", "+y"): (BW, UP),
    ("+z", "-y"): (FW, DN), ("-z", "-y"): (BW, DN),
    ("+x", "+z"): (LF, FW), ("-x", "+z"): (RG, FW),
    ("+x", "-z"): (LF, BW), ("-x", "-z"): (RG, BW),
}

# CORNERS: empirical, unanimous over ~700 corner blocks in the game's own
# fighter/tutorialship/localship plans (exposed faces = the cut octant).
# cut octant (sorted int codes) -> the game's canonical (look, up)...
_CORNER_CUT_TO_LU = {
    (1, 3, 4): (BW, RG),   # cut +x +y +z
    (0, 3, 4): (BW, UP),   # cut +x +y -z
    (1, 2, 4): (BW, DN),   # cut +x -y +z
    (0, 2, 4): (BW, LF),   # cut +x -y -z
    (1, 3, 5): (FW, UP),   # cut -x +y +z
    (0, 3, 5): (FW, LF),   # cut -x +y -z
    (1, 2, 5): (FW, RG),   # cut -x -y +z
    (0, 2, 5): (FW, DN),   # cut -x -y -z
}
# ... and every (look, up) observed in those plans -> its cut octant, for
# decoding foreign designs (reference turrets, workshop ships).
_CORNER_LU_TO_CUT = {
    (0, 2): (1, 2, 4), (0, 3): (0, 3, 4), (0, 4): (0, 2, 4), (0, 5): (1, 3, 4),
    (1, 2): (0, 2, 5), (1, 3): (1, 3, 5), (1, 4): (0, 3, 5), (1, 5): (1, 2, 5),
    (2, 1): (1, 3, 4), (2, 4): (0, 3, 4), (2, 5): (1, 3, 5),
    (3, 0): (1, 2, 5),
    (4, 0): (1, 3, 5), (4, 1): (1, 2, 4), (4, 2): (1, 2, 5), (4, 3): (1, 3, 4),
    (5, 2): (0, 2, 4), (5, 3): (0, 3, 5),
}

CORNER_LOOKUP = {
    ("+x", "+y", "-z"): (RG, UP),  # topBackLeft
    ("+x", "+y", "+z"): (BW, UP),  # topFrontLeft
    ("-x", "+y", "-z"): (FW, UP),  # topBackRight
    ("-x", "+y", "+z"): (LF, UP),  # topFrontRight
    ("+x", "-y", "+z"): (RG, DN),  # bottomFrontLeft
    ("-x", "-y", "+z"): (BW, DN),  # bottomFrontRight
    ("+x", "-y", "-z"): (FW, DN),  # bottomBackLeft
    ("-x", "-y", "-z"): (LF, DN),  # bottomBackRight
}


def edge_orientation(face_a: str, face_b: str):
    return EDGE_LOOKUP.get((face_a, face_b)) or EDGE_LOOKUP.get((face_b, face_a))


AXIS_OF = {0: 2, 1: 2, 2: 1, 3: 1, 4: 0, 5: 0}  # int code -> axis (0=x,1=y,2=z)


def bevel_edge_orient(d1: int, d2: int) -> tuple[int, int]:
    """(look, up) for a wedge that chamfers a convex edge whose two exposed
    faces point in directions ``d1``/``d2`` (int codes).

    GAME RULE, proven by data/plans/arrow.xml (its head renders as a clean
    arrow only this way): the wedge removes the (+look, +up) corner and the
    encoding is symmetric under swapping look and up."""
    # prefer the vertical (y) exposed face as `up`, like the game's plans
    if AXIS_OF[d1] == 1:
        return d2, d1
    return d1, d2


def bevel_corner_orient(d1: int, d2: int, d3: int):
    """(look, up) for a corner block whose three exposed faces are d1,d2,d3.

    Empirical table from the game's own plans — the mapping is NOT a fixed
    sign pattern in the (look, up, cross) frame, so no formula: just look
    up the cut octant."""
    return _CORNER_CUT_TO_LU.get(tuple(sorted((d1, d2, d3))))


# ---- geometry for the preview (render shapes as real solids) -----------------
# Shape block-type indices (families: Hull 100-103, Armour 104-107, Glow 151-154).
EDGE_TYPES = {100, 104, 151}
CORNER_TYPES = {101, 102, 103, 105, 106, 107, 152, 153, 154}
SHAPE_TYPES = EDGE_TYPES | CORNER_TYPES


# which (look,up)-plane corner the wedge slope cuts away, as (sign_look, sign_up).
# Proven by re-rendering data/plans/arrow.xml: remove the (+look, +up) corner.
_EDGE_CUT = (1, 1)


def edge_faces(lo, hi, look: int, up: int):
    """Triangular-prism faces for an Edge (wedge) block filling box lo..hi.

    The wedge cross-section in the (look, up) plane is a right triangle: the
    square minus the ``_EDGE_CUT`` corner; extruded fully along the right axis.
    """
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    c = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    L, U, R = frame(look, up)
    cl, cu = _EDGE_CUT

    def P(sl, su, sr):
        return c + half * (sl * L + su * U + sr * R)

    # kept triangle = 3 square corners except (cl, cu); slope connects the two
    # neighbours of the cut corner: (cl,-cu) and (-cl,cu).
    keep = P(-cl, -cu, -1), P(cl, -cu, -1), P(-cl, cu, -1)       # right=-1 cap
    keep1 = P(-cl, -cu, +1), P(cl, -cu, +1), P(-cl, cu, +1)      # right=+1 cap
    a0, b0, s0 = keep
    a1, b1, s1 = keep1
    faces = [
        [a0, b0, s0], [a1, b1, s1],   # end caps
        [a0, b0, b1, a1],
        [a0, s0, s1, a1],
        [b0, s0, s1, b1],             # slope
    ]
    return faces


def corner_faces(lo, hi, look: int, up: int):
    """Tetrahedron-ish faces for a Corner block (approx: cut to a single vertex).

    The cut octant comes from the empirical plan-derived table; the solid
    tetrahedron keeps the opposite box corner and its three neighbours."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    c = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    cut = _CORNER_LU_TO_CUT.get((look, up))
    if cut is None:                      # unseen encoding: default octant
        cut = (1, 3, 4)
    sx = -1 if 4 in cut else 1           # keep = opposite of the cut octant
    sy = -1 if 3 in cut else 1
    sz = -1 if 1 in cut else 1

    def P(ax, ay, az):
        s = np.array([ax, ay, az], dtype=float)
        return c + half * s

    o = P(sx, sy, sz)
    a = P(-sx, sy, sz)
    b = P(sx, -sy, sz)
    d = P(sx, sy, -sz)
    faces = [[o, a, b], [o, a, d], [o, b, d], [a, b, d]]
    return faces
