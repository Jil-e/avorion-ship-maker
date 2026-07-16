"""Block orientation + shape geometry for Avorion shape blocks (edges/corners).

Avorion orients a block with two integer codes ``look`` and ``up`` (each 0-5,
selecting one of six axis directions). For full cubes the codes are always the
default ``look=1, up=3``; for shape blocks (wedges/corners) they aim the slope.

The integer->axis mapping was derived from the reference ships (default cube =
look1/up3 => +z / +y); the wedge/corner slope semantics are taken VERBATIM from
the game's own data/scripts/plangenerator/lib/generator.lua: a wedge with
(look, up) chamfers the corner between its −look and +up faces, and a corner
block cuts the (−look, +up, −right) octant. Emission (bevel_*_orient) and
preview decoding (edge_faces/corner_faces) both live here so they can never
drift apart.
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
# VERIFIED against the game's own data/scripts/plangenerator/lib/generator.lua
# (lines 161-190): a wedge with (look, up) chamfers the edge between its −look
# and +up faces; the generator puts the y face (else the x face) in `up` and
# the other exposed face, NEGATED, in `look`. Keys use the two outward faces.
EDGE_LOOKUP = {
    # (Z edges) top/bottom running along Z, on +x/-x
    ("+x", "+y"): (RG, UP),   # topLeft     -> rgup
    ("-x", "+y"): (LF, UP),   # topRight    -> lfup
    ("+x", "-y"): (RG, DN),   # bottomLeft  -> rgdn
    ("-x", "-y"): (LF, DN),   # bottomRight -> lfdn
    # (X edges) top/bottom running along X, on +z/-z
    ("+z", "+y"): (BW, UP),   # topFront    -> bwup
    ("-z", "+y"): (FW, UP),   # topBack     -> fwup
    ("+z", "-y"): (BW, DN),   # bottomFront -> bwdn
    ("-z", "-y"): (FW, DN),   # bottomBack  -> fwdn
    # (Y edges) vertical, on +x/-x x +z/-z
    ("+x", "+z"): (BW, LF),   # frontLeft   -> bwlf
    ("-x", "+z"): (BW, RG),   # frontRight  -> bwrg
    ("+x", "-z"): (FW, LF),   # backLeft    -> fwlf
    ("-x", "-z"): (FW, RG),   # backRight   -> fwrg
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


_OPP = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}


def bevel_edge_orient(d1: int, d2: int) -> tuple[int, int]:
    """(look, up) for a wedge that chamfers a convex edge whose two exposed
    faces point in directions ``d1``/``d2`` (int codes).

    GAME RULE (plangenerator/lib/generator.lua): the wedge with (look, up)
    chamfers the edge between its −look and +up faces. The generator puts
    the y face (else the x face) in ``up`` and the other face, negated, in
    ``look`` — e.g. the (+x,+y) edge is ``rgup`` = (look=−x, up=+y)."""
    if AXIS_OF[d2] == 1 or (AXIS_OF[d1] != 1 and AXIS_OF[d2] == 0):
        up, other = d2, d1
    else:
        up, other = d1, d2
    return _OPP[other], up


def bevel_corner_orient(d1: int, d2: int, d3: int):
    """(look, up) for a corner block whose three exposed faces are d1,d2,d3.

    GAME RULE (generator.lua corners, all 8 verified): the cut octant is
    (−look, +up, −right) with right = cross(up, look); the y face always
    sits in ``up`` — e.g. topBackLeft (+x,+y,−z) is ``rgup``."""
    faces = {AXIS_OF[d]: d for d in (d1, d2, d3)}
    if len(faces) < 3:
        return None
    up = faces[1]
    for cut_axis, third_axis in ((0, 2), (2, 0)):
        look = _OPP[faces[cut_axis]]
        _, _, R = frame(look, up)
        if tuple(int(v) for v in R) == INT2VEC[_OPP[faces[third_axis]]]:
            return look, up
    return None


# ---- geometry for the preview (render shapes as real solids) -----------------
# Shape block-type indices (families: Hull 100-103, Armour 104-107, Glow 151-154).
EDGE_TYPES = {100, 104, 151}
CORNER_TYPES = {101, 102, 103, 105, 106, 107, 152, 153, 154}
SHAPE_TYPES = EDGE_TYPES | CORNER_TYPES


# which (look,up)-plane corner the wedge slope cuts away, as (sign_look, sign_up).
# GAME RULE from generator.lua: the wedge removes the (−look, +up) corner.
_EDGE_CUT = (-1, 1)


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
    """Tetrahedron-ish faces for a Corner block (approx: cut to a single vertex)."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    c = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    L, U, R = frame(look, up)
    def P(sl, su, sr):
        return c + half * (sl * L + su * U + sr * R)
    # cut octant is (−L, +U, −R) per generator.lua, so the solid tetrahedron
    # keeps the opposite corner (+L, −U, +R) and its three box neighbours
    o = P(+1, -1, +1)
    a = P(-1, -1, +1)
    b = P(+1, +1, +1)
    d = P(+1, -1, -1)
    faces = [[o, a, b], [o, a, d], [o, b, d], [a, b, d]]
    return faces
