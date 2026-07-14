"""Block orientation + shape geometry for Avorion shape blocks (edges/corners).

Avorion orients a block with two integer codes ``look`` and ``up`` (each 0-5,
selecting one of six axis directions). For full cubes the codes are always the
default ``look=1, up=3``; for shape blocks (wedges/corners) they aim the slope.

The integer->axis mapping is not published in any game file; it was derived from
the reference ships (default cube = look1/up3 => +z / +y) and validated by
re-rendering data/plans/arrow.xml. Keep it in one place so a single tweak (should
the in-game check disagree) fixes both emission and preview.
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


# ---- edge/corner orientation tables (from generator.lua rounded()/corners) ----
# Each maps a geometric edge/corner of a box to the (look, up) codes that aim a
# wedge/corner block's slope outward there. Keys use axis-sign tuples.
# Edges are named by the two outward faces they sit on.
EDGE_LOOKUP = {
    # (Y edges) top/bottom running along Z, on +x/-x
    ("+x", "+y"): (RG, UP),   # topLeft   -> rgup
    ("-x", "+y"): (LF, UP),   # topRight  -> lfup
    ("+x", "-y"): (RG, DN),   # bottomLeft
    ("-x", "-y"): (LF, DN),
    # (Y edges) top/bottom running along X, on +z/-z
    ("+z", "+y"): (BW, UP),   # topFront  -> bwup
    ("-z", "+y"): (FW, UP),   # topBack   -> fwup
    ("+z", "-y"): (BW, DN),
    ("-z", "-y"): (FW, DN),
    # (Z edges) front/back running along Y, on +x/-x  (derived from arrow.xml)
    ("+x", "+z"): (RG, FW),
    ("-x", "+z"): (FW, LF),   # arrow left tip: look=1(+z=FW), up=5(+x=LF)
    ("+x", "-z"): (BW, RG),
    ("-x", "-z"): (LF, BW),
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
    """(look, up) for a wedge that chamfers a convex edge whose two exposed faces
    point in directions ``d1``/``d2`` (int codes). Removes the (d1,d2) corner."""
    # prefer the vertical (y) exposed face as `up` for a natural-looking slope
    if AXIS_OF[d1] == 1:
        return d2, d1
    return d1, d2


def bevel_corner_orient(d1: int, d2: int, d3: int):
    """(look, up) for a corner block cutting the octant toward faces d1,d2,d3.

    Our corner geometry removes the (+look,+up,+right) octant with
    right = cross(up, look); pick the permutation whose right matches the 3rd face.
    """
    faces = (d1, d2, d3)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            k = 3 - i - j
            look, up, third = faces[i], faces[j], faces[k]
            _, _, R = frame(look, up)
            if tuple(int(x) for x in R) == INT2VEC[third]:
                return look, up
    return None


# ---- geometry for the preview (render shapes as real solids) -----------------
# Shape block-type indices (families: Hull 100-103, Armour 104-107, Glow 151-154).
EDGE_TYPES = {100, 104, 151}
CORNER_TYPES = {101, 102, 103, 105, 106, 107, 152, 153, 154}
SHAPE_TYPES = EDGE_TYPES | CORNER_TYPES


# which (look,up)-plane corner the wedge slope cuts away, as (sign_look, sign_up).
# Calibrated against arrow.xml.
_EDGE_CUT = (1, 1)  # remove the (+look, +up) corner


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
    # keep the corner at (-L,-U,-R); slope face connects (+L,-U,-R),(-L,+U,-R),(-L,-U,+R)
    o = P(-1, -1, -1)
    a = P(+1, -1, -1)
    b = P(-1, +1, -1)
    d = P(-1, -1, +1)
    faces = [[o, a, b], [o, a, d], [o, b, d], [a, b, d]]
    return faces
