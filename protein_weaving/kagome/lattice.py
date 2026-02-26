"""Flat triangular Kagome lattice construction and singularity insertion.

The flat triangular lattice is the canonical substrate for a Kagome weave: every
interior vertex has degree 6 (six strips meeting), which gives zero Gaussian
curvature everywhere.  Singularities are introduced by local mesh surgery:

  5-valent  — remove the edge between two ring neighbours of the target vertex,
               merging two adjacent face slots into one.  Cone angle = 5×60° = 300°.
               Positive Gaussian curvature (dome / convex).

  7-valent  — split one face adjacent to the target vertex by inserting a new
               centroid vertex.  Cone angle = 7×60° = 420°.
               Negative Gaussian curvature (saddle).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# Lattice construction
# ---------------------------------------------------------------------------

def build_flat_kagome_mesh(rows: int, cols: int) -> trimesh.Trimesh:
    """Build a flat triangular lattice.

    Vertices are placed on a regular triangular grid using axial coordinates:
    vertex at grid position (col c, row r) sits at 3D position
    (c + r/2,  r·√3/2,  0).

    Every *interior* vertex has exactly six neighbours (degree 6 = flat Kagome).
    Boundary vertices have degree 3–5 depending on their position.

    Parameters
    ----------
    rows, cols:
        Vertex grid dimensions.  Use ≥ 5 in each direction to get a
        meaningful interior region.

    Returns
    -------
    trimesh.Trimesh  (process=False, z=0 for all vertices)
    """
    if rows < 3 or cols < 3:
        raise ValueError(f"Need rows≥3, cols≥3; got {rows}×{cols}.")

    sqrt3_2 = np.sqrt(3.0) / 2.0
    verts = np.array(
        [[c + r * 0.5, r * sqrt3_2, 0.0] for r in range(rows) for c in range(cols)],
        dtype=np.float64,
    )

    def idx(c: int, r: int) -> int:
        return r * cols + c

    faces: list[list[int]] = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            # Up-pointing triangle
            faces.append([idx(c, r), idx(c + 1, r), idx(c, r + 1)])
            # Down-pointing triangle
            faces.append([idx(c + 1, r), idx(c + 1, r + 1), idx(c, r + 1)])

    return trimesh.Trimesh(
        vertices=verts,
        faces=np.array(faces, dtype=np.int64),
        process=False,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_boundary_vertices(mesh: trimesh.Trimesh) -> set[int]:
    """Return vertex indices that lie on a mesh boundary edge."""
    edge_face_count: dict[tuple[int, int], int] = defaultdict(int)
    for f in mesh.faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            edge_face_count[(min(a, b), max(a, b))] += 1
    boundary: set[int] = set()
    for (a, b), count in edge_face_count.items():
        if count == 1:
            boundary.add(a)
            boundary.add(b)
    return boundary


def find_interior_vertex(
    mesh: trimesh.Trimesh,
    row_frac: float = 0.5,
    col_frac: float = 0.5,
) -> int:
    """Return the interior vertex closest to a relative (col, row) position.

    Parameters
    ----------
    row_frac, col_frac:
        Fractional position in [0, 1].  (0.5, 0.5) targets the mesh centre.

    Raises
    ------
    ValueError  if no interior vertex exists (increase grid size).
    """
    verts = np.asarray(mesh.vertices)
    boundary = _find_boundary_vertices(mesh)

    xmin, xmax = float(verts[:, 0].min()), float(verts[:, 0].max())
    ymin, ymax = float(verts[:, 1].min()), float(verts[:, 1].max())
    tx = xmin + col_frac * (xmax - xmin)
    ty = ymin + row_frac * (ymax - ymin)

    best_idx = -1
    best_d = np.inf
    for i in range(len(verts)):
        if i in boundary:
            continue
        d = (verts[i, 0] - tx) ** 2 + (verts[i, 1] - ty) ** 2
        if d < best_d:
            best_d = d
            best_idx = i

    if best_idx == -1:
        raise ValueError(
            "No interior vertex found — increase rows/cols or move singularity "
            "away from the mesh boundary."
        )
    return best_idx


def _get_cyclic_ring(mesh: trimesh.Trimesh, v: int) -> list[int]:
    """Return the cyclic ring of neighbours of vertex *v* in CCW order.

    Uses face winding: for face containing (v, a, b) in CCW order the edge
    a→b encodes the directed arc of the ring, so next_in_ring[a] = b.
    This yields a consistent CCW ordering provided the mesh has uniform
    winding (true for meshes produced by :func:`build_flat_kagome_mesh`).
    """
    faces = mesh.faces
    face_mask = np.any(faces == v, axis=1)
    ring_face_indices = np.where(face_mask)[0]

    next_in_ring: dict[int, int] = {}
    for fi in ring_face_indices:
        f = faces[fi].tolist()
        v_pos = f.index(v)
        a = f[(v_pos + 1) % 3]
        b = f[(v_pos + 2) % 3]
        next_in_ring[a] = b

    if not next_in_ring:
        return []

    start = next(iter(next_in_ring))
    ring = [start]
    current = next_in_ring.get(start)
    for _ in range(len(next_in_ring)):
        if current is None or current == start:
            break
        ring.append(current)
        current = next_in_ring.get(current)

    return ring


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def introduce_singularity(
    mesh: trimesh.Trimesh,
    vertex_idx: int,
    singularity_type: int,
) -> trimesh.Trimesh:
    """Introduce a Kagome weave singularity at *vertex_idx*.

    Parameters
    ----------
    mesh:
        Triangulated mesh.  *vertex_idx* should be an interior vertex with
        degree ≥ 6 (produced by :func:`build_flat_kagome_mesh`).
    vertex_idx:
        Target vertex.
    singularity_type:
        **5** — remove one strip (positive Gaussian curvature, dome).
        **7** — add one strip (negative Gaussian curvature, saddle).

    Returns
    -------
    A *new* trimesh.Trimesh with the singularity applied.
    """
    if singularity_type == 5:
        return _singularity_5(mesh, vertex_idx)
    elif singularity_type == 7:
        return _singularity_7(mesh, vertex_idx)
    else:
        raise ValueError(
            f"singularity_type must be 5 or 7, got {singularity_type!r}."
        )


# ---------------------------------------------------------------------------
# Internal mesh surgery
# ---------------------------------------------------------------------------

def _singularity_5(mesh: trimesh.Trimesh, v: int) -> trimesh.Trimesh:
    """Reduce degree of vertex v from 6 → 5.

    Removes the two faces that share the edge (v, a5) — where a5 is the last
    neighbour in the CCW ring — and replaces them with a single face (v, a4, a0)
    that skips a5.  The edge v–a5 is thereby eliminated, dropping the degree by 1.
    The mesh remains closed (no boundary hole is introduced).
    """
    ring = _get_cyclic_ring(mesh, v)
    if len(ring) < 6:
        raise ValueError(
            f"Vertex {v} has only {len(ring)} ring neighbours (expected ≥ 6). "
            "Use an interior vertex of a lattice with rows ≥ 5, cols ≥ 5."
        )

    a0, a4, a5 = ring[0], ring[-2], ring[-1]

    # The two faces to remove share the edge v–a5
    remove = {frozenset([v, a4, a5]), frozenset([v, a5, a0])}

    new_faces: list[list[int]] = []
    gap_filled = False
    for f in mesh.faces.tolist():
        if frozenset(f) in remove:
            if not gap_filled:
                # Close the gap: (v, a4, a0) in CCW winding order
                new_faces.append([v, a4, a0])
                gap_filled = True
            # Second matching face is simply dropped
        else:
            new_faces.append(f)

    return trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=np.array(new_faces, dtype=np.int64),
        process=False,
    )


def _singularity_7(mesh: trimesh.Trimesh, v: int) -> trimesh.Trimesh:
    """Increase degree of vertex v from 6 → 7.

    Splits one face (v, a0, a1) by inserting a new vertex w at its centroid,
    producing three child faces (v, a0, w), (a0, a1, w), (v, w, a1).
    Vertex v gains the new neighbour w, increasing its degree to 7.
    """
    ring = _get_cyclic_ring(mesh, v)
    if len(ring) < 6:
        raise ValueError(
            f"Vertex {v} has only {len(ring)} ring neighbours (expected ≥ 6)."
        )

    a0, a1 = ring[0], ring[1]

    # New vertex w at centroid of face (v, a0, a1)
    verts = mesh.vertices.tolist()
    w_pos = (
        np.array(verts[v]) + np.array(verts[a0]) + np.array(verts[a1])
    ) / 3.0
    w = len(verts)
    verts.append(w_pos.tolist())

    target = frozenset([v, a0, a1])
    new_faces: list[list[int]] = []
    for f in mesh.faces.tolist():
        if frozenset(f) == target:
            # Three child triangles — all CCW (w is inside CCW parent)
            new_faces.append([v, a0, w])
            new_faces.append([a0, a1, w])
            new_faces.append([v, w, a1])
        else:
            new_faces.append(f)

    return trimesh.Trimesh(
        vertices=np.array(verts, dtype=np.float64),
        faces=np.array(new_faces, dtype=np.int64),
        process=False,
    )
