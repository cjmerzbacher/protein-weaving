"""Quad weaving: centroid-subdivision of a polygonal mesh → pure quad mesh.

Algorithm (from Section 5.4 of the reference paper):
  1. For every face f, add centroid c_f as a new vertex (colour 1).
  2. Original vertices keep colour 0.
  3. For each shared edge (v_i, v_j) between faces f and g, create quad
     [v_i, c_f, v_j, c_g] (alternating 0-1-0-1 colours).
  4. Strands trace same-colour diagonals; over/under by quad-index parity.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import trimesh

from protein_weaving.weaving.base import WeavingScheme


@dataclass
class QuadMesh:
    """Intermediate quad-mesh representation produced by QuadWeaving."""
    # All vertex positions (original + centroids)
    vertices: np.ndarray          # (M, 3)
    vertex_color: np.ndarray      # (M,)  0 = original, 1 = centroid
    # Each quad: (v0, c_f, v1, c_g) — alternating colours 0-1-0-1
    quads: np.ndarray             # (Q, 4)  integer vertex indices
    # Map from (face_a_idx, face_b_idx) → quad index (ordered pair, a < b)
    edge_quad_map: dict[tuple[int, int], int] = field(default_factory=dict)
    # Original trimesh for context
    source_mesh: trimesh.Trimesh | None = None


class QuadWeaving(WeavingScheme):
    """Centroid-subdivision quad weaving scheme."""

    def build_weave_mesh(self, mesh: trimesh.Trimesh) -> QuadMesh:
        """Convert *mesh* into a QuadMesh with alternating 0/1 vertex colours."""
        verts = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int64)
        n_orig = len(verts)

        # --- Step 1 & 2: add face centroids --------------------------------
        centroids = verts[faces].mean(axis=1)          # (F, 3)
        all_verts = np.vstack([verts, centroids])       # (n_orig + F, 3)

        vertex_color = np.zeros(len(all_verts), dtype=np.int8)
        centroid_start = n_orig
        vertex_color[centroid_start:] = 1              # centroids are colour 1

        # --- Step 3: build quads from shared edges --------------------------
        # face_adjacency: (E, 2) array of face pairs sharing an edge
        # face_adjacency_edges: (E, 2) the two vertex indices of each shared edge
        face_adj = mesh.face_adjacency                  # (E, 2) face index pairs
        face_adj_edges = mesh.face_adjacency_edges      # (E, 2) vertex index pairs

        quads = []
        edge_quad_map: dict[tuple[int, int], int] = {}

        for q_idx, ((fa, fb), (va, vb)) in enumerate(
            zip(face_adj, face_adj_edges)
        ):
            # centroid vertex indices
            ca = centroid_start + fa
            cb = centroid_start + fb
            # Quad: v_a, c_f_a, v_b, c_f_b  → colours 0,1,0,1
            quads.append([va, ca, vb, cb])
            key = (min(fa, fb), max(fa, fb))
            edge_quad_map[key] = q_idx

        quads_arr = np.array(quads, dtype=np.int64) if quads else np.empty(
            (0, 4), dtype=np.int64
        )

        return QuadMesh(
            vertices=all_verts,
            vertex_color=vertex_color,
            quads=quads_arr,
            edge_quad_map=edge_quad_map,
            source_mesh=mesh,
        )
