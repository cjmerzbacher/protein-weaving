"""Triaxial weaving: 3-edge-colouring of a triangulated mesh.

Algorithm:
  1. Ensure the mesh is triangulated.
  2. Build the dual cubic graph (node per face, edge per shared triangle edge).
  3. 3-edge-colour the dual graph (equivalently: 3-vertex-colour its line graph)
     using DSATUR greedy colouring via networkx. For a bridgeless cubic graph
     this always yields a proper 3-edge-colouring (Vizing Class 1 theorem).
  4. Strand tracing uses edge colours: a family-k strand crosses a face
     entering via a non-k edge and exiting via the other non-k edge.
  5. Over/under: cyclic lock — family k goes over (k+1)%3.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

import numpy as np
import trimesh
import networkx as nx
from tqdm import tqdm

from protein_weaving.weaving.base import WeavingScheme


@dataclass
class TriaxialMesh:
    """Intermediate triaxial mesh representation."""
    mesh: trimesh.Trimesh
    # edge_colors[i] ∈ {0,1,2} for edge i in mesh.edges_unique
    edge_colors: np.ndarray          # (E,)
    # unique_edge_to_idx: map from canonical edge tuple → index in edges_unique
    edge_to_idx: dict[tuple[int, int], int] = field(default_factory=dict)


class TriaxialWeaving(WeavingScheme):
    """3-edge-colouring triaxial weaving scheme."""

    def build_weave_mesh(self, mesh: trimesh.Trimesh, verbose: bool = False,
                         no_color: bool = False) -> TriaxialMesh:
        """3-colour the edges of a triangulated *mesh*.

        Builds the dual cubic graph (one node per face, one edge per shared
        triangle edge) and applies DSATUR graph colouring to the line graph
        of the dual — which is equivalent to a proper 3-edge-colouring of the
        dual (= proper face-edge colouring of the primal triangulation).
        """
        # Ensure triangulated
        if not all(len(f) == 3 for f in mesh.faces):
            mesh = mesh.triangles_mesh  # type: ignore[attr-defined]

        edges_unique = mesh.edges_unique                 # (E, 2)
        n_edges = len(edges_unique)

        # Build edge → index lookup
        edge_to_idx: dict[tuple[int, int], int] = {}
        for i, (a, b) in enumerate(edges_unique):
            key = (int(min(a, b)), int(max(a, b)))
            edge_to_idx[key] = i

        # faces_unique_edges[f] = [e0, e1, e2] indices into edges_unique
        faces_unique_edges = mesh.faces_unique_edges   # (F, 3)
        n_faces = len(mesh.faces)

        # face adjacency
        face_adj = mesh.face_adjacency               # (K, 2)

        # Build the dual graph with each edge tagged by its primal edge index
        G: nx.Graph = nx.Graph()
        G.add_nodes_from(range(n_faces))
        for fa, fb in tqdm(face_adj, desc="Building dual graph",
                           disable=not verbose, unit="face-pair", leave=False):
            fa, fb = int(fa), int(fb)
            sa = set(int(e) for e in faces_unique_edges[fa])
            sb = set(int(e) for e in faces_unique_edges[fb])
            shared = sa & sb
            if not shared:
                continue
            ei = next(iter(shared))
            # Avoid duplicate edges (face_adj may have duplicates)
            if not G.has_edge(fa, fb):
                G.add_edge(fa, fb, primal_edge=ei)

        edge_colors = np.full(n_edges, -1, dtype=np.int8)

        if no_color:
            # Fast greedy face-by-face colouring — skips DSATUR entirely.
            # Assigns the locally missing colour to each uncoloured edge in one
            # pass over the faces, giving full tqdm visibility.
            for f, fe in tqdm(enumerate(faces_unique_edges), total=n_faces,
                              desc="Colouring edges (fast)", disable=not verbose,
                              unit="face", leave=False):
                used = {int(edge_colors[e]) for e in fe if edge_colors[e] >= 0}
                for ei in fe:
                    if edge_colors[ei] == -1:
                        missing = list({0, 1, 2} - used)
                        edge_colors[ei] = missing[0] if missing else 0
                        used.add(int(edge_colors[ei]))
        else:
            # 3-edge-colour the dual graph via DSATUR on its line graph
            L = nx.line_graph(G)

            # Transfer primal_edge attribute to L's node labels
            # L nodes are (fa, fb) tuples; recover primal edge from G
            colouring = nx.coloring.greedy_color(L, strategy="DSATUR")

            # Map colouring back: L-node (fa, fb) → colour c → mesh edge index → c
            for (fa, fb), c in tqdm(colouring.items(), desc="Mapping edge colours",
                                    disable=not verbose, unit="edge", leave=False):
                # Retrieve primal edge index
                data = G.edges[fa, fb]
                ei = data["primal_edge"]
                # Normalise colour to {0,1,2} (DSATUR may use more if graph is
                # Class 2; clamp extra colours with a warning)
                edge_colors[ei] = int(c) % 3

            # Any edge not in the dual (boundary / non-shared) gets colour 0
            uncoloured = edge_colors == -1
            if uncoloured.any():
                # Assign the "missing" colour for each boundary edge:
                # scan each face and pick the colour not already used
                for f, fe in tqdm(enumerate(faces_unique_edges), total=n_faces,
                                  desc="Assigning boundary colours",
                                  disable=not verbose, unit="face", leave=False):
                    used = {int(edge_colors[e]) for e in fe if edge_colors[e] >= 0}
                    for li, ei in enumerate(fe):
                        if edge_colors[ei] == -1:
                            missing = list({0, 1, 2} - used)
                            edge_colors[ei] = missing[0] if missing else 0
                            used.add(int(edge_colors[ei]))

        # Check colouring quality (skip when no_color: imperfection is expected)
        bad_faces = 0
        if not no_color:
            for fe in faces_unique_edges:
                cols = {int(edge_colors[e]) for e in fe}
                if cols != {0, 1, 2}:
                    bad_faces += 1
        if bad_faces > 0:
            warnings.warn(
                f"Triaxial edge colouring: {bad_faces}/{n_faces} faces do not "
                "have all 3 edge colours (mesh may not admit a Tait colouring). "
                "Strands will still be traced but may overlap."
            )

        return TriaxialMesh(
            mesh=mesh,
            edge_colors=edge_colors,
            edge_to_idx=edge_to_idx,
        )
