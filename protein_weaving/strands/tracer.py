"""Strand tracing and over/under assignment for both weaving schemes."""

from __future__ import annotations
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from protein_weaving.strands.models import Strand, Crossing, WeavingPattern

if TYPE_CHECKING:
    from protein_weaving.weaving.quad import QuadMesh
    from protein_weaving.weaving.triaxial import TriaxialMesh


def _edge_midpoint(vertices: np.ndarray, a: int, b: int) -> np.ndarray:
    return 0.5 * (vertices[a] + vertices[b])


# ---------------------------------------------------------------------------
# Quad strand tracing
# ---------------------------------------------------------------------------

def trace_quad_strands(
    quad_mesh: "QuadMesh",
    pattern: WeavingPattern,
) -> WeavingPattern:
    """Trace strands and assign crossings for a QuadMesh.

    Each quad Q = [v0(col=0), c0(col=1), v1(col=0), c1(col=1)] has 4 edges:
      0: v0–c0,  1: c0–v1,  2: v1–c1,  3: c1–v0

    Exit rule: exit_edge = (entry_edge + 2) % 4  (opposite edge in the quad).

    Two strand families:
      Family 0 – strands that start at an even entry edge (0 or 2).
      Family 1 – strands that start at an odd entry edge (1 or 3).

    Over/under: even-entry strand goes over at even quads; odd-entry strand
    goes over at odd quads (checkerboard by quad index parity).

    Each physical strand is traced once (forward direction only); the reverse
    direction is marked visited automatically to prevent duplication.
    """
    verts = quad_mesh.vertices
    quads = quad_mesh.quads
    Q = len(quads)

    # --- Build adjacency: edge-key → list of (quad_idx, local_edge_idx) ---
    edge_to_quads: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for q, (v0, c0, v1, c1) in enumerate(quads):
        for li, (a, b) in enumerate([(v0, c0), (c0, v1), (v1, c1), (c1, v0)]):
            key = (min(a, b), max(a, b))
            edge_to_quads[key].append((q, li))

    def find_neighbor_and_entry(q: int, exit_local: int):
        v0, c0, v1, c1 = quads[q]
        edges = [(v0, c0), (c0, v1), (v1, c1), (c1, v0)]
        a, b = edges[exit_local]
        key = (min(a, b), max(a, b))
        for nq, nli in edge_to_quads[key]:
            if nq != q:
                return nq, nli
        return None, None

    visited: set[tuple[int, int]] = set()   # (quad, entry_edge)

    strands: list[Strand] = []
    crossings: list[Crossing] = []
    strand_id = 0
    crossing_id = 0

    # One crossing per quad; keyed by quad index
    crossing_quad_map: dict[int, int] = {}   # quad_idx → crossing_id
    # Track entry parity per (crossing_id, strand_id) for over/under
    cr_parity: dict[tuple[int, int], int] = {}   # (crossing_id, strand_id) → entry_parity

    # Iterate all 4 starting entry edges; reverse entries are marked visited
    # during tracing so each physical crossing direction is traced exactly once.
    for start_e in range(4):
        for start_q in range(Q):
            if (start_q, start_e) in visited:
                continue

            waypoints_3d: list[np.ndarray] = []
            crossing_ids: list[int] = []
            is_closed = False

            cur_q, cur_e = start_q, start_e

            for _ in range(Q * 4 + 10):
                k = (cur_q, cur_e)

                # Detect loop closure back to our own start
                if k == (start_q, start_e) and waypoints_3d:
                    is_closed = True
                    break

                # Stop if already claimed by a previously completed strand
                if k in visited:
                    break

                # Mark this entry AND its reverse as visited so the same
                # physical crossing is not traced again in the opposite direction
                visited.add(k)
                visited.add((cur_q, (cur_e + 2) % 4))

                exit_e = (cur_e + 2) % 4

                # Waypoints: midpoints of entry and exit edges
                v0, c0, v1, c1 = quads[cur_q]
                edge_verts = [(v0, c0), (c0, v1), (v1, c1), (c1, v0)]
                a_en, b_en = edge_verts[cur_e]
                a_ex, b_ex = edge_verts[exit_e]
                waypoints_3d.append(_edge_midpoint(verts, int(a_en), int(b_en)))
                waypoints_3d.append(_edge_midpoint(verts, int(a_ex), int(b_ex)))

                # Register crossing at this quad (one Crossing object per quad)
                crid = crossing_quad_map.get(cur_q)
                if crid is None:
                    pos3d = verts[list(quads[cur_q])].mean(axis=0)
                    cr = Crossing(
                        crossing_id=crossing_id,
                        face_idx=cur_q,
                        strand_a_id=strand_id,
                        strand_b_id=-1,
                        over_strand_id=-1,
                        position_3d=pos3d,
                        position_uv=np.zeros(2),
                    )
                    crossings.append(cr)
                    crossing_quad_map[cur_q] = crossing_id
                    crid = crossing_id
                    crossing_id += 1
                crossing_ids.append(crid)
                cr_parity[(crid, strand_id)] = cur_e % 2

                nq, ne = find_neighbor_and_entry(cur_q, exit_e)
                if nq is None:
                    break
                cur_q, cur_e = nq, ne

            if waypoints_3d:
                strands.append(Strand(
                    strand_id=strand_id,
                    family=strand_id % 2,   # alternating colour for visualisation
                    vertex_positions_3d=waypoints_3d,
                    vertex_positions_uv=[],
                    crossings=crossing_ids,
                    is_closed=is_closed,
                ))
                strand_id += 1

    # --- Resolve strand_b_id and over/under for every crossing ---
    # Build: crossing_id → list of strand_ids that pass through it
    cr_strands: dict[int, list[int]] = defaultdict(list)
    for s in strands:
        for cid in s.crossings:
            if s.strand_id not in cr_strands[cid]:
                cr_strands[cid].append(s.strand_id)

    for cr in crossings:
        involved = cr_strands.get(cr.crossing_id, [])
        q = cr.face_idx
        if len(involved) >= 2:
            cr.strand_a_id = involved[0]
            cr.strand_b_id = involved[1]
            # Over/under: at quad q, the strand that entered with even parity
            # goes over on even quads; odd-parity strand goes over on odd quads.
            over_parity = q % 2
            for sid in involved:
                if cr_parity.get((cr.crossing_id, sid), 0) == over_parity:
                    cr.over_strand_id = sid
                    break
            else:
                cr.over_strand_id = involved[0]
        elif len(involved) == 1:
            cr.strand_a_id = involved[0]
            cr.strand_b_id = involved[0]
            cr.over_strand_id = involved[0]

    pattern.strands = strands
    pattern.crossings = crossings
    return pattern


# ---------------------------------------------------------------------------
# Triaxial strand tracing
# ---------------------------------------------------------------------------

def trace_triaxial_strands(
    tri_mesh: "TriaxialMesh",
    pattern: WeavingPattern,
) -> WeavingPattern:
    """Trace strands and assign crossings for a TriaxialMesh.

    For family k, a strand enters a face via a non-k edge and exits via the
    other non-k edge (crossing between midpoints of those two edges).
    Over/under: family k goes over (k+1)%3.
    """
    mesh = tri_mesh.mesh
    verts = np.array(mesh.vertices, dtype=np.float64)
    faces = mesh.faces          # (F, 3)
    edge_colors = tri_mesh.edge_colors
    faces_unique_edges = mesh.faces_unique_edges   # (F, 3)
    edges_unique = mesh.edges_unique               # (E, 2)

    # Build face-to-face neighbour lookup via shared edges
    # face_neighbors[f][local_e] = (nb_face, nb_local_e)
    face_neighbors: dict[int, dict[int, tuple[int, int]]] = defaultdict(dict)
    face_adj = mesh.face_adjacency

    for fa, fb in face_adj:
        fa, fb = int(fa), int(fb)
        sa = set(int(e) for e in faces_unique_edges[fa])
        sb = set(int(e) for e in faces_unique_edges[fb])
        shared = sa & sb
        if not shared:
            continue
        ei = next(iter(shared))
        la = list(int(e) for e in faces_unique_edges[fa]).index(ei)
        lb = list(int(e) for e in faces_unique_edges[fb]).index(ei)
        face_neighbors[fa][la] = (fb, lb)
        face_neighbors[fb][lb] = (fa, la)

    strands: list[Strand] = []
    strand_id = 0

    # (face, family) → strand_id  — used to build crossings in post-processing
    face_family_strand: dict[tuple[int, int], int] = {}
    # strand_id → list of face indices it visits  — for building strand.crossings
    strand_face_visits: dict[int, list[int]] = {}

    # visited is keyed by (face_idx, local_entry, family) so different families
    # can independently use the same local edge index of the same face.
    # Both entry AND exit are marked so reverse strands are not created.
    visited: set[tuple[int, int, int]] = set()

    for family in range(3):
        for start_f in range(len(faces)):
            face_ec = [int(edge_colors[faces_unique_edges[start_f][i]])
                       for i in range(3)]
            non_fam = [i for i, c in enumerate(face_ec) if c != family]
            if len(non_fam) < 2:
                continue

            for start_entry in non_fam:
                if (start_f, start_entry, family) in visited:
                    continue

                waypoints_3d: list[np.ndarray] = []
                is_closed = False
                face_visits: list[int] = []

                cur_f, cur_entry = start_f, start_entry

                for _ in range(len(faces) * 2 + 10):
                    if (cur_f, cur_entry) == (start_f, start_entry) and waypoints_3d:
                        is_closed = True
                        break

                    if (cur_f, cur_entry, family) in visited:
                        break

                    # Find exit before marking so we can mark both directions
                    face_ec_cur = [
                        int(edge_colors[faces_unique_edges[cur_f][i]])
                        for i in range(3)
                    ]
                    non_fam_cur = [
                        i for i, c in enumerate(face_ec_cur) if c != family
                    ]
                    if len(non_fam_cur) < 2:
                        break
                    exit_local = (
                        non_fam_cur[1] if cur_entry == non_fam_cur[0]
                        else non_fam_cur[0]
                    )

                    # Mark both entry and exit to prevent reverse strands
                    visited.add((cur_f, cur_entry, family))
                    visited.add((cur_f, exit_local, family))

                    # Record face visit
                    face_visits.append(cur_f)
                    face_family_strand[(cur_f, family)] = strand_id

                    # Waypoints
                    ei_entry = int(faces_unique_edges[cur_f][cur_entry])
                    ei_exit = int(faces_unique_edges[cur_f][exit_local])
                    va_e, vb_e = edges_unique[ei_entry]
                    va_x, vb_x = edges_unique[ei_exit]
                    waypoints_3d.append(
                        _edge_midpoint(verts, int(va_e), int(vb_e)))
                    waypoints_3d.append(
                        _edge_midpoint(verts, int(va_x), int(vb_x)))

                    # Advance
                    if exit_local in face_neighbors[cur_f]:
                        nb_f, nb_entry = face_neighbors[cur_f][exit_local]
                        cur_f, cur_entry = nb_f, nb_entry
                    else:
                        break  # boundary

                if waypoints_3d:
                    strand_face_visits[strand_id] = face_visits
                    strands.append(Strand(
                        strand_id=strand_id,
                        family=family,
                        vertex_positions_3d=waypoints_3d,
                        vertex_positions_uv=[],
                        crossings=[],          # filled below
                        is_closed=is_closed,
                    ))
                    strand_id += 1

    # --- Build crossings in post-processing ---
    # Each crossing is between two strands of different families at the same face.
    # Families (k1, k2) with k1 < k2: over-strand = k1 if k1 == (k2+2)%3 else k2.
    crossings: list[Crossing] = []
    crossing_id = 0
    face_pair_crossing: dict[tuple[int, int, int], int] = {}  # (f, k1, k2) → cid

    for f in range(len(faces)):
        for k1, k2 in ((0, 1), (1, 2), (0, 2)):
            s1 = face_family_strand.get((f, k1))
            s2 = face_family_strand.get((f, k2))
            if s1 is None or s2 is None:
                continue
            pos3d = verts[faces[f]].mean(axis=0)
            over_s = s1 if k1 == (k2 + 2) % 3 else s2
            cr = Crossing(
                crossing_id=crossing_id,
                face_idx=f,
                strand_a_id=s1,
                strand_b_id=s2,
                over_strand_id=over_s,
                position_3d=pos3d,
                position_uv=np.zeros(2),
            )
            crossings.append(cr)
            face_pair_crossing[(f, k1, k2)] = crossing_id
            crossing_id += 1

    # Populate strand.crossings with the relevant crossing_ids
    for s in strands:
        k = s.family
        cr_ids: list[int] = []
        for f in strand_face_visits.get(s.strand_id, []):
            for k2 in range(3):
                if k2 == k:
                    continue
                k1_key, k2_key = (k, k2) if k < k2 else (k2, k)
                cid = face_pair_crossing.get((f, k1_key, k2_key))
                if cid is not None:
                    cr_ids.append(cid)
        s.crossings = cr_ids

    pattern.strands = strands
    pattern.crossings = crossings
    return pattern
