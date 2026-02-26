"""3D visualisation of strands on the mesh surface."""

from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 – registers 3D projection

from protein_weaving.strands.models import WeavingPattern

_FAMILY_COLOURS = ["#e63946", "#457b9d", "#2a9d8f"]


def _compute_displaced_positions(
    pattern: WeavingPattern, delta: float
) -> dict[int, list[np.ndarray]]:
    """Return per-strand waypoint lists with over/under displacement applied.

    Each strand has 2 waypoints per face visit (entry/exit edge midpoints).
    At each crossing, the over strand's pair is shifted +delta along the face
    normal; the under strand's pair is shifted -delta.
    """
    mesh = pattern.mesh

    # Deep copy all waypoints so the original WeavingPattern is never mutated
    displaced = {
        s.strand_id: [np.array(pt) for pt in s.vertex_positions_3d]
        for s in pattern.strands
    }

    for crossing in pattern.crossings:
        normal = mesh.face_normals[crossing.face_idx]
        face_center = crossing.position_3d

        for strand_id in {crossing.strand_a_id, crossing.strand_b_id}:
            pts = displaced[strand_id]
            N = len(pts) // 2
            if N == 0:
                continue

            # Find the pair whose midpoint is closest to the crossing position
            mids = np.array([0.5 * (pts[2 * k] + pts[2 * k + 1]) for k in range(N)])
            best_k = int(np.argmin(np.linalg.norm(mids - face_center, axis=1)))

            sign = 1.0 if strand_id == crossing.over_strand_id else -1.0
            pts[2 * best_k]     = pts[2 * best_k]     + sign * delta * normal
            pts[2 * best_k + 1] = pts[2 * best_k + 1] + sign * delta * normal

    return displaced


def render_3d_png(
    pattern: WeavingPattern,
    output_path: str | Path,
    dpi: int = 150,
    verbose: bool = False,
    crossing_displacement: float | None = None,
) -> Path:
    """Render a matplotlib 3D view of strands on the mesh."""
    output_path = Path(output_path)

    # Determine displacement delta
    if crossing_displacement is None:
        crossing_displacement = 0.15 * float(pattern.mesh.edges_unique_length.mean())
    delta = crossing_displacement

    displaced = _compute_displaced_positions(pattern, delta) if delta > 0 else None

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw mesh as semi-transparent surface
    verts = np.array(pattern.mesh.vertices)
    faces = np.array(pattern.mesh.faces)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    mesh_polys = verts[faces]
    poly_col = Poly3DCollection(mesh_polys, alpha=0.08, facecolor="#cccccc",
                                edgecolor="none")
    ax.add_collection3d(poly_col)

    # Draw strands
    plotted = 0
    for strand in pattern.strands:
        if len(strand.vertex_positions_3d) < 2:
            continue
        if displaced is not None:
            pts = np.array(displaced[strand.strand_id])
        else:
            pts = np.array(strand.vertex_positions_3d)
        col = _FAMILY_COLOURS[strand.family % len(_FAMILY_COLOURS)]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color=col, linewidth=0.8, alpha=0.7)
        plotted += 1

    # Axis limits from mesh
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.set_title(f"Weaving strands ({pattern.scheme}, {plotted} strands)")
    ax.set_box_aspect([hi[i] - lo[i] for i in range(3)])

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=dpi)
    plt.close(fig)

    if verbose:
        print(f"  3D PNG written → {output_path}")
    return output_path


def render_3d_html(
    pattern: WeavingPattern,
    output_path: str | Path,
    verbose: bool = False,
    crossing_displacement: float | None = None,
) -> Path | None:
    """Render an interactive Plotly HTML view of the strands.

    Returns None and warns if plotly is not installed.
    """
    output_path = Path(output_path)

    # Determine displacement delta
    if crossing_displacement is None:
        crossing_displacement = 0.15 * float(pattern.mesh.edges_unique_length.mean())
    delta = crossing_displacement

    displaced = _compute_displaced_positions(pattern, delta) if delta > 0 else None

    try:
        import plotly.graph_objects as go
    except ImportError:
        warnings.warn(
            "plotly is not installed; HTML output skipped.  "
            "Install with: pip install plotly"
        )
        return None

    fig = go.Figure()

    verts = np.array(pattern.mesh.vertices)
    faces = np.array(pattern.mesh.faces)

    # Mesh surface
    fig.add_trace(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        opacity=0.15,
        color="lightgrey",
        name="surface",
        showlegend=False,
    ))

    # Strand lines
    plotted_families: set[int] = set()
    for strand in pattern.strands:
        if len(strand.vertex_positions_3d) < 2:
            continue
        if displaced is not None:
            pts = np.array(displaced[strand.strand_id])
        else:
            pts = np.array(strand.vertex_positions_3d)
        col = _FAMILY_COLOURS[strand.family % len(_FAMILY_COLOURS)]
        show = strand.family not in plotted_families
        plotted_families.add(strand.family)
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="lines",
            line={"color": col, "width": 2},
            name=f"family {strand.family}",
            showlegend=show,
            legendgroup=f"family {strand.family}",
        ))

    fig.update_layout(
        title=f"Protein weaving — {pattern.scheme} scheme",
        scene={"aspectmode": "data"},
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    fig.write_html(str(output_path))

    if verbose:
        print(f"  HTML written → {output_path}")
    return output_path
