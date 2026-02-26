"""UV unwrap and SVG/PDF flat-pattern rendering with over/under gap cues."""

from __future__ import annotations
import warnings
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import trimesh

from protein_weaving.strands.models import WeavingPattern, Strand, Crossing

# SVG namespace
_SVG_NS = "http://www.w3.org/2000/svg"

# Strand family colours (up to 3 families)
_FAMILY_COLOURS = ["#e63946", "#457b9d", "#2a9d8f"]

# Gap in under-strand at each crossing (in UV units, scaled to canvas)
_GAP_FRAC = 0.015


def _compute_uv(pattern: WeavingPattern, verbose: bool = False) -> np.ndarray:
    """Compute or retrieve UV coordinates for the mesh vertices.

    Tries trimesh's built-in UV unwrap (xatlas if available, else LSCM).
    Returns (N, 2) array of UV coords in [0, 1].
    """
    mesh = pattern.mesh
    if pattern.uv_coords is not None:
        return pattern.uv_coords

    try:
        # trimesh ≥4 exposes angle-based flattening via xatlas
        uv = trimesh.util.concatenate  # just to check trimesh is available
        result = mesh.unwrap()
        if result is not None and hasattr(result, "visual"):
            # result is a new mesh with UV coords
            uv_coords = result.visual.uv
        else:
            raise AttributeError("unwrap returned None")
    except Exception as exc:
        if verbose:
            warnings.warn(f"UV unwrap failed ({exc}); using naive cylindrical projection")
        # Fallback: cylindrical projection
        v = np.array(mesh.vertices)
        cx, cy = v[:, 0].mean(), v[:, 1].mean()
        theta = np.arctan2(v[:, 1] - cy, v[:, 0] - cx)
        z = v[:, 2]
        u = (theta + np.pi) / (2 * np.pi)
        vv = (z - z.min()) / (np.ptp(z) + 1e-9)
        uv_coords = np.stack([u, vv], axis=1)

    # Normalise to [0,1]
    uv_coords = uv_coords - uv_coords.min(axis=0)
    rng = uv_coords.max(axis=0)
    rng[rng == 0] = 1.0
    uv_coords = uv_coords / rng

    pattern.uv_coords = uv_coords
    return uv_coords


def _build_kd_tree(mesh: trimesh.Trimesh):
    from scipy.spatial import cKDTree
    return cKDTree(np.array(mesh.vertices))


def _map_points_to_uv(
    points: np.ndarray,
    uv_coords: np.ndarray,
    kd_tree,
    k: int = 2,
) -> np.ndarray:
    """Batch-map 3D points to UV by querying the KD-tree for k nearest vertices."""
    _, idx = kd_tree.query(points, k=k)
    if k == 1:
        return uv_coords[idx]
    return uv_coords[idx].mean(axis=1)


def _strand_uv_waypoints(
    strand: Strand,
    uv_coords: np.ndarray,
    kd_tree,
) -> list[np.ndarray]:
    """Map 3D waypoints to UV via KD-tree nearest-vertex lookup."""
    if strand.vertex_positions_uv:
        return strand.vertex_positions_uv

    pts = np.array(strand.vertex_positions_3d)
    uv_arr = _map_points_to_uv(pts, uv_coords, kd_tree, k=2)
    strand.vertex_positions_uv = list(uv_arr)
    return strand.vertex_positions_uv


def _crossing_uv(
    crossing: Crossing,
    uv_coords: np.ndarray,
    kd_tree,
) -> np.ndarray:
    """Map a crossing's 3D position to UV."""
    if np.any(crossing.position_uv != 0):
        return crossing.position_uv
    pt = crossing.position_3d.reshape(1, -1)
    crossing.position_uv = _map_points_to_uv(pt, uv_coords, kd_tree, k=1)[0]
    return crossing.position_uv


def _polyline_with_gaps(
    pts: list[np.ndarray],
    crossing_uvs: list[np.ndarray],
    gap: float,
    canvas_size: float,
) -> list[str]:
    """Return a list of SVG path 'd' attributes for one strand.

    Inserts gaps (broken segments) near crossing positions where this strand
    goes under another.
    """
    if len(pts) < 2:
        return []

    pts_arr = np.array(pts) * canvas_size          # (N, 2)
    gap_px = gap * canvas_size

    if not crossing_uvs:
        # No gaps needed — emit one path
        coords = " ".join(
            f"{'M' if j == 0 else 'L'}{pt[0]:.3f},{pt[1]:.3f}"
            for j, pt in enumerate(pts_arr)
        )
        return [coords]

    cross_arr = np.array(crossing_uvs) * canvas_size  # (C, 2)

    # For each segment, find if a crossing projects onto it within the gap radius.
    # Vectorised: work with all crossings at once for each segment.
    p0 = pts_arr[:-1]   # (N-1, 2)
    p1 = pts_arr[1:]    # (N-1, 2)
    seg_vec = p1 - p0                                   # (N-1, 2)
    seg_len = np.linalg.norm(seg_vec, axis=1)           # (N-1,)

    # gap_t[i] = parameter along segment i where a gap starts (or -1 if none)
    gap_t0 = np.full(len(p0), -1.0)
    gap_t1 = np.full(len(p0), -1.0)

    valid = seg_len > 1e-6
    if valid.any():
        d = np.where(valid[:, None], seg_vec / np.where(valid[:, None], seg_len[:, None], 1.0), 0.0)
        # Project each crossing onto each segment: t[i,c] = dot(cross_c - p0_i, d_i)
        diff = cross_arr[None, :, :] - p0[:, None, :]   # (N-1, C, 2)
        t = np.einsum("ics,is->ic", diff, d)             # (N-1, C)
        # Perpendicular distance from crossing to segment line
        proj = p0[:, None, :] + t[:, :, None] * d[:, None, :]
        perp = np.linalg.norm(cross_arr[None, :, :] - proj, axis=2)  # (N-1, C)
        # A crossing hits segment i if t in [0, seg_len] and perp < gap_px
        in_range = (t >= 0) & (t <= seg_len[:, None]) & (perp < gap_px) & valid[:, None]
        # Take first hitting crossing per segment
        hit = in_range.any(axis=1)  # (N-1,)
        for i in np.where(hit)[0]:
            c = np.argmax(in_range[i])
            tc = float(t[i, c])
            gap_t0[i] = max(0.0, tc - gap_px / 2)
            gap_t1[i] = min(float(seg_len[i]), tc + gap_px / 2)

    # Build sub-paths respecting the gaps
    segments: list[list[np.ndarray]] = []
    seg: list[np.ndarray] = [pts_arr[0]]

    for i in range(len(p0)):
        if gap_t0[i] >= 0:
            sl = float(seg_len[i])
            if sl < 1e-6:
                seg.append(p1[i])
                continue
            dv = seg_vec[i] / sl
            if gap_t0[i] > 0:
                seg.append(p0[i] + dv * gap_t0[i])
            segments.append(seg)
            seg = [p0[i] + dv * gap_t1[i]]
        else:
            seg.append(p1[i])

    if seg:
        segments.append(seg)

    paths = []
    for s in segments:
        if len(s) < 2:
            continue
        coords = " ".join(
            f"{'M' if j == 0 else 'L'}{pt[0]:.3f},{pt[1]:.3f}"
            for j, pt in enumerate(s)
        )
        paths.append(coords)
    return paths


def render_svg(
    pattern: WeavingPattern,
    output_path: str | Path,
    strand_width: float = 2.0,
    canvas_size: float = 800.0,
    verbose: bool = False,
) -> Path:
    """Render a flat weaving pattern to an SVG file.

    Parameters
    ----------
    pattern:
        Completed WeavingPattern with strands and crossings.
    output_path:
        Path for the output SVG file.
    strand_width:
        Line width in SVG user units (points).
    canvas_size:
        Width and height of the SVG canvas in pixels.
    """
    output_path = Path(output_path)
    uv = _compute_uv(pattern, verbose=verbose)
    mesh = pattern.mesh

    # Build KD-tree once for all nearest-vertex lookups
    kd_tree = _build_kd_tree(mesh)

    # Precompute UV for all crossings in one pass
    for cr in pattern.crossings:
        _crossing_uv(cr, uv, kd_tree)

    # Build: over-strand set per crossing
    under_crossings: dict[int, list[np.ndarray]] = {}  # strand_id → list of crossing UVs
    for cr in pattern.crossings:
        # The under-strand gets a gap
        under_id = (
            cr.strand_b_id if cr.over_strand_id == cr.strand_a_id
            else cr.strand_a_id
        )
        if under_id not in under_crossings:
            under_crossings[under_id] = []
        under_crossings[under_id].append(cr.position_uv.copy())

    # SVG root
    svg = ET.Element("svg", {
        "xmlns": _SVG_NS,
        "width": str(canvas_size),
        "height": str(canvas_size),
        "viewBox": f"0 0 {canvas_size} {canvas_size}",
    })
    # Background
    ET.SubElement(svg, "rect", {
        "width": str(canvas_size), "height": str(canvas_size),
        "fill": "white",
    })

    for strand in pattern.strands:
        uv_pts = _strand_uv_waypoints(strand, uv, kd_tree)
        if not uv_pts:
            continue

        colour = _FAMILY_COLOURS[strand.family % len(_FAMILY_COLOURS)]
        # Determine under-crossings for this strand
        gap_uvs = under_crossings.get(strand.strand_id, [])
        paths = _polyline_with_gaps(
            uv_pts, gap_uvs, _GAP_FRAC, canvas_size
        )

        for d in paths:
            ET.SubElement(svg, "path", {
                "d": d,
                "stroke": colour,
                "stroke-width": str(strand_width),
                "fill": "none",
                "stroke-linecap": "round",
                "stroke-linejoin": "round",
                "opacity": "0.85",
            })

    tree = ET.ElementTree(svg)
    ET.indent(tree, space="  ")
    tree.write(str(output_path), encoding="unicode", xml_declaration=False)

    if verbose:
        print(f"  SVG written → {output_path}")
    return output_path


def render_pdf(
    pattern: WeavingPattern,
    output_path: str | Path,
    strand_width: float = 2.0,
    canvas_size: float = 800.0,
    verbose: bool = False,
) -> Path | None:
    """Convert the SVG flat pattern to PDF via cairosvg.

    Falls back to SVG-only output with a warning if cairosvg is not installed.
    """
    output_path = Path(output_path)
    svg_path = output_path.with_suffix(".svg")
    render_svg(pattern, svg_path, strand_width=strand_width,
               canvas_size=canvas_size, verbose=verbose)

    try:
        import cairosvg
        cairosvg.svg2pdf(url=str(svg_path), write_to=str(output_path))
        if verbose:
            print(f"  PDF written → {output_path}")
        return output_path
    except ImportError:
        warnings.warn(
            "cairosvg is not installed; PDF output skipped.  "
            "Install with: pip install cairosvg"
        )
        return None
