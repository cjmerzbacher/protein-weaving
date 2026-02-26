"""Generate human-readable and JSON weaving instructions."""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from protein_weaving.strands.models import WeavingPattern


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


def _strand_summary(s) -> dict:
    return {
        "strand_id": s.strand_id,
        "family": s.family,
        "is_closed": s.is_closed,
        "num_waypoints": len(s.vertex_positions_3d),
        "crossing_ids": s.crossings,
    }


def _crossing_summary(c) -> dict:
    return {
        "crossing_id": c.crossing_id,
        "face_idx": c.face_idx,
        "strand_a": c.strand_a_id,
        "strand_b": c.strand_b_id,
        "over_strand": c.over_strand_id,
        "position_3d": c.position_3d,
    }


def write_json(
    pattern: WeavingPattern,
    output_path: str | Path,
    verbose: bool = False,
) -> Path:
    """Write machine-readable weaving instructions as JSON."""
    output_path = Path(output_path)

    data = {
        "scheme": pattern.scheme,
        "mesh": {
            "num_vertices": len(pattern.mesh.vertices),
            "num_faces": len(pattern.mesh.faces),
            "is_watertight": bool(pattern.mesh.is_watertight),
        },
        "summary": {
            "num_strands": len(pattern.strands),
            "num_crossings": len(pattern.crossings),
            "families": sorted({s.family for s in pattern.strands}),
            "closed_strands": sum(1 for s in pattern.strands if s.is_closed),
            "open_strands": sum(1 for s in pattern.strands if not s.is_closed),
        },
        "strands": [_strand_summary(s) for s in pattern.strands],
        "crossings": [_crossing_summary(c) for c in pattern.crossings],
    }

    output_path.write_text(
        json.dumps(data, indent=2, cls=_NumpyEncoder), encoding="utf-8"
    )
    if verbose:
        print(f"  JSON written → {output_path}")
    return output_path


def write_txt(
    pattern: WeavingPattern,
    output_path: str | Path,
    verbose: bool = False,
) -> Path:
    """Write human-readable weaving instructions as plain text."""
    output_path = Path(output_path)

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("PROTEIN WEAVING INSTRUCTIONS")
    lines.append("=" * 60)
    lines.append(f"Scheme     : {pattern.scheme}")
    lines.append(
        f"Mesh       : {len(pattern.mesh.vertices)} vertices, "
        f"{len(pattern.mesh.faces)} faces"
    )
    lines.append(f"Strands    : {len(pattern.strands)}")
    lines.append(f"Crossings  : {len(pattern.crossings)}")
    lines.append("")

    families = sorted({s.family for s in pattern.strands})
    if pattern.scheme == "quad":
        lines.append("STRAND FAMILIES")
        lines.append("  Family 0 (red)   — diagonal strands, goes OVER on even quads")
        lines.append("  Family 1 (blue)  — cross-diagonal strands, goes OVER on odd quads")
    else:
        lines.append("STRAND FAMILIES (triaxial)")
        lines.append("  Family 0 (red)   — goes over family 1")
        lines.append("  Family 1 (blue)  — goes over family 2")
        lines.append("  Family 2 (teal)  — goes over family 0")
    lines.append("")

    open_strands = [s for s in pattern.strands if not s.is_closed]
    if open_strands:
        lines.append(
            f"NOTE: {len(open_strands)} strand(s) are open (not closed loops)."
        )
        lines.append("  These terminate at mesh boundary edges.")
        lines.append("")

    lines.append("STRAND DETAILS")
    lines.append("-" * 40)
    for s in pattern.strands:
        status = "closed loop" if s.is_closed else "open strand"
        lines.append(
            f"Strand {s.strand_id:4d}  family={s.family}  {status}  "
            f"waypoints={len(s.vertex_positions_3d)}  "
            f"crossings={len(s.crossings)}"
        )

    lines.append("")
    lines.append("CROSSING DETAILS")
    lines.append("-" * 40)
    lines.append(
        "  Format: Crossing ID | Face | Strand A | Strand B | OVER strand"
    )
    for c in pattern.crossings:
        lines.append(
            f"  Crossing {c.crossing_id:4d} | face {c.face_idx:6d} | "
            f"strand {c.strand_a_id:4d} | strand {c.strand_b_id:4d} | "
            f"OVER={c.over_strand_id}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if verbose:
        print(f"  TXT written → {output_path}")
    return output_path
