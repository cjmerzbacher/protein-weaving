"""Kagome singularity weaving pipeline.

Entry point: :func:`weave_kagome`.

Pipeline stages
---------------
1. Build a flat triangular Kagome lattice (all interior vertices degree 6).
2. Introduce singularities: degree-5 nodes (positive curvature) or degree-7
   nodes (negative curvature) at user-specified positions.
3. Optionally embed the mesh in 3D using Gaussian height bumps at each
   singularity.
4. Run the triaxial 3-edge-colouring weave algorithm.
5. Trace strands and assign over/under crossings.
6. Write output files (SVG flat pattern, JSON, TXT, and optionally PNG/HTML).

Singularity specification
-------------------------
Each singularity is a dict with a ``"type"`` key (5 or 7) and either:

* ``"row_frac"`` / ``"col_frac"``  — relative position in [0, 1]; the nearest
  interior vertex is selected automatically, **or**
* ``"vertex_idx"`` — an explicit vertex index in the lattice.

Example::

    weave_kagome(
        rows=14, cols=14,
        singularities=[
            {"type": 5, "row_frac": 0.5, "col_frac": 0.5},   # dome at centre
            {"type": 7, "row_frac": 0.25, "col_frac": 0.75},  # saddle off-centre
        ],
        embed_3d=True,
        formats=["svg", "json", "txt", "html"],
        output_dir="out_kagome",
        verbose=True,
    )
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from protein_weaving.strands.models import WeavingPattern


def weave_kagome(
    rows: int = 12,
    cols: int = 12,
    singularities: list[dict] | None = None,
    embed_3d: bool = True,
    bump_height: float = 3.0,
    bump_sigma: float | None = None,
    output_dir: str | Path = ".",
    formats: list[str] | None = None,
    strand_width: float = 2.0,
    verbose: bool = False,
) -> WeavingPattern:
    """Build a Kagome weave pattern from a singularity specification.

    Parameters
    ----------
    rows, cols:
        Vertex dimensions of the flat Kagome lattice.  Use ≥ 5 so there are
        interior vertices to place singularities on.
    singularities:
        List of singularity dicts (see module docstring).  Pass ``None`` or
        an empty list for a flat, singularity-free weave.
    embed_3d:
        Lift the mesh into 3D with Gaussian bumps centred at each singularity.
        Has no effect on the strand topology; only affects 3D/PNG/HTML output.
    bump_height:
        Peak z-displacement of each Gaussian bump (mesh units).
    bump_sigma:
        Gaussian σ (mesh units).  Defaults to mesh_diameter / 4.
    output_dir:
        Directory for output files (created if absent).
    formats:
        Output formats.  Any subset of ``{"svg", "pdf", "json", "txt",
        "png", "html", "obj"}``.  Defaults to ``["svg", "json", "txt"]``.
    strand_width:
        SVG stroke width in points.
    verbose:
        Print progress messages to stdout.

    Returns
    -------
    WeavingPattern  (strands + crossings + embedded mesh)
    """
    import trimesh

    from protein_weaving.kagome.lattice import (
        build_flat_kagome_mesh,
        find_interior_vertex,
        introduce_singularity,
    )
    from protein_weaving.kagome.embed import embed_kagome_3d
    from protein_weaving.weaving.triaxial import TriaxialWeaving
    from protein_weaving.strands.tracer import trace_triaxial_strands
    from protein_weaving.output.svg_pdf import render_svg, render_pdf
    from protein_weaving.output.viz3d import render_3d_png, render_3d_html
    from protein_weaving.output.instructions import write_json, write_txt

    if formats is None:
        formats = ["svg", "json", "txt"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_sing = len(singularities) if singularities else 0
    total_steps = 5

    # ------------------------------------------------------------------
    # 1. Build flat lattice
    # ------------------------------------------------------------------
    step = 1
    if verbose:
        print(f"[{step}/{total_steps}] Building flat Kagome lattice ({rows}×{cols}) …")
    mesh = build_flat_kagome_mesh(rows, cols)
    if verbose:
        print(f"      {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    # ------------------------------------------------------------------
    # 2. Introduce singularities
    # ------------------------------------------------------------------
    step = 2
    if verbose:
        print(
            f"[{step}/{total_steps}] Introducing {n_sing} singularity/ies …"
        )

    singularity_specs_for_embed: list[dict] = []

    if singularities:
        for i, spec in enumerate(singularities):
            stype = int(spec["type"])
            if stype not in (5, 7):
                raise ValueError(
                    f"Singularity {i}: type must be 5 or 7, got {stype!r}."
                )

            if "vertex_idx" in spec:
                vi = int(spec["vertex_idx"])
            else:
                row_frac = float(spec.get("row_frac", 0.5))
                col_frac = float(spec.get("col_frac", 0.5))
                vi = find_interior_vertex(
                    mesh, row_frac=row_frac, col_frac=col_frac
                )

            if verbose:
                xy = mesh.vertices[vi, :2]
                print(
                    f"      [{i}] type-{stype} at vertex {vi} "
                    f"(x={xy[0]:.2f}, y={xy[1]:.2f})"
                )

            mesh = introduce_singularity(mesh, vi, singularity_type=stype)
            # For type-5 the vertex index is unchanged; for type-7 vertex vi
            # is still present (a new vertex w is appended to the end).
            singularity_specs_for_embed.append(
                {"vertex_idx": vi, "type": stype}
            )

    # ------------------------------------------------------------------
    # 3. 3D embedding
    # ------------------------------------------------------------------
    step = 3
    if embed_3d and singularity_specs_for_embed:
        if verbose:
            print(f"[{step}/{total_steps}] Embedding in 3D …")
        mesh = embed_kagome_3d(
            mesh,
            singularity_specs_for_embed,
            bump_height=bump_height,
            bump_sigma=bump_sigma,
        )
    elif verbose:
        print(f"[{step}/{total_steps}] Keeping flat (embed_3d=False or no singularities).")

    # ------------------------------------------------------------------
    # 4. Build triaxial weave (3-edge colouring)
    # ------------------------------------------------------------------
    step = 4
    if verbose:
        print(f"[{step}/{total_steps}] Building triaxial weave …")
    tw = TriaxialWeaving()
    tri_mesh = tw.build_weave_mesh(mesh)
    if verbose:
        print(f"      {len(tri_mesh.edge_colors)} edges coloured")

    # ------------------------------------------------------------------
    # 5. Trace strands
    # ------------------------------------------------------------------
    step = 5
    if verbose:
        print(f"[{step}/{total_steps}] Tracing strands …")
    pattern = WeavingPattern(mesh=mesh, scheme="triaxial")
    pattern = trace_triaxial_strands(tri_mesh, pattern, verbose=verbose)
    if verbose:
        n_closed = sum(1 for s in pattern.strands if s.is_closed)
        print(
            f"      {len(pattern.strands)} strands "
            f"({n_closed} closed, {len(pattern.strands) - n_closed} open), "
            f"{len(pattern.crossings)} crossings"
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    stem = "weaving_kagome"
    if "svg" in formats:
        render_svg(
            pattern, output_dir / f"{stem}.svg",
            strand_width=strand_width, verbose=verbose,
        )
    if "pdf" in formats:
        render_pdf(
            pattern, output_dir / f"{stem}.pdf",
            strand_width=strand_width, verbose=verbose,
        )
    if "png" in formats:
        render_3d_png(pattern, output_dir / f"{stem}_3d.png", verbose=verbose)
    if "html" in formats:
        render_3d_html(pattern, output_dir / f"{stem}_3d.html", verbose=verbose)
    if "json" in formats:
        write_json(
            pattern, output_dir / f"{stem}_instructions.json", verbose=verbose
        )
    if "txt" in formats:
        write_txt(
            pattern, output_dir / f"{stem}_instructions.txt", verbose=verbose
        )
    if "obj" in formats:
        obj_path = output_dir / f"{stem}.obj"
        mesh.export(str(obj_path))
        if verbose:
            print(f"      Mesh exported → {obj_path}")

    if verbose:
        print("Done.")

    return pattern
