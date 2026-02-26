"""Top-level pipeline orchestration for protein_weaving."""

from __future__ import annotations
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh

from protein_weaving.strands.models import WeavingPattern


def weave_pdb(
    pdb_path: str | Path,
    scheme: Literal["quad", "triaxial"] = "quad",
    resolution: float = 0.5,
    simplify_factor: float = 0.1,
    output_dir: str | Path = ".",
    formats: list[str] | None = None,
    strand_width: float = 2.0,
    fill_voids: bool = False,
    smooth_sigma: float = 1.5,
    crochet_stitch_mm: float | None = None,
    verbose: bool = False,
) -> WeavingPattern:
    """Full pipeline: PDB file → weaving pattern + output files.

    Parameters
    ----------
    pdb_path:
        Path to a PDB file.
    scheme:
        "quad" or "triaxial" weaving scheme.
    resolution:
        vdW grid spacing in Angstroms.
    simplify_factor:
        Fraction of faces to keep after simplification.
    output_dir:
        Directory for output files.
    formats:
        List of output formats to generate, e.g. ["svg", "json", "txt", "html",
        "crochet"].  Defaults to ["svg", "json", "txt"].
    strand_width:
        SVG strand line width in points.
    fill_voids:
        Fill enclosed surface voids before running marching cubes.
    smooth_sigma:
        Gaussian blur sigma in voxels after void-filling.
    crochet_stitch_mm:
        Physical stitch width in mm for crochet pattern header.
    verbose:
        Print progress messages.
    """
    from protein_weaving.surface.vdw_grid import build_vdw_grid
    from protein_weaving.surface.marching_cubes import mesh_from_grid
    from protein_weaving.mesh.clean import clean_mesh, validate_mesh

    pdb_path = Path(pdb_path)
    suffix = pdb_path.suffix.lower()

    if verbose:
        print(f"[1/5] Parsing structure: {pdb_path}")

    if suffix in (".cif", ".mmcif"):
        from protein_weaving.io.cif_parser import parse_cif
        atoms = parse_cif(pdb_path)
    else:
        from protein_weaving.io.pdb_parser import parse_pdb
        atoms = parse_pdb(pdb_path)
    if verbose:
        print(f"      {len(atoms)} atoms loaded")

    if verbose:
        print("[2/5] Building vdW scalar field …")
    grid, origin = build_vdw_grid(atoms, spacing=resolution)

    if verbose:
        fill_msg = " (fill_voids=True)" if fill_voids else ""
        print(f"[3/5] Running marching cubes{fill_msg} …")
    raw_mesh = mesh_from_grid(
        grid, origin, spacing=resolution,
        fill_voids=fill_voids, smooth_sigma=smooth_sigma,
    )
    if verbose:
        print(f"      Raw mesh: {len(raw_mesh.vertices)} verts, "
              f"{len(raw_mesh.faces)} faces")

    if verbose:
        print("[4/5] Cleaning mesh …")
    mesh = clean_mesh(raw_mesh, simplify_factor=simplify_factor, verbose=verbose)
    validate_mesh(mesh, name=str(pdb_path))
    if verbose:
        print(f"      Clean mesh: {len(mesh.vertices)} verts, "
              f"{len(mesh.faces)} faces")

    return weave_mesh(
        mesh,
        scheme=scheme,
        output_dir=output_dir,
        formats=formats,
        strand_width=strand_width,
        crochet_stitch_mm=crochet_stitch_mm,
        verbose=verbose,
        _step_offset=4,
        _total_steps=3,
    )


def weave_mesh(
    mesh: trimesh.Trimesh,
    scheme: Literal["quad", "triaxial"] = "quad",
    output_dir: str | Path = ".",
    formats: list[str] | None = None,
    strand_width: float = 2.0,
    crochet_stitch_mm: float | None = None,
    verbose: bool = False,
    _step_offset: int = 0,
    _total_steps: int = 3,
) -> WeavingPattern:
    """Weaving pipeline starting from an existing mesh.

    Parameters
    ----------
    mesh:
        A trimesh.Trimesh surface mesh.
    scheme:
        "quad" or "triaxial".
    output_dir:
        Directory for output files.
    formats:
        List of output formats.  Defaults to ["svg", "json", "txt"].
        Include "crochet" to generate an amigurumi pattern text file.
    strand_width:
        SVG strand line width in points.
    crochet_stitch_mm:
        Physical stitch width in mm for the crochet pattern header.
    verbose:
        Print progress messages.
    """
    from protein_weaving.weaving.quad import QuadWeaving
    from protein_weaving.weaving.triaxial import TriaxialWeaving
    from protein_weaving.strands.tracer import trace_quad_strands, trace_triaxial_strands
    from protein_weaving.output.svg_pdf import render_svg, render_pdf
    from protein_weaving.output.viz3d import render_3d_png, render_3d_html
    from protein_weaving.output.instructions import write_json, write_txt
    from protein_weaving.mesh.clean import validate_mesh

    if formats is None:
        formats = ["svg", "json", "txt"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = _step_offset + _total_steps
    step = _step_offset + 1

    # Crochet-only path: skip weaving entirely
    crochet_only = set(formats) == {"crochet"}

    if not crochet_only:
        if verbose:
            print(f"[{step}/{total}] Building weave mesh (scheme={scheme}) …")

        pattern = WeavingPattern(mesh=mesh, scheme=scheme)

        if scheme == "quad":
            qw = QuadWeaving()
            qm = qw.build_weave_mesh(mesh)
            if verbose:
                print(f"      Quad mesh: {len(qm.quads)} quads")
            step += 1
            if verbose:
                print(f"[{step}/{total}] Tracing quad strands …")
            pattern = trace_quad_strands(qm, pattern)

        elif scheme == "triaxial":
            tw = TriaxialWeaving()
            tm = tw.build_weave_mesh(mesh)
            if verbose:
                print(f"      Triaxial mesh: {len(tm.edge_colors)} edges coloured")
            step += 1
            if verbose:
                print(f"[{step}/{total}] Tracing triaxial strands …")
            pattern = trace_triaxial_strands(tm, pattern)

        else:
            raise ValueError(f"Unknown scheme: {scheme!r}. Use 'quad' or 'triaxial'.")

        if verbose:
            n_closed = sum(1 for s in pattern.strands if s.is_closed)
            print(f"      {len(pattern.strands)} strands "
                  f"({n_closed} closed, "
                  f"{len(pattern.strands) - n_closed} open), "
                  f"{len(pattern.crossings)} crossings")
    else:
        pattern = WeavingPattern(mesh=mesh, scheme=scheme)

    # --- Output ---
    step += 1
    if verbose:
        print(f"[{step}/{total}] Writing output …")

    stem = f"weaving_{scheme}"

    if "svg" in formats:
        render_svg(pattern, output_dir / f"{stem}.svg",
                   strand_width=strand_width, verbose=verbose)
    if "pdf" in formats:
        render_pdf(pattern, output_dir / f"{stem}.pdf",
                   strand_width=strand_width, verbose=verbose)
    if "png" in formats:
        render_3d_png(pattern, output_dir / f"{stem}_3d.png", verbose=verbose)
    if "html" in formats:
        render_3d_html(pattern, output_dir / f"{stem}_3d.html", verbose=verbose)
    if "json" in formats:
        write_json(pattern, output_dir / f"{stem}_instructions.json",
                   verbose=verbose)
    if "txt" in formats:
        write_txt(pattern, output_dir / f"{stem}_instructions.txt",
                  verbose=verbose)
    if "obj" in formats:
        obj_path = output_dir / f"{stem}.obj"
        mesh.export(str(obj_path))
        if verbose:
            print(f"      Mesh exported → {obj_path}")
    if "crochet" in formats:
        from protein_weaving.crochet.amigurumi import (
            generate_amigurumi_pattern,
            write_amigurumi_txt,
        )
        cp = generate_amigurumi_pattern(
            mesh, stitch_width_mm=crochet_stitch_mm, verbose=verbose,
        )
        write_amigurumi_txt(cp, output_dir / f"{stem}_crochet.txt", verbose=verbose)

    if verbose:
        print("Done.")

    return pattern
