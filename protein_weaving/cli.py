"""Click CLI entry points for protein_weaving."""

from __future__ import annotations
from pathlib import Path

import click


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--scheme", default="quad", show_default=True,
    type=click.Choice(["quad", "triaxial"]),
    help="Weaving scheme.",
)
@click.option(
    "--resolution", default=0.5, show_default=True, type=float,
    help="vdW grid spacing in Å (PDB inputs only).",
)
@click.option(
    "--simplify", default=0.1, show_default=True, type=float,
    help="Fraction of faces to keep after simplification.",
)
@click.option(
    "--output-dir", default=".", show_default=True,
    type=click.Path(file_okay=False),
    help="Directory for output files.",
)
@click.option(
    "--formats", default="svg,json,txt", show_default=True,
    help="Comma-separated list of output formats: svg,pdf,json,txt,png,html,obj,crochet.",
)
@click.option(
    "--strand-width", default=2.0, show_default=True, type=float,
    help="SVG strand line width in points.",
)
@click.option("--fill-voids", is_flag=True, default=False,
              help="Fill enclosed surface voids (PDB inputs only).")
@click.option("--smooth-sigma", default=1.5, show_default=True, type=float,
              help="Gaussian blur sigma in voxels after void-filling.")
@click.option("--stitch-width", default=None, type=float,
              help="Physical stitch width in mm for crochet pattern header (default 6.0).")
@click.option("--verbose", is_flag=True, help="Print progress messages.")
@click.option("--no-color", is_flag=True, default=False,
              help="Skip DSATUR edge colouring; render all strands white on black.")
@click.option("--no-color-alpha", default=0.35, show_default=True, type=float,
              help="Strand opacity used with --no-color (0=transparent, 1=opaque).")
def main(
    input: str,
    scheme: str,
    resolution: float,
    simplify: float,
    output_dir: str,
    formats: str,
    strand_width: float,
    fill_voids: bool,
    smooth_sigma: float,
    stitch_width: float | None,
    verbose: bool,
    no_color: bool,
    no_color_alpha: float,
) -> None:
    """Convert INPUT (.pdb or mesh file) into a weavable basketry pattern.

    INPUT may be a .pdb file (full surface-extraction pipeline) or a mesh
    file (.obj, .ply, .stl, etc.) which skips the surface extraction steps.
    """
    fmt_list = [f.strip().lower() for f in formats.split(",")]
    input_path = Path(input)
    suffix = input_path.suffix.lower()

    if suffix in (".pdb", ".cif", ".mmcif"):
        from protein_weaving.pipeline import weave_pdb
        weave_pdb(
            pdb_path=input_path,
            scheme=scheme,
            resolution=resolution,
            simplify_factor=simplify,
            output_dir=output_dir,
            formats=fmt_list,
            strand_width=strand_width,
            fill_voids=fill_voids,
            smooth_sigma=smooth_sigma,
            crochet_stitch_mm=stitch_width,
            verbose=verbose,
            no_color=no_color,
            no_color_alpha=no_color_alpha,
        )
    else:
        # Assume mesh file
        import trimesh
        from protein_weaving.pipeline import weave_mesh
        from protein_weaving.mesh.clean import clean_mesh, validate_mesh

        if verbose:
            click.echo(f"Loading mesh: {input_path}")
        raw_mesh = trimesh.load(str(input_path), force="mesh")
        if verbose:
            click.echo(f"  {len(raw_mesh.vertices)} verts, {len(raw_mesh.faces)} faces")
            click.echo(f"Cleaning mesh (simplify={simplify}) …")
        mesh = clean_mesh(raw_mesh, simplify_factor=simplify, verbose=verbose)
        validate_mesh(mesh, name=str(input_path))

        weave_mesh(
            mesh=mesh,
            scheme=scheme,
            output_dir=output_dir,
            formats=fmt_list,
            strand_width=strand_width,
            crochet_stitch_mm=stitch_width,
            verbose=verbose,
            no_color=no_color,
            no_color_alpha=no_color_alpha,
        )


# ---------------------------------------------------------------------------
# Kagome singularity weaving CLI
# ---------------------------------------------------------------------------

def _parse_singularity(value: str) -> dict:
    """Parse a singularity spec string.

    Accepted formats:
      ``row_frac,col_frac,type``    e.g. ``0.5,0.5,5``
      ``v=INDEX,type``              e.g. ``v=42,7``
    """
    parts = [p.strip() for p in value.split(",")]
    if len(parts) == 3 and not parts[0].startswith("v="):
        return {
            "row_frac": float(parts[0]),
            "col_frac": float(parts[1]),
            "type": int(parts[2]),
        }
    if len(parts) == 2 and parts[0].startswith("v="):
        return {
            "vertex_idx": int(parts[0][2:]),
            "type": int(parts[1]),
        }
    raise click.BadParameter(
        f"Cannot parse singularity {value!r}. "
        "Use 'row_frac,col_frac,type' (e.g. 0.5,0.5,5) "
        "or 'v=INDEX,type' (e.g. v=42,7)."
    )


@click.command("kagome")
@click.option(
    "--rows", default=12, show_default=True, type=int,
    help="Vertex rows in the flat Kagome lattice.",
)
@click.option(
    "--cols", default=12, show_default=True, type=int,
    help="Vertex columns in the flat Kagome lattice.",
)
@click.option(
    "--singularity", "singularity_strs",
    multiple=True, metavar="SPEC",
    help=(
        "Singularity specification.  Repeat for multiple singularities.  "
        "Format: 'row_frac,col_frac,type'  e.g. '--singularity 0.5,0.5,5' "
        "or 'v=INDEX,type'  e.g. '--singularity v=42,7'."
    ),
)
@click.option(
    "--no-embed", is_flag=True, default=False,
    help="Keep the mesh flat (disable 3D Gaussian bump embedding).",
)
@click.option(
    "--bump-height", default=3.0, show_default=True, type=float,
    help="Peak height of 3D Gaussian bump per singularity (mesh units).",
)
@click.option(
    "--bump-sigma", default=None, type=float,
    help="Gaussian sigma for 3D bumps (default: mesh_diameter / 4).",
)
@click.option(
    "--output-dir", default=".", show_default=True,
    type=click.Path(file_okay=False),
    help="Directory for output files.",
)
@click.option(
    "--formats", default="svg,json,txt", show_default=True,
    help="Comma-separated output formats: svg,pdf,json,txt,png,html,obj.",
)
@click.option(
    "--strand-width", default=2.0, show_default=True, type=float,
    help="SVG strand line width in points.",
)
@click.option("--verbose", is_flag=True, help="Print progress messages.")
@click.option("--no-color", is_flag=True, default=False,
              help="Skip DSATUR edge colouring; render all strands white on black.")
@click.option("--no-color-alpha", default=0.35, show_default=True, type=float,
              help="Strand opacity used with --no-color (0=transparent, 1=opaque).")
def kagome_main(
    rows: int,
    cols: int,
    singularity_strs: tuple[str, ...],
    no_embed: bool,
    bump_height: float,
    bump_sigma: float | None,
    output_dir: str,
    formats: str,
    strand_width: float,
    verbose: bool,
    no_color: bool,
    no_color_alpha: float,
) -> None:
    """Generate a triaxial Kagome weaving pattern with singularities.

    Builds a flat triangular Kagome lattice, introduces the requested
    singularities, optionally embeds in 3D, and writes the weaving pattern.

    \b
    Examples
    --------
    Flat weave (no singularities):
      kagome_weaving --rows 10 --cols 10

    Dome (5-valent singularity at centre):
      kagome_weaving --rows 14 --cols 14 --singularity 0.5,0.5,5

    Dome + saddle:
      kagome_weaving --rows 16 --cols 16 \\
          --singularity 0.5,0.5,5 --singularity 0.25,0.75,7
    """
    from protein_weaving.kagome.pipeline import weave_kagome

    singularities = [_parse_singularity(s) for s in singularity_strs]
    fmt_list = [f.strip().lower() for f in formats.split(",")]

    weave_kagome(
        rows=rows,
        cols=cols,
        singularities=singularities if singularities else None,
        embed_3d=not no_embed,
        bump_height=bump_height,
        bump_sigma=bump_sigma,
        output_dir=output_dir,
        formats=fmt_list,
        strand_width=strand_width,
        verbose=verbose,
        no_color=no_color,
        no_color_alpha=no_color_alpha,
    )
