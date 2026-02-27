# protein-weaving

Convert 3D protein structures and arbitrary meshes into physically-weavable basketry patterns. Generate flat weaving schematics, 3D visualizations, step-by-step weaving instructions, and crochet amigurumi patterns from PDB files or any triangle mesh.

---

## Packages

This repository provides two CLI tools and a Python API:

- **`protein_weaving`** — weave PDB/CIF protein structures or arbitrary mesh files
- **`kagome_weaving`** — generate Kagome lattice weaves with optional curvature singularities

---

## Installation

```bash
pip install -e .
```

**Optional dependencies** (install as needed):

```bash
pip install -e ".[optional]"      # plotly (interactive HTML), cairosvg (PDF), xatlas (UV unwrapping)
pip install -e ".[dev]"           # pytest for running tests
pip install -e ".[optional,dev]"  # everything
```

---

## Quick Start

```bash
# Weave a protein from a PDB file
protein_weaving structure.pdb --formats svg,html,txt --verbose

# Weave an existing mesh
protein_weaving mesh.obj --scheme triaxial --output-dir ./output

# Generate a flat Kagome weave
kagome_weaving --rows 12 --cols 12 --formats svg,txt

# Kagome dome (5-valent singularity at centre)
kagome_weaving --rows 14 --cols 14 --singularity 0.5,0.5,5
```

---

## `protein_weaving` — Protein & Mesh Weaving

### Usage

```
protein_weaving INPUT [OPTIONS]
```

`INPUT` can be any of: `.pdb`, `.cif`, `.mmcif`, `.obj`, `.ply`, `.stl`

### Options

| Option | Default | Description |
|---|---|---|
| `--scheme` | `quad` | Weaving scheme: `quad` or `triaxial` |
| `--resolution` | `0.5` | Van der Waals grid spacing in Ångströms (PDB/CIF only) |
| `--simplify` | `0.1` | Fraction of faces to keep after simplification |
| `--output-dir` | `.` | Directory for output files |
| `--formats` | `svg,json,txt` | Comma-separated list of output formats (see below) |
| `--strand-width` | `2.0` | SVG strand line width in points |
| `--fill-voids` | off | Fill enclosed surface voids before meshing (PDB/CIF only) |
| `--smooth-sigma` | `1.5` | Gaussian blur sigma in voxels after void-filling |
| `--stitch-width` | — | Physical stitch width in mm for crochet pattern header |
| `--verbose` | off | Print progress messages |

### Output Formats

| Format | Flag | Description |
|---|---|---|
| SVG | `svg` | Flat weaving schematic with colour-coded strands and gap visualization at under-crossings |
| PDF | `pdf` | PDF version of the SVG (requires `cairosvg`) |
| PNG | `png` | Matplotlib 3D rendering with semi-transparent mesh surface |
| HTML | `html` | Interactive Plotly 3D visualization (requires `plotly`) |
| JSON | `json` | Machine-readable weaving instructions |
| TXT | `txt` | Human-readable weaving instructions |
| OBJ | `obj` | Mesh export |
| Crochet | `crochet` | Amigurumi crochet pattern |

### Examples

```bash
# Default: quad weave, SVG + JSON + TXT
protein_weaving 1CRN.cif --verbose

# Triaxial weave with all visualizations
protein_weaving 4PE5.cif --scheme triaxial --formats svg,pdf,png,html,json,txt --verbose

# Mesh input, simplified more aggressively
protein_weaving surface.obj --scheme triaxial --simplify 0.3 --output-dir ./out

# Include crochet pattern with physical gauge
protein_weaving structure.pdb --formats crochet,txt --stitch-width 5.0

# Fill enclosed voids before meshing (useful for hollow proteins)
protein_weaving structure.pdb --fill-voids --smooth-sigma 2.0 --verbose
```

---

## `kagome_weaving` — Kagome Lattice Weaving

### Usage

```
kagome_weaving [OPTIONS]
```

Generates a Kagome triangular lattice and optionally introduces curvature singularities — 5-valent vertices (domes) and 7-valent vertices (saddles) — before weaving.

### Options

| Option | Default | Description |
|---|---|---|
| `--rows` | `12` | Number of lattice rows |
| `--cols` | `12` | Number of lattice columns |
| `--singularity` | — | Add a curvature singularity (repeat for multiple) |
| `--no-embed` | off | Keep the mesh flat; disable 3D bumps |
| `--bump-height` | `3.0` | Peak height of Gaussian 3D bumps |
| `--bump-sigma` | auto | Gaussian sigma for bumps (default: mesh diameter / 4) |
| `--output-dir` | `.` | Output directory |
| `--formats` | `svg,json,txt` | Output formats (same options as `protein_weaving`) |
| `--strand-width` | `2.0` | SVG strand line width in points |
| `--verbose` | off | Print progress messages |

### Singularity Specification

Singularities are specified with `--singularity SPEC`. Two formats are accepted:

```
row_frac,col_frac,type     # position by fractional grid coordinates
v=INDEX,type               # position by exact vertex index
```

**Singularity types:**
- `5` — 5-valent vertex: removes one edge, reduces valence 6→5, creates **positive curvature** (dome)
- `7` — 7-valent vertex: inserts a centroid vertex, increases valence 6→7, creates **negative curvature** (saddle)

### Examples

```bash
# Flat weave, no singularities
kagome_weaving --rows 10 --cols 10 --no-embed

# Dome at the centre
kagome_weaving --rows 14 --cols 14 --singularity 0.5,0.5,5

# Dome at centre, saddle at offset position
kagome_weaving --rows 16 --cols 16 \
    --singularity 0.5,0.5,5 \
    --singularity 0.25,0.75,7

# Multiple singularities with tall bumps
kagome_weaving --rows 20 --cols 20 \
    --singularity 0.5,0.5,5 \
    --singularity 0.2,0.2,7 \
    --singularity 0.8,0.8,7 \
    --bump-height 5.0 \
    --formats svg,html,json --verbose
```

---

## Python API

Import the package directly for programmatic use:

```python
from protein_weaving import weave_pdb, weave_mesh, weave_kagome, WeavingPattern

# Weave from a PDB/CIF file
pattern = weave_pdb(
    pdb_path="structure.pdb",
    scheme="triaxial",
    resolution=0.5,
    simplify_factor=0.1,
    verbose=True,
)

# Weave from an existing trimesh
import trimesh
mesh = trimesh.load("surface.obj")
pattern = weave_mesh(mesh, scheme="quad")

# Kagome lattice
pattern = weave_kagome(
    rows=14,
    cols=14,
    singularities=[(0.5, 0.5, 5)],   # (row_frac, col_frac, type)
    embed_3d=True,
    bump_height=3.0,
)

# Inspect the result
print(f"Strands: {len(pattern.strands)}")
print(f"Crossings: {len(pattern.crossings)}")
print(f"Scheme: {pattern.scheme}")
```

### `WeavingPattern` fields

| Field | Type | Description |
|---|---|---|
| `mesh` | `trimesh.Trimesh` | The source mesh |
| `scheme` | `"quad"` \| `"triaxial"` | Weaving scheme used |
| `strands` | `list[Strand]` | All strands with waypoints and crossing references |
| `crossings` | `list[Crossing]` | All crossings with over/under assignments |
| `uv_coords` | `np.ndarray \| None` | UV coordinates if computed |

---

## Weaving Schemes

### Quad Scheme

Centroid-subdivision algorithm. For each mesh face a centroid vertex is added; strands follow same-coloured diagonals through the resulting quads. Uses 2 strand families with a checkerboard over/under assignment.

Best for: meshes of any topology where a simple two-strand interlace is desired.

### Triaxial Scheme

3-edge colouring algorithm. The mesh is triangulated, then a proper 3-edge colouring is computed on the dual cubic graph (via DSATUR greedy colouring on the line graph). Strands follow each edge-colour family, with a cyclic lock: family *k* passes over family *(k+1) mod 3*.

Best for: triangulated meshes; produces the balanced three-direction structure of traditional triaxial basketry.

---

## Output File Naming

Output files are named after the input file and scheme:

```
weaving_<scheme>_instructions.txt
weaving_<scheme>_instructions.json
weaving_<scheme>.svg
weaving_<scheme>.pdf
weaving_<scheme>_3d.png
weaving_<scheme>_3d.html
weaving_<scheme>.obj
weaving_<scheme>_amigurumi.txt
```

For `kagome_weaving`, the prefix is `kagome_weaving`.

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical arrays |
| `trimesh` | 3D mesh processing |
| `scikit-image` | Marching cubes surface extraction |
| `scipy` | KDTree, Gaussian smoothing, Poisson solvers |
| `matplotlib` | 3D PNG rendering |
| `click` | CLI argument parsing |
| `networkx` | Graph algorithms for 3-edge colouring |
| `plotly` *(optional)* | Interactive HTML visualization |
| `cairosvg` *(optional)* | SVG → PDF conversion |
| `xatlas` *(optional)* | Advanced UV unwrapping |

---

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=protein_weaving
```
