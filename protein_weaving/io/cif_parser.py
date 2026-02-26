"""Stdlib-only mmCIF parser for ATOM/HETATM records (_atom_site loop)."""

from __future__ import annotations
from pathlib import Path
from protein_weaving.strands.models import Atom
import numpy as np

# Same vdW radii table as pdb_parser
_VDW_RADII: dict[str, float] = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
    "P": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "FE": 1.80, "ZN": 1.39, "CA": 1.74, "MG": 1.73, "NA": 2.27,
    "K": 2.75, "MN": 1.73, "NI": 1.63, "CU": 1.40, "CO": 1.63,
    "SE": 1.90,
}
_DEFAULT_RADIUS = 1.70


def _vdw_radius(element: str) -> float:
    return _VDW_RADII.get(element.upper(), _DEFAULT_RADIUS)


def parse_cif(path: str | Path) -> list[Atom]:
    """Parse an mmCIF file and return a list of Atom objects.

    Reads the first `_atom_site` loop block and extracts ATOM/HETATM records.
    Uses `_atom_site.type_symbol` for element; falls back to
    `_atom_site.label_atom_id` if absent.
    """
    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()

    # --- Locate the _atom_site loop ---
    # Find the line "loop_" followed by "_atom_site.*" column headers
    col_names: list[str] = []
    data_start = -1
    in_atom_loop = False
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == "loop_":
            # Peek ahead for atom_site columns
            j = i + 1
            peek_cols: list[str] = []
            while j < len(lines) and lines[j].strip().startswith("_atom_site."):
                peek_cols.append(lines[j].strip())
                j += 1
            if peek_cols:
                col_names = [c.split(".")[-1] for c in peek_cols]
                data_start = j
                in_atom_loop = True
                break
        i += 1

    if not in_atom_loop or data_start < 0:
        raise ValueError(f"No _atom_site loop found in {path}")

    # Build column index lookup
    def col(name: str) -> int | None:
        try:
            return col_names.index(name)
        except ValueError:
            return None

    idx_group = col("group_PDB")
    idx_elem = col("type_symbol")
    idx_atom = col("label_atom_id")
    idx_x = col("Cartn_x")
    idx_y = col("Cartn_y")
    idx_z = col("Cartn_z")

    if idx_x is None or idx_y is None or idx_z is None:
        raise ValueError("_atom_site loop is missing Cartn_x/y/z columns")

    atoms: list[Atom] = []
    for line in lines[data_start:]:
        stripped = line.strip()
        # End of this loop block
        if stripped.startswith("_") or stripped == "loop_" or stripped.startswith("#"):
            if stripped.startswith("_") or stripped == "loop_":
                break
            continue
        if not stripped:
            continue

        fields = stripped.split()
        if len(fields) <= max(f for f in [idx_x, idx_y, idx_z] if f is not None):
            continue

        # Filter by record type
        if idx_group is not None and idx_group < len(fields):
            rec = fields[idx_group]
            if rec not in ("ATOM", "HETATM"):
                continue

        try:
            x = float(fields[idx_x])
            y = float(fields[idx_y])
            z = float(fields[idx_z])
        except (ValueError, IndexError):
            continue

        # Element
        element = ""
        if idx_elem is not None and idx_elem < len(fields):
            element = fields[idx_elem].strip("'\"").upper()
        if not element or element in (".", "?"):
            if idx_atom is not None and idx_atom < len(fields):
                raw = fields[idx_atom].strip("'\"")
                element = "".join(c for c in raw if c.isalpha()).upper()
                if len(element) > 1 and element[:2] not in _VDW_RADII:
                    element = element[0]
            if not element:
                element = "C"

        atoms.append(Atom(
            element=element,
            pos=np.array([x, y, z], dtype=np.float64),
            radius=_vdw_radius(element),
        ))

    if not atoms:
        raise ValueError(f"No ATOM/HETATM records parsed from {path}")
    return atoms
