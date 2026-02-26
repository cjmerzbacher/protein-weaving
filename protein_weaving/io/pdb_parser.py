"""Stdlib-only PDB parser.  Reads ATOM/HETATM records and returns a list of Atom."""

from __future__ import annotations
import re
from pathlib import Path
from protein_weaving.strands.models import Atom
import numpy as np

# Standard vdW radii in Angstroms (Bondi 1964 + common extensions)
_VDW_RADII: dict[str, float] = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
    "P": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "FE": 1.80, "ZN": 1.39, "CA": 1.74, "MG": 1.73, "NA": 2.27,
    "K": 2.75, "MN": 1.73, "NI": 1.63, "CU": 1.40, "CO": 1.63,
    "SE": 1.90,
}
_DEFAULT_RADIUS = 1.70


def _infer_element(atom_name: str) -> str:
    """Infer element symbol from PDB atom name when ELEMENT column is absent."""
    # Strip leading digits and whitespace, keep only letters
    stripped = re.sub(r"[^A-Za-z]", "", atom_name).upper()
    # Remote common prefixes: 1HB → H, CA → C, etc.
    if not stripped:
        return "C"
    # Two-char elements come first (FE, CL, BR, …)
    if len(stripped) >= 2 and stripped[:2] in _VDW_RADII:
        return stripped[:2]
    return stripped[0]


def _vdw_radius(element: str) -> float:
    return _VDW_RADII.get(element.upper(), _DEFAULT_RADIUS)


def parse_pdb(path: str | Path) -> list[Atom]:
    """Parse a PDB file and return a list of Atom objects.

    Reads ATOM and HETATM records.  Uses the standard ELEMENT column (cols 77-78)
    when present; falls back to inferring element from the atom-name column.
    """
    atoms: list[Atom] = []
    path = Path(path)
    with path.open("r", errors="replace") as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            # ELEMENT column: cols 77-78 (0-indexed 76-77)
            element_col = line[76:78].strip() if len(line) > 76 else ""
            if element_col:
                element = element_col.upper()
            else:
                atom_name = line[12:16].strip()
                element = _infer_element(atom_name)

            atoms.append(Atom(
                element=element,
                pos=np.array([x, y, z], dtype=np.float64),
                radius=_vdw_radius(element),
            ))

    if not atoms:
        raise ValueError(f"No ATOM/HETATM records found in {path}")
    return atoms
