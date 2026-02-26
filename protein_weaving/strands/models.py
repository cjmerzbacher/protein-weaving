"""Central data structures for the protein_weaving package."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
import trimesh


@dataclass
class Atom:
    element: str
    pos: np.ndarray       # shape (3,), Angstroms
    radius: float         # vdW radius in Angstroms


@dataclass
class Crossing:
    crossing_id: int
    face_idx: int
    strand_a_id: int
    strand_b_id: int
    over_strand_id: int   # which of strand_a / strand_b goes over
    position_3d: np.ndarray   # shape (3,)
    position_uv: np.ndarray   # shape (2,), may be zeros if UV not computed


@dataclass
class Strand:
    strand_id: int
    family: int                              # 0/1 for quad; 0/1/2 for triaxial
    vertex_positions_3d: list[np.ndarray]    # waypoints in 3D
    vertex_positions_uv: list[np.ndarray]    # waypoints in UV space
    crossings: list[int]                     # crossing_ids in order along strand
    is_closed: bool


@dataclass
class WeavingPattern:
    mesh: trimesh.Trimesh
    scheme: Literal["quad", "triaxial"]
    strands: list[Strand] = field(default_factory=list)
    crossings: list[Crossing] = field(default_factory=list)
    uv_coords: np.ndarray | None = None      # (N, 2) UV for mesh vertices
