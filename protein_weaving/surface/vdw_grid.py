"""Build a 3D scalar field from a list of Atoms via Gaussian-sum vdW surface."""

from __future__ import annotations
import numpy as np
from scipy.spatial import KDTree
from protein_weaving.strands.models import Atom


def build_vdw_grid(
    atoms: list[Atom],
    spacing: float = 0.5,
    padding: float = 3.0,
    gaussian_c: float = 2.0,
    cutoff_factor: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a 3D scalar field and its grid origin.

    Parameters
    ----------
    atoms:
        List of Atom objects with positions and vdW radii.
    spacing:
        Grid voxel size in Angstroms.
    padding:
        Extra space (Å) added on each side of the atom bounding box.
    gaussian_c:
        Width parameter for Gaussian kernel: exp(-c * (r/radius)²).
    cutoff_factor:
        Only accumulate within this many radii of each atom (performance).

    Returns
    -------
    grid : ndarray, shape (nx, ny, nz)
    origin : ndarray, shape (3,)   — world coords of grid voxel (0,0,0)
    """
    positions = np.array([a.pos for a in atoms], dtype=np.float64)
    radii = np.array([a.radius for a in atoms], dtype=np.float64)

    lo = positions.min(axis=0) - padding
    hi = positions.max(axis=0) + padding

    # Grid dimensions
    dims = np.ceil((hi - lo) / spacing).astype(int) + 1
    nx, ny, nz = dims

    # Voxel centres along each axis
    xs = lo[0] + np.arange(nx) * spacing
    ys = lo[1] + np.arange(ny) * spacing
    zs = lo[2] + np.arange(nz) * spacing

    grid = np.zeros((nx, ny, nz), dtype=np.float32)

    tree = KDTree(positions)

    # Accumulate each atom's Gaussian onto nearby voxels
    for i, (pos, radius) in enumerate(zip(positions, radii)):
        cutoff = cutoff_factor * radius
        # Integer index range
        lo_idx = np.floor((pos - cutoff - lo) / spacing).astype(int)
        hi_idx = np.ceil((pos + cutoff - lo) / spacing).astype(int) + 1
        lo_idx = np.clip(lo_idx, 0, dims - 1)
        hi_idx = np.clip(hi_idx, 0, dims)

        ix = np.arange(lo_idx[0], hi_idx[0])
        iy = np.arange(lo_idx[1], hi_idx[1])
        iz = np.arange(lo_idx[2], hi_idx[2])

        # Broadcast distance computation
        dx = (xs[ix] - pos[0])[:, None, None]
        dy = (ys[iy] - pos[1])[None, :, None]
        dz = (zs[iz] - pos[2])[None, None, :]
        r2 = dx**2 + dy**2 + dz**2
        gauss = np.exp(-gaussian_c * r2 / (radius**2)).astype(np.float32)
        grid[
            lo_idx[0]:hi_idx[0],
            lo_idx[1]:hi_idx[1],
            lo_idx[2]:hi_idx[2],
        ] += gauss

    return grid, lo
