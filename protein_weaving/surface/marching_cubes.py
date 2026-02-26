"""Convert a 3D scalar field to a trimesh surface via marching cubes."""

from __future__ import annotations
import numpy as np
import trimesh
from skimage.measure import marching_cubes


def mesh_from_grid(
    grid: np.ndarray,
    origin: np.ndarray,
    spacing: float = 0.5,
    isovalue: float | None = None,
    fill_voids: bool = False,
    smooth_sigma: float = 1.5,
) -> trimesh.Trimesh:
    """Run marching cubes on *grid* and return a trimesh.Trimesh.

    Parameters
    ----------
    grid:
        3D scalar field, shape (nx, ny, nz).
    origin:
        World coordinates of voxel index (0,0,0).
    spacing:
        Voxel size in Angstroms; used to convert voxel coords to world coords.
    isovalue:
        Surface threshold.  Defaults to 0.5 * grid.max().
    fill_voids:
        If True, threshold the field to binary, fill enclosed voids with
        scipy.ndimage.binary_fill_holes, Gaussian-smooth the result, and
        re-run marching cubes.  Produces a smooth balloon-like outer surface.
    smooth_sigma:
        Gaussian blur sigma in voxels applied after void-filling.
    """
    if fill_voids:
        from scipy.ndimage import binary_fill_holes, gaussian_filter
        isovalue = isovalue or float(grid.max()) * 0.5
        binary_mask = grid > isovalue
        # Pad with a False border so binary_fill_holes always has an accessible
        # "outside", even when the protein surface extends to the grid boundary.
        padded = np.pad(binary_mask, pad_width=1, mode='constant',
                        constant_values=False)
        filled_mask = binary_fill_holes(padded)[1:-1, 1:-1, 1:-1]
        grid = gaussian_filter(filled_mask.astype(np.float32), sigma=smooth_sigma)
        # Clamp 0.5 to the actual data range in case Gaussian blur shifts it
        gmin, gmax = float(grid.min()), float(grid.max())
        isovalue = float(np.clip(0.5, gmin + 1e-6, gmax - 1e-6))
    else:
        if isovalue is None:
            isovalue = float(grid.max()) * 0.5

    verts, faces, normals, _ = marching_cubes(
        grid, level=isovalue, spacing=(spacing, spacing, spacing)
    )
    # marching_cubes returns coords in voxel-scaled space starting at 0;
    # shift by origin
    verts = verts + origin.astype(np.float32)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals,
                           process=False)
    return mesh
