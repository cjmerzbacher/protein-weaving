"""3D embedding of the Kagome mesh using singularity-driven Gaussian bumps.

This is a *visual* approximation: each 5-valent singularity (positive Gaussian
curvature) raises a dome around its location; each 7-valent singularity
(negative Gaussian curvature) creates a depression.  The shapes are
superimposed Gaussians centred at the singularity vertices.

A physically exact embedding would require solving a discrete Poisson equation
or running discrete Ricci flow; the Gaussian approach is sufficient for the
visualisation goal of making the curvature effect legible.
"""

from __future__ import annotations

import numpy as np
import trimesh


def embed_kagome_3d(
    mesh: trimesh.Trimesh,
    singularity_specs: list[dict],
    bump_height: float = 3.0,
    bump_sigma: float | None = None,
) -> trimesh.Trimesh:
    """Lift the flat Kagome mesh into 3D by adding Gaussian height bumps.

    Parameters
    ----------
    mesh:
        Flat (z ≈ 0) Kagome mesh after singularity insertion.
    singularity_specs:
        List of dicts, each with keys:

        ``vertex_idx`` (int)
            The singularity vertex index in *mesh*.
        ``type`` (int)
            5 (dome, z > 0) or 7 (depression, z < 0).

    bump_height:
        Peak absolute z-displacement in mesh units.
    bump_sigma:
        Gaussian standard deviation in mesh units.  Defaults to one quarter
        of the mesh diameter (XY only).

    Returns
    -------
    A new trimesh.Trimesh with updated z coordinates.
    """
    verts = np.array(mesh.vertices, dtype=np.float64)

    if not singularity_specs:
        return mesh

    if bump_sigma is None:
        xy_extent = np.linalg.norm(
            verts[:, :2].max(axis=0) - verts[:, :2].min(axis=0)
        )
        bump_sigma = max(xy_extent / 4.0, 1e-6)

    z = np.zeros(len(verts))
    for spec in singularity_specs:
        vi = spec["vertex_idx"]
        sign = 1.0 if spec["type"] == 5 else -1.0
        cx, cy = float(verts[vi, 0]), float(verts[vi, 1])
        r2 = (verts[:, 0] - cx) ** 2 + (verts[:, 1] - cy) ** 2
        z += sign * bump_height * np.exp(-r2 / (2.0 * bump_sigma ** 2))

    new_verts = verts.copy()
    new_verts[:, 2] = z

    return trimesh.Trimesh(
        vertices=new_verts,
        faces=mesh.faces.copy(),
        process=False,
    )
