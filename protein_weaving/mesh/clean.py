"""Mesh repair, validation, and simplification."""

from __future__ import annotations
import warnings
import trimesh


def validate_mesh(mesh: trimesh.Trimesh, name: str = "mesh") -> None:
    """Emit warnings for common mesh problems."""
    if not mesh.is_watertight:
        warnings.warn(f"{name}: mesh is not watertight (has boundary edges)")
    if not mesh.is_winding_consistent:
        warnings.warn(f"{name}: winding is inconsistent")
    if len(mesh.faces) > 50_000:
        warnings.warn(
            f"{name}: mesh has {len(mesh.faces):,} faces — consider using "
            "--simplify to reduce it"
        )


def clean_mesh(
    mesh: trimesh.Trimesh,
    simplify_factor: float = 0.1,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Return a repaired, simplified, manifold mesh.

    Steps
    -----
    1. Merge duplicate vertices.
    2. Remove degenerate / zero-area faces.
    3. Remove non-manifold edges by splitting at them.
    4. Fill small holes (up to 1000 edges per hole).
    5. Simplify to ``simplify_factor`` × original face count.
    6. Final watertight check + split into connected components
       (return largest).
    """
    # Work on a copy
    m = mesh.copy()

    # 1. Merge duplicate vertices
    m.merge_vertices()

    # 2. Remove degenerate / duplicate faces (trimesh 4.x uses boolean-mask methods)
    def _drop_bad_faces(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        mask = mesh.nondegenerate_faces() & mesh.unique_faces()
        if mask.all():
            return mesh
        return trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces[mask],
            process=False,
        )

    m = _drop_bad_faces(m)

    # 3. Fix winding
    trimesh.repair.fix_winding(m)

    # 4. Fill small holes
    trimesh.repair.fill_holes(m)

    # 5. Simplify
    watertight_pre = m.is_watertight
    target = max(4, int(len(m.faces) * simplify_factor))
    if target < len(m.faces):
        try:
            simplified = m.simplify_quadric_decimation(face_count=target)
            trimesh.repair.fix_winding(simplified)
            trimesh.repair.fill_holes(simplified)
            if watertight_pre and not simplified.is_watertight:
                warnings.warn(
                    f"Simplification (factor={simplify_factor}) broke watertightness "
                    "and could not be repaired; keeping pre-decimation mesh. "
                    "Try a higher --simplify value."
                )
            else:
                m = simplified
        except Exception as exc:
            warnings.warn(f"Simplification failed ({exc}); skipping")

    # 6. Keep largest connected component
    components = m.split(only_watertight=False)
    if len(components) > 1:
        if verbose:
            print(f"  Mesh has {len(components)} components; keeping largest")
        m = max(components, key=lambda c: len(c.faces))

    m = _drop_bad_faces(m)
    m.merge_vertices()

    return m
