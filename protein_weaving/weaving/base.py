"""Abstract base class for weaving schemes."""

from __future__ import annotations
from abc import ABC, abstractmethod
import trimesh


class WeavingScheme(ABC):
    """Abstract base for quad and triaxial weaving schemes."""

    @abstractmethod
    def build_weave_mesh(self, mesh: trimesh.Trimesh) -> object:
        """Convert a trimesh into a scheme-specific weave mesh structure."""
        ...
