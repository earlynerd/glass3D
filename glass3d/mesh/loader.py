"""Mesh loading utilities using trimesh.

This module handles loading 3D models from various formats (STL, OBJ, PLY, etc.)
and provides utilities for mesh analysis and preparation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import trimesh

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MeshLoader:
    """Load and prepare 3D meshes for point cloud generation."""
    
    SUPPORTED_FORMATS = {".stl", ".obj", ".ply", ".off", ".glb", ".gltf"}
    
    def __init__(self, path: str | Path):
        """Load a mesh from file.
        
        Args:
            path: Path to mesh file (STL, OBJ, PLY, etc.)
        """
        self.path = Path(path)
        
        if not self.path.exists():
            raise FileNotFoundError(f"Mesh file not found: {self.path}")
        
        if self.path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {self.path.suffix}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )
        
        self._mesh = trimesh.load_mesh(str(self.path))
        
        # Handle scenes (multiple meshes) by concatenating
        if isinstance(self._mesh, trimesh.Scene):
            meshes = [
                geom for geom in self._mesh.geometry.values()
                if isinstance(geom, trimesh.Trimesh)
            ]
            if not meshes:
                raise ValueError("No valid meshes found in scene")
            self._mesh = trimesh.util.concatenate(meshes)
    
    @property
    def mesh(self) -> trimesh.Trimesh:
        """Return the loaded trimesh object."""
        return self._mesh
    
    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (min, max) bounding box coordinates."""
        return self._mesh.bounds[0], self._mesh.bounds[1]
    
    @property
    def size(self) -> NDArray[np.float64]:
        """Return size of bounding box (x, y, z)."""
        return self._mesh.bounds[1] - self._mesh.bounds[0]
    
    @property
    def center(self) -> NDArray[np.float64]:
        """Return center of bounding box."""
        return (self._mesh.bounds[0] + self._mesh.bounds[1]) / 2
    
    @property
    def is_watertight(self) -> bool:
        """Check if mesh is watertight (closed)."""
        return self._mesh.is_watertight
    
    @property
    def volume(self) -> float:
        """Return mesh volume (only meaningful for watertight meshes)."""
        return float(self._mesh.volume)
    
    @property
    def num_vertices(self) -> int:
        """Return number of vertices."""
        return len(self._mesh.vertices)
    
    @property
    def num_faces(self) -> int:
        """Return number of faces."""
        return len(self._mesh.faces)
    
    def center_at_origin(self) -> MeshLoader:
        """Center the mesh at the origin (in-place)."""
        self._mesh.vertices -= self.center
        return self
    
    def scale_to_fit(self, max_size: float) -> MeshLoader:
        """Scale mesh to fit within max_size (preserving aspect ratio).
        
        Args:
            max_size: Maximum dimension in any axis (in mm)
        """
        current_max = self.size.max()
        if current_max > 0:
            scale_factor = max_size / current_max
            self._mesh.vertices *= scale_factor
        return self
    
    def scale(self, factor: float) -> MeshLoader:
        """Scale mesh by uniform factor (in-place)."""
        self._mesh.vertices *= factor
        return self
    
    def translate(self, offset: tuple[float, float, float]) -> MeshLoader:
        """Translate mesh by offset (in-place)."""
        self._mesh.vertices += np.array(offset)
        return self
    
    def repair(self) -> MeshLoader:
        """Attempt to repair mesh issues (in-place).
        
        This fixes:
        - Duplicate vertices
        - Degenerate faces
        - Inverted normals (for watertight meshes)
        """
        # Remove duplicate vertices
        self._mesh.merge_vertices()
        
        # Remove degenerate faces
        self._mesh.remove_degenerate_faces()
        
        # Fix normals for watertight meshes
        if self._mesh.is_watertight:
            self._mesh.fix_normals()
        
        return self
    
    def stats(self) -> dict:
        """Return statistics about the mesh."""
        return {
            "path": str(self.path),
            "num_vertices": self.num_vertices,
            "num_faces": self.num_faces,
            "is_watertight": self.is_watertight,
            "bounds_min": self.bounds[0].tolist(),
            "bounds_max": self.bounds[1].tolist(),
            "size": self.size.tolist(),
            "volume": self.volume if self.is_watertight else None,
        }
    
    def __repr__(self) -> str:
        return (
            f"MeshLoader({self.path.name}, "
            f"{self.num_vertices} vertices, "
            f"{self.num_faces} faces, "
            f"size={self.size.round(2)})"
        )


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Convenience function to load a mesh directly.
    
    Args:
        path: Path to mesh file
        
    Returns:
        trimesh.Trimesh object
    """
    return MeshLoader(path).mesh
