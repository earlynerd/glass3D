"""Mesh slicing for layer-based point cloud generation.

This module provides functionality to slice 3D meshes into 2D cross-sections
at specified Z heights, which can then be used to generate point clouds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import numpy as np
import trimesh
from numpy.typing import NDArray
from trimesh.path import Path2D


@dataclass
class SliceResult:
    """Result of slicing a mesh at a specific Z height.
    
    Attributes:
        z_height: The Z coordinate of this slice
        layer_index: Index of this layer (0-based)
        path_2d: 2D path object from trimesh (may contain multiple polygons)
        transform: 4x4 matrix to transform back to 3D
    """
    
    z_height: float
    layer_index: int
    path_2d: trimesh.path.Path2D | None
    transform: NDArray[np.float64]
    
    @property
    def is_empty(self) -> bool:
        """Check if this slice has no geometry."""
        return self.path_2d is None or len(self.path_2d.entities) == 0
    
    @property
    def polygons(self) -> list:
        """Return list of shapely polygons for this slice."""
        if self.path_2d is None or len(self.path_2d.entities) == 0:
            return []
        return list(self.path_2d.polygons_full)

    @property
    def area(self) -> float:
        """Return total area of all polygons in this slice."""
        if self.is_empty:
            return 0.0
        return float(sum(p.area for p in self.polygons))


class MeshSlicer:
    """Slice a mesh into 2D cross-sections at specified Z heights."""
    
    def __init__(self, mesh: trimesh.Trimesh):
        """Initialize slicer with a mesh.
        
        Args:
            mesh: The trimesh object to slice
        """
        self.mesh = mesh
        self._bounds = mesh.bounds
    
    @property
    def z_min(self) -> float:
        """Minimum Z coordinate of mesh."""
        return float(self._bounds[0, 2])
    
    @property
    def z_max(self) -> float:
        """Maximum Z coordinate of mesh."""
        return float(self._bounds[1, 2])
    
    @property
    def z_height(self) -> float:
        """Total height in Z direction."""
        return self.z_max - self.z_min
    
    def slice_at(self, z: float) -> SliceResult:
        """Slice mesh at a specific Z height.
        
        Args:
            z: Z coordinate to slice at
            
        Returns:
            SliceResult containing the 2D cross-section
        """
        # Get the 3D path at this Z height
        section = self.mesh.section(
            plane_origin=[0, 0, z],
            plane_normal=[0, 0, 1]
        )
        
        if section is None:
            return SliceResult(
                z_height=z,
                layer_index=-1,  # Will be set by caller
                path_2d=None,
                transform=np.eye(4),
            )
        
        # Convert to 2D path
        path_2d, transform = section.to_planar()
        
        return SliceResult(
            z_height=z,
            layer_index=-1,  # Will be set by caller
            path_2d=path_2d,
            transform=transform,
        )
    
    def slice_uniform(
        self,
        layer_height: float,
        start_offset: float = 0.0,
        end_offset: float = 0.0,
    ) -> list[SliceResult]:
        """Slice mesh at uniform Z intervals.
        
        Args:
            layer_height: Distance between slices
            start_offset: Offset from bottom of mesh to start
            end_offset: Offset from top of mesh to end
            
        Returns:
            List of SliceResult objects, one per layer
        """
        z_start = self.z_min + start_offset
        z_end = self.z_max - end_offset
        
        if z_start >= z_end:
            return []
        
        z_levels = np.arange(z_start, z_end + layer_height / 2, layer_height)
        return self.slice_at_heights(z_levels)
    
    def slice_at_heights(self, z_heights: NDArray[np.float64] | list[float]) -> list[SliceResult]:
        """Slice mesh at specific Z heights.
        
        Args:
            z_heights: Array of Z coordinates to slice at
            
        Returns:
            List of SliceResult objects
        """
        z_heights = np.asarray(z_heights)
        results = []
        
        for i, z in enumerate(z_heights):
            result = self.slice_at(z)
            result.layer_index = i
            results.append(result)
        
        return results
    
    def slice_adaptive(
        self,
        min_layer_height: float,
        max_layer_height: float,
        curvature_threshold: float = 0.1,
    ) -> list[SliceResult]:
        """Slice with adaptive layer height based on geometry.
        
        Uses finer layers where there's more detail/curvature.
        
        Args:
            min_layer_height: Minimum layer height
            max_layer_height: Maximum layer height
            curvature_threshold: Threshold for switching layer heights
            
        Returns:
            List of SliceResult objects
        """
        # Start with coarse sampling to analyze shape
        coarse_heights = np.linspace(self.z_min, self.z_max, 50)
        coarse_slices = self.slice_at_heights(coarse_heights)
        
        # Compute area changes between consecutive slices
        areas = np.array([s.area for s in coarse_slices])
        area_changes = np.abs(np.diff(areas))
        max_change = area_changes.max() if len(area_changes) > 0 else 0
        
        # Build adaptive height list
        z_heights = [self.z_min]
        current_z = self.z_min
        
        while current_z < self.z_max:
            # Find the area change rate at this height
            idx = np.searchsorted(coarse_heights, current_z)
            idx = min(idx, len(area_changes) - 1)
            
            if max_change > 0:
                change_ratio = area_changes[idx] / max_change
            else:
                change_ratio = 0
            
            # Interpolate layer height based on change ratio
            if change_ratio > curvature_threshold:
                layer_h = min_layer_height
            else:
                layer_h = max_layer_height
            
            current_z += layer_h
            if current_z <= self.z_max:
                z_heights.append(current_z)
        
        return self.slice_at_heights(z_heights)
    
    def iter_slices(
        self,
        layer_height: float,
        **kwargs: float,
    ) -> Iterator[SliceResult]:
        """Iterate over slices (memory-efficient for large meshes).
        
        Args:
            layer_height: Distance between slices
            **kwargs: Additional arguments for slice_uniform
            
        Yields:
            SliceResult objects one at a time
        """
        z_start = self.z_min + kwargs.get("start_offset", 0.0)
        z_end = self.z_max - kwargs.get("end_offset", 0.0)
        
        z_levels = np.arange(z_start, z_end + layer_height / 2, layer_height)
        
        for i, z in enumerate(z_levels):
            result = self.slice_at(z)
            result.layer_index = i
            yield result
    
    def multiplane_slice(self, layer_height: float) -> list[Path2D | None]:
        """Use trimesh's optimized multiplane slicing.

        This is faster than individual slices for many layers.

        Args:
            layer_height: Distance between slices

        Returns:
            List of Path2D objects (or None for empty slices) at each height.
            Each path's metadata['to_3D'] contains the transform back to 3D.
        """
        z_levels = np.arange(0, self.z_height + layer_height / 2, layer_height)
        
        return self.mesh.section_multiplane(
            plane_origin=self._bounds[0],
            plane_normal=[0, 0, 1],
            heights=z_levels,
        )
    
    def estimate_point_count(
        self,
        layer_height: float,
        point_spacing: float,
    ) -> int:
        """Estimate total number of points for given parameters.
        
        This is useful for progress estimation and memory planning.
        
        Args:
            layer_height: Distance between layers
            point_spacing: Distance between points
            
        Returns:
            Estimated point count
        """
        # Sample a few layers to estimate average area
        sample_heights = np.linspace(self.z_min, self.z_max, min(20, int(self.z_height / layer_height)))
        
        total_area = 0.0
        valid_samples = 0
        
        for z in sample_heights:
            result = self.slice_at(z)
            if not result.is_empty:
                total_area += result.area
                valid_samples += 1
        
        if valid_samples == 0:
            return 0
        
        avg_area = total_area / valid_samples
        num_layers = int(self.z_height / layer_height)
        points_per_layer = avg_area / (point_spacing ** 2)
        
        return int(num_layers * points_per_layer)
