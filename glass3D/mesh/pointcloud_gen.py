"""Point cloud generation strategies.

This module provides different strategies for converting meshes into point clouds.
Each strategy produces different visual effects in the final engraving.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

if TYPE_CHECKING:
    import trimesh
    from ..core.config import PointCloudParams

from ..core.point_cloud import PointCloud
from .slicer import MeshSlicer


class BaseStrategy(ABC):
    """Abstract base class for point generation strategies."""
    
    @abstractmethod
    def generate(
        self,
        mesh: trimesh.Trimesh,
        params: PointCloudParams,
    ) -> PointCloud:
        """Generate point cloud from mesh.
        
        Args:
            mesh: The input mesh
            params: Point cloud generation parameters
            
        Returns:
            Generated PointCloud
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for display/logging."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of this strategy."""
        pass


class SurfaceStrategy(BaseStrategy):
    """Generate points only on the surface of the mesh.
    
    This creates a "shell" effect where only the outer surface is visible.
    Good for objects where you want to see the shape outline clearly.
    """
    
    @property
    def name(self) -> str:
        return "surface"
    
    @property
    def description(self) -> str:
        return "Points on mesh surface only (shell effect)"
    
    def generate(
        self,
        mesh: trimesh.Trimesh,
        params: PointCloudParams,
    ) -> PointCloud:
        """Generate surface points using mesh sampling."""
        # Calculate number of points based on surface area and spacing
        surface_area = mesh.area
        points_per_area = 1.0 / (params.point_spacing_mm ** 2)
        num_points = int(surface_area * points_per_area)
        
        # Sample points on the surface
        points, face_indices = mesh.sample(num_points, return_index=True)
        
        # Optionally offset points inward
        if params.surface_offset_mm != 0:
            # Get face normals for the sampled points
            normals = mesh.face_normals[face_indices]
            # Offset along normal (negative = inward)
            points = points + normals * params.surface_offset_mm
        
        # Assign layer indices based on Z coordinate
        z_min = points[:, 2].min()
        layer_indices = ((points[:, 2] - z_min) / params.layer_height_mm).astype(np.int32)
        
        return PointCloud(
            points=points,
            layer_indices=layer_indices,
        )


class SolidStrategy(BaseStrategy):
    """Generate points throughout the volume of the mesh.
    
    This creates a solid-fill effect. Point density is uniform throughout.
    Good for objects that should appear as solid blocks of points.
    """
    
    @property
    def name(self) -> str:
        return "solid"
    
    @property
    def description(self) -> str:
        return "Points fill entire volume (solid effect)"
    
    def generate(
        self,
        mesh: trimesh.Trimesh,
        params: PointCloudParams,
    ) -> PointCloud:
        """Generate solid fill using slice-based approach."""
        slicer = MeshSlicer(mesh)
        all_points = []
        all_layers = []
        
        layer_height = params.layer_height_mm
        point_spacing = params.point_spacing_mm * (1.0 / params.solid_density)
        
        # Slice the mesh at each layer height
        for slice_result in slicer.iter_slices(layer_height):
            if slice_result.is_empty:
                continue
            
            layer_points = self._fill_slice(
                slice_result.polygons,
                slice_result.z_height,
                point_spacing,
            )
            
            if len(layer_points) > 0:
                all_points.append(layer_points)
                all_layers.append(
                    np.full(len(layer_points), slice_result.layer_index, dtype=np.int32)
                )
        
        if not all_points:
            return PointCloud(points=np.empty((0, 3)))
        
        points = np.vstack(all_points)
        layer_indices = np.concatenate(all_layers)
        
        return PointCloud(points=points, layer_indices=layer_indices)
    
    def _fill_slice(
        self,
        polygons: list,
        z_height: float,
        spacing: float,
    ) -> NDArray[np.float64]:
        """Fill a 2D slice with points.
        
        Args:
            polygons: List of shapely polygons
            z_height: Z coordinate for all points
            spacing: Distance between points
            
        Returns:
            Nx3 array of points
        """
        if not polygons:
            return np.empty((0, 3))
        
        points = []
        
        for polygon in polygons:
            if polygon.is_empty:
                continue
            
            # Get bounding box
            minx, miny, maxx, maxy = polygon.bounds
            
            # Create grid of candidate points
            x_coords = np.arange(minx, maxx + spacing, spacing)
            y_coords = np.arange(miny, maxy + spacing, spacing)
            xx, yy = np.meshgrid(x_coords, y_coords)
            candidates = np.column_stack([xx.ravel(), yy.ravel()])
            
            # Filter to points inside polygon
            from shapely.geometry import Point
            from shapely.vectorized import contains
            
            # Use vectorized contains for speed
            try:
                inside = contains(polygon, candidates[:, 0], candidates[:, 1])
            except Exception:
                # Fallback to slower method
                inside = np.array([polygon.contains(Point(x, y)) for x, y in candidates])
            
            interior_points = candidates[inside]
            
            # Add Z coordinate
            if len(interior_points) > 0:
                z_col = np.full((len(interior_points), 1), z_height)
                points_3d = np.hstack([interior_points, z_col])
                points.append(points_3d)
        
        if not points:
            return np.empty((0, 3))
        
        return np.vstack(points)


class GrayscaleStrategy(BaseStrategy):
    """Generate points with variable density for grayscale effects.
    
    Point density varies based on distance from surface or other criteria,
    creating the illusion of shading or depth.
    """
    
    @property
    def name(self) -> str:
        return "grayscale"
    
    @property
    def description(self) -> str:
        return "Variable density for shading effects"
    
    def generate(
        self,
        mesh: trimesh.Trimesh,
        params: PointCloudParams,
    ) -> PointCloud:
        """Generate points with density based on distance from surface."""
        # First generate a dense solid fill
        solid_strategy = SolidStrategy()
        
        # Use higher density for initial generation
        dense_params = params.model_copy()
        dense_params.point_spacing_mm = params.point_spacing_mm * 0.5
        dense_params.solid_density = 2.0
        
        dense_cloud = solid_strategy.generate(mesh, dense_params)
        
        if len(dense_cloud) == 0:
            return dense_cloud
        
        # Calculate distance from surface for each point
        # Use trimesh's proximity query
        closest_points, distances, _ = mesh.nearest.on_surface(dense_cloud.points)
        
        # Normalize distances
        max_dist = distances.max()
        if max_dist > 0:
            normalized_dist = distances / max_dist
        else:
            normalized_dist = np.zeros_like(distances)
        
        # Points near surface have higher probability of being kept
        # Points far from surface (interior) have lower probability
        keep_probability = 1.0 - (normalized_dist * 0.8)  # Keep 20% minimum
        
        # Random sampling based on probability
        rng = np.random.default_rng(42)
        keep_mask = rng.random(len(dense_cloud)) < keep_probability
        
        filtered = dense_cloud[keep_mask]
        
        # Set intensities based on distance (for potential future use)
        filtered.intensities = keep_probability[keep_mask]
        
        return filtered


class ContourStrategy(BaseStrategy):
    """Generate points along contour lines only.
    
    Creates a "topographic map" effect with distinct layer outlines.
    """
    
    @property
    def name(self) -> str:
        return "contour"
    
    @property
    def description(self) -> str:
        return "Points along layer contours only"
    
    def generate(
        self,
        mesh: trimesh.Trimesh,
        params: PointCloudParams,
    ) -> PointCloud:
        """Generate points along slice contours."""
        slicer = MeshSlicer(mesh)
        all_points = []
        all_layers = []
        
        for slice_result in slicer.iter_slices(params.layer_height_mm):
            if slice_result.is_empty:
                continue
            
            layer_points = self._sample_contours(
                slice_result.polygons,
                slice_result.z_height,
                params.point_spacing_mm,
            )
            
            if len(layer_points) > 0:
                all_points.append(layer_points)
                all_layers.append(
                    np.full(len(layer_points), slice_result.layer_index, dtype=np.int32)
                )
        
        if not all_points:
            return PointCloud(points=np.empty((0, 3)))
        
        points = np.vstack(all_points)
        layer_indices = np.concatenate(all_layers)
        
        return PointCloud(points=points, layer_indices=layer_indices)
    
    def _sample_contours(
        self,
        polygons: list,
        z_height: float,
        spacing: float,
    ) -> NDArray[np.float64]:
        """Sample points along polygon contours.
        
        Args:
            polygons: List of shapely polygons
            z_height: Z coordinate for all points
            spacing: Distance between points along contour
            
        Returns:
            Nx3 array of points
        """
        from shapely.geometry import LineString
        
        points = []
        
        for polygon in polygons:
            if polygon.is_empty:
                continue
            
            # Process exterior ring
            exterior = polygon.exterior
            points.extend(self._sample_ring(exterior, z_height, spacing))
            
            # Process interior rings (holes)
            for interior in polygon.interiors:
                points.extend(self._sample_ring(interior, z_height, spacing))
        
        if not points:
            return np.empty((0, 3))
        
        return np.array(points)
    
    def _sample_ring(
        self,
        ring,
        z_height: float,
        spacing: float,
    ) -> list[tuple[float, float, float]]:
        """Sample points along a ring at regular intervals."""
        length = ring.length
        if length == 0:
            return []
        
        num_points = max(1, int(length / spacing))
        
        points = []
        for i in range(num_points):
            distance = (i / num_points) * length
            point = ring.interpolate(distance)
            points.append((point.x, point.y, z_height))
        
        return points


# Strategy registry
STRATEGIES = {
    "surface": SurfaceStrategy,
    "solid": SolidStrategy,
    "grayscale": GrayscaleStrategy,
    "contour": ContourStrategy,
}


def get_strategy(name: str) -> BaseStrategy:
    """Get a strategy instance by name.
    
    Args:
        name: Strategy name
        
    Returns:
        Strategy instance
        
    Raises:
        ValueError: If strategy name is unknown
    """
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    
    return STRATEGIES[name]()


def list_strategies() -> list[dict]:
    """List all available strategies with descriptions.
    
    Returns:
        List of dicts with 'name' and 'description' keys
    """
    return [
        {"name": cls().name, "description": cls().description}
        for cls in STRATEGIES.values()
    ]
