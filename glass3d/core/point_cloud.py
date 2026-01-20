"""Point cloud data structure for Glass3D.

The PointCloud class is the central data structure representing the 3D points
to be engraved. It stores XYZ coordinates and optional per-point attributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass
class PointCloud:
    """A collection of 3D points for laser engraving.
    
    Points are stored as a numpy array of shape (N, 3) for XYZ coordinates.
    Additional per-point attributes (like intensity) can be stored separately.
    
    Attributes:
        points: Nx3 array of XYZ coordinates in mm
        intensities: Optional N-length array of intensity values (0-1)
        layer_indices: Optional N-length array mapping points to layer numbers
    """
    
    points: NDArray[np.float64]
    intensities: NDArray[np.float64] | None = None
    layer_indices: NDArray[np.int32] | None = None
    
    # Metadata
    source_file: str | None = None
    
    def __post_init__(self) -> None:
        """Validate and normalize data after initialization."""
        self.points = np.asarray(self.points, dtype=np.float64)
        
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"Points must be Nx3 array, got shape {self.points.shape}")
        
        if self.intensities is not None:
            self.intensities = np.asarray(self.intensities, dtype=np.float64)
            if len(self.intensities) != len(self.points):
                raise ValueError("Intensities length must match points length")
        
        if self.layer_indices is not None:
            self.layer_indices = np.asarray(self.layer_indices, dtype=np.int32)
            if len(self.layer_indices) != len(self.points):
                raise ValueError("Layer indices length must match points length")
    
    def __len__(self) -> int:
        """Return number of points."""
        return len(self.points)
    
    def __iter__(self) -> Iterator[NDArray[np.float64]]:
        """Iterate over individual points."""
        return iter(self.points)
    
    def __getitem__(self, idx: int | slice | NDArray) -> PointCloud:
        """Get subset of points by index."""
        new_points = self.points[idx]
        new_intensities = self.intensities[idx] if self.intensities is not None else None
        new_layers = self.layer_indices[idx] if self.layer_indices is not None else None
        
        # Handle single point case
        if new_points.ndim == 1:
            new_points = new_points.reshape(1, 3)
            if new_intensities is not None:
                new_intensities = np.array([new_intensities])
            if new_layers is not None:
                new_layers = np.array([new_layers])
        
        return PointCloud(
            points=new_points,
            intensities=new_intensities,
            layer_indices=new_layers,
            source_file=self.source_file,
        )
    
    @property
    def x(self) -> NDArray[np.float64]:
        """X coordinates."""
        return self.points[:, 0]
    
    @property
    def y(self) -> NDArray[np.float64]:
        """Y coordinates."""
        return self.points[:, 1]
    
    @property
    def z(self) -> NDArray[np.float64]:
        """Z coordinates."""
        return self.points[:, 2]
    
    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (min_xyz, max_xyz) bounding box."""
        return self.points.min(axis=0), self.points.max(axis=0)
    
    @property
    def center(self) -> NDArray[np.float64]:
        """Return center point of bounding box."""
        min_pt, max_pt = self.bounds
        return (min_pt + max_pt) / 2
    
    @property
    def size(self) -> NDArray[np.float64]:
        """Return size of bounding box (width, height, depth)."""
        min_pt, max_pt = self.bounds
        return max_pt - min_pt
    
    @property
    def num_layers(self) -> int:
        """Return number of unique layers."""
        if self.layer_indices is None:
            return 1
        return len(np.unique(self.layer_indices))
    
    def get_layer(self, layer_idx: int) -> PointCloud:
        """Get points belonging to a specific layer."""
        if self.layer_indices is None:
            if layer_idx != 0:
                raise ValueError("No layer indices assigned, only layer 0 exists")
            return self
        
        mask = self.layer_indices == layer_idx
        return self[mask]
    
    def iter_layers(self) -> Iterator[tuple[int, PointCloud]]:
        """Iterate over layers, yielding (layer_index, points) tuples."""
        if self.layer_indices is None:
            yield 0, self
            return
        
        for layer_idx in np.unique(self.layer_indices):
            yield int(layer_idx), self.get_layer(layer_idx)
    
    # -------------------------------------------------------------------------
    # Transformations
    # -------------------------------------------------------------------------
    
    def translate(self, offset: Sequence[float]) -> PointCloud:
        """Return new PointCloud translated by offset."""
        offset = np.asarray(offset, dtype=np.float64)
        return PointCloud(
            points=self.points + offset,
            intensities=self.intensities.copy() if self.intensities is not None else None,
            layer_indices=self.layer_indices.copy() if self.layer_indices is not None else None,
            source_file=self.source_file,
        )
    
    def scale(self, factor: float | Sequence[float]) -> PointCloud:
        """Return new PointCloud scaled by factor (uniform or per-axis)."""
        if isinstance(factor, (int, float)):
            factor = np.array([factor, factor, factor], dtype=np.float64)
        else:
            factor = np.asarray(factor, dtype=np.float64)
        
        return PointCloud(
            points=self.points * factor,
            intensities=self.intensities.copy() if self.intensities is not None else None,
            layer_indices=self.layer_indices.copy() if self.layer_indices is not None else None,
            source_file=self.source_file,
        )
    
    def center_at_origin(self) -> PointCloud:
        """Return new PointCloud centered at origin."""
        return self.translate(-self.center)
    
    def move_to(self, position: Sequence[float]) -> PointCloud:
        """Return new PointCloud with center at given position."""
        return self.center_at_origin().translate(position)
    
    def sort_by_z(self, ascending: bool = True) -> PointCloud:
        """Return new PointCloud with points sorted by Z coordinate.
        
        For SSLE, ascending=True (bottom-up) is recommended to avoid
        engraving through already-fractured material.
        """
        if ascending:
            order = np.argsort(self.z)
        else:
            order = np.argsort(self.z)[::-1]
        
        return self[order]
    
    def filter_bounds(
        self,
        min_bounds: Sequence[float] | None = None,
        max_bounds: Sequence[float] | None = None,
    ) -> PointCloud:
        """Return new PointCloud with points within bounds."""
        mask = np.ones(len(self), dtype=bool)
        
        if min_bounds is not None:
            min_bounds = np.asarray(min_bounds)
            mask &= np.all(self.points >= min_bounds, axis=1)
        
        if max_bounds is not None:
            max_bounds = np.asarray(max_bounds)
            mask &= np.all(self.points <= max_bounds, axis=1)
        
        return self[mask]
    
    def subsample(self, factor: int) -> PointCloud:
        """Return new PointCloud with every Nth point."""
        return self[::factor]
    
    def random_sample(self, n: int, seed: int | None = None) -> PointCloud:
        """Return new PointCloud with n randomly sampled points."""
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self), size=min(n, len(self)), replace=False)
        return self[indices]

    def remove_duplicates(self, tolerance: float = 1e-6) -> PointCloud:
        """Return new PointCloud with duplicate points removed.

        Points are considered duplicates if all coordinates are within
        the given tolerance.

        Args:
            tolerance: Distance threshold for duplicate detection

        Returns:
            PointCloud with unique points only
        """
        if len(self) == 0:
            return self

        # Round to tolerance precision and find unique rows
        scale = 1.0 / tolerance if tolerance > 0 else 1e10
        rounded = np.round(self.points * scale).astype(np.int64)

        # Find unique rows and their indices
        _, unique_indices = np.unique(rounded, axis=0, return_index=True)

        # Sort to preserve original order
        unique_indices = np.sort(unique_indices)

        return self[unique_indices]

    # -------------------------------------------------------------------------
    # Combining
    # -------------------------------------------------------------------------
    
    def append(self, other: PointCloud) -> PointCloud:
        """Return new PointCloud with other's points appended."""
        new_points = np.vstack([self.points, other.points])
        
        # Handle intensities
        if self.intensities is not None and other.intensities is not None:
            new_intensities = np.concatenate([self.intensities, other.intensities])
        else:
            new_intensities = None
        
        # Handle layer indices
        if self.layer_indices is not None and other.layer_indices is not None:
            new_layers = np.concatenate([self.layer_indices, other.layer_indices])
        else:
            new_layers = None
        
        return PointCloud(
            points=new_points,
            intensities=new_intensities,
            layer_indices=new_layers,
        )
    
    @classmethod
    def concatenate(cls, clouds: Sequence[PointCloud]) -> PointCloud:
        """Concatenate multiple point clouds into one."""
        if not clouds:
            return cls(points=np.empty((0, 3)))
        
        result = clouds[0]
        for cloud in clouds[1:]:
            result = result.append(cloud)
        return result
    
    # -------------------------------------------------------------------------
    # I/O
    # -------------------------------------------------------------------------
    
    def to_numpy(self) -> NDArray[np.float64]:
        """Export as raw numpy array."""
        return self.points.copy()
    
    @classmethod
    def from_numpy(cls, arr: NDArray) -> PointCloud:
        """Create from numpy array."""
        return cls(points=arr)
    
    def save_xyz(self, path: str) -> None:
        """Save to XYZ text file format."""
        np.savetxt(path, self.points, fmt="%.6f", delimiter=" ")
    
    @classmethod
    def load_xyz(cls, path: str) -> PointCloud:
        """Load from XYZ text file format."""
        points = np.loadtxt(path, dtype=np.float64)
        return cls(points=points, source_file=path)
    
    def save_npz(self, path: str) -> None:
        """Save to numpy compressed format (preserves all attributes)."""
        data = {"points": self.points}
        if self.intensities is not None:
            data["intensities"] = self.intensities
        if self.layer_indices is not None:
            data["layer_indices"] = self.layer_indices
        np.savez_compressed(path, **data)
    
    @classmethod
    def load_npz(cls, path: str) -> PointCloud:
        """Load from numpy compressed format."""
        data = np.load(path)
        return cls(
            points=data["points"],
            intensities=data.get("intensities"),
            layer_indices=data.get("layer_indices"),
            source_file=path,
        )
    
    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    
    def stats(self) -> dict:
        """Return statistics about the point cloud."""
        min_pt, max_pt = self.bounds
        return {
            "num_points": len(self),
            "num_layers": self.num_layers,
            "bounds_min": min_pt.tolist(),
            "bounds_max": max_pt.tolist(),
            "size": self.size.tolist(),
            "center": self.center.tolist(),
        }
    
    def __repr__(self) -> str:
        return f"PointCloud({len(self)} points, size={self.size})"
