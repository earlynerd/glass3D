"""Scene and model placement data structures.

This module provides the core data models for managing multiple 3D models
within a workspace, including Scene, ModelPlacement, and WorkspaceBounds.

Supports loading from 3MF files exported from slicer software (PrusaSlicer,
Cura, Bambu Studio, etc.) for familiar model arrangement workflows.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import trimesh
from pydantic import BaseModel, Field

from .transform import Transform3D

if TYPE_CHECKING:
    from ..core.config import MachineParams, PointCloudParams
    from ..core.point_cloud import PointCloud


class WorkspaceBounds(BaseModel):
    """Physical workspace constraints.

    Defines the valid region where models can be placed.
    Based on the galvo field size and Z-axis range.

    Uses corner-origin coordinates (0,0,0 at corner) to match
    slicer software conventions where the bed origin is at the corner.
    """

    x_range: tuple[float, float] = Field(
        default=(0.0, 110.0),
        description="X-axis bounds in mm (0 at corner)"
    )
    y_range: tuple[float, float] = Field(
        default=(0.0, 110.0),
        description="Y-axis bounds in mm (0 at corner)"
    )
    z_range: tuple[float, float] = Field(
        default=(0.0, 100.0),
        description="Z-axis bounds in mm (depth into glass)"
    )

    @classmethod
    def from_machine_params(cls, params: MachineParams) -> WorkspaceBounds:
        """Create WorkspaceBounds from MachineParams.

        Args:
            params: Machine parameters with field_size_mm and z_range_mm

        Returns:
            WorkspaceBounds matching the machine configuration
        """
        return cls(
            x_range=(0.0, params.field_size_mm[0]),
            y_range=(0.0, params.field_size_mm[1]),
            z_range=params.z_range_mm,
        )

    @property
    def size(self) -> tuple[float, float, float]:
        """Return workspace dimensions (width, height, depth)."""
        return (
            self.x_range[1] - self.x_range[0],
            self.y_range[1] - self.y_range[0],
            self.z_range[1] - self.z_range[0],
        )

    @property
    def center(self) -> tuple[float, float, float]:
        """Return workspace center point."""
        return (
            (self.x_range[0] + self.x_range[1]) / 2,
            (self.y_range[0] + self.y_range[1]) / 2,
            (self.z_range[0] + self.z_range[1]) / 2,
        )

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """Check if a point is within workspace bounds."""
        return (
            self.x_range[0] <= x <= self.x_range[1] and
            self.y_range[0] <= y <= self.y_range[1] and
            self.z_range[0] <= z <= self.z_range[1]
        )

    def contains_bounds(
        self,
        min_pt: Sequence[float],
        max_pt: Sequence[float],
    ) -> bool:
        """Check if a bounding box is fully within workspace bounds."""
        return (
            self.x_range[0] <= min_pt[0] and max_pt[0] <= self.x_range[1] and
            self.y_range[0] <= min_pt[1] and max_pt[1] <= self.y_range[1] and
            self.z_range[0] <= min_pt[2] and max_pt[2] <= self.z_range[1]
        )


class ModelPlacement(BaseModel):
    """A single model with its placement in the scene.

    Stores the path to a mesh file along with transformation parameters
    (position, rotation, scale) and optional override settings.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique identifier for this placement"
    )
    source_path: str = Field(description="Path to source mesh file (STL/OBJ/etc)")
    name: str = Field(default="", description="Display name for the model")
    transform: Transform3D = Field(
        default_factory=Transform3D,
        description="Position, rotation, and scale"
    )

    # Generation settings (can override scene defaults)
    strategy: str | None = Field(
        default=None,
        description="Override point generation strategy"
    )
    point_spacing_mm: float | None = Field(
        default=None,
        description="Override point spacing"
    )

    # Visual settings for viewer
    color: tuple[float, float, float] = Field(
        default=(0.7, 0.7, 0.9),
        description="RGB color for viewer (0-1 range)"
    )
    visible: bool = Field(default=True, description="Whether model is visible")

    model_config = {"frozen": False}

    def get_absolute_path(self, base_dir: Path | None = None) -> Path:
        """Resolve the source path to an absolute path.

        Args:
            base_dir: Base directory for relative paths

        Returns:
            Absolute path to the source file
        """
        path = Path(self.source_path)
        if path.is_absolute():
            return path
        if base_dir is not None:
            return (base_dir / path).resolve()
        return path.resolve()


class Scene(BaseModel):
    """A scene containing multiple models for arrangement.

    The Scene is the top-level container that holds all models to be engraved
    together. It manages workspace bounds and default generation parameters.
    """

    name: str = Field(default="Untitled Scene", description="Scene name")
    version: str = Field(default="1.0", description="Scene file version")

    # Models in the scene
    models: list[ModelPlacement] = Field(
        default_factory=list,
        description="List of models with their placements"
    )

    # Workspace
    workspace: WorkspaceBounds = Field(
        default_factory=WorkspaceBounds,
        description="Workspace bounds"
    )

    # Default generation parameters
    default_strategy: str = Field(
        default="surface",
        description="Default point generation strategy"
    )
    default_point_spacing_mm: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Default point spacing in mm"
    )
    default_layer_height_mm: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Default layer height in mm"
    )

    model_config = {"frozen": False}

    def add_model(
        self,
        source_path: str | Path,
        name: str | None = None,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: float = 1.0,
        **kwargs,
    ) -> ModelPlacement:
        """Add a model to the scene.

        Args:
            source_path: Path to the mesh file
            name: Display name (defaults to filename)
            position: XYZ position in mm
            rotation: XYZ rotation in degrees
            scale: Uniform scale factor
            **kwargs: Additional ModelPlacement parameters

        Returns:
            The created ModelPlacement
        """
        path = Path(source_path)
        model = ModelPlacement(
            source_path=str(path),
            name=name or path.stem,
            transform=Transform3D(
                position=position,
                rotation=rotation,
                scale=scale,
            ),
            **kwargs,
        )
        self.models.append(model)
        return model

    def remove_model(self, model_id: str) -> bool:
        """Remove a model from the scene by ID.

        Args:
            model_id: ID of the model to remove

        Returns:
            True if model was removed, False if not found
        """
        for i, model in enumerate(self.models):
            if model.id == model_id:
                self.models.pop(i)
                return True
        return False

    def get_model(self, model_id: str) -> ModelPlacement | None:
        """Get a model by ID.

        Args:
            model_id: ID of the model to find

        Returns:
            ModelPlacement if found, None otherwise
        """
        for model in self.models:
            if model.id == model_id:
                return model
        return None

    def validate_bounds(self) -> tuple[bool, list[str]]:
        """Validate that all models are within workspace bounds.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        from ..mesh.loader import MeshLoader

        errors = []

        for model in self.models:
            if not model.visible:
                continue

            try:
                # Load mesh to get bounds
                path = model.get_absolute_path()
                loader = MeshLoader(path)

                # Get mesh bounds
                mesh_min, mesh_max = loader.bounds

                # Apply transform to bounding box corners
                corners = np.array([
                    [mesh_min[0], mesh_min[1], mesh_min[2]],
                    [mesh_min[0], mesh_min[1], mesh_max[2]],
                    [mesh_min[0], mesh_max[1], mesh_min[2]],
                    [mesh_min[0], mesh_max[1], mesh_max[2]],
                    [mesh_max[0], mesh_min[1], mesh_min[2]],
                    [mesh_max[0], mesh_min[1], mesh_max[2]],
                    [mesh_max[0], mesh_max[1], mesh_min[2]],
                    [mesh_max[0], mesh_max[1], mesh_max[2]],
                ])

                transformed_corners = model.transform.apply_to_points(corners)
                transformed_min = transformed_corners.min(axis=0)
                transformed_max = transformed_corners.max(axis=0)

                if not self.workspace.contains_bounds(transformed_min, transformed_max):
                    errors.append(
                        f"Model '{model.name}' ({model.id}) is outside workspace bounds. "
                        f"Bounds: [{transformed_min.tolist()}, {transformed_max.tolist()}]"
                    )

            except Exception as e:
                errors.append(f"Could not validate model '{model.name}': {e}")

        return len(errors) == 0, errors

    def save(self, path: str | Path) -> None:
        """Save scene to a JSON file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> Scene:
        """Load scene from a JSON file.

        Args:
            path: Input file path

        Returns:
            Loaded Scene
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        return cls.model_validate(data)

    @classmethod
    def from_3mf(cls, path: str | Path, workspace: WorkspaceBounds | None = None) -> Scene:
        """Load scene from a 3MF file exported from slicer software.

        3MF files from PrusaSlicer, Cura, Bambu Studio, etc. contain
        multiple models with their placement transforms (position,
        rotation, scale) already applied.

        Args:
            path: Path to the 3MF file
            workspace: Optional workspace bounds (uses defaults if None)

        Returns:
            Scene with models and transforms from the 3MF file
        """
        path = Path(path)

        # Load 3MF using trimesh (returns a Scene for multi-object files)
        loaded = trimesh.load(str(path))

        # Try to extract model names from slicer metadata
        name_map = cls._extract_slicer_names(path)

        # Create scene with workspace bounds
        scene = cls(
            name=path.stem,
            workspace=workspace or WorkspaceBounds(),
        )

        # Handle single mesh vs scene
        if isinstance(loaded, trimesh.Trimesh):
            # Single mesh - add with identity transform
            scene._add_trimesh(
                mesh=loaded,
                name=path.stem,
                source_path=str(path),
            )
        elif isinstance(loaded, trimesh.Scene):
            # Multi-object scene - extract each geometry with its transform
            scene._load_from_trimesh_scene(loaded, str(path), name_map)
        else:
            raise ValueError(f"Unexpected type from trimesh.load: {type(loaded)}")

        return scene

    @staticmethod
    def _extract_slicer_names(path: Path) -> dict[str, str]:
        """Extract model names from slicer-specific metadata in 3MF file.

        3MF files from PrusaSlicer/Slic3r store model names in a separate
        config file (Metadata/Slic3r_PE_model.config) rather than in the
        main 3D model file.

        Args:
            path: Path to the 3MF file

        Returns:
            Dict mapping object IDs to model names
        """
        import zipfile
        import xml.etree.ElementTree as ET

        name_map: dict[str, str] = {}

        try:
            with zipfile.ZipFile(path, 'r') as zf:
                # Try PrusaSlicer/Slic3r format
                if 'Metadata/Slic3r_PE_model.config' in zf.namelist():
                    content = zf.read('Metadata/Slic3r_PE_model.config').decode('utf-8')
                    root = ET.fromstring(content)

                    for obj in root.findall('object'):
                        obj_id = obj.get('id')
                        if obj_id:
                            # Look for name in object metadata
                            for meta in obj.findall('metadata'):
                                if meta.get('key') == 'name':
                                    name_map[obj_id] = meta.get('value', obj_id)
                                    break

        except Exception:
            # If metadata extraction fails, fall back to using IDs
            pass

        return name_map

    def _load_from_trimesh_scene(
        self,
        tm_scene: trimesh.Scene,
        source_path: str,
        name_map: dict[str, str] | None = None,
    ) -> None:
        """Load models from a trimesh Scene object.

        Args:
            tm_scene: trimesh Scene containing geometries and transforms
            source_path: Original file path for reference
            name_map: Optional mapping from object IDs to display names
        """
        if name_map is None:
            name_map = {}

        # Get the scene graph to access transforms
        graph = tm_scene.graph

        # Iterate through geometry instances in the scene
        # Each node in the graph can reference a geometry with a transform
        for node_name in graph.nodes_geometry:
            # Get the geometry name and transform for this node
            transform_matrix, geometry_name = graph.get(node_name)

            # Get the actual mesh
            geometry = tm_scene.geometry.get(geometry_name)
            if geometry is None:
                continue

            # Skip non-mesh geometry (e.g., PointCloud, Path)
            if not isinstance(geometry, trimesh.Trimesh):
                continue

            # Extract transform components from the 4x4 matrix
            transform = Transform3D.from_matrix(transform_matrix)

            # Create model placement
            # Try to get name from slicer metadata, fall back to node/geometry name
            display_name = name_map.get(node_name) or name_map.get(geometry_name)
            if not display_name:
                display_name = node_name if node_name != geometry_name else geometry_name

            model = ModelPlacement(
                source_path=source_path,
                name=display_name,
                transform=transform,
            )

            # Store the actual mesh data for later use
            # We need to store the base geometry (without scene transform)
            model._mesh = geometry.copy()

            self.models.append(model)

    def _add_trimesh(
        self,
        mesh: trimesh.Trimesh,
        name: str,
        source_path: str,
        transform: Transform3D | None = None,
    ) -> ModelPlacement:
        """Add a trimesh directly to the scene.

        Args:
            mesh: The trimesh geometry
            name: Display name
            source_path: Source file path
            transform: Optional transform (identity if None)

        Returns:
            The created ModelPlacement
        """
        model = ModelPlacement(
            source_path=source_path,
            name=name,
            transform=transform or Transform3D(),
        )
        model._mesh = mesh.copy()
        self.models.append(model)
        return model

    @staticmethod
    def is_anchor_model(name: str) -> bool:
        """Check if a model name indicates it's an anchor (to be skipped).

        Anchor models are used in slicers to hold floating parts at Z=0
        but should be excluded from point cloud generation.

        A model is considered an anchor if "anchor" appears anywhere in the
        name (case-insensitive). This handles various naming conventions:
        - "anchor"
        - "anchor.stl"
        - "MyAssembly_anchor"
        - "anchor_plate"
        - "_anchor"

        Args:
            name: Model name to check

        Returns:
            True if this is an anchor model
        """
        return "anchor" in name.lower()

    @staticmethod
    def is_anchor_geometry(mesh: trimesh.Trimesh, z_tolerance: float = 0.5, max_height: float = 2.0) -> bool:
        """Check if a mesh looks like an anchor based on its geometry.

        A mesh is considered an anchor if:
        - Its minimum Z is close to 0 (touching the bed)
        - Its total height (Z extent) is small

        This is a fallback for when anchors aren't properly named.

        Args:
            mesh: The transformed mesh to check
            z_tolerance: How close to Z=0 the bottom must be (default 0.5mm)
            max_height: Maximum height to be considered an anchor (default 2.0mm)

        Returns:
            True if this mesh appears to be an anchor
        """
        bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        min_z = bounds[0][2]
        max_z = bounds[1][2]
        height = max_z - min_z

        return min_z <= z_tolerance and height <= max_height

    def get_anchor_status(self, model: ModelPlacement) -> tuple[bool, str]:
        """Check if a model should be skipped and return the reason.

        Args:
            model: The model to check

        Returns:
            Tuple of (is_anchor, reason_string)
            reason_string is empty if not an anchor, or one of:
            - "name": anchor detected by name
            - "geometry": entire mesh is anchor geometry
            - "partial": mesh contains anchor components that will be filtered
        """
        # Check by name first
        if self.is_anchor_model(model.name):
            return True, "name"

        # Check by geometry (need to load and transform mesh)
        try:
            if hasattr(model, '_mesh') and model._mesh is not None:
                mesh = model._mesh.copy()
            else:
                from ..mesh.loader import MeshLoader
                path = model.get_absolute_path()
                loader = MeshLoader(path)
                mesh = loader.mesh

            transform_matrix = model.transform.to_matrix()
            mesh.apply_transform(transform_matrix)

            # Check if entire mesh is an anchor
            if self.is_anchor_geometry(mesh):
                return True, "geometry"

            # Check for anchor components within the mesh (assemblies)
            if hasattr(mesh, 'split'):
                components = mesh.split()
                if len(components) > 1:
                    anchor_count = sum(1 for c in components if self.is_anchor_geometry(c))
                    if anchor_count == len(components):
                        return True, "geometry"
                    elif anchor_count > 0:
                        return False, f"partial:{anchor_count}"
        except Exception:
            pass

        return False, ""

    def to_combined_point_cloud(
        self,
        params: PointCloudParams | None = None,
    ) -> PointCloud:
        """Generate combined point cloud from all models.

        Loads each model, applies its transform, generates a point cloud,
        then combines all point clouds into one.

        For 3MF-loaded scenes, uses the cached mesh data directly.
        For file-based scenes, loads meshes from disk.

        Models named "anchor" or starting with "_anchor"/"anchor_" are
        automatically skipped (used as slicer anchor plates).

        Args:
            params: Point cloud parameters (uses scene defaults if None)

        Returns:
            Combined PointCloud from all visible models
        """
        from ..core.config import PointCloudParams
        from ..core.point_cloud import PointCloud
        from ..mesh.loader import MeshLoader
        from ..mesh.pointcloud_gen import get_strategy

        # Create default params if not provided
        if params is None:
            params = PointCloudParams(
                point_spacing_mm=self.default_point_spacing_mm,
                layer_height_mm=self.default_layer_height_mm,
                strategy=self.default_strategy,
            )

        clouds = []

        for model in self.models:
            if not model.visible:
                continue

            # Skip anchor models (used for slicer positioning only)
            if self.is_anchor_model(model.name):
                continue

            # Get mesh - either from cache (3MF) or load from file
            if hasattr(model, '_mesh') and model._mesh is not None:
                # Use cached mesh from 3MF loading
                mesh = model._mesh.copy()
            else:
                # Load from file
                path = model.get_absolute_path()
                loader = MeshLoader(path)
                mesh = loader.mesh

            # Apply transform to mesh
            transform_matrix = model.transform.to_matrix()
            mesh.apply_transform(transform_matrix)

            # Skip if entire geometry looks like an anchor (flat object at Z=0)
            if self.is_anchor_geometry(mesh):
                continue

            # Split mesh into disconnected components and filter out anchor parts
            # This handles assemblies where anchor + model are merged into one mesh
            if hasattr(mesh, 'split'):
                components = mesh.split()
                if len(components) > 1:
                    # Filter out anchor-like components
                    non_anchor_components = [
                        c for c in components
                        if not self.is_anchor_geometry(c)
                    ]
                    if len(non_anchor_components) == 0:
                        # All components are anchors, skip entire mesh
                        continue
                    elif len(non_anchor_components) < len(components):
                        # Some components were anchors, recombine the rest
                        mesh = trimesh.util.concatenate(non_anchor_components)

            # Determine strategy (model override or scene default)
            strategy_name = model.strategy or params.strategy

            # Determine point spacing (model override or params)
            model_params = params.model_copy()
            if model.point_spacing_mm is not None:
                model_params.point_spacing_mm = model.point_spacing_mm

            # Generate point cloud
            strategy = get_strategy(strategy_name)
            cloud = strategy.generate(mesh, model_params)

            clouds.append(cloud)

        if not clouds:
            return PointCloud(points=np.empty((0, 3)))

        # Combine all point clouds
        combined = PointCloud.concatenate(clouds)

        # Sort by Z for proper SSLE engraving (bottom-up)
        combined = combined.sort_by_z(ascending=True)

        return combined

    def __repr__(self) -> str:
        return f"Scene('{self.name}', {len(self.models)} models)"
