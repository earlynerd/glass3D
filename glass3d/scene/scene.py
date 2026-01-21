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
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
import trimesh
from pydantic import BaseModel, Field, PrivateAttr

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

    # Private attribute for cached mesh data (not serialized)
    _mesh: trimesh.Trimesh | None = PrivateAttr(default=None)

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
    default_strategy: Literal["surface", "solid", "grayscale", "contour"] = Field(
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
        **kwargs: Any,
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

        # Create scene with workspace bounds
        scene = cls(
            name=path.stem,
            workspace=workspace or WorkspaceBounds(),
        )

        # Try BambuStudio format first (handles objectid references correctly)
        if cls._is_bambu_3mf(path):
            scene._load_bambu_3mf(path)
            return scene

        # Fall back to trimesh-based loading for other formats
        loaded = trimesh.load(str(path))

        # Try to extract volume info from slicer metadata (includes face ranges)
        volume_info = cls._extract_slicer_volume_info(path)

        # Fall back to simple name map if no volume info
        name_map = cls._extract_slicer_names(path) if not volume_info else {}

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
            scene._load_from_trimesh_scene(loaded, str(path), name_map, volume_info)
        else:
            raise ValueError(f"Unexpected type from trimesh.load: {type(loaded)}")

        return scene

    @staticmethod
    def _is_bambu_3mf(path: Path) -> bool:
        """Check if a 3MF file is in BambuStudio format.

        BambuStudio 3MF files have:
        - Metadata/model_settings.config with part names
        - 3D/Objects/*.model files with multiple objects
        - Components that reference objectid within model files
        """
        import zipfile

        try:
            with zipfile.ZipFile(path, 'r') as zf:
                names = zf.namelist()
                has_bambu_config = 'Metadata/model_settings.config' in names
                has_object_models = any(n.startswith('3D/Objects/') and n.endswith('.model') for n in names)
                return has_bambu_config and has_object_models
        except Exception:
            return False

    def _load_bambu_3mf(self, path: Path) -> None:
        """Load a BambuStudio format 3MF file.

        BambuStudio stores multiple meshes in single .model files and references
        them by objectid in component tags. This method correctly loads each mesh
        with its proper transform.
        """
        import zipfile
        import xml.etree.ElementTree as ET

        ns = {
            'm': 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02',
            'p': 'http://schemas.microsoft.com/3dmanufacturing/production/2015/06',
        }

        # Extract model names from BambuStudio metadata
        name_map = self._extract_bambu_names(path)

        with zipfile.ZipFile(path, 'r') as zf:
            # Parse main model to get component references
            main_content = zf.read('3D/3dmodel.model').decode('utf-8')
            main_root = ET.fromstring(main_content)

            # Get build-level transform (placement on build plate)
            build_transform = np.eye(4)
            for item in main_root.findall('.//m:item', ns):
                transform_str = item.get('transform')
                if transform_str:
                    vals = [float(x) for x in transform_str.split()]
                    if len(vals) == 12:
                        build_transform[:3, :] = np.array(vals).reshape(4, 3).T
                break  # Only use first build item

            # Cache for loaded model files (path -> {objectid -> mesh})
            model_cache: dict[str, dict[str, trimesh.Trimesh]] = {}

            # Find all components and their transforms
            for comp in main_root.findall('.//m:component', ns):
                model_path = comp.get(f'{{{ns["p"]}}}path')
                object_id = comp.get('objectid')
                transform_str = comp.get('transform')

                if not model_path or not object_id:
                    continue

                # Normalize path (remove leading /)
                model_path = model_path.lstrip('/')

                # Load mesh from the model file if not cached
                if model_path not in model_cache:
                    model_cache[model_path] = self._parse_3mf_model_file(
                        zf.read(model_path).decode('utf-8')
                    )

                mesh = model_cache[model_path].get(object_id)
                if mesh is None:
                    continue

                # Parse component transform (3x4 matrix in column-major order per 3MF spec:
                # m00 m10 m20 m01 m11 m21 m02 m12 m22 m03 m13 m23)
                component_transform = np.eye(4)
                if transform_str:
                    vals = [float(x) for x in transform_str.split()]
                    if len(vals) == 12:
                        # Reshape as 4 columns of 3 elements, then transpose to get 3x4
                        component_transform[:3, :] = np.array(vals).reshape(4, 3).T

                # Combined transform: build_transform @ component_transform
                combined_transform = build_transform @ component_transform

                # Get name from metadata or use object_id
                display_name = name_map.get(object_id, object_id)

                # Try to decompose transform, or bake it into mesh if not possible
                try:
                    transform = Transform3D.from_matrix(combined_transform)
                    mesh_to_store = mesh.copy()
                except ValueError:
                    mesh_to_store = mesh.copy()
                    mesh_to_store.apply_transform(combined_transform)
                    transform = Transform3D()

                model = ModelPlacement(
                    source_path=str(path),
                    name=display_name,
                    transform=transform,
                )
                model._mesh = mesh_to_store
                self.models.append(model)

    @staticmethod
    def _parse_3mf_model_file(content: str) -> dict[str, trimesh.Trimesh]:
        """Parse a 3MF model file and return meshes by object ID.

        Args:
            content: XML content of the .model file

        Returns:
            Dict mapping object IDs to trimesh meshes
        """
        import xml.etree.ElementTree as ET

        ns = {'m': 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'}
        root = ET.fromstring(content)

        meshes: dict[str, trimesh.Trimesh] = {}

        for obj in root.findall('.//m:object', ns):
            obj_id = obj.get('id')
            if not obj_id:
                continue

            mesh_elem = obj.find('m:mesh', ns)
            if mesh_elem is None:
                continue

            # Parse vertices
            vertices = []
            verts_elem = mesh_elem.find('m:vertices', ns)
            if verts_elem is not None:
                for v in verts_elem.findall('m:vertex', ns):
                    vertices.append([
                        float(v.get('x', 0)),
                        float(v.get('y', 0)),
                        float(v.get('z', 0)),
                    ])

            # Parse triangles
            faces = []
            tris_elem = mesh_elem.find('m:triangles', ns)
            if tris_elem is not None:
                for t in tris_elem.findall('m:triangle', ns):
                    faces.append([
                        int(t.get('v1', 0)),
                        int(t.get('v2', 0)),
                        int(t.get('v3', 0)),
                    ])

            if vertices and faces:
                meshes[obj_id] = trimesh.Trimesh(
                    vertices=np.array(vertices),
                    faces=np.array(faces),
                )

        return meshes

    @staticmethod
    def _extract_bambu_names(path: Path) -> dict[str, str]:
        """Extract model names from BambuStudio model_settings.config.

        Args:
            path: Path to the 3MF file

        Returns:
            Dict mapping part IDs to display names
        """
        import zipfile
        import xml.etree.ElementTree as ET

        name_map: dict[str, str] = {}

        try:
            with zipfile.ZipFile(path, 'r') as zf:
                if 'Metadata/model_settings.config' not in zf.namelist():
                    return name_map

                content = zf.read('Metadata/model_settings.config').decode('utf-8')
                root = ET.fromstring(content)

                # Find all parts and their names
                for part in root.findall('.//part'):
                    part_id = part.get('id')
                    if not part_id:
                        continue

                    for meta in part.findall('metadata'):
                        if meta.get('key') == 'name':
                            name_map[part_id] = meta.get('value', part_id)
                            break

        except Exception:
            pass

        return name_map

    @staticmethod
    def _extract_slicer_volume_info(path: Path) -> dict[str, list[dict[str, Any]]]:
        """Extract volume info from slicer-specific metadata in 3MF file.

        3MF files from PrusaSlicer/Slic3r store model names and volume
        information in a separate config file (Metadata/Slic3r_PE_model.config).
        When models are added as subparts in the slicer, each object contains
        multiple volumes that need to be split.

        Args:
            path: Path to the 3MF file

        Returns:
            Dict mapping object IDs to list of volume info dicts.
            Each volume dict has: name, firstid, lastid, matrix (optional)
        """
        import zipfile
        import xml.etree.ElementTree as ET

        volume_info: dict[str, list[dict[str, Any]]] = {}

        try:
            with zipfile.ZipFile(path, 'r') as zf:
                # Try PrusaSlicer/Slic3r format
                if 'Metadata/Slic3r_PE_model.config' in zf.namelist():
                    content = zf.read('Metadata/Slic3r_PE_model.config').decode('utf-8')
                    root = ET.fromstring(content)

                    for obj in root.findall('object'):
                        obj_id = obj.get('id')
                        if not obj_id:
                            continue

                        volumes: list[dict[str, Any]] = []
                        for vol in obj.findall('volume'):
                            vol_data: dict[str, Any] = {
                                'firstid': int(vol.get('firstid', 0)),
                                'lastid': int(vol.get('lastid', 0)),
                            }

                            # Extract volume metadata
                            for meta in vol.findall('metadata'):
                                key = meta.get('key')
                                value = meta.get('value')
                                if key == 'name':
                                    vol_data['name'] = value
                                elif key == 'matrix':
                                    # Parse 4x4 matrix from space-separated string
                                    # Matrix is stored row-major (standard order)
                                    vals = [float(x) for x in value.split()]
                                    if len(vals) == 16:
                                        mat = np.array(vals).reshape(4, 4)
                                        vol_data['matrix'] = mat

                            volumes.append(vol_data)

                        if volumes:
                            volume_info[obj_id] = volumes

        except Exception:
            # If metadata extraction fails, fall back to no volume info
            pass

        return volume_info

    @staticmethod
    def _extract_slicer_names(path: Path) -> dict[str, str]:
        """Extract model names from slicer-specific metadata in 3MF file.

        This is a simplified version that just returns object-level names.
        For full volume support, use _extract_slicer_volume_info instead.

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
        volume_info: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        """Load models from a trimesh Scene object.

        Args:
            tm_scene: trimesh Scene containing geometries and transforms
            source_path: Original file path for reference
            name_map: Optional mapping from object IDs to display names
            volume_info: Optional mapping from object IDs to volume info (for splitting)
        """
        if name_map is None:
            name_map = {}
        if volume_info is None:
            volume_info = {}

        # Get the scene graph to access transforms
        graph = tm_scene.graph

        # Iterate through geometry instances in the scene
        # Each node in the graph can reference a geometry with a transform
        for node_name in graph.nodes_geometry:
            # Get the geometry name and transform for this node
            object_transform, geometry_name = graph.get(node_name)

            # Get the actual mesh
            geometry = tm_scene.geometry.get(geometry_name)
            if geometry is None:
                continue

            # Skip non-mesh geometry (e.g., PointCloud, Path)
            if not isinstance(geometry, trimesh.Trimesh):
                continue

            # Check if we have volume info for this object (multiple subparts)
            volumes = volume_info.get(node_name) or volume_info.get(geometry_name)

            if volumes and len(volumes) > 1:
                # Split mesh into separate volumes and add each
                self._add_split_volumes(
                    geometry, volumes, object_transform, source_path
                )
            else:
                # Single mesh - try to decompose the transform
                # If decomposition fails (reflection, non-uniform scale, etc.),
                # apply the transform directly to the mesh vertices
                try:
                    transform = Transform3D.from_matrix(object_transform)
                    mesh_to_store = geometry.copy()
                except ValueError:
                    # Transform can't be represented in Transform3D
                    # Apply it directly to mesh vertices instead
                    mesh_to_store = geometry.copy()
                    mesh_to_store.apply_transform(object_transform)
                    transform = Transform3D()

                # Get name from volume info, name_map, or fall back to node/geometry name
                if volumes and len(volumes) == 1:
                    display_name = volumes[0].get('name', geometry_name)
                else:
                    display_name = name_map.get(node_name) or name_map.get(geometry_name)
                    if not display_name:
                        display_name = node_name if node_name != geometry_name else geometry_name

                model = ModelPlacement(
                    source_path=source_path,
                    name=display_name,
                    transform=transform,
                )

                # Store the mesh (already copied, with transform baked in if needed)
                model._mesh = mesh_to_store

                self.models.append(model)

    def _add_split_volumes(
        self,
        merged_mesh: trimesh.Trimesh,
        volumes: list[dict[str, Any]],
        object_transform: np.ndarray,
        source_path: str,
    ) -> None:
        """Split a merged mesh into separate volumes and add each as a model.

        When PrusaSlicer adds models as subparts of an object, trimesh loads
        them as a single merged mesh. This method uses the face range info
        from the slicer config to split them back into separate models.

        Args:
            merged_mesh: The merged mesh containing all volumes
            volumes: List of volume info dicts with name, firstid, lastid, matrix
            object_transform: The object-level transform from the scene graph
            source_path: Original file path for reference
        """
        for vol in volumes:
            name = vol.get('name', 'unnamed')
            first_face = vol.get('firstid', 0)
            last_face = vol.get('lastid', len(merged_mesh.faces) - 1)

            # Extract faces for this volume
            face_mask = np.zeros(len(merged_mesh.faces), dtype=bool)
            face_mask[first_face:last_face + 1] = True

            # Create submesh with only these faces
            submesh = merged_mesh.submesh([face_mask], append=True)

            # NOTE: trimesh already applies volume-level transforms when loading 3MF,
            # so we only need to apply the object-level transform here.
            # The vol_matrix from slicer config is for reference/metadata only.

            # Try to decompose the object transform
            # If it fails (reflection, non-uniform scale, etc.), apply directly to mesh
            try:
                transform = Transform3D.from_matrix(object_transform)
            except ValueError:
                # Transform can't be represented in Transform3D
                # Apply it directly to mesh vertices instead
                submesh.apply_transform(object_transform)
                transform = Transform3D()

            model = ModelPlacement(
                source_path=source_path,
                name=name,
                transform=transform,
            )
            model._mesh = submesh

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

        return bool(min_z <= z_tolerance and height <= max_height)

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
