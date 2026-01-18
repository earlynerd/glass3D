"""Tests for Scene, ModelPlacement, Transform3D, and WorkspaceBounds."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from glass3d.scene.transform import Transform3D
from glass3d.scene.scene import Scene, ModelPlacement, WorkspaceBounds


class TestTransform3D:
    """Test Transform3D functionality."""

    def test_default_transform(self):
        """Test default transform is identity-like."""
        t = Transform3D()
        assert t.position == (0.0, 0.0, 0.0)
        assert t.rotation == (0.0, 0.0, 0.0)
        assert t.scale == 1.0

    def test_to_matrix_identity(self):
        """Test identity transform produces identity matrix."""
        t = Transform3D()
        matrix = t.to_matrix()

        expected = np.eye(4)
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_to_matrix_translation(self):
        """Test translation-only transform."""
        t = Transform3D(position=(10.0, 20.0, 30.0))
        matrix = t.to_matrix()

        # Check translation part
        np.testing.assert_array_almost_equal(matrix[:3, 3], [10.0, 20.0, 30.0])

    def test_to_matrix_scale(self):
        """Test scale-only transform."""
        t = Transform3D(scale=2.0)
        matrix = t.to_matrix()

        # Check diagonal (should be 2, 2, 2, 1)
        np.testing.assert_array_almost_equal(
            np.diag(matrix), [2.0, 2.0, 2.0, 1.0]
        )

    def test_to_matrix_rotation_z(self):
        """Test Z-rotation transform."""
        t = Transform3D(rotation=(0.0, 0.0, 90.0))
        matrix = t.to_matrix()

        # 90 degree Z rotation: x -> y, y -> -x
        point = np.array([1.0, 0.0, 0.0, 1.0])
        result = matrix @ point
        np.testing.assert_array_almost_equal(result[:3], [0.0, 1.0, 0.0])

    def test_apply_to_points(self):
        """Test applying transform to points."""
        t = Transform3D(position=(10.0, 0.0, 0.0), scale=2.0)
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])

        result = t.apply_to_points(points)

        # Point at origin: scaled (stays at 0), then translated by 10
        np.testing.assert_array_almost_equal(result[0], [10.0, 0.0, 0.0])
        # Point at (1,0,0): scaled to (2,0,0), then translated to (12,0,0)
        np.testing.assert_array_almost_equal(result[1], [12.0, 0.0, 0.0])

    def test_from_matrix_roundtrip(self):
        """Test that from_matrix inverts to_matrix."""
        original = Transform3D(
            position=(5.0, 10.0, 15.0),
            rotation=(30.0, 45.0, 60.0),
            scale=1.5,
        )
        matrix = original.to_matrix()
        recovered = Transform3D.from_matrix(matrix)

        assert abs(recovered.scale - original.scale) < 0.01
        for i in range(3):
            assert abs(recovered.position[i] - original.position[i]) < 0.01
            assert abs(recovered.rotation[i] - original.rotation[i]) < 1.0  # Degrees

    def test_compose(self):
        """Test composing transforms."""
        t1 = Transform3D(position=(10.0, 0.0, 0.0))
        t2 = Transform3D(scale=2.0)

        # t1 then t2: first translate, then scale
        composed = t1.compose(t2)

        point = np.array([[0.0, 0.0, 0.0]])
        result = composed.apply_to_points(point)

        # Original at (0,0,0), translate to (10,0,0), scale to (20,0,0)
        np.testing.assert_array_almost_equal(result[0], [20.0, 0.0, 0.0])

    def test_inverse(self):
        """Test inverse transform."""
        t = Transform3D(
            position=(10.0, 20.0, 30.0),
            rotation=(0.0, 0.0, 90.0),
            scale=2.0,
        )
        inv = t.inverse()

        # Composing with inverse should give identity
        composed = t.compose(inv)
        matrix = composed.to_matrix()

        np.testing.assert_array_almost_equal(matrix, np.eye(4), decimal=5)


class TestWorkspaceBounds:
    """Test WorkspaceBounds functionality."""

    def test_default_bounds(self):
        """Test default workspace bounds."""
        ws = WorkspaceBounds()
        assert ws.x_range == (-55.0, 55.0)
        assert ws.y_range == (-55.0, 55.0)
        assert ws.z_range == (0.0, 100.0)

    def test_size(self):
        """Test size calculation."""
        ws = WorkspaceBounds()
        assert ws.size == (110.0, 110.0, 100.0)

    def test_center(self):
        """Test center calculation."""
        ws = WorkspaceBounds()
        assert ws.center == (0.0, 0.0, 50.0)

    def test_contains_point_inside(self):
        """Test that points inside bounds return True."""
        ws = WorkspaceBounds()
        assert ws.contains_point(0, 0, 50) is True
        assert ws.contains_point(-50, 50, 0) is True

    def test_contains_point_outside(self):
        """Test that points outside bounds return False."""
        ws = WorkspaceBounds()
        assert ws.contains_point(100, 0, 0) is False
        assert ws.contains_point(0, 0, -10) is False

    def test_contains_bounds_inside(self):
        """Test bounding box containment."""
        ws = WorkspaceBounds()
        assert ws.contains_bounds([-10, -10, 10], [10, 10, 20]) is True

    def test_contains_bounds_outside(self):
        """Test bounding box outside workspace."""
        ws = WorkspaceBounds()
        assert ws.contains_bounds([-100, -10, 10], [10, 10, 20]) is False


class TestModelPlacement:
    """Test ModelPlacement functionality."""

    def test_create_minimal(self):
        """Test creating a model with minimal parameters."""
        model = ModelPlacement(source_path="test.stl")
        assert model.source_path == "test.stl"
        assert model.name == ""
        assert model.transform.position == (0.0, 0.0, 0.0)

    def test_create_with_transform(self):
        """Test creating a model with transform."""
        model = ModelPlacement(
            source_path="test.stl",
            name="Test Model",
            transform=Transform3D(
                position=(10.0, 20.0, 30.0),
                scale=2.0,
            ),
        )
        assert model.name == "Test Model"
        assert model.transform.position == (10.0, 20.0, 30.0)
        assert model.transform.scale == 2.0

    def test_unique_id(self):
        """Test that each model gets a unique ID."""
        model1 = ModelPlacement(source_path="a.stl")
        model2 = ModelPlacement(source_path="b.stl")
        assert model1.id != model2.id

    def test_get_absolute_path_absolute(self):
        """Test getting absolute path when already absolute."""
        # Use a platform-appropriate absolute path
        import sys
        if sys.platform == "win32":
            abs_path = "C:\\absolute\\path\\test.stl"
        else:
            abs_path = "/absolute/path/test.stl"
        model = ModelPlacement(source_path=abs_path)
        path = model.get_absolute_path()
        assert path == Path(abs_path)

    def test_get_absolute_path_relative(self):
        """Test getting absolute path from relative with base dir."""
        model = ModelPlacement(source_path="models/test.stl")
        path = model.get_absolute_path(Path("/project"))
        assert path == Path("/project/models/test.stl").resolve()


class TestScene:
    """Test Scene functionality."""

    def test_create_empty(self):
        """Test creating an empty scene."""
        scene = Scene()
        assert scene.name == "Untitled Scene"
        assert len(scene.models) == 0

    def test_create_with_name(self):
        """Test creating a named scene."""
        scene = Scene(name="My Project")
        assert scene.name == "My Project"

    def test_add_model(self):
        """Test adding a model to a scene."""
        scene = Scene()
        model = scene.add_model(
            source_path="test.stl",
            name="Test",
            position=(10.0, 0.0, 0.0),
        )

        assert len(scene.models) == 1
        assert model.name == "Test"
        assert model.transform.position == (10.0, 0.0, 0.0)

    def test_add_model_auto_name(self):
        """Test that add_model uses filename as default name."""
        scene = Scene()
        model = scene.add_model(source_path="path/to/my_model.stl")

        assert model.name == "my_model"

    def test_remove_model(self):
        """Test removing a model from a scene."""
        scene = Scene()
        model = scene.add_model(source_path="test.stl")

        assert len(scene.models) == 1
        result = scene.remove_model(model.id)
        assert result is True
        assert len(scene.models) == 0

    def test_remove_model_not_found(self):
        """Test removing a non-existent model."""
        scene = Scene()
        result = scene.remove_model("nonexistent")
        assert result is False

    def test_get_model(self):
        """Test getting a model by ID."""
        scene = Scene()
        model = scene.add_model(source_path="test.stl", name="Test")

        found = scene.get_model(model.id)
        assert found is not None
        assert found.name == "Test"

    def test_get_model_not_found(self):
        """Test getting a non-existent model."""
        scene = Scene()
        found = scene.get_model("nonexistent")
        assert found is None

    def test_save_and_load(self):
        """Test saving and loading a scene."""
        scene = Scene(name="Test Scene")
        scene.add_model(
            source_path="model1.stl",
            name="Model 1",
            position=(10.0, 20.0, 30.0),
            rotation=(0.0, 45.0, 0.0),
            scale=1.5,
        )
        scene.add_model(
            source_path="model2.stl",
            name="Model 2",
        )

        with tempfile.NamedTemporaryFile(suffix=".g3scene", delete=False) as f:
            scene_file = f.name

        try:
            scene.save(scene_file)

            loaded = Scene.load(scene_file)

            assert loaded.name == "Test Scene"
            assert len(loaded.models) == 2
            assert loaded.models[0].name == "Model 1"
            assert loaded.models[0].transform.position == (10.0, 20.0, 30.0)
            assert loaded.models[0].transform.rotation == (0.0, 45.0, 0.0)
            assert loaded.models[0].transform.scale == 1.5
            assert loaded.models[1].name == "Model 2"
        finally:
            Path(scene_file).unlink()

    def test_scene_file_format(self):
        """Test that scene file is valid JSON with expected structure."""
        scene = Scene(name="JSON Test")
        scene.add_model(source_path="test.stl", position=(5.0, 0.0, 0.0))

        with tempfile.NamedTemporaryFile(suffix=".g3scene", delete=False) as f:
            scene_file = f.name

        try:
            scene.save(scene_file)

            with open(scene_file) as f:
                data = json.load(f)

            assert "name" in data
            assert "models" in data
            assert "workspace" in data
            assert data["name"] == "JSON Test"
            assert len(data["models"]) == 1
            assert data["models"][0]["transform"]["position"] == [5.0, 0.0, 0.0]
        finally:
            Path(scene_file).unlink()

    def test_default_parameters(self):
        """Test default generation parameters."""
        scene = Scene()
        assert scene.default_strategy == "surface"
        assert scene.default_point_spacing_mm == 0.1
        assert scene.default_layer_height_mm == 0.1

    def test_custom_workspace(self):
        """Test scene with custom workspace bounds."""
        scene = Scene(
            workspace=WorkspaceBounds(
                x_range=(-25.0, 25.0),
                y_range=(-25.0, 25.0),
                z_range=(0.0, 50.0),
            )
        )

        assert scene.workspace.size == (50.0, 50.0, 50.0)
