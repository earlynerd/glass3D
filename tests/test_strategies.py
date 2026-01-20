"""Tests for point cloud generation strategies."""

import numpy as np
import pytest
import trimesh

from glass3d.core.config import PointCloudParams
from glass3d.mesh.pointcloud_gen import (
    get_strategy,
    list_strategies,
    SurfaceStrategy,
    SolidStrategy,
    GrayscaleStrategy,
    ShellStrategy,
    ContourStrategy,
)

# Check if rtree is available (needed for some trimesh operations)
try:
    import rtree
    HAS_RTREE = True
except ImportError:
    HAS_RTREE = False

requires_rtree = pytest.mark.skipif(not HAS_RTREE, reason="rtree module not installed")


@pytest.fixture
def simple_box() -> trimesh.Trimesh:
    """Create a simple box mesh for testing."""
    return trimesh.creation.box(extents=[10, 10, 10])


@pytest.fixture
def default_params() -> PointCloudParams:
    """Create default point cloud params."""
    return PointCloudParams(
        point_spacing_mm=1.0,  # Coarse for fast tests
        layer_height_mm=1.0,
    )


class TestStrategyRegistry:
    """Test strategy registration and lookup."""

    def test_list_strategies(self):
        """Test that all strategies are registered."""
        strategies = list_strategies()
        strategy_names = [s["name"] for s in strategies]
        assert "surface" in strategy_names
        assert "solid" in strategy_names
        assert "grayscale" in strategy_names
        assert "shell" in strategy_names
        assert "contour" in strategy_names

    def test_get_strategy_valid(self):
        """Test getting valid strategies by name."""
        surface = get_strategy("surface")
        assert isinstance(surface, SurfaceStrategy)

        solid = get_strategy("solid")
        assert isinstance(solid, SolidStrategy)

    def test_get_strategy_invalid(self):
        """Test getting invalid strategy raises error."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("nonexistent")


class TestSurfaceStrategy:
    """Test surface point generation strategy."""

    def test_generates_points(self, simple_box, default_params):
        """Test that surface strategy generates points."""
        strategy = SurfaceStrategy()
        cloud = strategy.generate(simple_box, default_params)

        assert len(cloud) > 0
        assert cloud.points.shape[1] == 3

    @requires_rtree
    def test_points_near_surface(self, simple_box, default_params):
        """Test that points are near the mesh surface."""
        strategy = SurfaceStrategy()
        cloud = strategy.generate(simple_box, default_params)

        # Check that all points are within reasonable distance of surface
        _, distances, _ = simple_box.nearest.on_surface(cloud.points)
        assert distances.max() < default_params.point_spacing_mm * 2


class TestSolidStrategy:
    """Test solid fill point generation strategy."""

    def test_generates_points(self, simple_box, default_params):
        """Test that solid strategy generates points."""
        strategy = SolidStrategy()
        cloud = strategy.generate(simple_box, default_params)

        assert len(cloud) > 0
        assert cloud.points.shape[1] == 3

    @requires_rtree
    def test_fills_interior(self, simple_box, default_params):
        """Test that solid strategy fills interior."""
        strategy = SolidStrategy()
        cloud = strategy.generate(simple_box, default_params)

        # Check that some points are inside (not on surface)
        _, distances, _ = simple_box.nearest.on_surface(cloud.points)
        interior_points = distances > 0.1
        assert interior_points.any(), "Expected some interior points"


class TestGrayscaleStrategy:
    """Test grayscale point generation strategy."""

    @requires_rtree
    def test_generates_points(self, simple_box, default_params):
        """Test that grayscale strategy generates points."""
        strategy = GrayscaleStrategy()
        cloud = strategy.generate(simple_box, default_params)

        assert len(cloud) > 0
        assert cloud.points.shape[1] == 3

    @requires_rtree
    def test_deterministic_with_seed(self, simple_box):
        """Test that same seed produces identical output."""
        params = PointCloudParams(
            point_spacing_mm=1.0,
            layer_height_mm=1.0,
            randomization_seed=42,
        )

        strategy = GrayscaleStrategy()
        cloud1 = strategy.generate(simple_box, params)
        cloud2 = strategy.generate(simple_box, params)

        np.testing.assert_array_equal(cloud1.points, cloud2.points)

    @requires_rtree
    def test_different_without_seed(self, simple_box):
        """Test that None seed can produce different output."""
        params = PointCloudParams(
            point_spacing_mm=1.0,
            layer_height_mm=1.0,
            randomization_seed=None,
        )

        strategy = GrayscaleStrategy()
        cloud1 = strategy.generate(simple_box, params)
        cloud2 = strategy.generate(simple_box, params)

        # With None seed, results should differ (though there's a tiny chance they match)
        # We check that at least the counts differ or points differ
        # Note: This test may occasionally fail due to randomness
        if len(cloud1) == len(cloud2):
            # If same length, points should differ
            assert not np.allclose(cloud1.points, cloud2.points)

    @requires_rtree
    def test_different_seeds_differ(self, simple_box):
        """Test that different seeds produce different results."""
        params1 = PointCloudParams(
            point_spacing_mm=1.0,
            layer_height_mm=1.0,
            randomization_seed=42,
        )
        params2 = PointCloudParams(
            point_spacing_mm=1.0,
            layer_height_mm=1.0,
            randomization_seed=123,
        )

        strategy = GrayscaleStrategy()
        cloud1 = strategy.generate(simple_box, params1)
        cloud2 = strategy.generate(simple_box, params2)

        # Different seeds should produce different results
        assert not np.array_equal(cloud1.points, cloud2.points)

    @requires_rtree
    def test_has_intensities(self, simple_box, default_params):
        """Test that grayscale strategy sets intensities."""
        default_params.randomization_seed = 42
        strategy = GrayscaleStrategy()
        cloud = strategy.generate(simple_box, default_params)

        assert cloud.intensities is not None
        assert len(cloud.intensities) == len(cloud)


class TestShellStrategy:
    """Test shell/wall point generation strategy."""

    def test_generates_points(self, simple_box, default_params):
        """Test that shell strategy generates points."""
        strategy = ShellStrategy()
        cloud = strategy.generate(simple_box, default_params)

        assert len(cloud) > 0
        assert cloud.points.shape[1] == 3

    def test_shell_count_affects_output(self, simple_box):
        """Test that shell count parameter affects point count."""
        params_1_shell = PointCloudParams(
            point_spacing_mm=1.0,
            layer_height_mm=1.0,
            shell_count=1,
        )
        params_3_shells = PointCloudParams(
            point_spacing_mm=1.0,
            layer_height_mm=1.0,
            shell_count=3,
        )

        strategy = ShellStrategy()
        cloud_1 = strategy.generate(simple_box, params_1_shell)
        cloud_3 = strategy.generate(simple_box, params_3_shells)

        # More shells should produce more points
        assert len(cloud_3) > len(cloud_1)


class TestContourStrategy:
    """Test contour point generation strategy."""

    def test_generates_points(self, simple_box, default_params):
        """Test that contour strategy generates points."""
        strategy = ContourStrategy()
        cloud = strategy.generate(simple_box, default_params)

        assert len(cloud) > 0
        assert cloud.points.shape[1] == 3

    def test_has_layers(self, simple_box, default_params):
        """Test that contour strategy produces layered output."""
        strategy = ContourStrategy()
        cloud = strategy.generate(simple_box, default_params)

        assert cloud.layer_indices is not None
        assert cloud.num_layers > 1
