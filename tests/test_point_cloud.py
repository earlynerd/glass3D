"""Tests for PointCloud class."""

import numpy as np
import pytest

from glass3d.core.point_cloud import PointCloud


class TestPointCloudBasics:
    """Test basic PointCloud functionality."""
    
    def test_create_empty(self):
        """Test creating an empty point cloud."""
        cloud = PointCloud(points=np.empty((0, 3)))
        assert len(cloud) == 0
    
    def test_create_with_points(self):
        """Test creating a point cloud with data."""
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        cloud = PointCloud(points=points)
        assert len(cloud) == 4
    
    def test_invalid_shape_raises(self):
        """Test that invalid point arrays raise errors."""
        with pytest.raises(ValueError):
            PointCloud(points=np.array([[0, 0]]))  # Only 2D
    
    def test_bounds(self):
        """Test bounding box calculation."""
        points = np.array([
            [0, 0, 0],
            [10, 20, 30],
            [5, 10, 15],
        ])
        cloud = PointCloud(points=points)
        
        min_pt, max_pt = cloud.bounds
        np.testing.assert_array_equal(min_pt, [0, 0, 0])
        np.testing.assert_array_equal(max_pt, [10, 20, 30])
    
    def test_center(self):
        """Test center calculation."""
        points = np.array([
            [0, 0, 0],
            [10, 10, 10],
        ])
        cloud = PointCloud(points=points)
        
        np.testing.assert_array_equal(cloud.center, [5, 5, 5])
    
    def test_size(self):
        """Test size calculation."""
        points = np.array([
            [0, 0, 0],
            [10, 20, 30],
        ])
        cloud = PointCloud(points=points)
        
        np.testing.assert_array_equal(cloud.size, [10, 20, 30])


class TestPointCloudTransforms:
    """Test PointCloud transformations."""
    
    def test_translate(self):
        """Test translation."""
        points = np.array([[0, 0, 0], [1, 1, 1]])
        cloud = PointCloud(points=points)
        
        translated = cloud.translate([10, 20, 30])
        
        np.testing.assert_array_equal(translated.points[0], [10, 20, 30])
        np.testing.assert_array_equal(translated.points[1], [11, 21, 31])
        # Original unchanged
        np.testing.assert_array_equal(cloud.points[0], [0, 0, 0])
    
    def test_scale_uniform(self):
        """Test uniform scaling."""
        points = np.array([[1, 2, 3]])
        cloud = PointCloud(points=points)
        
        scaled = cloud.scale(2.0)
        
        np.testing.assert_array_equal(scaled.points[0], [2, 4, 6])
    
    def test_scale_per_axis(self):
        """Test per-axis scaling."""
        points = np.array([[1, 1, 1]])
        cloud = PointCloud(points=points)
        
        scaled = cloud.scale([1, 2, 3])
        
        np.testing.assert_array_equal(scaled.points[0], [1, 2, 3])
    
    def test_center_at_origin(self):
        """Test centering at origin."""
        points = np.array([
            [10, 20, 30],
            [20, 40, 60],
        ])
        cloud = PointCloud(points=points)
        
        centered = cloud.center_at_origin()
        
        np.testing.assert_array_almost_equal(centered.center, [0, 0, 0])
    
    def test_sort_by_z(self):
        """Test Z-sorting."""
        points = np.array([
            [0, 0, 30],
            [0, 0, 10],
            [0, 0, 20],
        ])
        cloud = PointCloud(points=points)
        
        sorted_asc = cloud.sort_by_z(ascending=True)
        assert sorted_asc.z[0] == 10
        assert sorted_asc.z[1] == 20
        assert sorted_asc.z[2] == 30
        
        sorted_desc = cloud.sort_by_z(ascending=False)
        assert sorted_desc.z[0] == 30
        assert sorted_desc.z[2] == 10


class TestPointCloudIndexing:
    """Test PointCloud indexing and slicing."""
    
    def test_getitem_single(self):
        """Test single item indexing."""
        points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        cloud = PointCloud(points=points)
        
        subset = cloud[1]
        assert len(subset) == 1
        np.testing.assert_array_equal(subset.points[0], [1, 1, 1])
    
    def test_getitem_slice(self):
        """Test slice indexing."""
        points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
        cloud = PointCloud(points=points)
        
        subset = cloud[1:3]
        assert len(subset) == 2
    
    def test_getitem_mask(self):
        """Test boolean mask indexing."""
        points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        cloud = PointCloud(points=points)
        
        mask = np.array([True, False, True])
        subset = cloud[mask]
        assert len(subset) == 2


class TestPointCloudLayers:
    """Test layer functionality."""
    
    def test_no_layers(self):
        """Test point cloud without layer indices."""
        points = np.array([[0, 0, 0], [1, 1, 1]])
        cloud = PointCloud(points=points)
        
        assert cloud.num_layers == 1
    
    def test_with_layers(self):
        """Test point cloud with layer indices."""
        points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        layers = np.array([0, 0, 1])
        cloud = PointCloud(points=points, layer_indices=layers)
        
        assert cloud.num_layers == 2
    
    def test_get_layer(self):
        """Test getting a specific layer."""
        points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        layers = np.array([0, 0, 1])
        cloud = PointCloud(points=points, layer_indices=layers)
        
        layer_0 = cloud.get_layer(0)
        assert len(layer_0) == 2
        
        layer_1 = cloud.get_layer(1)
        assert len(layer_1) == 1
    
    def test_iter_layers(self):
        """Test iterating over layers."""
        points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        layers = np.array([0, 1, 1])
        cloud = PointCloud(points=points, layer_indices=layers)
        
        layer_list = list(cloud.iter_layers())
        assert len(layer_list) == 2
        assert layer_list[0][0] == 0  # Layer index
        assert len(layer_list[0][1]) == 1  # Points in layer


class TestPointCloudCombining:
    """Test combining point clouds."""
    
    def test_append(self):
        """Test appending two point clouds."""
        cloud1 = PointCloud(points=np.array([[0, 0, 0]]))
        cloud2 = PointCloud(points=np.array([[1, 1, 1]]))
        
        combined = cloud1.append(cloud2)
        assert len(combined) == 2
    
    def test_concatenate(self):
        """Test concatenating multiple point clouds."""
        clouds = [
            PointCloud(points=np.array([[i, i, i]]))
            for i in range(5)
        ]
        
        combined = PointCloud.concatenate(clouds)
        assert len(combined) == 5
