"""Glass3D - Subsurface laser engraving software.

A Python application for creating 3D subsurface laser engravings (SSLE)
in glass/crystal blocks using BJJCZ galvo lasers.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .core.config import Glass3DConfig
from .core.point_cloud import PointCloud
from .mesh.loader import MeshLoader, load_mesh
from .mesh.slicer import MeshSlicer
from .mesh.pointcloud_gen import get_strategy, list_strategies
from .laser.controller import LaserController

__all__ = [
    "Glass3DConfig",
    "PointCloud",
    "MeshLoader",
    "load_mesh",
    "MeshSlicer",
    "get_strategy",
    "list_strategies",
    "LaserController",
]
