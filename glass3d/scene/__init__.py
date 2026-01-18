"""Scene management for multi-model arrangement.

This module provides data structures for managing multiple 3D models
within a workspace, including positioning, rotation, and scaling.
"""

from .transform import Transform3D
from .scene import Scene, ModelPlacement, WorkspaceBounds

__all__ = [
    "Transform3D",
    "Scene",
    "ModelPlacement",
    "WorkspaceBounds",
]
