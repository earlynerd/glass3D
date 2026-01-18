"""3D transformation utilities for scene management.

Provides Transform3D class for representing position, rotation, and scale,
with conversion to 4x4 homogeneous transformation matrices.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from scipy.spatial.transform import Rotation


class Transform3D(BaseModel):
    """3D transformation: position + rotation + scale.

    Attributes:
        position: XYZ position in mm (relative to workspace center)
        rotation: XYZ Euler angles in degrees (applied in XYZ order)
        scale: Uniform scale factor
    """

    position: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="XYZ position in mm"
    )
    rotation: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="XYZ rotation in degrees (Euler angles)"
    )
    scale: float = Field(
        default=1.0,
        ge=0.01,
        le=100.0,
        description="Uniform scale factor"
    )

    model_config = {"frozen": False}

    def to_matrix(self) -> NDArray[np.float64]:
        """Convert to 4x4 homogeneous transformation matrix.

        The transformation order is: Scale -> Rotate -> Translate
        This means the matrix is built as: T @ R @ S

        Returns:
            4x4 transformation matrix
        """
        # Scale matrix
        s = np.eye(4, dtype=np.float64)
        s[0, 0] = s[1, 1] = s[2, 2] = self.scale

        # Rotation matrix (XYZ Euler angles)
        rot = Rotation.from_euler('xyz', self.rotation, degrees=True)
        r = np.eye(4, dtype=np.float64)
        r[:3, :3] = rot.as_matrix()

        # Translation matrix
        t = np.eye(4, dtype=np.float64)
        t[:3, 3] = self.position

        # Combined: T @ R @ S (applied right to left to vertices)
        return t @ r @ s

    @classmethod
    def from_matrix(cls, matrix: NDArray[np.float64]) -> Transform3D:
        """Create Transform3D from a 4x4 transformation matrix.

        Note: This assumes the matrix represents uniform scaling.
        Non-uniform scaling will be approximated.

        Args:
            matrix: 4x4 homogeneous transformation matrix

        Returns:
            Transform3D instance
        """
        # Extract translation
        position = tuple(matrix[:3, 3].tolist())

        # Extract scale (from the length of the first column of rotation part)
        scale = float(np.linalg.norm(matrix[:3, 0]))

        # Extract rotation (normalize the rotation part)
        rot_matrix = matrix[:3, :3] / scale
        rot = Rotation.from_matrix(rot_matrix)
        rotation = tuple(rot.as_euler('xyz', degrees=True).tolist())

        return cls(position=position, rotation=rotation, scale=scale)

    def apply_to_points(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply transformation to an Nx3 array of points.

        Args:
            points: Nx3 array of XYZ coordinates

        Returns:
            Transformed Nx3 array of points
        """
        matrix = self.to_matrix()

        # Convert to homogeneous coordinates (Nx4)
        ones = np.ones((len(points), 1), dtype=np.float64)
        homogeneous = np.hstack([points, ones])

        # Apply transformation
        transformed = (matrix @ homogeneous.T).T

        # Return XYZ only
        return transformed[:, :3]

    def compose(self, other: Transform3D) -> Transform3D:
        """Compose this transform with another.

        The result applies self first, then other.

        Args:
            other: Transform to apply after this one

        Returns:
            Combined Transform3D
        """
        combined_matrix = other.to_matrix() @ self.to_matrix()
        return Transform3D.from_matrix(combined_matrix)

    def inverse(self) -> Transform3D:
        """Return the inverse transformation.

        Returns:
            Inverse Transform3D
        """
        matrix = self.to_matrix()
        inv_matrix = np.linalg.inv(matrix)
        return Transform3D.from_matrix(inv_matrix)

    @classmethod
    def identity(cls) -> Transform3D:
        """Return identity transform (no transformation)."""
        return cls()

    def __repr__(self) -> str:
        return (
            f"Transform3D(pos={self.position}, "
            f"rot={self.rotation}, scale={self.scale:.2f})"
        )
