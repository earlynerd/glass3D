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
    def from_matrix(
        cls,
        matrix: NDArray[np.float64],
        scale_tolerance: float = 0.01,
    ) -> Transform3D:
        """Create Transform3D from a 4x4 transformation matrix.

        This method only supports matrices with uniform scaling and proper
        rotations (no reflections). For matrices with non-uniform scaling,
        reflections, or other unsupported transforms, a ValueError is raised.

        Args:
            matrix: 4x4 homogeneous transformation matrix
            scale_tolerance: Relative tolerance for uniform scale check

        Returns:
            Transform3D instance

        Raises:
            ValueError: If matrix contains reflection, non-uniform scaling,
                       zero scaling, or other unsupported transforms
        """
        rot_part = matrix[:3, :3]

        # Check for reflection (negative determinant)
        det = np.linalg.det(rot_part)
        if det < 0:
            raise ValueError(
                "Matrix contains a reflection (negative determinant). "
                "Transform3D only supports proper rotations."
            )

        # Extract scale from each column
        scales = np.array([np.linalg.norm(rot_part[:, i]) for i in range(3)])

        # Check for zero scale
        if np.any(scales < 1e-10):
            raise ValueError(
                f"Matrix contains zero or near-zero scale: {scales}. "
                "Transform3D requires positive scaling."
            )

        # Check for uniform scale
        mean_scale = np.mean(scales)
        if not np.allclose(scales, mean_scale, rtol=scale_tolerance):
            raise ValueError(
                f"Matrix contains non-uniform scaling: X={scales[0]:.4f}, "
                f"Y={scales[1]:.4f}, Z={scales[2]:.4f}. "
                "Transform3D only supports uniform scaling."
            )

        scale = float(mean_scale)

        # Extract rotation (normalize the rotation part)
        rot_matrix = rot_part / scale
        rot = Rotation.from_matrix(rot_matrix)
        rotation = tuple(rot.as_euler('xyz', degrees=True).tolist())

        # Extract translation
        position = tuple(matrix[:3, 3].tolist())

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
