"""Laser controller wrapper for Glass3D.

This module provides a safe, high-level interface to the galvoplotter library
for controlling BJJCZ galvo lasers. It handles coordinate transformation,
safety checks, and provides progress tracking.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterator

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..core.config import Glass3DConfig
    from ..core.point_cloud import PointCloud

logger = logging.getLogger(__name__)


@dataclass
class EngraveProgress:
    """Progress information during engraving."""
    
    total_points: int
    completed_points: int
    current_layer: int
    total_layers: int
    elapsed_seconds: float
    
    @property
    def percent_complete(self) -> float:
        """Percentage of points completed."""
        if self.total_points == 0:
            return 100.0
        return (self.completed_points / self.total_points) * 100
    
    @property
    def estimated_remaining_seconds(self) -> float:
        """Estimated time remaining in seconds."""
        if self.completed_points == 0:
            return 0.0
        rate = self.completed_points / self.elapsed_seconds
        remaining = self.total_points - self.completed_points
        return remaining / rate if rate > 0 else 0.0


class CoordinateTransformer:
    """Transform between model coordinates (mm) and galvo coordinates (16-bit).

    Supports both centered coordinates (0,0 at field center) and corner-origin
    coordinates (0,0 at field corner, matching slicer software).
    """

    def __init__(
        self,
        field_size_mm: tuple[float, float],
        galvo_bits: int = 16,
        offset_mm: tuple[float, float] = (0.0, 0.0),
        corner_origin: bool = True,
    ):
        """Initialize transformer.

        Args:
            field_size_mm: Physical size of galvo field (x, y) in mm
            galvo_bits: Resolution of galvo DAC (typically 16)
            offset_mm: Additional offset from field center in mm
            corner_origin: If True, input coordinates use corner-origin (0,0 at corner).
                          If False, input coordinates are centered (0,0 at field center).
        """
        self.field_size_mm = field_size_mm
        self.galvo_max = (2 ** galvo_bits) - 1
        self.galvo_center = 2 ** (galvo_bits - 1)
        self.offset_mm = offset_mm
        self.corner_origin = corner_origin

        # Conversion factors
        self.mm_to_galvo = (
            self.galvo_max / field_size_mm[0],
            self.galvo_max / field_size_mm[1],
        )
        self.galvo_to_mm = (
            field_size_mm[0] / self.galvo_max,
            field_size_mm[1] / self.galvo_max,
        )

        # Corner-origin offset (converts corner-origin to centered)
        if corner_origin:
            self._origin_offset = (
                -field_size_mm[0] / 2,
                -field_size_mm[1] / 2,
            )
        else:
            self._origin_offset = (0.0, 0.0)
    
    def mm_to_galvo_coords(self, x_mm: float, y_mm: float) -> tuple[int, int]:
        """Convert mm coordinates to galvo coordinates.

        Args:
            x_mm: X position in mm. If corner_origin=True, 0 = field corner.
                  If corner_origin=False, 0 = field center.
            y_mm: Y position in mm (same coordinate system as x_mm)

        Returns:
            Tuple of (x_galvo, y_galvo) as integers
        """
        # Convert corner-origin to centered if needed
        x_mm = x_mm + self._origin_offset[0]
        y_mm = y_mm + self._origin_offset[1]

        # Apply additional user offset
        x_mm = x_mm + self.offset_mm[0]
        y_mm = y_mm + self.offset_mm[1]

        # Convert to galvo units (centered at galvo_center)
        x_galvo = int(self.galvo_center + x_mm * self.mm_to_galvo[0])
        y_galvo = int(self.galvo_center + y_mm * self.mm_to_galvo[1])

        return x_galvo, y_galvo
    
    def galvo_to_mm_coords(self, x_galvo: int, y_galvo: int) -> tuple[float, float]:
        """Convert galvo coordinates to mm.

        Args:
            x_galvo: X galvo position
            y_galvo: Y galvo position

        Returns:
            Tuple of (x_mm, y_mm) in the same coordinate system as input
            (corner-origin if corner_origin=True, else centered)
        """
        x_mm = (x_galvo - self.galvo_center) * self.galvo_to_mm[0]
        y_mm = (y_galvo - self.galvo_center) * self.galvo_to_mm[1]

        # Remove user offset
        x_mm = x_mm - self.offset_mm[0]
        y_mm = y_mm - self.offset_mm[1]

        # Convert back from centered to corner-origin if needed
        x_mm = x_mm - self._origin_offset[0]
        y_mm = y_mm - self._origin_offset[1]

        return x_mm, y_mm
    
    def is_in_bounds(self, x_mm: float, y_mm: float, margin_mm: float = 0.0) -> bool:
        """Check if mm coordinates are within galvo field.

        Args:
            x_mm: X position in mm (in the same coordinate system as inputs)
            y_mm: Y position in mm
            margin_mm: Safety margin from edges

        Returns:
            True if position is valid
        """
        if self.corner_origin:
            # Corner-origin: valid range is [margin, field_size - margin]
            return (
                margin_mm <= x_mm <= self.field_size_mm[0] - margin_mm
                and margin_mm <= y_mm <= self.field_size_mm[1] - margin_mm
            )
        else:
            # Centered: valid range is [-half + margin, half - margin]
            half_x = (self.field_size_mm[0] / 2) - margin_mm
            half_y = (self.field_size_mm[1] / 2) - margin_mm
            return abs(x_mm) <= half_x and abs(y_mm) <= half_y
    
    def clamp_to_bounds(
        self,
        x_mm: float,
        y_mm: float,
        margin_mm: float = 0.0,
    ) -> tuple[float, float]:
        """Clamp mm coordinates to galvo field bounds.

        Args:
            x_mm: X position in mm
            y_mm: Y position in mm
            margin_mm: Safety margin from edges

        Returns:
            Clamped (x_mm, y_mm)
        """
        if self.corner_origin:
            # Corner-origin: clamp to [margin, field_size - margin]
            x_mm = max(margin_mm, min(self.field_size_mm[0] - margin_mm, x_mm))
            y_mm = max(margin_mm, min(self.field_size_mm[1] - margin_mm, y_mm))
        else:
            # Centered: clamp to [-half + margin, half - margin]
            half_x = (self.field_size_mm[0] / 2) - margin_mm
            half_y = (self.field_size_mm[1] / 2) - margin_mm
            x_mm = max(-half_x, min(half_x, x_mm))
            y_mm = max(-half_y, min(half_y, y_mm))

        return x_mm, y_mm


class LaserController:
    """High-level laser controller for subsurface engraving."""
    
    def __init__(self, config: Glass3DConfig):
        """Initialize controller.
        
        Args:
            config: Glass3D configuration
        """
        self.config = config
        self._controller = None
        self._connected = False
        self._abort_requested = False
        
        # Initialize coordinate transformer
        self.transformer = CoordinateTransformer(
            field_size_mm=config.machine.field_size_mm,
            galvo_bits=config.machine.galvo_bits,
        )
        
        # Z-axis state
        self._current_z_mm = 0.0
    
    def connect(self) -> None:
        """Connect to the laser controller."""
        if self._connected:
            return
        
        try:
            from galvo import GalvoController

            if self.config.mock_laser:
                logger.info("Using mock laser connection")
                self._controller = GalvoController(mock=True)
            elif self.config.settings_file:
                logger.info(f"Connecting with settings: {self.config.settings_file}")
                self._controller = GalvoController(
                    settings_file=str(self.config.settings_file)
                )
            else:
                logger.info("Connecting with default settings")
                self._controller = GalvoController()
            
            self._connected = True
            logger.info("Laser controller connected")
            
        except ImportError:
            raise RuntimeError(
                "galvoplotter not installed. Install with: pip install galvoplotter (imports as 'galvo')"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to laser: {e}")
    
    def disconnect(self) -> None:
        """Disconnect from the laser controller."""
        if self._controller is not None:
            try:
                self._controller.shutdown()
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            finally:
                self._controller = None
                self._connected = False
    
    @property
    def is_connected(self) -> bool:
        """Check if controller is connected."""
        return self._connected
    
    def abort(self) -> None:
        """Request abort of current operation."""
        self._abort_requested = True
        if self._controller:
            try:
                self._controller.abort()
            except Exception as e:
                logger.error(f"Error during abort: {e}")
    
    def _check_abort(self) -> None:
        """Raise exception if abort was requested."""
        if self._abort_requested:
            self._abort_requested = False
            raise InterruptedError("Operation aborted by user")
    
    def _apply_laser_params(self) -> None:
        """Apply laser parameters from config."""
        if not self._controller:
            return
        
        params = self.config.laser
        self._controller.set(
            power=params.power,
            frequency=params.frequency,
        )
    
    def set_z_position(self, z_mm: float) -> None:
        """Set the Z-axis position (focus depth).
        
        Note: Actual Z control depends on your hardware setup.
        This may need customization for your specific Z-axis control.
        
        Args:
            z_mm: Z position in mm
        """
        self._current_z_mm = z_mm
        # TODO: Implement actual Z-axis control
        # This depends on whether Z is controlled via:
        # - Separate stepper motor
        # - Controller GPIO
        # - Serial command
        logger.debug(f"Z position set to {z_mm:.3f}mm")
    
    def validate_point_cloud(self, cloud: PointCloud) -> tuple[bool, list[str]]:
        """Validate that point cloud is safe to engrave.

        Args:
            cloud: Point cloud to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        margin = self.config.machine.edge_margin_mm
        field_size = self.config.machine.field_size_mm

        # Check XY bounds (corner-origin: valid range is [margin, field_size - margin])
        min_pt, max_pt = cloud.bounds
        x_min_valid = margin
        x_max_valid = field_size[0] - margin
        y_min_valid = margin
        y_max_valid = field_size[1] - margin

        if min_pt[0] < x_min_valid or max_pt[0] > x_max_valid:
            issues.append(
                f"X coordinates out of bounds (valid range: {x_min_valid:.1f} to {x_max_valid:.1f}mm, "
                f"got: {min_pt[0]:.1f} to {max_pt[0]:.1f}mm)"
            )

        if min_pt[1] < y_min_valid or max_pt[1] > y_max_valid:
            issues.append(
                f"Y coordinates out of bounds (valid range: {y_min_valid:.1f} to {y_max_valid:.1f}mm, "
                f"got: {min_pt[1]:.1f} to {max_pt[1]:.1f}mm)"
            )
        
        # Check Z bounds
        z_min, z_max = self.config.machine.z_range_mm
        if min_pt[2] < z_min:
            issues.append(f"Z minimum {min_pt[2]:.1f}mm below range (min {z_min:.1f}mm)")
        if max_pt[2] > z_max:
            issues.append(f"Z maximum {max_pt[2]:.1f}mm above range (max {z_max:.1f}mm)")
        
        # Check point spacing
        # (simplified - full check would use spatial data structure)
        if len(cloud) > 1:
            # Sample-based check
            sample_size = min(1000, len(cloud))
            indices = np.random.choice(len(cloud), sample_size, replace=False)
            sample = cloud.points[indices]
            
            # Check distances between consecutive sampled points
            diffs = np.diff(sample, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            min_dist = dists.min()
            
            if min_dist < self.config.material.min_point_spacing_mm:
                issues.append(
                    f"Point spacing {min_dist:.3f}mm below minimum "
                    f"({self.config.material.min_point_spacing_mm:.3f}mm)"
                )
        
        return len(issues) == 0, issues
    
    def engrave_point_cloud(
        self,
        cloud: PointCloud,
        progress_callback: Callable[[EngraveProgress], None] | None = None,
        dry_run: bool = False,
    ) -> None:
        """Engrave a point cloud.
        
        Points are engraved in order (caller should pre-sort).
        For SSLE, points should be sorted bottom-up (ascending Z).
        
        Args:
            cloud: Point cloud to engrave
            progress_callback: Optional callback for progress updates
            dry_run: If True, simulate without firing laser
        """
        if not self._connected:
            self.connect()
        
        # Validate
        if self.config.engrave.validate_bounds:
            valid, issues = self.validate_point_cloud(cloud)
            if not valid:
                raise ValueError(f"Point cloud validation failed:\n" + "\n".join(issues))
        
        # Reset abort flag
        self._abort_requested = False
        
        # Apply parameters
        self._apply_laser_params()
        
        total_points = len(cloud)
        completed = 0
        start_time = time.time()
        
        chunk_size = self.config.engrave.chunk_size
        dwell_ms = self.config.laser.point_dwell_ms
        
        logger.info(f"Starting engrave: {total_points} points")
        
        try:
            if dry_run:
                context = self._controller.lighting()
            else:
                context = self._controller.marking()
            
            with context as c:
                current_z = None
                points_since_pause = 0
                
                for i, point in enumerate(cloud.points):
                    self._check_abort()
                    
                    x_mm, y_mm, z_mm = point
                    
                    # Update Z if changed
                    if current_z != z_mm:
                        # In a real implementation, this would move the Z axis
                        self.set_z_position(z_mm)
                        current_z = z_mm
                    
                    # Convert to galvo coordinates
                    x_galvo, y_galvo = self.transformer.mm_to_galvo_coords(x_mm, y_mm)
                    
                    # Move to position and fire
                    c.goto(x_galvo, y_galvo)
                    
                    if dry_run:
                        c.light(x_galvo, y_galvo)
                        time.sleep(0.0001)  # Brief delay for preview
                    else:
                        c.dwell(dwell_ms)
                    
                    completed += 1
                    points_since_pause += 1
                    
                    # Thermal pause if needed
                    if points_since_pause >= self.config.material.max_continuous_points:
                        logger.debug("Thermal pause")
                        time.sleep(self.config.material.thermal_pause_ms / 1000)
                        points_since_pause = 0
                    
                    # Progress callback
                    if progress_callback and (i % chunk_size == 0 or i == total_points - 1):
                        progress = EngraveProgress(
                            total_points=total_points,
                            completed_points=completed,
                            current_layer=int(cloud.layer_indices[i]) if cloud.layer_indices is not None else 0,
                            total_layers=cloud.num_layers,
                            elapsed_seconds=time.time() - start_time,
                        )
                        progress_callback(progress)
            
            # Wait for completion
            self._controller.wait_for_machine_idle()
            
            elapsed = time.time() - start_time
            logger.info(f"Engrave complete: {total_points} points in {elapsed:.1f}s")
            
        except InterruptedError:
            logger.warning("Engrave aborted by user")
            raise
        except Exception as e:
            logger.error(f"Engrave failed: {e}")
            raise
    
    def preview_bounds(self, cloud: PointCloud) -> None:
        """Preview the bounding box with the red dot laser.
        
        Traces the XY bounding box at the min and max Z levels.
        
        Args:
            cloud: Point cloud to preview bounds of
        """
        if not self._connected:
            self.connect()
        
        min_pt, max_pt = cloud.bounds
        
        # Corner coordinates
        corners = [
            (min_pt[0], min_pt[1]),
            (max_pt[0], min_pt[1]),
            (max_pt[0], max_pt[1]),
            (min_pt[0], max_pt[1]),
            (min_pt[0], min_pt[1]),  # Close the loop
        ]
        
        with self._controller.lighting() as c:
            # Move to first corner
            x, y = self.transformer.mm_to_galvo_coords(*corners[0])
            c.dark(x, y)
            
            # Trace rectangle
            for corner in corners[1:]:
                x, y = self.transformer.mm_to_galvo_coords(*corner)
                c.light(x, y)
        
        self._controller.wait_for_machine_idle()
    
    def __enter__(self) -> LaserController:
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()
