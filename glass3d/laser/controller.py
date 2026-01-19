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
from types import TracebackType
from typing import TYPE_CHECKING, Callable, Iterator

import numpy as np
from numpy.typing import NDArray

from ..core.config import LensCorrection

if TYPE_CHECKING:
    from ..core.config import Glass3DConfig
    from ..core.point_cloud import PointCloud

logger = logging.getLogger(__name__)


def _apply_lens_correction(
    x_norm: float,
    y_norm: float,
    correction: LensCorrection,
) -> tuple[float, float]:
    """Apply lens distortion correction to normalized coordinates.

    Coordinates should be in range [-1, 1] representing the galvo field.
    Corrections are applied in order: mirror → rotation → scale → skew → trapezoid → bulge

    Args:
        x_norm: Normalized X coordinate (-1 to 1)
        y_norm: Normalized Y coordinate (-1 to 1)
        correction: Lens correction parameters

    Returns:
        Corrected (x_norm, y_norm)
    """
    x, y = x_norm, y_norm

    # 1. Mirror
    if correction.mirror_x:
        x = -x
    if correction.mirror_y:
        y = -y

    # 2. Field rotation
    if correction.field_angle_deg != 0:
        angle_rad = np.radians(correction.field_angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        x, y = x * cos_a - y * sin_a, x * sin_a + y * cos_a

    # 3. Scale (per-axis calibration)
    x = x * correction.x_axis.scale
    y = y * correction.y_axis.scale

    # 4. Sign (axis direction)
    x = x * correction.x_axis.sign
    y = y * correction.y_axis.sign

    # 5. Skew correction (parallelogram distortion)
    # Skew affects the perpendicularity of axes
    if correction.x_axis.skew != 1.0:
        x = x + (correction.x_axis.skew - 1.0) * y
    if correction.y_axis.skew != 1.0:
        y = y + (correction.y_axis.skew - 1.0) * x

    # 6. Trapezoid correction (keystone)
    # Applies position-dependent scaling
    if correction.x_axis.trapezoid != 1.0:
        trap_factor = 1.0 + (correction.x_axis.trapezoid - 1.0) * y
        x = x * trap_factor
    if correction.y_axis.trapezoid != 1.0:
        trap_factor = 1.0 + (correction.y_axis.trapezoid - 1.0) * x
        y = y * trap_factor

    # 7. Bulge correction (barrel/pincushion distortion)
    # Radial distortion based on distance from center
    r_squared = x * x + y * y
    if correction.x_axis.bulge != 1.0:
        bulge_factor = 1.0 + (correction.x_axis.bulge - 1.0) * r_squared
        x = x * bulge_factor
    if correction.y_axis.bulge != 1.0:
        bulge_factor = 1.0 + (correction.y_axis.bulge - 1.0) * r_squared
        y = y * bulge_factor

    return x, y


def optimize_path_nearest_neighbor(
    points: NDArray[np.float64],
    start_point: tuple[float, float] | None = None,
) -> NDArray[np.intp]:
    """Optimize point order using nearest-neighbor algorithm.

    This is a greedy approach that always moves to the closest unvisited point.
    Not globally optimal but fast O(n²) and produces reasonable results.

    Args:
        points: Nx3 array of XYZ coordinates (only XY used for distance)
        start_point: Optional (x, y) to start from. If None, starts from first point.

    Returns:
        Array of indices representing the optimized visit order
    """
    n = len(points)
    if n <= 2:
        return np.arange(n)

    # Work with XY coordinates only
    xy = points[:, :2]

    visited = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=np.intp)

    # Find starting point
    if start_point is not None:
        # Start from point closest to start_point
        start_xy = np.array(start_point)
        dists = np.sum((xy - start_xy) ** 2, axis=1)
        current = int(np.argmin(dists))
    else:
        current = 0

    order[0] = current
    visited[current] = True

    for i in range(1, n):
        # Find nearest unvisited point
        current_xy = xy[current]
        dists = np.sum((xy - current_xy) ** 2, axis=1)
        dists[visited] = np.inf  # Exclude visited points

        current = int(np.argmin(dists))
        order[i] = current
        visited[current] = True

    return order


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

    Optionally applies lens distortion correction when LensCorrection is provided.
    """

    def __init__(
        self,
        field_size_mm: tuple[float, float],
        galvo_bits: int = 16,
        offset_mm: tuple[float, float] = (0.0, 0.0),
        corner_origin: bool = True,
        lens_correction: LensCorrection | None = None,
    ):
        """Initialize transformer.

        Args:
            field_size_mm: Physical size of galvo field (x, y) in mm
            galvo_bits: Resolution of galvo DAC (typically 16)
            offset_mm: Additional offset from field center in mm
            corner_origin: If True, input coordinates use corner-origin (0,0 at corner).
                          If False, input coordinates are centered (0,0 at field center).
            lens_correction: Optional lens distortion correction parameters
        """
        self.field_size_mm = field_size_mm
        self.galvo_max = (2 ** galvo_bits) - 1
        self.galvo_center = 2 ** (galvo_bits - 1)
        self.offset_mm = offset_mm
        self.corner_origin = corner_origin
        self.lens_correction = lens_correction

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

        # Apply lens correction if enabled
        if self.lens_correction is not None and self.lens_correction.enabled:
            # Apply field offset from lens correction
            x_mm = x_mm + self.lens_correction.field_offset_mm[0]
            y_mm = y_mm + self.lens_correction.field_offset_mm[1]

            # Normalize to [-1, 1] range for distortion correction
            half_x = self.field_size_mm[0] / 2
            half_y = self.field_size_mm[1] / 2
            x_norm = x_mm / half_x
            y_norm = y_mm / half_y

            # Apply distortion correction
            x_norm, y_norm = _apply_lens_correction(x_norm, y_norm, self.lens_correction)

            # Convert back to mm
            x_mm = x_norm * half_x
            y_mm = y_norm * half_y

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
        
        # Initialize coordinate transformer with lens correction if configured
        lens_correction = config.machine.lens_correction
        if lens_correction.enabled:
            logger.info("Lens correction enabled")

        self.transformer = CoordinateTransformer(
            field_size_mm=config.machine.field_size_mm,
            galvo_bits=config.machine.galvo_bits,
            lens_correction=lens_correction if lens_correction.enabled else None,
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
        """Apply laser and timing parameters from config."""
        if not self._controller:
            return

        laser = self.config.laser
        speed = self.config.speed

        # Set laser parameters
        self._controller.set(
            power=laser.power,
            frequency=laser.frequency,
        )

        # Set speed parameters
        self._controller.travel_speed = speed.travel_speed
        self._controller.mark_speed = speed.mark_speed

        # Set timing delays (galvo settling and laser timing compensation)
        self._controller.delay_jump_short = speed.jump_delay_min
        self._controller.delay_jump_long = speed.jump_delay_max
        self._controller.delay_laser_on = speed.laser_on_delay
        self._controller.delay_laser_off = speed.laser_off_delay
        self._controller.delay_polygon = speed.polygon_delay

        logger.debug(
            f"Applied timing: travel={speed.travel_speed}mm/s, "
            f"jump_delay={speed.jump_delay_min}-{speed.jump_delay_max}µs, "
            f"laser_on/off={speed.laser_on_delay}/{speed.laser_off_delay}µs"
        )
    
    def _mm_to_z_steps(self, z_mm: float) -> int:
        """Convert Z position in mm to stepper steps.

        Args:
            z_mm: Z position in mm

        Returns:
            Position in steps (integer)
        """
        return int(z_mm * self.config.machine.z_steps_per_mm)

    def set_z_position(self, z_mm: float, wait: bool = True) -> None:
        """Set the Z-axis position (focus depth).

        Moves the Z-axis stepper motor to the specified position using
        the BJJCZ controller's auxiliary axis control.

        Args:
            z_mm: Z position in mm (absolute position)
            wait: If True, block until movement completes
        """
        if not self._controller:
            logger.warning("Cannot set Z position: controller not connected")
            return

        # Validate Z is within range
        z_min, z_max = self.config.machine.z_range_mm
        if z_mm < z_min or z_mm > z_max:
            raise ValueError(
                f"Z position {z_mm:.2f}mm out of range [{z_min:.1f}, {z_max:.1f}]mm"
            )

        # Convert mm to steps
        z_steps = self._mm_to_z_steps(z_mm)

        # Get motion parameters from config
        min_speed = self.config.machine.z_axis_min_speed
        max_speed = self.config.machine.z_axis_max_speed
        acc_time = self.config.machine.z_axis_acc_time

        logger.debug(
            f"Moving Z to {z_mm:.3f}mm ({z_steps} steps), "
            f"speed: {min_speed}-{max_speed}, acc: {acc_time}ms"
        )

        # Use galvoplotter's rotary/axis control
        # This sets motion params, converts position, moves, and waits
        self._controller.set_axis_motion_param(min_speed & 0xFFFF, max_speed & 0xFFFF)
        self._controller.set_axis_origin_param(acc_time)

        # Convert position to 32-bit format (handle negative values)
        pos = z_steps if z_steps >= 0 else -z_steps + 0x80000000
        p1 = (pos >> 16) & 0xFFFF
        p0 = pos & 0xFFFF
        self._controller.move_axis_to(p0, p1)

        if wait:
            self._controller.wait_axis()

        self._current_z_mm = z_mm
        logger.debug(f"Z position set to {z_mm:.3f}mm")

    def z_home(self, wait: bool = True) -> None:
        """Home the Z-axis to its origin position.

        This sends the axis to its home/origin position as configured
        in the BJJCZ controller.

        Args:
            wait: If True, block until homing completes
        """
        if not self._controller:
            logger.warning("Cannot home Z: controller not connected")
            return

        logger.info("Homing Z-axis...")
        self._controller.axis_go_origin()

        if wait:
            self._controller.wait_axis()

        self._current_z_mm = 0.0
        logger.info("Z-axis homed")

    def get_z_position(self) -> float:
        """Get the current Z-axis position from the controller.

        Returns:
            Current Z position in mm
        """
        if not self._controller:
            logger.warning("Cannot get Z position: controller not connected")
            return self._current_z_mm

        # Read position from controller (returns tuple with position data)
        result = self._controller.get_axis_pos(0)
        if result and len(result) >= 3:
            # Position is split across result[1] and result[2]
            pos = (result[1] << 16) | result[2]
            # Handle negative values (sign bit at 0x80000000)
            if pos >= 0x80000000:
                pos = -(pos - 0x80000000)
            z_mm = pos / self.config.machine.z_steps_per_mm
            self._current_z_mm = z_mm
            return z_mm

        return self._current_z_mm

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

        assert self._controller is not None, "Controller not initialized after connect()"

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

            optimize_xy = self.config.engrave.optimize_path
            total_layers = cloud.num_layers

            with context as c:
                points_since_pause = 0
                last_xy: tuple[float, float] | None = None

                # Iterate layer by layer (sorted by Z ascending)
                for layer_idx, layer_cloud in cloud.iter_layers():
                    self._check_abort()

                    layer_points = layer_cloud.points
                    if len(layer_points) == 0:
                        continue

                    # Move Z axis once at start of layer
                    # Use mean Z of layer to handle any floating-point variations
                    layer_z = float(np.mean(layer_points[:, 2]))
                    self.set_z_position(layer_z)

                    # Wait for mechanical settling (brake disengage, motor movement, vibration)
                    z_settle_ms = self.config.machine.z_axis_settle_ms
                    if z_settle_ms > 0:
                        time.sleep(z_settle_ms / 1000.0)

                    logger.debug(f"Layer {layer_idx}: Z={layer_z:.3f}mm, {len(layer_points)} points")

                    # Optimize XY path within this layer
                    if optimize_xy and len(layer_points) > 2:
                        order = optimize_path_nearest_neighbor(layer_points, last_xy)
                        layer_points = layer_points[order]

                    # Engrave all points in this layer
                    for point in layer_points:
                        self._check_abort()

                        x_mm, y_mm = point[0], point[1]

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
                        if progress_callback and (completed % chunk_size == 0 or completed == total_points):
                            progress = EngraveProgress(
                                total_points=total_points,
                                completed_points=completed,
                                current_layer=layer_idx,
                                total_layers=total_layers,
                                elapsed_seconds=time.time() - start_time,
                            )
                            progress_callback(progress)

                    # Remember last position for next layer's path optimization
                    if len(layer_points) > 0:
                        last_xy = (float(layer_points[-1, 0]), float(layer_points[-1, 1]))

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

        assert self._controller is not None, "Controller not initialized after connect()"

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
    
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit."""
        self.disconnect()
