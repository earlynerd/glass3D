"""Configuration management for Glass3D.

This module defines all configuration models using Pydantic for validation.
Configuration can be loaded from JSON files or constructed programmatically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class LaserParams(BaseModel):
    """Laser firing parameters.
    
    These control how the laser fires when creating each point.
    Values need to be tuned for specific materials.
    """
    
    power: float = Field(default=50.0, ge=0, le=100, description="Laser power percentage")
    frequency: float = Field(default=30.0, ge=1, le=200, description="Pulse frequency in kHz")
    pulse_width: float = Field(default=4.0, ge=0.1, le=500, description="Pulse width in microseconds")
    q_switch_period: float = Field(default=50.0, ge=1, description="Q-switch period in nanoseconds")
    
    # Dwell time per point - critical for SSLE
    point_dwell_ms: float = Field(default=0.5, ge=0.1, le=100, description="Time to dwell at each point in ms")


class SpeedParams(BaseModel):
    """Movement speed parameters."""

    mark_speed: float = Field(default=500.0, ge=1, le=10000, description="Speed while marking in mm/s")
    travel_speed: float = Field(default=2000.0, ge=1, le=10000, description="Speed while traveling in mm/s")

    # Jump delays (settling time after galvo move completes)
    jump_delay_min: float = Field(default=200.0, ge=0, description="Min delay after short jumps (µs)")
    jump_delay_max: float = Field(default=400.0, ge=0, description="Max delay after long jumps (µs)")
    jump_distance_threshold: float = Field(default=10.0, ge=0, description="Distance threshold for long jump delay (mm)")

    # Laser timing compensation
    laser_on_delay: float = Field(default=100.0, ge=0, description="Delay before laser fires (µs)")
    laser_off_delay: float = Field(default=100.0, ge=0, description="Delay after laser stops (µs)")
    polygon_delay: float = Field(default=100.0, ge=0, description="Delay at polygon corners (µs)")


class GalvoAxisCorrection(BaseModel):
    """Lens correction parameters for a single galvo axis.

    These parameters compensate for optical distortions in the F-theta lens.
    Values are typically calibrated using LightBurn or EzCad software.
    """

    scale: float = Field(default=1.0, description="Scale factor for this axis")
    bulge: float = Field(default=1.0, description="Barrel/pincushion distortion (>1 = pincushion)")
    trapezoid: float = Field(default=1.0, description="Keystone correction factor")
    skew: float = Field(default=1.0, description="Parallelogram distortion correction")
    sign: int = Field(default=1, ge=-1, le=1, description="Axis direction (+1 or -1)")


class LensCorrection(BaseModel):
    """Lens distortion correction parameters.

    F-theta lenses introduce various distortions that need compensation:
    - Barrel/pincushion: radial distortion from lens curvature
    - Trapezoid: keystone effect from non-perpendicular mounting
    - Skew: parallelogram distortion from mirror alignment
    - Scale: calibration for actual vs expected field size

    These can be imported from LightBurn device exports (.lbzip files).
    """

    enabled: bool = Field(default=False, description="Enable lens correction")

    # Per-axis correction (LightBurn uses Galvo_1 for Y, Galvo_2 for X typically)
    x_axis: GalvoAxisCorrection = Field(default_factory=GalvoAxisCorrection)
    y_axis: GalvoAxisCorrection = Field(default_factory=GalvoAxisCorrection)

    # Field rotation and offset
    field_angle_deg: float = Field(default=0.0, description="Field rotation in degrees")
    field_offset_mm: tuple[float, float] = Field(
        default=(0.0, 0.0),
        description="Field offset from center (x, y) in mm"
    )

    # Mirror settings
    mirror_x: bool = Field(default=False, description="Mirror X axis")
    mirror_y: bool = Field(default=False, description="Mirror Y axis")


class MachineParams(BaseModel):
    """Physical machine parameters."""

    # Galvo field parameters
    field_size_mm: tuple[float, float] = Field(
        default=(110.0, 110.0),
        description="Physical size of galvo field in mm (x, y)"
    )
    galvo_bits: int = Field(default=16, description="Resolution of galvo DAC")

    # Lens correction
    lens_correction: LensCorrection = Field(
        default_factory=LensCorrection,
        description="F-theta lens distortion correction"
    )
    
    # Z-axis parameters
    z_range_mm: tuple[float, float] = Field(
        default=(0.0, 100.0),
        description="Z-axis range in mm (min, max)"
    )
    z_steps_per_mm: float = Field(default=100.0, ge=1, description="Z-axis steps per mm")
    z_axis_min_speed: int = Field(default=100, ge=1, description="Z-axis minimum speed (steps/sec)")
    z_axis_max_speed: int = Field(default=5000, ge=1, description="Z-axis maximum speed (steps/sec)")
    z_axis_acc_time: int = Field(default=100, ge=1, description="Z-axis acceleration time (ms)")
    z_axis_settle_ms: int = Field(default=500, ge=0, description="Delay after Z move for mechanical settling (ms)")

    # Safety margins
    edge_margin_mm: float = Field(default=2.0, ge=0, description="Safety margin from field edges")
    
    @property
    def galvo_max(self) -> int:
        """Maximum galvo coordinate value."""
        return int((2 ** self.galvo_bits) - 1)

    @property
    def galvo_center(self) -> int:
        """Center galvo coordinate value."""
        return int(2 ** (self.galvo_bits - 1))
    
    @property
    def mm_per_galvo_unit(self) -> tuple[float, float]:
        """Conversion factor from galvo units to mm."""
        return (
            self.field_size_mm[0] / self.galvo_max,
            self.field_size_mm[1] / self.galvo_max,
        )


class MaterialParams(BaseModel):
    """Material-specific parameters for glass/crystal."""
    
    name: str = Field(default="K9 Glass", description="Material name")
    refractive_index: float = Field(default=1.5168, ge=1.0, le=3.0, description="Refractive index")
    
    # Recommended laser parameters for this material
    recommended_power: tuple[float, float] = Field(
        default=(40.0, 60.0),
        description="Recommended power range (min, max)"
    )
    
    # Point spacing limits
    min_point_spacing_mm: float = Field(
        default=0.05,
        ge=0.01,
        description="Minimum spacing to avoid clashing"
    )
    max_point_spacing_mm: float = Field(
        default=0.3,
        description="Maximum spacing before image is too sparse"
    )
    
    # Thermal parameters
    max_continuous_points: int = Field(
        default=10000,
        ge=100,
        description="Max points before thermal pause"
    )
    thermal_pause_ms: float = Field(
        default=1000.0,
        ge=0,
        description="Pause duration for thermal management"
    )


class PointCloudParams(BaseModel):
    """Parameters for point cloud generation."""
    
    # Point spacing
    point_spacing_mm: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Base spacing between points"
    )
    
    # Layer parameters
    layer_height_mm: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Height between Z layers"
    )
    
    # Generation strategy
    strategy: Literal["surface", "solid", "grayscale", "contour", "shell"] = Field(
        default="surface",
        description="Point generation strategy"
    )

    # Surface strategy options
    surface_offset_mm: float = Field(
        default=0.0,
        description="Offset from actual surface (negative = inside)"
    )

    # Shell strategy options
    shell_count: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of shells/walls (1 = surface only)"
    )
    shell_spacing_mm: float = Field(
        default=0.15,
        ge=0.05,
        le=2.0,
        description="Distance between shells in mm"
    )

    # Solid strategy options
    solid_density: float = Field(
        default=1.0,
        ge=0.1,
        le=2.0,
        description="Density multiplier for solid fill"
    )

    # Randomization
    randomization_seed: int | None = Field(
        default=None,
        description="Seed for random sampling in strategies. None = non-deterministic."
    )


class EngraveParams(BaseModel):
    """Parameters for the engraving process."""
    
    # Positioning
    center_in_field: bool = Field(default=True, description="Center model in galvo field")
    z_offset_mm: float = Field(default=5.0, description="Offset from glass surface to start")
    
    # Ordering
    engrave_direction: Literal["bottom_up", "top_down"] = Field(
        default="bottom_up",
        description="Direction to engrave layers (bottom_up recommended)"
    )
    optimize_path: bool = Field(default=True, description="Optimize point order within layers")
    
    # Progress
    chunk_size: int = Field(default=1000, ge=100, description="Points per progress chunk")
    
    # Safety
    dry_run: bool = Field(default=False, description="Preview without firing laser")
    validate_bounds: bool = Field(default=True, description="Validate all points in bounds")


class Glass3DConfig(BaseModel):
    """Main configuration container."""
    
    laser: LaserParams = Field(default_factory=LaserParams)
    speed: SpeedParams = Field(default_factory=SpeedParams)
    machine: MachineParams = Field(default_factory=MachineParams)
    material: MaterialParams = Field(default_factory=MaterialParams)
    point_cloud: PointCloudParams = Field(default_factory=PointCloudParams)
    engrave: EngraveParams = Field(default_factory=EngraveParams)
    
    # Connection settings
    settings_file: Path | None = Field(
        default=None,
        description="Path to galvoplotter settings JSON"
    )
    mock_laser: bool = Field(default=False, description="Use mock connection for testing")
    
    @classmethod
    def from_file(cls, path: Path | str) -> Glass3DConfig:
        """Load configuration from a JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls.model_validate(data)
    
    def to_file(self, path: Path | str) -> None:
        """Save configuration to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)
    
    @classmethod
    def default(cls) -> Glass3DConfig:
        """Create a default configuration."""
        return cls()
    
    @classmethod
    def for_material(cls, material_name: str) -> Glass3DConfig:
        """Create configuration with material-specific defaults."""
        materials = {
            "k9": MaterialParams(
                name="K9 Borosilicate Glass",
                refractive_index=1.5168,
                recommended_power=(40.0, 60.0),
                min_point_spacing_mm=0.05,
            ),
            "bk7": MaterialParams(
                name="Schott BK7",
                refractive_index=1.5168,
                recommended_power=(35.0, 55.0),
                min_point_spacing_mm=0.05,
            ),
            "fused_silica": MaterialParams(
                name="Fused Silica",
                refractive_index=1.458,
                recommended_power=(50.0, 70.0),
                min_point_spacing_mm=0.04,
            ),
        }
        
        material_key = material_name.lower().replace(" ", "_").replace("-", "_")
        if material_key not in materials:
            raise ValueError(f"Unknown material: {material_name}. Known: {list(materials.keys())}")
        
        return cls(material=materials[material_key])
