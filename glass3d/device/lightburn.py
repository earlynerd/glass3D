"""LightBurn device import for Glass3D.

Parses LightBurn device export files (.lbzip) to extract lens correction
parameters and machine settings.
"""

from __future__ import annotations

import json
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.config import (
    GalvoAxisCorrection,
    Glass3DConfig,
    LensCorrection,
    MachineParams,
    LaserParams,
    SpeedParams,
)

logger = logging.getLogger(__name__)


@dataclass
class LightBurnDevice:
    """Parsed LightBurn device settings."""

    name: str
    device_type: str
    width_mm: float
    height_mm: float

    # Galvo 1 correction (typically Y axis)
    galvo_1_scale: float
    galvo_1_bulge: float
    galvo_1_trapezoid: float
    galvo_1_skew: float
    galvo_1_sign: int
    galvo_1_is_x: bool

    # Galvo 2 correction (typically X axis)
    galvo_2_scale: float
    galvo_2_bulge: float
    galvo_2_trapezoid: float
    galvo_2_skew: float
    galvo_2_sign: int

    # Field settings
    field_angle: float
    field_offset_x: float
    field_offset_y: float
    mirror_x: bool
    mirror_y: bool

    # Speed/timing defaults
    max_speed: float
    default_jump_speed: float
    frame_speed: float

    # Laser settings
    laser_min_freq: float
    laser_max_freq: float

    @classmethod
    def from_lbdev(cls, data: dict[str, Any]) -> LightBurnDevice:
        """Parse from .lbdev JSON data."""
        device_list = data.get("DeviceList", [])
        if not device_list:
            raise ValueError("No devices found in LightBurn export")

        device = device_list[0]
        settings = device.get("Settings", {})

        return cls(
            name=device.get("DisplayName", "Unknown"),
            device_type=device.get("Name", "Unknown"),
            width_mm=float(device.get("Width", 110)),
            height_mm=float(device.get("Height", 110)),
            # Galvo 1 (check Galvo_1_is_X to determine axis mapping)
            galvo_1_scale=float(settings.get("Galvo_1_Scale", 1.0)),
            galvo_1_bulge=float(settings.get("Galvo_1_Bulge", 1.0)),
            galvo_1_trapezoid=float(settings.get("Galvo_1_Trapezoid", 1.0)),
            galvo_1_skew=float(settings.get("Galvo_1_Skew", 1.0)),
            galvo_1_sign=int(settings.get("Galvo_1_Sign", 1)),
            galvo_1_is_x=bool(settings.get("Galvo_1_is_X", False)),
            # Galvo 2
            galvo_2_scale=float(settings.get("Galvo_2_Scale", 1.0)),
            galvo_2_bulge=float(settings.get("Galvo_2_Bulge", 1.0)),
            galvo_2_trapezoid=float(settings.get("Galvo_2_Trapezoid", 1.0)),
            galvo_2_skew=float(settings.get("Galvo_2_Skew", 1.0)),
            galvo_2_sign=int(settings.get("Galvo_2_Sign", 1)),
            # Field settings
            field_angle=float(settings.get("FieldAngle", 0.0)),
            field_offset_x=float(settings.get("FieldOffsetX", 0.0)),
            field_offset_y=float(settings.get("FieldOffsetY", 0.0)),
            mirror_x=bool(device.get("MirrorX", False)),
            mirror_y=bool(device.get("MirrorY", False)),
            # Speeds
            max_speed=float(settings.get("MaxSpeed", 7000)),
            default_jump_speed=float(settings.get("Default_JumpSpeed", 4000)),
            frame_speed=float(settings.get("FrameSpeed", 4000)),
            # Laser
            laser_min_freq=float(settings.get("Laser_MinFreq", 1)),
            laser_max_freq=float(settings.get("Laser_MaxFreq", 80)),
        )

    def to_lens_correction(self) -> LensCorrection:
        """Convert to Glass3D LensCorrection config."""
        # Map galvo axes based on Galvo_1_is_X setting
        if self.galvo_1_is_x:
            x_axis = GalvoAxisCorrection(
                scale=self.galvo_1_scale,
                bulge=self.galvo_1_bulge,
                trapezoid=self.galvo_1_trapezoid,
                skew=self.galvo_1_skew,
                sign=self.galvo_1_sign,
            )
            y_axis = GalvoAxisCorrection(
                scale=self.galvo_2_scale,
                bulge=self.galvo_2_bulge,
                trapezoid=self.galvo_2_trapezoid,
                skew=self.galvo_2_skew,
                sign=self.galvo_2_sign,
            )
        else:
            # Galvo_1 is Y (typical for JCZ controllers)
            y_axis = GalvoAxisCorrection(
                scale=self.galvo_1_scale,
                bulge=self.galvo_1_bulge,
                trapezoid=self.galvo_1_trapezoid,
                skew=self.galvo_1_skew,
                sign=self.galvo_1_sign,
            )
            x_axis = GalvoAxisCorrection(
                scale=self.galvo_2_scale,
                bulge=self.galvo_2_bulge,
                trapezoid=self.galvo_2_trapezoid,
                skew=self.galvo_2_skew,
                sign=self.galvo_2_sign,
            )

        return LensCorrection(
            enabled=True,
            x_axis=x_axis,
            y_axis=y_axis,
            field_angle_deg=self.field_angle,
            field_offset_mm=(self.field_offset_x, self.field_offset_y),
            mirror_x=self.mirror_x,
            mirror_y=self.mirror_y,
        )

    def to_machine_params(self) -> MachineParams:
        """Convert to Glass3D MachineParams config."""
        return MachineParams(
            field_size_mm=(self.width_mm, self.height_mm),
            lens_correction=self.to_lens_correction(),
        )

    def to_config(self, base_config: Glass3DConfig | None = None) -> Glass3DConfig:
        """Convert to full Glass3D config, optionally merging with existing."""
        if base_config is None:
            base_config = Glass3DConfig.default()

        # Update machine params with imported settings
        machine = self.to_machine_params()

        # Preserve Z-axis settings from base config
        machine_dict = machine.model_dump()
        machine_dict["z_range_mm"] = base_config.machine.z_range_mm
        machine_dict["z_steps_per_mm"] = base_config.machine.z_steps_per_mm
        machine_dict["z_axis_min_speed"] = base_config.machine.z_axis_min_speed
        machine_dict["z_axis_max_speed"] = base_config.machine.z_axis_max_speed
        machine_dict["z_axis_acc_time"] = base_config.machine.z_axis_acc_time
        machine_dict["z_axis_settle_ms"] = base_config.machine.z_axis_settle_ms
        machine_dict["edge_margin_mm"] = base_config.machine.edge_margin_mm

        updated_machine = MachineParams.model_validate(machine_dict)

        # Update speed params with imported defaults
        speed = SpeedParams(
            travel_speed=self.default_jump_speed,
            mark_speed=base_config.speed.mark_speed,
            jump_delay=base_config.speed.jump_delay,
            mark_delay=base_config.speed.mark_delay,
        )

        return Glass3DConfig(
            laser=base_config.laser,
            speed=speed,
            machine=updated_machine,
            material=base_config.material,
            point_cloud=base_config.point_cloud,
            engrave=base_config.engrave,
            settings_file=base_config.settings_file,
            mock_laser=base_config.mock_laser,
        )


def load_lightburn_device(path: Path | str) -> LightBurnDevice:
    """Load a LightBurn device export file.

    Args:
        path: Path to .lbzip file

    Returns:
        Parsed LightBurnDevice

    Raises:
        ValueError: If file is invalid or missing device data
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LightBurn export not found: {path}")

    logger.info(f"Loading LightBurn device from: {path}")

    with zipfile.ZipFile(path, "r") as zf:
        # Find the .lbdev file (device settings)
        lbdev_files = [n for n in zf.namelist() if n.endswith(".lbdev")]
        if not lbdev_files:
            raise ValueError(f"No .lbdev file found in {path}")

        lbdev_name = lbdev_files[0]
        logger.debug(f"Reading device file: {lbdev_name}")

        with zf.open(lbdev_name) as f:
            data = json.load(f)

    device = LightBurnDevice.from_lbdev(data)
    logger.info(
        f"Loaded device '{device.name}' ({device.device_type}): "
        f"{device.width_mm}x{device.height_mm}mm"
    )

    return device
