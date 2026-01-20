"""Calibration and comparison tools for Glass3D.

Provides utilities for comparing coordinate transformations with LightBurn
to verify lens correction parameters produce correct output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ..core.config import Glass3DConfig
from ..laser.controller import CoordinateTransformer

logger = logging.getLogger(__name__)


@dataclass
class CalibrationPoint:
    """A single calibration point with input and output coordinates."""

    input_mm: tuple[float, float]
    output_galvo: tuple[int, int]

    @property
    def output_galvo_hex(self) -> tuple[str, str]:
        """Output galvo coordinates as hex strings."""
        return (f"0x{self.output_galvo[0]:04X}", f"0x{self.output_galvo[1]:04X}")

    @property
    def output_galvo_signed(self) -> tuple[int, int]:
        """Output galvo coordinates as signed offset from center (0x8000)."""
        center = 0x8000
        return (self.output_galvo[0] - center, self.output_galvo[1] - center)


def generate_calibration_grid(
    field_size_mm: tuple[float, float],
    grid_size: int = 5,
    corner_origin: bool = True,
) -> Iterator[tuple[float, float]]:
    """Generate a grid of test points for calibration comparison.

    Args:
        field_size_mm: Field size in mm (width, height)
        grid_size: Number of points per axis (e.g., 5 = 5x5 = 25 points)
        corner_origin: If True, grid spans (0,0) to field_size.
                      If False, grid is centered around (0,0).

    Yields:
        (x_mm, y_mm) coordinate pairs
    """
    if corner_origin:
        # Corner origin: 0 to field_size
        x_values = np.linspace(0, field_size_mm[0], grid_size)
        y_values = np.linspace(0, field_size_mm[1], grid_size)
    else:
        # Centered: -half to +half
        half_x = field_size_mm[0] / 2
        half_y = field_size_mm[1] / 2
        x_values = np.linspace(-half_x, half_x, grid_size)
        y_values = np.linspace(-half_y, half_y, grid_size)

    for y in y_values:
        for x in x_values:
            yield (float(x), float(y))


def compare_coordinates(
    config: Glass3DConfig,
    grid_size: int = 5,
) -> list[CalibrationPoint]:
    """Transform grid points and return comparison data.

    Creates a CoordinateTransformer with the config's lens correction
    settings and transforms a grid of test points.

    Args:
        config: Glass3D configuration with lens correction parameters
        grid_size: Number of points per axis

    Returns:
        List of CalibrationPoint objects with input/output coordinates
    """
    field_size = config.machine.field_size_mm

    # Create transformer with lens correction from config
    transformer = CoordinateTransformer(
        field_size_mm=field_size,
        corner_origin=True,  # Match slicer conventions
        lens_correction=config.lens_correction,
    )

    results = []
    for x_mm, y_mm in generate_calibration_grid(field_size, grid_size):
        x_galvo, y_galvo = transformer.mm_to_galvo_coords(x_mm, y_mm)
        results.append(CalibrationPoint(
            input_mm=(x_mm, y_mm),
            output_galvo=(x_galvo, y_galvo),
        ))

    return results


def format_comparison_table(points: list[CalibrationPoint]) -> str:
    """Format calibration points as a text table.

    Args:
        points: List of CalibrationPoint objects

    Returns:
        Formatted table string
    """
    lines = []
    lines.append("Input (mm)          | Output (galvo)      | Output (hex)")
    lines.append("-" * 60)

    for pt in points:
        x_mm, y_mm = pt.input_mm
        x_g, y_g = pt.output_galvo
        x_hex, y_hex = pt.output_galvo_hex
        lines.append(
            f"({x_mm:7.2f}, {y_mm:7.2f}) | ({x_g:5d}, {y_g:5d}) | ({x_hex}, {y_hex})"
        )

    return "\n".join(lines)


def format_comparison_csv(points: list[CalibrationPoint]) -> str:
    """Format calibration points as CSV.

    Args:
        points: List of CalibrationPoint objects

    Returns:
        CSV formatted string
    """
    lines = ["input_x_mm,input_y_mm,output_x_galvo,output_y_galvo,output_x_hex,output_y_hex"]

    for pt in points:
        x_mm, y_mm = pt.input_mm
        x_g, y_g = pt.output_galvo
        x_hex, y_hex = pt.output_galvo_hex
        lines.append(f"{x_mm:.3f},{y_mm:.3f},{x_g},{y_g},{x_hex},{y_hex}")

    return "\n".join(lines)
