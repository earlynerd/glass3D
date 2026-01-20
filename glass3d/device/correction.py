"""Lens correction table generation for BJJCZ galvo controllers.

This module generates 65x65 correction lookup tables from LightBurn-style
distortion parameters (bulge, trapezoid, skew, scale).

The BJJCZ controller uses these tables to hardware-correct galvo positions,
which is more accurate than software pre-distortion.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ..core.config import LensCorrection

logger = logging.getLogger(__name__)

# Correction table is always 65x65
GRID_SIZE = 65
GRID_CENTER = 32  # Index of center point


@dataclass
class CorrectionTable:
    """A 65x65 lens correction table for BJJCZ controllers.

    Each entry contains (dx, dy) offsets in galvo units to apply
    at that grid position. The grid spans the full field with
    position (32, 32) at the center.
    """

    # Offsets stored as (65, 65, 2) array: [y, x, (dx, dy)]
    offsets: np.ndarray  # Shape (65, 65, 2), dtype int16

    @classmethod
    def from_lens_correction(
        cls,
        correction: LensCorrection,
        field_size_mm: tuple[float, float],
        galvo_bits: int = 16,
    ) -> CorrectionTable:
        """Generate correction table from LensCorrection parameters.

        Uses LightBurn-compatible algorithm with cosine-modulated bulge
        and cross-term trapezoid. Scale is NOT included in the correction
        table (it should be applied separately in coordinate transformation).

        Args:
            correction: Lens correction parameters (bulge, trapezoid, etc.)
            field_size_mm: Physical field size in mm
            galvo_bits: Resolution of galvo DAC (typically 16)

        Returns:
            CorrectionTable with computed offsets
        """
        galvo_max = (2 ** galvo_bits) - 1

        # Galvo units per normalized unit (-1 to +1 spans full field)
        galvo_per_norm = galvo_max / 2

        offsets = np.zeros((GRID_SIZE, GRID_SIZE, 2), dtype=np.int16)

        for j in range(GRID_SIZE):  # Y grid index
            for i in range(GRID_SIZE):  # X grid index
                # Normalized position: -1 to +1
                x_norm = (i - GRID_CENTER) / GRID_CENTER
                y_norm = (j - GRID_CENTER) / GRID_CENTER

                # Compute correction offsets using LightBurn algorithm
                dx_norm, dy_norm = _compute_correction_lightburn(
                    x_norm, y_norm, correction
                )

                # Convert to galvo units
                dx_galvo = int(round(dx_norm * galvo_per_norm))
                dy_galvo = int(round(dy_norm * galvo_per_norm))

                offsets[j, i, 0] = dx_galvo
                offsets[j, i, 1] = dy_galvo

        return cls(offsets=offsets)

    @classmethod
    def from_cor_file(cls, path: Path | str) -> CorrectionTable:
        """Load correction table from a .cor file.

        Supports both the 5-byte format and standard LMC1 formats.

        Args:
            path: Path to .cor file

        Returns:
            CorrectionTable with loaded offsets
        """
        path = Path(path)
        data = path.read_bytes()

        offsets = np.zeros((GRID_SIZE, GRID_SIZE, 2), dtype=np.int16)

        # Check file size to determine format
        if len(data) == GRID_SIZE * GRID_SIZE * 5:
            # 5-byte format: 2-byte dx + 2-byte dy + 1-byte flag
            logger.debug("Reading 5-byte correction format")
            for j in range(GRID_SIZE):
                for i in range(GRID_SIZE):
                    idx = (j * GRID_SIZE + i) * 5
                    block = data[idx:idx + 5]

                    dx_raw = struct.unpack('<H', block[0:2])[0]
                    dy_raw = struct.unpack('<H', block[2:4])[0]
                    # flag = block[4]  # Not used when loading

                    # High bit indicates negative
                    dx = -(dx_raw - 0x8000) if dx_raw >= 0x8000 else dx_raw
                    dy = -(dy_raw - 0x8000) if dy_raw >= 0x8000 else dy_raw

                    offsets[j, i, 0] = dx
                    offsets[j, i, 1] = dy
        else:
            # Try standard LMC1 format (handled by galvoplotter)
            raise NotImplementedError(
                f"Unknown .cor format (size={len(data)}). "
                "Expected 5-byte format (21125 bytes) or LMC1 format."
            )

        return cls(offsets=offsets)

    def to_cor_file(self, path: Path | str) -> None:
        """Save correction table to a 5-byte format .cor file.

        Args:
            path: Output path for .cor file
        """
        path = Path(path)

        data = bytearray(GRID_SIZE * GRID_SIZE * 5)

        for j in range(GRID_SIZE):
            for i in range(GRID_SIZE):
                idx = (j * GRID_SIZE + i) * 5
                dx = int(self.offsets[j, i, 0])
                dy = int(self.offsets[j, i, 1])

                # Encode: negative values use high bit
                if dx < 0:
                    dx_raw = 0x8000 + (-dx)
                else:
                    dx_raw = dx

                if dy < 0:
                    dy_raw = 0x8000 + (-dy)
                else:
                    dy_raw = dy

                # Flag: 0 at origin, 1 elsewhere
                flag = 0 if (i == 0 and j == 0) else 1

                struct.pack_into('<HHB', data, idx, dx_raw & 0xFFFF, dy_raw & 0xFFFF, flag)

        path.write_bytes(data)
        logger.info(f"Wrote correction table to {path}")

    def to_galvoplotter_table(self) -> list[tuple[int, int]]:
        """Convert to galvoplotter format for _write_correction_table().

        Returns:
            List of 4225 (dx, dy) tuples in galvoplotter format
        """
        table = []
        for j in range(GRID_SIZE):
            for i in range(GRID_SIZE):
                dx = int(self.offsets[j, i, 0])
                dy = int(self.offsets[j, i, 1])

                # Galvoplotter uses: positive as-is, negative as (0x8000 + abs)
                if dx < 0:
                    dx = 0x8000 + (-dx)
                if dy < 0:
                    dy = 0x8000 + (-dy)

                table.append((dx & 0xFFFF, dy & 0xFFFF))

        return table

    def get_offset(self, x_norm: float, y_norm: float) -> tuple[int, int]:
        """Get interpolated offset for a normalized position.

        Args:
            x_norm: X position in range [-1, 1]
            y_norm: Y position in range [-1, 1]

        Returns:
            (dx, dy) offset in galvo units
        """
        # Convert to grid coordinates
        x_grid = (x_norm + 1) * GRID_CENTER
        y_grid = (y_norm + 1) * GRID_CENTER

        # Clamp to valid range
        x_grid = max(0, min(GRID_SIZE - 1.001, x_grid))
        y_grid = max(0, min(GRID_SIZE - 1.001, y_grid))

        # Bilinear interpolation
        x0, y0 = int(x_grid), int(y_grid)
        x1, y1 = min(x0 + 1, GRID_SIZE - 1), min(y0 + 1, GRID_SIZE - 1)

        fx, fy = x_grid - x0, y_grid - y0

        # Interpolate dx
        dx00 = self.offsets[y0, x0, 0]
        dx10 = self.offsets[y0, x1, 0]
        dx01 = self.offsets[y1, x0, 0]
        dx11 = self.offsets[y1, x1, 0]
        dx = dx00 * (1-fx) * (1-fy) + dx10 * fx * (1-fy) + dx01 * (1-fx) * fy + dx11 * fx * fy

        # Interpolate dy
        dy00 = self.offsets[y0, x0, 1]
        dy10 = self.offsets[y0, x1, 1]
        dy01 = self.offsets[y1, x0, 1]
        dy11 = self.offsets[y1, x1, 1]
        dy = dy00 * (1-fx) * (1-fy) + dy10 * fx * (1-fy) + dy01 * (1-fx) * fy + dy11 * fx * fy

        return int(round(dx)), int(round(dy))

    def summary(self) -> str:
        """Get a text summary of the correction table."""
        dx_min, dx_max = self.offsets[:, :, 0].min(), self.offsets[:, :, 0].max()
        dy_min, dy_max = self.offsets[:, :, 1].min(), self.offsets[:, :, 1].max()

        center_dx = self.offsets[GRID_CENTER, GRID_CENTER, 0]
        center_dy = self.offsets[GRID_CENTER, GRID_CENTER, 1]

        lines = [
            f"Correction Table ({GRID_SIZE}x{GRID_SIZE})",
            f"  DX range: {dx_min:+5} to {dx_max:+5} galvo units",
            f"  DY range: {dy_min:+5} to {dy_max:+5} galvo units",
            f"  Center offset: ({center_dx:+5}, {center_dy:+5})",
        ]
        return "\n".join(lines)


def _compute_correction_lightburn(
    x_norm: float,
    y_norm: float,
    correction: LensCorrection,
) -> tuple[float, float]:
    """Compute correction offsets using LightBurn-compatible algorithm.

    This matches LightBurn's correction table generation algorithm,
    which uses cosine-modulated bulge and cross-term trapezoid.

    The formula is:
        dx = (bulge_x - 1) * x * cos(y * pi/2) + (trap_y - 1) * x * y
        dy = (bulge_y - 1) * y * cos(x * pi/2) + (trap_x - 1) * x * y

    Args:
        x_norm: Normalized X (-1 to 1)
        y_norm: Normalized Y (-1 to 1)
        correction: Lens correction parameters

    Returns:
        (dx, dy) correction offsets in normalized units
    """
    x, y = x_norm, y_norm

    # Apply mirror before computing correction
    if correction.mirror_x:
        x = -x
    if correction.mirror_y:
        y = -y

    # Apply sign
    x = x * correction.x_axis.sign
    y = y * correction.y_axis.sign

    # Bulge correction (cosine-modulated per-axis)
    # Each axis's bulge affects its own direction but is modulated
    # by the perpendicular axis position via cosine
    dx_bulge = (correction.x_axis.bulge - 1.0) * x * np.cos(y * np.pi / 2)
    dy_bulge = (correction.y_axis.bulge - 1.0) * y * np.cos(x * np.pi / 2)

    # Trapezoid correction (cross-term)
    # Each axis's trapezoid creates a correction proportional to x*y
    # X trapezoid affects dx, Y trapezoid affects dy
    dx_trap = (correction.x_axis.trapezoid - 1.0) * x * y
    dy_trap = (correction.y_axis.trapezoid - 1.0) * x * y

    # Skew correction (if any)
    dx_skew = 0.0
    dy_skew = 0.0
    if correction.x_axis.skew != 1.0:
        dx_skew = (correction.x_axis.skew - 1.0) * y
    if correction.y_axis.skew != 1.0:
        dy_skew = (correction.y_axis.skew - 1.0) * x

    dx = dx_bulge + dx_trap + dx_skew
    dy = dy_bulge + dy_trap + dy_skew

    return dx, dy


def generate_correction_table(
    correction: LensCorrection,
    field_size_mm: tuple[float, float],
    galvo_bits: int = 16,
) -> CorrectionTable:
    """Generate a correction table from LensCorrection parameters.

    This is the main entry point for creating correction tables.

    Args:
        correction: Lens correction parameters from LightBurn import
        field_size_mm: Physical field size in mm
        galvo_bits: DAC resolution (typically 16)

    Returns:
        CorrectionTable ready for board upload or file export
    """
    return CorrectionTable.from_lens_correction(
        correction, field_size_mm, galvo_bits
    )


def write_correction_to_controller(
    controller,  # galvo.GalvoController
    table: CorrectionTable,
) -> None:
    """Write a correction table to the BJJCZ controller.

    This uploads the 65x65 correction table to the board's memory.
    The board will then apply these corrections automatically to
    all coordinates sent to the galvos.

    Args:
        controller: galvoplotter GalvoController instance
        table: CorrectionTable to upload

    Note:
        After uploading, coordinates sent to the controller should
        NOT be pre-corrected in software - the board handles it.
    """
    gp_table = table.to_galvoplotter_table()
    logger.info(f"Writing correction table to controller ({len(gp_table)} entries)")

    # Use galvoplotter's internal method
    controller._write_correction_table(gp_table)
    logger.info("Correction table uploaded successfully")
