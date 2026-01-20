"""Tests for coordinate transformation and lens correction."""

import numpy as np
import pytest

from glass3d.core.config import GalvoAxisCorrection, LensCorrection
from glass3d.laser.controller import CoordinateTransformer, _apply_lens_correction


class TestCoordinateTransformer:
    """Test CoordinateTransformer class."""

    def test_center_point_maps_to_galvo_center(self):
        """Test that center of field maps to galvo center (0x8000)."""
        transformer = CoordinateTransformer(
            field_size_mm=(110.0, 110.0),
            corner_origin=False,  # Centered coordinates
        )

        x_galvo, y_galvo = transformer.mm_to_galvo_coords(0.0, 0.0)

        assert x_galvo == 0x8000
        assert y_galvo == 0x8000

    def test_corner_origin_center_maps_correctly(self):
        """Test corner origin mode maps field center to galvo center."""
        transformer = CoordinateTransformer(
            field_size_mm=(110.0, 110.0),
            corner_origin=True,
        )

        # Field center at corner origin is (55, 55)
        x_galvo, y_galvo = transformer.mm_to_galvo_coords(55.0, 55.0)

        assert x_galvo == 0x8000
        assert y_galvo == 0x8000

    def test_corner_origin_zero_offset(self):
        """Test corner origin (0,0) maps to expected galvo position."""
        transformer = CoordinateTransformer(
            field_size_mm=(110.0, 110.0),
            corner_origin=True,
        )

        # Corner (0,0) should map to negative half of galvo range
        x_galvo, y_galvo = transformer.mm_to_galvo_coords(0.0, 0.0)

        # Should be less than center
        assert x_galvo < 0x8000
        assert y_galvo < 0x8000

    def test_galvo_to_mm_inverse(self):
        """Test that galvo_to_mm inverts mm_to_galvo."""
        transformer = CoordinateTransformer(
            field_size_mm=(110.0, 110.0),
            corner_origin=True,
        )

        original_mm = (25.0, 75.0)
        x_galvo, y_galvo = transformer.mm_to_galvo_coords(*original_mm)
        recovered_mm = transformer.galvo_to_mm_coords(x_galvo, y_galvo)

        assert abs(recovered_mm[0] - original_mm[0]) < 0.1
        assert abs(recovered_mm[1] - original_mm[1]) < 0.1

    def test_is_in_bounds_inside(self):
        """Test that points inside field are detected."""
        transformer = CoordinateTransformer(
            field_size_mm=(110.0, 110.0),
            corner_origin=True,
        )

        assert transformer.is_in_bounds(55.0, 55.0) is True
        assert transformer.is_in_bounds(10.0, 100.0) is True

    def test_is_in_bounds_outside(self):
        """Test that points outside field are detected."""
        transformer = CoordinateTransformer(
            field_size_mm=(110.0, 110.0),
            corner_origin=True,
        )

        assert transformer.is_in_bounds(-5.0, 55.0) is False
        assert transformer.is_in_bounds(55.0, 120.0) is False

    def test_is_in_bounds_with_margin(self):
        """Test bounds checking with safety margin."""
        transformer = CoordinateTransformer(
            field_size_mm=(110.0, 110.0),
            corner_origin=True,
        )

        # Point at edge (1mm from boundary)
        assert transformer.is_in_bounds(1.0, 55.0, margin_mm=0.0) is True
        assert transformer.is_in_bounds(1.0, 55.0, margin_mm=2.0) is False

    def test_clamp_to_bounds(self):
        """Test clamping coordinates to field bounds."""
        transformer = CoordinateTransformer(
            field_size_mm=(110.0, 110.0),
            corner_origin=True,
        )

        # Point outside should be clamped
        x, y = transformer.clamp_to_bounds(-10.0, 120.0)

        assert x == 0.0
        assert y == 110.0


class TestLensCorrection:
    """Test lens correction application."""

    def test_identity_correction(self):
        """Test that default correction params produce identity transform."""
        correction = LensCorrection(enabled=True)

        # Test a few points
        for x, y in [(0.0, 0.0), (0.5, 0.5), (-0.5, 0.3)]:
            x_out, y_out = _apply_lens_correction(x, y, correction)
            assert abs(x_out - x) < 1e-10
            assert abs(y_out - y) < 1e-10

    def test_mirror_x(self):
        """Test X mirror flips X coordinate."""
        correction = LensCorrection(enabled=True, mirror_x=True)

        x_out, y_out = _apply_lens_correction(0.5, 0.3, correction)

        assert abs(x_out - (-0.5)) < 1e-10
        assert abs(y_out - 0.3) < 1e-10

    def test_mirror_y(self):
        """Test Y mirror flips Y coordinate."""
        correction = LensCorrection(enabled=True, mirror_y=True)

        x_out, y_out = _apply_lens_correction(0.5, 0.3, correction)

        assert abs(x_out - 0.5) < 1e-10
        assert abs(y_out - (-0.3)) < 1e-10

    def test_scale_x(self):
        """Test X scale multiplies X coordinate."""
        correction = LensCorrection(
            enabled=True,
            x_axis=GalvoAxisCorrection(scale=1.1),
        )

        x_out, y_out = _apply_lens_correction(0.5, 0.3, correction)

        assert abs(x_out - 0.55) < 1e-10
        assert abs(y_out - 0.3) < 1e-10

    def test_scale_y(self):
        """Test Y scale multiplies Y coordinate."""
        correction = LensCorrection(
            enabled=True,
            y_axis=GalvoAxisCorrection(scale=0.9),
        )

        x_out, y_out = _apply_lens_correction(0.5, 0.3, correction)

        assert abs(x_out - 0.5) < 1e-10
        assert abs(y_out - 0.27) < 1e-10

    def test_sign_negative(self):
        """Test negative sign inverts axis."""
        correction = LensCorrection(
            enabled=True,
            x_axis=GalvoAxisCorrection(sign=-1),
        )

        x_out, y_out = _apply_lens_correction(0.5, 0.3, correction)

        assert abs(x_out - (-0.5)) < 1e-10
        assert abs(y_out - 0.3) < 1e-10

    def test_field_rotation(self):
        """Test field rotation rotates both axes."""
        correction = LensCorrection(enabled=True, field_angle_deg=90.0)

        # 90 degree rotation: (1, 0) -> (0, 1)
        x_out, y_out = _apply_lens_correction(1.0, 0.0, correction)

        assert abs(x_out - 0.0) < 1e-10
        assert abs(y_out - 1.0) < 1e-10

    def test_skew_x(self):
        """Test X skew correction."""
        correction = LensCorrection(
            enabled=True,
            x_axis=GalvoAxisCorrection(skew=1.1),
        )

        # X skew adds (skew-1)*y to x
        x_out, y_out = _apply_lens_correction(0.5, 1.0, correction)

        # x = 0.5 + (1.1-1.0)*1.0 = 0.5 + 0.1 = 0.6
        assert abs(x_out - 0.6) < 1e-10
        assert abs(y_out - 1.0) < 1e-10

    def test_bulge_at_center(self):
        """Test bulge has no effect at center."""
        correction = LensCorrection(
            enabled=True,
            x_axis=GalvoAxisCorrection(bulge=1.5),
            y_axis=GalvoAxisCorrection(bulge=1.5),
        )

        # At center (0,0), r²=0, so bulge has no effect
        x_out, y_out = _apply_lens_correction(0.0, 0.0, correction)

        assert abs(x_out) < 1e-10
        assert abs(y_out) < 1e-10

    def test_bulge_at_edge(self):
        """Test bulge affects points away from center."""
        correction = LensCorrection(
            enabled=True,
            x_axis=GalvoAxisCorrection(bulge=1.1),
        )

        # At (0.5, 0), r²=0.25, bulge_factor = 1 + 0.1*0.25 = 1.025
        x_out, y_out = _apply_lens_correction(0.5, 0.0, correction)

        expected_x = 0.5 * 1.025  # 0.5125
        assert abs(x_out - expected_x) < 1e-10

    def test_correction_order_documented(self):
        """Test that demonstrates order dependency of skew corrections.

        This test documents the behavior where Y-skew uses already-modified X.
        If the order changes, this test will fail.
        """
        # Apply both X and Y skew
        correction = LensCorrection(
            enabled=True,
            x_axis=GalvoAxisCorrection(skew=1.1),
            y_axis=GalvoAxisCorrection(skew=1.1),
        )

        x_out, y_out = _apply_lens_correction(1.0, 1.0, correction)

        # X skew: x = 1.0 + (1.1-1.0)*1.0 = 1.1
        # Y skew (uses modified x=1.1): y = 1.0 + (1.1-1.0)*1.1 = 1.11
        expected_x = 1.1
        expected_y = 1.11

        assert abs(x_out - expected_x) < 1e-10
        assert abs(y_out - expected_y) < 1e-10


class TestTransformerWithLensCorrection:
    """Test CoordinateTransformer with lens correction enabled."""

    def test_lens_correction_applied(self):
        """Test that lens correction is actually applied."""
        correction = LensCorrection(
            enabled=True,
            x_axis=GalvoAxisCorrection(scale=1.1),
        )

        transformer_with = CoordinateTransformer(
            field_size_mm=(110.0, 110.0),
            corner_origin=True,
            lens_correction=correction,
        )

        transformer_without = CoordinateTransformer(
            field_size_mm=(110.0, 110.0),
            corner_origin=True,
            lens_correction=None,
        )

        # Same input point
        x_with, y_with = transformer_with.mm_to_galvo_coords(55.0, 55.0)
        x_without, y_without = transformer_without.mm_to_galvo_coords(55.0, 55.0)

        # Center point should still map to center (scale doesn't affect center)
        assert x_with == x_without == 0x8000

        # Try an off-center point
        x_with, y_with = transformer_with.mm_to_galvo_coords(80.0, 55.0)
        x_without, y_without = transformer_without.mm_to_galvo_coords(80.0, 55.0)

        # X should differ due to scale correction
        assert x_with != x_without

    def test_disabled_correction_has_no_effect(self):
        """Test that disabled correction doesn't change output."""
        correction = LensCorrection(
            enabled=False,  # Disabled
            x_axis=GalvoAxisCorrection(scale=1.5),  # Would change X
        )

        transformer_with = CoordinateTransformer(
            field_size_mm=(110.0, 110.0),
            corner_origin=True,
            lens_correction=correction,
        )

        transformer_without = CoordinateTransformer(
            field_size_mm=(110.0, 110.0),
            corner_origin=True,
            lens_correction=None,
        )

        x_with, y_with = transformer_with.mm_to_galvo_coords(80.0, 55.0)
        x_without, y_without = transformer_without.mm_to_galvo_coords(80.0, 55.0)

        # Should be identical when correction is disabled
        assert x_with == x_without
        assert y_with == y_without
