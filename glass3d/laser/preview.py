"""Live preview visualization for engraving progress."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class LivePreview:
    """Matplotlib-based live preview of engraving progress.

    Shows a 2D scatter plot of points, updated once per layer.
    Completed layers shown in green, current layer in red.
    """

    def __init__(
        self,
        field_size_mm: tuple[float, float] = (110.0, 110.0),
        title: str = "Engraving Progress",
    ):
        """Initialize the live preview window.

        Args:
            field_size_mm: Workspace size for axis limits
            title: Window title
        """
        self.field_size_mm = field_size_mm
        self.title = title

        self._fig = None
        self._ax = None
        self._completed_scatter = None
        self._current_scatter = None
        self._initialized = False

        # Store completed points for cumulative display
        self._completed_points: list[NDArray[np.float64]] = []

    def _ensure_initialized(self) -> bool:
        """Initialize matplotlib if not already done.

        Returns:
            True if initialization successful, False otherwise.
        """
        if self._initialized:
            return True

        try:
            import matplotlib.pyplot as plt

            # Enable interactive mode for non-blocking updates
            plt.ion()

            self._fig, self._ax = plt.subplots(figsize=(8, 8))
            self._ax.set_xlim(0, self.field_size_mm[0])
            self._ax.set_ylim(0, self.field_size_mm[1])
            self._ax.set_aspect('equal')
            self._ax.set_xlabel('X (mm)')
            self._ax.set_ylabel('Y (mm)')
            self._ax.set_title(self.title)
            self._ax.grid(True, alpha=0.3)

            # Initialize empty scatter plots
            self._completed_scatter = self._ax.scatter(
                [], [], s=0.5, c='green', alpha=0.5, label='Completed'
            )
            self._current_scatter = self._ax.scatter(
                [], [], s=1, c='red', alpha=0.8, label='Current Layer'
            )
            self._ax.legend(loc='upper right', markerscale=10)

            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)  # Give time for window to appear

            self._initialized = True
            logger.info("Live preview initialized")
            return True

        except ImportError:
            logger.warning("matplotlib not available - live preview disabled")
            return False
        except Exception as e:
            logger.warning(f"Failed to initialize live preview: {e}")
            return False

    def update(
        self,
        current_layer_points: NDArray[np.float64],
        layer_idx: int,
        total_layers: int,
        completed_points: int,
        total_points: int,
    ) -> None:
        """Update the preview with current layer.

        Args:
            current_layer_points: Nx3 array of points in current layer
            layer_idx: Current layer index (0-based)
            total_layers: Total number of layers
            completed_points: Number of points completed so far
            total_points: Total points to engrave
        """
        if not self._ensure_initialized():
            return

        try:
            import matplotlib.pyplot as plt

            # Add previous layer to completed points (if we have one)
            # Note: We update at the START of each layer, so the "current"
            # from the previous call is now completed

            # Update current layer scatter
            if len(current_layer_points) > 0:
                self._current_scatter.set_offsets(current_layer_points[:, :2])
            else:
                self._current_scatter.set_offsets(np.empty((0, 2)))

            # Update completed points scatter
            if self._completed_points:
                all_completed = np.vstack(self._completed_points)
                self._completed_scatter.set_offsets(all_completed[:, :2])

            # Update title with progress
            pct = (completed_points / total_points * 100) if total_points > 0 else 0
            self._ax.set_title(
                f"{self.title}\n"
                f"Layer {layer_idx + 1}/{total_layers} | "
                f"{completed_points:,}/{total_points:,} points ({pct:.1f}%)"
            )

            # Refresh display
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
            plt.pause(0.01)  # Brief pause to allow update

        except Exception as e:
            logger.debug(f"Preview update failed: {e}")

    def mark_layer_complete(self, layer_points: NDArray[np.float64]) -> None:
        """Mark a layer as completed (will show as green on next update).

        Args:
            layer_points: Points from the completed layer
        """
        if len(layer_points) > 0:
            self._completed_points.append(layer_points.copy())

    def close(self) -> None:
        """Close the preview window."""
        if self._initialized:
            try:
                import matplotlib.pyplot as plt
                plt.close(self._fig)
                plt.ioff()
            except Exception:
                pass
            self._initialized = False
            logger.info("Live preview closed")
