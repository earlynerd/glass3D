"""Checkpoint system for resumable engraving jobs.

This module provides fault tolerance for long-running engrave operations.
Checkpoints are saved periodically (at layer boundaries) so that jobs can
be resumed after crashes, power outages, or intentional stops.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .point_cloud import PointCloud

logger = logging.getLogger(__name__)


def _hash_point_cloud(cloud: PointCloud) -> str:
    """Compute a hash of point cloud data for verification.

    Uses SHA-256 of the points array bytes. This allows us to verify
    that we're resuming with the same point cloud.

    Args:
        cloud: Point cloud to hash

    Returns:
        Hex string of SHA-256 hash
    """
    hasher = hashlib.sha256()
    hasher.update(cloud.points.tobytes())
    if cloud.layer_indices is not None:
        hasher.update(cloud.layer_indices.tobytes())
    return hasher.hexdigest()[:16]  # Truncate for readability


class CheckpointData(BaseModel):
    """Data stored in a checkpoint file.

    This contains all information needed to resume an interrupted engrave job.
    """

    # Job identification
    job_id: str = Field(description="Unique identifier for this job")
    created_at: str = Field(description="ISO timestamp when job started")
    updated_at: str = Field(description="ISO timestamp of last checkpoint")

    # Source tracking
    source_file: str | None = Field(
        default=None,
        description="Original model/scene file path",
    )
    point_cloud_hash: str = Field(
        description="Hash of point cloud for verification",
    )

    # Progress tracking
    total_points: int = Field(description="Total points in the job")
    completed_points: int = Field(
        default=0,
        description="Number of points successfully engraved",
    )
    current_layer: int = Field(
        default=0,
        description="Current layer index (0-based)",
    )
    total_layers: int = Field(description="Total number of layers")

    # For precise resumption within a layer
    layer_start_point_index: int = Field(
        default=0,
        description="Global point index where current layer starts",
    )

    # Configuration snapshot (for reference, not used on resume)
    config_snapshot: dict = Field(
        default_factory=dict,
        description="Copy of key config values for reference",
    )

    # Status
    status: str = Field(
        default="in_progress",
        description="Job status: in_progress, completed, aborted",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if job failed",
    )

    @classmethod
    def create_new(
        cls,
        cloud: PointCloud,
        source_file: str | None = None,
        config_snapshot: dict | None = None,
    ) -> CheckpointData:
        """Create a new checkpoint for a fresh job.

        Args:
            cloud: Point cloud being engraved
            source_file: Original source file path
            config_snapshot: Key configuration values to record

        Returns:
            New CheckpointData instance
        """
        now = datetime.now().isoformat()

        # Count layers
        if cloud.layer_indices is not None:
            total_layers = int(cloud.layer_indices.max()) + 1
        else:
            total_layers = 1

        return cls(
            job_id=str(uuid.uuid4())[:8],
            created_at=now,
            updated_at=now,
            source_file=source_file,
            point_cloud_hash=_hash_point_cloud(cloud),
            total_points=len(cloud),
            completed_points=0,
            current_layer=0,
            total_layers=total_layers,
            layer_start_point_index=0,
            config_snapshot=config_snapshot or {},
            status="in_progress",
        )

    def update_progress(
        self,
        completed_points: int,
        current_layer: int,
        layer_start_point_index: int,
    ) -> None:
        """Update progress in the checkpoint.

        Args:
            completed_points: Total points completed so far
            current_layer: Current layer index
            layer_start_point_index: Global index where this layer starts
        """
        self.completed_points = completed_points
        self.current_layer = current_layer
        self.layer_start_point_index = layer_start_point_index
        self.updated_at = datetime.now().isoformat()

    def mark_completed(self) -> None:
        """Mark the job as successfully completed."""
        self.status = "completed"
        self.updated_at = datetime.now().isoformat()

    def mark_aborted(self, error_message: str | None = None) -> None:
        """Mark the job as aborted.

        Args:
            error_message: Optional error description
        """
        self.status = "aborted"
        self.error_message = error_message
        self.updated_at = datetime.now().isoformat()

    @property
    def percent_complete(self) -> float:
        """Calculate completion percentage."""
        if self.total_points == 0:
            return 100.0
        return (self.completed_points / self.total_points) * 100

    @property
    def is_resumable(self) -> bool:
        """Check if this checkpoint can be resumed."""
        return self.status == "in_progress" and self.completed_points < self.total_points


class CheckpointManager:
    """Manages checkpoint files for engrave jobs.

    Handles saving, loading, and validating checkpoints. Checkpoints are
    stored as JSON files for human readability and easy debugging.
    """

    def __init__(self, checkpoint_dir: Path | str | None = None):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints.
                           Defaults to ./glass3d_checkpoints/
        """
        if checkpoint_dir is None:
            checkpoint_dir = Path.cwd() / "glass3d_checkpoints"
        self.checkpoint_dir = Path(checkpoint_dir)

    def _ensure_dir(self) -> None:
        """Ensure checkpoint directory exists."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint_path(self, job_id: str) -> Path:
        """Get the path for a checkpoint file.

        Args:
            job_id: Job identifier

        Returns:
            Path to checkpoint JSON file
        """
        return self.checkpoint_dir / f"checkpoint_{job_id}.json"

    def save(self, checkpoint: CheckpointData) -> Path:
        """Save checkpoint to disk.

        Args:
            checkpoint: Checkpoint data to save

        Returns:
            Path where checkpoint was saved
        """
        self._ensure_dir()
        path = self.checkpoint_path(checkpoint.job_id)

        # Write atomically by writing to temp file first
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(checkpoint.model_dump(), f, indent=2)

        # Atomic rename
        temp_path.replace(path)

        logger.debug(f"Checkpoint saved: {path}")
        return path

    def load(self, job_id_or_path: str | Path) -> CheckpointData:
        """Load checkpoint from disk.

        Args:
            job_id_or_path: Either a job ID or full path to checkpoint file

        Returns:
            Loaded CheckpointData

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            ValueError: If checkpoint file is invalid
        """
        path = Path(job_id_or_path)
        if not path.exists():
            # Try as job ID
            path = self.checkpoint_path(str(job_id_or_path))

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {job_id_or_path}")

        with open(path) as f:
            data = json.load(f)

        return CheckpointData.model_validate(data)

    def validate_for_resume(
        self,
        checkpoint: CheckpointData,
        cloud: PointCloud,
    ) -> tuple[bool, str]:
        """Validate that a checkpoint can be used with a point cloud.

        Checks that:
        - Checkpoint is resumable (not completed/failed)
        - Point cloud hash matches
        - Point counts match

        Args:
            checkpoint: Checkpoint to validate
            cloud: Point cloud to resume with

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not checkpoint.is_resumable:
            return False, f"Checkpoint status is '{checkpoint.status}', not resumable"

        # Verify point cloud matches
        cloud_hash = _hash_point_cloud(cloud)
        if cloud_hash != checkpoint.point_cloud_hash:
            return False, (
                f"Point cloud hash mismatch. Expected {checkpoint.point_cloud_hash}, "
                f"got {cloud_hash}. The point cloud may have changed."
            )

        if len(cloud) != checkpoint.total_points:
            return False, (
                f"Point count mismatch. Expected {checkpoint.total_points}, "
                f"got {len(cloud)}. The point cloud may have changed."
            )

        return True, ""

    def delete(self, job_id: str) -> bool:
        """Delete a checkpoint file.

        Args:
            job_id: Job identifier

        Returns:
            True if deleted, False if not found
        """
        path = self.checkpoint_path(job_id)
        if path.exists():
            path.unlink()
            logger.info(f"Deleted checkpoint: {path}")
            return True
        return False

    def list_checkpoints(self) -> list[CheckpointData]:
        """List all checkpoints in the checkpoint directory.

        Returns:
            List of checkpoint data, sorted by updated_at descending
        """
        if not self.checkpoint_dir.exists():
            return []

        checkpoints = []
        for path in self.checkpoint_dir.glob("checkpoint_*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                checkpoints.append(CheckpointData.model_validate(data))
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {path}: {e}")

        # Sort by updated_at, most recent first
        checkpoints.sort(key=lambda c: c.updated_at, reverse=True)
        return checkpoints

    def find_resumable(self, source_file: str | None = None) -> list[CheckpointData]:
        """Find checkpoints that can be resumed.

        Args:
            source_file: Optionally filter by source file

        Returns:
            List of resumable checkpoints
        """
        checkpoints = self.list_checkpoints()
        resumable = [c for c in checkpoints if c.is_resumable]

        if source_file:
            # Normalize path for comparison
            source_path = str(Path(source_file).resolve())
            resumable = [
                c for c in resumable
                if c.source_file and str(Path(c.source_file).resolve()) == source_path
            ]

        return resumable
