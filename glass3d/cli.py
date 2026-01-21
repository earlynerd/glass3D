"""Command-line interface for Glass3D.

Usage:
    glass3d engrave model.stl [options]
    glass3d preview model.stl [options]
    glass3d info model.stl
    glass3d calibrate [options]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal, cast

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from .core.checkpoint import CheckpointData, CheckpointManager
from .core.config import Glass3DConfig
from .core.point_cloud import PointCloud
from .mesh.loader import MeshLoader
from .mesh.pointcloud_gen import get_strategy, list_strategies
from .laser.controller import LaserController, EngraveProgress

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False)],
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """Glass3D - Subsurface laser engraving software."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose)


def _is_3mf_file(path: str) -> bool:
    """Check if a file is a 3MF file by extension."""
    return Path(path).suffix.lower() == ".3mf"


def _estimate_engrave_time(cloud: PointCloud, cfg: Glass3DConfig) -> float:
    """Estimate total engrave time in seconds.

    Accounts for:
    - Point dwell time
    - Average travel time between points
    - Jump delays (settling time)
    - Laser on/off delays
    - Z-axis settling per layer
    - Thermal pauses

    Args:
        cloud: Point cloud to engrave
        cfg: Configuration with timing parameters

    Returns:
        Estimated time in seconds
    """
    import numpy as np

    num_points = len(cloud)
    if num_points == 0:
        return 0.0

    # Time components (all in seconds)
    dwell_time_per_point = cfg.laser.point_dwell_ms / 1000.0

    # Average jump delay (use midpoint of min/max)
    avg_jump_delay = (cfg.speed.jump_delay_min + cfg.speed.jump_delay_max) / 2.0 / 1_000_000.0

    # Laser on/off delays (per point)
    laser_delay = (cfg.speed.laser_on_delay + cfg.speed.laser_off_delay) / 1_000_000.0

    # Estimate average travel distance between points
    # Use a sample to avoid computing all pairwise distances
    points = cloud.points
    if num_points > 1000:
        sample_idx = np.random.choice(num_points, 1000, replace=False)
        sample = points[sample_idx]
    else:
        sample = points

    # Compute distances between consecutive points in sample
    if len(sample) > 1:
        diffs = np.diff(sample[:, :2], axis=0)  # XY only
        distances = np.linalg.norm(diffs, axis=1)
        avg_distance_mm = float(np.mean(distances))
    else:
        avg_distance_mm = 1.0  # Fallback

    # Travel time per point
    travel_time_per_point = avg_distance_mm / cfg.speed.travel_speed

    # Per-point time
    time_per_point = (
        dwell_time_per_point
        + travel_time_per_point
        + avg_jump_delay
        + laser_delay
    )

    # Total point time
    total_point_time = num_points * time_per_point

    # Z-axis settling time (per layer)
    num_layers = cloud.num_layers
    z_settle_time = num_layers * (cfg.machine.z_axis_settle_ms / 1000.0)

    # Thermal pauses
    thermal_pause_count = num_points // cfg.material.max_continuous_points
    thermal_pause_time = thermal_pause_count * (cfg.material.thermal_pause_ms / 1000.0)

    total_time = total_point_time + z_settle_time + thermal_pause_time

    return total_time


def _load_and_generate_point_cloud(
    model_path: str,
    cfg: Glass3DConfig,
    scale: float,
    max_size: float | None,
) -> tuple[PointCloud, dict]:
    """Load model(s) and generate point cloud.

    Handles both single mesh files (STL/OBJ) and 3MF scene files.

    Returns:
        Tuple of (point_cloud, info_dict)
    """
    from .scene import Scene

    path = Path(model_path)
    is_3mf = path.suffix.lower() == ".3mf"

    if is_3mf:
        # Load 3MF scene with multiple models
        with console.status("Loading 3MF scene..."):
            scene = Scene.from_3mf(path)

        # Show scene info
        info = {
            "type": "3mf",
            "name": scene.name,
            "num_models": len(scene.models),
            "models": [
                {
                    "name": m.name,
                    "position": m.transform.position,
                    "rotation": m.transform.rotation,
                    "scale": m.transform.scale,
                }
                for m in scene.models
            ],
        }

        console.print(f"[cyan]Loaded 3MF scene: {scene.name}[/cyan]")
        console.print(f"Models: {len(scene.models)}")

        # Check anchor status for all models (by name and geometry)
        anchor_statuses = {m.id: scene.get_anchor_status(m) for m in scene.models}
        anchor_count = sum(1 for is_anchor, _ in anchor_statuses.values() if is_anchor)
        if anchor_count > 0:
            console.print(f"[dim]({anchor_count} anchor model(s) will be excluded)[/dim]")

        # Show model table
        table = Table(title="Models in Scene")
        table.add_column("Name", style="cyan")
        table.add_column("Position", style="green")
        table.add_column("Rotation", style="yellow")
        table.add_column("Scale", style="magenta")
        table.add_column("Status", style="dim")

        for model in scene.models:
            pos = model.transform.position
            rot = model.transform.rotation
            is_anchor, reason = anchor_statuses[model.id]
            if is_anchor:
                status = f"[yellow]anchor ({reason})[/yellow]"
            elif reason.startswith("partial:"):
                # Assembly with some anchor components
                count = reason.split(":")[1]
                status = f"[dim]{count} anchor part(s) filtered[/dim]"
            else:
                status = ""
            table.add_row(
                model.name,
                f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})",
                f"({rot[0]:.0f}, {rot[1]:.0f}, {rot[2]:.0f})",
                f"{model.transform.scale:.2f}",
                status,
            )
        console.print(table)

        # Generate combined point cloud
        console.print(f"\n[cyan]Generating point cloud ({cfg.point_cloud.strategy} strategy)...[/cyan]")

        with console.status("Processing models..."):
            cloud = scene.to_combined_point_cloud(cfg.point_cloud)

    else:
        # Single mesh file
        with console.status("Loading mesh..."):
            loader = MeshLoader(model_path)

            # Apply transformations
            loader.center_at_origin()

            if max_size:
                loader.scale_to_fit(max_size)
            elif scale != 1.0:
                loader.scale(scale)

            loader.repair()

        # Get mesh info
        stats = loader.stats()
        info = {
            "type": "mesh",
            "name": path.name,
            "num_vertices": stats["num_vertices"],
            "num_faces": stats["num_faces"],
            "size": stats["size"],
            "is_watertight": stats["is_watertight"],
        }

        # Show mesh info
        table = Table(title="Mesh Info")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("File", path.name)
        table.add_row("Vertices", str(stats["num_vertices"]))
        table.add_row("Faces", str(stats["num_faces"]))
        table.add_row("Size (mm)", f"{stats['size'][0]:.1f} x {stats['size'][1]:.1f} x {stats['size'][2]:.1f}")
        table.add_row("Watertight", "Yes" if stats["is_watertight"] else "No")
        console.print(table)

        # Generate point cloud
        console.print(f"\n[cyan]Generating point cloud ({cfg.point_cloud.strategy} strategy)...[/cyan]")

        with console.status("Generating points..."):
            strat = get_strategy(cfg.point_cloud.strategy)
            cloud = strat.generate(loader.mesh, cfg.point_cloud)

        # Sort points bottom-up for SSLE
        cloud = cloud.sort_by_z(ascending=True)

    return cloud, info


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.option(
    "--strategy", "-s",
    type=click.Choice(["surface", "solid", "grayscale", "contour", "shell"]),
    default="surface",
    help="Point generation strategy",
)
@click.option(
    "--spacing", "-p",
    type=float,
    default=0.1,
    help="Point spacing in mm",
)
@click.option(
    "--layer-height", "-l",
    type=float,
    default=0.1,
    help="Layer height in mm",
)
@click.option(
    "--shells",
    type=int,
    default=None,
    help="Number of shells for 'shell' strategy (default: 3)",
)
@click.option(
    "--shell-spacing",
    type=float,
    default=None,
    help="Distance between shells in mm for 'shell' strategy (default: 0.15)",
)
@click.option(
    "--scale",
    type=float,
    default=1.0,
    help="Scale factor for model (single mesh only, ignored for 3MF)",
)
@click.option(
    "--max-size",
    type=float,
    default=None,
    help="Scale model to fit within this size in mm (single mesh only, ignored for 3MF)",
)
@click.option(
    "--dry-run", "-n",
    is_flag=True,
    help="Preview with red dot, don't fire laser",
)
@click.option(
    "--mock",
    is_flag=True,
    help="Use mock laser connection (for testing)",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Save point cloud to file (skip engraving)",
)
@click.option(
    "--live-preview",
    is_flag=True,
    help="Show live matplotlib preview (updates per layer)",
)
@click.option(
    "--resume", "-r",
    type=str,
    default=None,
    help="Resume from checkpoint (job ID or path to checkpoint file)",
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(),
    default=None,
    help="Directory for checkpoint files (default: ./glass3d_checkpoints)",
)
@click.option(
    "--no-checkpoint",
    is_flag=True,
    help="Disable checkpoint saving (not recommended for long jobs)",
)
@click.option(
    "--cor-file",
    type=click.Path(exists=True),
    default=None,
    help="Hardware correction file (.cor) to upload to controller",
)
@click.pass_context
def engrave(
    ctx: click.Context,
    model_path: str,
    config: str | None,
    strategy: str,
    spacing: float,
    layer_height: float,
    shells: int | None,
    shell_spacing: float | None,
    scale: float,
    max_size: float | None,
    dry_run: bool,
    mock: bool,
    output: str | None,
    live_preview: bool,
    resume: str | None,
    checkpoint_dir: str | None,
    no_checkpoint: bool,
    cor_file: str | None,
) -> None:
    """Engrave a 3D model or scene into glass.

    MODEL_PATH: Path to model file (STL/OBJ) or 3MF scene file

    For 3MF files exported from slicer software (PrusaSlicer, Cura, etc.),
    the model positions and transforms are preserved from the slicer.
    """
    # Load config
    if config:
        cfg = Glass3DConfig.from_file(config)
    else:
        cfg = Glass3DConfig.default()

    cfg.mock_laser = mock
    cfg.point_cloud.point_spacing_mm = spacing
    cfg.point_cloud.layer_height_mm = layer_height
    # Cast validated Click choice to Literal type
    cfg.point_cloud.strategy = cast(
        Literal["surface", "solid", "grayscale", "contour", "shell"], strategy
    )
    # Shell strategy options
    if shells is not None:
        cfg.point_cloud.shell_count = shells
    if shell_spacing is not None:
        cfg.point_cloud.shell_spacing_mm = shell_spacing
    cfg.engrave.dry_run = dry_run

    # When using hardware correction, disable software lens correction
    # (the board will apply correction from the uploaded table)
    if cor_file:
        cfg.machine.lens_correction.enabled = False

    console.print(f"\n[bold]Glass3D Engraver[/bold]\n")

    # Load model(s) and generate point cloud
    cloud, info = _load_and_generate_point_cloud(model_path, cfg, scale, max_size)

    # Remove duplicate points (can occur from overlapping generation passes)
    original_count = len(cloud)
    cloud = cloud.remove_duplicates(tolerance=0.001)  # 1 micron tolerance
    if len(cloud) < original_count:
        console.print(f"[dim]Removed {original_count - len(cloud):,} duplicate points[/dim]")

    # Show point cloud info
    cloud_stats = cloud.stats()
    table = Table(title="Point Cloud")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Points", f"{cloud_stats['num_points']:,}")
    table.add_row("Layers", str(cloud_stats["num_layers"]))
    table.add_row("Size (mm)", f"{cloud_stats['size'][0]:.1f} x {cloud_stats['size'][1]:.1f} x {cloud_stats['size'][2]:.1f}")
    console.print(table)
    
    # Save if requested
    if output:
        output_path = Path(output)
        console.print(f"\n[cyan]Saving point cloud to {output_path}...[/cyan]")

        if output_path.suffix == ".npz":
            cloud.save_npz(str(output_path))
        else:
            cloud.save_xyz(str(output_path))

        console.print(f"[green]Saved {len(cloud):,} points[/green]")
        return

    # --- Checkpoint setup ---
    checkpoint_mgr: CheckpointManager | None = None
    resume_ckpt: CheckpointData | None = None

    if not no_checkpoint:
        checkpoint_mgr = CheckpointManager(checkpoint_dir)

        if resume:
            # Load and validate checkpoint
            try:
                resume_ckpt = checkpoint_mgr.load(resume)
                valid, error = checkpoint_mgr.validate_for_resume(resume_ckpt, cloud)
                if not valid:
                    console.print(f"[bold red]Cannot resume: {error}[/bold red]")
                    raise click.Abort()

                console.print(f"\n[cyan]Resuming job {resume_ckpt.job_id}[/cyan]")
                console.print(f"  Progress: {resume_ckpt.completed_points:,}/{resume_ckpt.total_points:,} points ({resume_ckpt.percent_complete:.1f}%)")
                console.print(f"  Starting from layer {resume_ckpt.current_layer + 1}/{resume_ckpt.total_layers}")
            except FileNotFoundError:
                console.print(f"[bold red]Checkpoint not found: {resume}[/bold red]")
                raise click.Abort()

    # Estimate time (accounting for travel, delays, thermal pauses)
    est_seconds = _estimate_engrave_time(cloud, cfg)

    # Adjust estimate for resumed jobs
    if resume_ckpt:
        remaining_fraction = 1.0 - (resume_ckpt.completed_points / resume_ckpt.total_points)
        est_seconds = est_seconds * remaining_fraction
        console.print(f"\n[yellow]Estimated remaining time: {est_seconds/60:.1f} minutes[/yellow]")
    elif est_seconds < 60:
        console.print(f"\n[yellow]Estimated time: {est_seconds:.0f} seconds[/yellow]")
    else:
        console.print(f"\n[yellow]Estimated time: {est_seconds/60:.1f} minutes[/yellow]")

    if not mock and not dry_run:
        if not click.confirm("\nProceed with engraving?"):
            console.print("[red]Aborted[/red]")
            return

    # Engrave
    action = "Resuming" if resume_ckpt else "Starting"
    console.print(f"\n[bold green]{'DRY RUN - ' if dry_run else ''}{action} engrave...[/bold green]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Engraving...", total=len(cloud))

        # Set initial progress for resumed jobs
        if resume_ckpt:
            progress.update(task, completed=resume_ckpt.completed_points)

        def update_progress(p: EngraveProgress) -> None:
            progress.update(
                task,
                completed=p.completed_points,
                description=f"Layer {p.current_layer + 1}/{p.total_layers}",
            )

        final_checkpoint: CheckpointData | None = None
        try:
            with LaserController(cfg) as laser:
                # Note: Hardware correction table is uploaded by engrave_point_cloud()
                # after entering the marking context to avoid galvoplotter re-init issues
                if cor_file:
                    console.print(f"[cyan]Hardware correction: {cor_file}[/cyan]")

                final_checkpoint = laser.engrave_point_cloud(
                    cloud,
                    progress_callback=update_progress,
                    dry_run=dry_run,
                    live_preview=live_preview,
                    checkpoint_manager=checkpoint_mgr,
                    resume_checkpoint=resume_ckpt,
                    source_file=str(Path(model_path).resolve()),
                )

            console.print("\n[bold green]Engrave complete![/bold green]")
            if final_checkpoint and checkpoint_mgr:
                console.print(f"[dim]Job {final_checkpoint.job_id} finished[/dim]")

        except InterruptedError:
            console.print("\n[bold yellow]Engrave aborted[/bold yellow]")
            if checkpoint_mgr:
                # Find the checkpoint that was just saved
                resumable = checkpoint_mgr.find_resumable(str(Path(model_path).resolve()))
                if resumable:
                    ckpt = resumable[0]
                    ckpt_path = checkpoint_mgr.checkpoint_path(ckpt.job_id)
                    console.print(f"[cyan]Checkpoint saved: {ckpt_path}[/cyan]")
                    console.print(f"[cyan]Resume with: glass3d engrave {model_path} --resume {ckpt.job_id}[/cyan]")
        except Exception as e:
            console.print(f"\n[bold red]Error: {e}[/bold red]")
            if checkpoint_mgr:
                # Find the most recent checkpoint for this source
                resumable = checkpoint_mgr.find_resumable(str(Path(model_path).resolve()))
                if resumable:
                    ckpt = resumable[0]
                    console.print(f"[cyan]Checkpoint available: {ckpt.job_id} ({ckpt.percent_complete:.1f}% complete)[/cyan]")
                    console.print(f"[cyan]Resume with: glass3d engrave {model_path} --resume {ckpt.job_id}[/cyan]")
            raise click.Abort()


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    help="Path to configuration file (for workspace bounds)",
)
@click.option(
    "--strategy", "-s",
    type=click.Choice(["surface", "solid", "grayscale", "contour", "shell"]),
    default="surface",
    help="Point generation strategy",
)
@click.option(
    "--spacing", "-p",
    type=float,
    default=0.1,
    help="Point spacing in mm",
)
@click.option(
    "--max-size",
    type=float,
    default=50.0,
    help="Scale model to fit within this size in mm (single mesh only, ignored for 3MF)",
)
@click.option(
    "--max-points",
    type=int,
    default=50000,
    help="Maximum points to display (default 50000)",
)
def preview(
    model_path: str,
    config: str | None,
    strategy: str,
    spacing: float,
    max_size: float,
    max_points: int,
) -> None:
    """Preview a model's point cloud without engraving.

    MODEL_PATH: Path to model file (STL/OBJ) or 3MF scene file

    Opens a 3D visualization of the generated points using matplotlib.
    Shows the full laser workspace bounds for context.
    """
    from .scene import WorkspaceBounds

    # Load config
    if config:
        cfg = Glass3DConfig.from_file(config)
    else:
        cfg = Glass3DConfig.default()
    cfg.point_cloud.point_spacing_mm = spacing
    # Cast validated Click choice to Literal type
    cfg.point_cloud.strategy = cast(
        Literal["surface", "solid", "grayscale", "contour", "shell"], strategy
    )

    # Get workspace bounds from config's machine params
    workspace = WorkspaceBounds.from_machine_params(cfg.machine)

    console.print(f"\n[bold]Point Cloud Preview[/bold]\n")

    # Load model(s) and generate point cloud
    cloud, info = _load_and_generate_point_cloud(model_path, cfg, scale=1.0, max_size=max_size)

    console.print(f"\nGenerated {len(cloud):,} points")
    console.print(f"Size: {cloud.size[0]:.1f} x {cloud.size[1]:.1f} x {cloud.size[2]:.1f} mm")

    # Try to visualize
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Subsample for visualization if too many points
        if len(cloud) > max_points:
            vis_cloud = cloud.random_sample(max_points)
            console.print(f"[dim](Showing {max_points:,} of {len(cloud):,} points)[/dim]")
        else:
            vis_cloud = cloud

        # Plot points with smaller markers
        ax.scatter(
            vis_cloud.x, vis_cloud.y, vis_cloud.z,
            s=0.3, alpha=0.6, c=vis_cloud.z, cmap='viridis'
        )

        # Draw workspace bounds as wireframe box
        x_min, x_max = workspace.x_range
        y_min, y_max = workspace.y_range
        z_min, z_max = workspace.z_range

        # Define the 8 corners of the workspace
        corners = np.array([
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max],
        ])

        # Define the 12 edges of the box
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
        ]

        for i, j in edges:
            ax.plot3D(
                [corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
                color='steelblue', linewidth=1.5, alpha=0.7
            )

        # Draw floor grid on Z=0 plane
        grid_spacing = 10  # mm
        for x in np.arange(x_min, x_max + grid_spacing, grid_spacing):
            ax.plot3D([x, x], [y_min, y_max], [z_min, z_min],
                     color='lightgray', linewidth=0.3, alpha=0.3)
        for y in np.arange(y_min, y_max + grid_spacing, grid_spacing):
            ax.plot3D([x_min, x_max], [y, y], [z_min, z_min],
                     color='lightgray', linewidth=0.3, alpha=0.3)

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'{Path(model_path).name} - {len(cloud):,} points')

        # Set axis limits with equal scaling to preserve proportions
        # Find the largest dimension and center all axes on that range
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        z_mid = (z_min + z_max) / 2

        ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
        ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
        ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

        plt.tight_layout()
        plt.show()

    except ImportError:
        console.print("[yellow]matplotlib not installed - cannot show preview[/yellow]")
        console.print("Install with: pip install matplotlib")


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
def info(model_path: str) -> None:
    """Show information about a 3D model or 3MF scene file.

    MODEL_PATH: Path to model file (STL/OBJ) or 3MF scene file
    """
    from .scene import Scene

    path = Path(model_path)
    is_3mf = path.suffix.lower() == ".3mf"

    console.print(f"\n[bold]{'Scene' if is_3mf else 'Model'} Info: {path.name}[/bold]\n")

    try:
        if is_3mf:
            # Load 3MF scene
            scene = Scene.from_3mf(path)

            console.print(f"[cyan]Scene name:[/cyan] {scene.name}")
            console.print(f"[cyan]Models:[/cyan] {len(scene.models)}")

            # Check anchor status for all models (by name and geometry)
            anchor_statuses = {m.id: scene.get_anchor_status(m) for m in scene.models}
            anchor_count = sum(1 for is_anchor, _ in anchor_statuses.values() if is_anchor)
            if anchor_count > 0:
                console.print(f"[dim]({anchor_count} anchor model(s) will be excluded)[/dim]")
            console.print()

            # Show model table
            table = Table(title="Models")
            table.add_column("Name", style="cyan")
            table.add_column("Position (mm)", style="green")
            table.add_column("Rotation (deg)", style="yellow")
            table.add_column("Scale", style="magenta")
            table.add_column("Status", style="dim")

            for model in scene.models:
                pos = model.transform.position
                rot = model.transform.rotation
                is_anchor, reason = anchor_statuses[model.id]
                if is_anchor:
                    status = f"[yellow]anchor ({reason})[/yellow]"
                elif reason.startswith("partial:"):
                    count = reason.split(":")[1]
                    status = f"[dim]{count} anchor part(s) filtered[/dim]"
                else:
                    status = ""
                table.add_row(
                    model.name,
                    f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})",
                    f"({rot[0]:.1f}, {rot[1]:.1f}, {rot[2]:.1f})",
                    f"{model.transform.scale:.2f}",
                    status,
                )
            console.print(table)

        else:
            # Single mesh file
            loader = MeshLoader(model_path)
            stats = loader.stats()

            table = Table()
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("File", str(stats["path"]))
            table.add_row("Vertices", f"{stats['num_vertices']:,}")
            table.add_row("Faces", f"{stats['num_faces']:,}")
            table.add_row("Watertight", "Yes" if stats["is_watertight"] else "No")
            table.add_row(
                "Bounds (min)",
                f"({stats['bounds_min'][0]:.2f}, {stats['bounds_min'][1]:.2f}, {stats['bounds_min'][2]:.2f})"
            )
            table.add_row(
                "Bounds (max)",
                f"({stats['bounds_max'][0]:.2f}, {stats['bounds_max'][1]:.2f}, {stats['bounds_max'][2]:.2f})"
            )
            table.add_row(
                "Size",
                f"{stats['size'][0]:.2f} x {stats['size'][1]:.2f} x {stats['size'][2]:.2f}"
            )

            if stats["volume"]:
                table.add_row("Volume", f"{stats['volume']:.2f} cubic units")

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error loading: {e}[/red]")
        raise click.Abort()


@main.command()
def strategies() -> None:
    """List available point generation strategies."""
    console.print("\n[bold]Available Strategies[/bold]\n")
    
    table = Table()
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    
    for strat in list_strategies():
        table.add_row(strat["name"], strat["description"])
    
    console.print(table)


@main.command()
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="glass3d_config.json",
    help="Output path for config file",
)
@click.option(
    "--material", "-m",
    type=click.Choice(["k9", "bk7", "fused_silica"]),
    default="k9",
    help="Material preset",
)
def init_config(output: str, material: str) -> None:
    """Generate a default configuration file."""
    try:
        cfg = Glass3DConfig.for_material(material)
        cfg.to_file(output)
        console.print(f"[green]Created config file: {output}[/green]")
        console.print(f"Material: {cfg.material.name}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@main.command("import-device")
@click.argument("device_file", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Output path for config file (if not specified, just shows device info)",
)
@click.option(
    "--base-config", "-b",
    type=click.Path(exists=True),
    default=None,
    help="Base config file to merge with (preserves laser/material settings)",
)
@click.option(
    "--cor-file",
    type=click.Path(),
    default=None,
    help="Output path for hardware correction file (.cor). Auto-generated if --output is set.",
)
def import_device(device_file: str, output: str | None, base_config: str | None, cor_file: str | None) -> None:
    """Import device settings from a LightBurn export file.

    DEVICE_FILE: Path to LightBurn export (.lbzip) file

    This imports lens correction parameters, workspace size, and other
    device-specific settings from LightBurn's device export format.

    To export from LightBurn:
      1. Edit > Device Settings
      2. Click "Export" button
      3. Save as .lbzip file

    Examples:
        # Show device info only
        glass3d import-device uvlaser_50mm.lbzip

        # Create new config file with device settings
        glass3d import-device uvlaser_50mm.lbzip -o config.json

        # Merge with existing config (preserves laser/material settings)
        glass3d import-device uvlaser_50mm.lbzip -b existing.json -o updated.json
    """
    from .device import load_lightburn_device

    console.print(f"\n[bold]Import LightBurn Device[/bold]\n")

    try:
        device = load_lightburn_device(device_file)
    except Exception as e:
        console.print(f"[red]Error loading device file: {e}[/red]")
        raise click.Abort()

    # Display device info
    table = Table(title=f"Device: {device.name}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Name", device.name)
    table.add_row("Type", device.device_type)
    table.add_row("Workspace Size", f"{device.width_mm} x {device.height_mm} mm")
    table.add_row("Mirror X/Y", f"{device.mirror_x} / {device.mirror_y}")
    table.add_row("Field Angle", f"{device.field_angle}°")
    table.add_row("Field Offset", f"({device.field_offset_x}, {device.field_offset_y}) mm")
    console.print(table)

    # Lens correction table
    console.print()
    lens_table = Table(title="Lens Correction Parameters")
    lens_table.add_column("Parameter", style="cyan")
    lens_table.add_column("X Axis (Galvo 2)" if not device.galvo_1_is_x else "X Axis (Galvo 1)", style="green")
    lens_table.add_column("Y Axis (Galvo 1)" if not device.galvo_1_is_x else "Y Axis (Galvo 2)", style="yellow")

    if device.galvo_1_is_x:
        x_vals = (device.galvo_1_scale, device.galvo_1_bulge, device.galvo_1_trapezoid, device.galvo_1_skew, device.galvo_1_sign)
        y_vals = (device.galvo_2_scale, device.galvo_2_bulge, device.galvo_2_trapezoid, device.galvo_2_skew, device.galvo_2_sign)
    else:
        x_vals = (device.galvo_2_scale, device.galvo_2_bulge, device.galvo_2_trapezoid, device.galvo_2_skew, device.galvo_2_sign)
        y_vals = (device.galvo_1_scale, device.galvo_1_bulge, device.galvo_1_trapezoid, device.galvo_1_skew, device.galvo_1_sign)

    lens_table.add_row("Scale", f"{x_vals[0]:.6f}", f"{y_vals[0]:.6f}")
    lens_table.add_row("Bulge", f"{x_vals[1]:.6f}", f"{y_vals[1]:.6f}")
    lens_table.add_row("Trapezoid", f"{x_vals[2]:.6f}", f"{y_vals[2]:.6f}")
    lens_table.add_row("Skew", f"{x_vals[3]:.6f}", f"{y_vals[3]:.6f}")
    lens_table.add_row("Sign", f"{x_vals[4]}", f"{y_vals[4]}")
    console.print(lens_table)

    # Speed info
    console.print()
    speed_table = Table(title="Speed Defaults")
    speed_table.add_column("Property", style="cyan")
    speed_table.add_column("Value", style="green")
    speed_table.add_row("Max Speed", f"{device.max_speed} mm/s")
    speed_table.add_row("Jump Speed", f"{device.default_jump_speed} mm/s")
    speed_table.add_row("Frame Speed", f"{device.frame_speed} mm/s")
    speed_table.add_row("Frequency Range", f"{device.laser_min_freq} - {device.laser_max_freq} kHz")
    console.print(speed_table)

    # Timing info
    console.print()
    timing_table = Table(title="Galvo Timing")
    timing_table.add_column("Property", style="cyan")
    timing_table.add_column("Value", style="green")
    timing_table.add_row("Jump Delay (short)", f"{device.jump_delay_min} µs")
    timing_table.add_row("Jump Delay (long)", f"{device.jump_delay_max} µs")
    timing_table.add_row("Jump Distance Threshold", f"{device.jump_distance_threshold} mm")
    timing_table.add_row("Laser On Delay", f"{device.laser_on_delay} µs")
    timing_table.add_row("Laser Off Delay", f"{device.laser_off_delay} µs")
    timing_table.add_row("Polygon Delay", f"{device.polygon_delay} µs")
    console.print(timing_table)

    # Save config if output specified
    if output:
        console.print()

        # Load base config or create default
        if base_config:
            base_cfg = Glass3DConfig.from_file(base_config)
            console.print(f"[dim]Merging with base config: {base_config}[/dim]")
        else:
            base_cfg = Glass3DConfig.default()

        # Generate .cor file for hardware correction first
        from .device.correction import generate_correction_table

        lens_correction = device.to_lens_correction()
        table = generate_correction_table(
            lens_correction,
            field_size_mm=(device.width_mm, device.height_mm),
        )

        # Determine cor file path
        if cor_file:
            cor_path = Path(cor_file)
        else:
            # Auto-generate path based on output config
            cor_path = Path(output).with_suffix(".cor")

        table.to_cor_file(cor_path)

        # Convert device to config with cor_file reference
        cfg = device.to_config(base_cfg)
        cfg = cfg.model_copy(update={"cor_file": cor_path})

        # Save config
        cfg.to_file(output)
        console.print(f"\n[green]Created config file: {output}[/green]")
        console.print(f"Workspace: {cfg.machine.field_size_mm[0]} x {cfg.machine.field_size_mm[1]} mm")
        console.print(f"Lens correction: [bold green]enabled[/bold green]")
        console.print(f"Hardware correction: [bold green]{cor_path}[/bold green]")

        console.print(f"\n[green]Created correction file: {cor_path}[/green]")
        console.print(table.summary())
        console.print()
        console.print("[cyan]Use with engrave:[/cyan]")
        console.print(f"  glass3d engrave model.stl -c {output}")
    else:
        # Just generate cor file if requested
        if cor_file:
            from .device.correction import generate_correction_table

            lens_correction = device.to_lens_correction()
            table = generate_correction_table(
                lens_correction,
                field_size_mm=(device.width_mm, device.height_mm),
            )
            table.to_cor_file(cor_file)
            console.print(f"\n[green]Created correction file: {cor_file}[/green]")
            console.print(table.summary())
        else:
            console.print("\n[dim]Use -o/--output to save as config file[/dim]")
            console.print("[dim]Use --cor-file to generate hardware correction file[/dim]")


@main.command("generate-anchor")
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="anchor.stl",
    help="Output path for anchor STL file",
)
@click.option(
    "--size",
    type=float,
    default=5.0,
    help="Size of anchor plate in mm (default 5mm square)",
)
@click.option(
    "--thickness",
    type=float,
    default=0.5,
    help="Thickness of anchor plate in mm (default 0.5mm)",
)
def generate_anchor(output: str, size: float, thickness: float) -> None:
    """Generate an anchor STL for use in slicer software.

    The anchor is a small flat plate that sits at Z=0. Import it into your
    slicer alongside your actual models to anchor floating parts to the bed.

    Models named "anchor" or starting with "_anchor"/"anchor_" are
    automatically excluded from point cloud generation.

    Example workflow:
        1. glass3d generate-anchor -o anchor.stl
        2. Import anchor.stl and your models into PrusaSlicer/Cura
        3. Position your models above the anchor (floating in Z)
        4. Export as 3MF
        5. glass3d engrave scene.3mf --mock --dry-run
    """
    import trimesh

    # Create a flat box (plate) at Z=0
    # The box is centered, so we offset it to sit on Z=0
    plate = trimesh.creation.box(extents=[size, size, thickness])

    # Move plate so bottom is at Z=0
    plate.apply_translation([0, 0, thickness / 2])

    # Export
    output_path = Path(output)
    plate.export(str(output_path))

    console.print(f"[green]Created anchor: {output_path}[/green]")
    console.print(f"Size: {size} x {size} x {thickness} mm")
    console.print()
    console.print("[cyan]Usage:[/cyan]")
    console.print("  1. Import this anchor into your slicer alongside your models")
    console.print("  2. Position your models above the anchor")
    console.print("  3. Export as 3MF")
    console.print("  4. The anchor will be automatically excluded from engraving")
    console.print()
    console.print("[dim]Tip: Name the anchor 'anchor' in your slicer for automatic exclusion[/dim]")


@main.command()
@click.option(
    "--mock",
    is_flag=True,
    help="Use mock laser connection",
)
def test_connection(mock: bool) -> None:
    """Test connection to the laser controller."""
    console.print("\n[bold]Testing Laser Connection[/bold]\n")

    cfg = Glass3DConfig.default()
    cfg.mock_laser = mock

    try:
        with console.status("Connecting..."):
            controller = LaserController(cfg)
            controller.connect()

        console.print("[green]Connection successful![/green]")

        if mock:
            console.print("[dim](Using mock connection)[/dim]")

        controller.disconnect()
        console.print("[green]Disconnected cleanly[/green]")

    except Exception as e:
        console.print(f"[red]Connection failed: {e}[/red]")
        raise click.Abort()


# -----------------------------------------------------------------------------
# Job/Checkpoint commands
# -----------------------------------------------------------------------------

@main.group()
def jobs() -> None:
    """Manage engrave job checkpoints for fault recovery."""
    pass


@jobs.command("list")
@click.option(
    "--checkpoint-dir",
    type=click.Path(),
    default=None,
    help="Directory for checkpoint files (default: ./glass3d_checkpoints)",
)
@click.option(
    "--all", "-a", "show_all",
    is_flag=True,
    help="Show all checkpoints (including completed)",
)
def jobs_list(checkpoint_dir: str | None, show_all: bool) -> None:
    """List saved job checkpoints."""
    mgr = CheckpointManager(checkpoint_dir)
    checkpoints = mgr.list_checkpoints()

    if not checkpoints:
        console.print("[dim]No checkpoints found[/dim]")
        console.print(f"[dim]Checkpoint directory: {mgr.checkpoint_dir}[/dim]")
        return

    if not show_all:
        checkpoints = [c for c in checkpoints if c.is_resumable]

    if not checkpoints:
        console.print("[dim]No resumable checkpoints (use --all to see completed jobs)[/dim]")
        return

    table = Table(title="Job Checkpoints")
    table.add_column("Job ID", style="cyan")
    table.add_column("Status", style="yellow")
    table.add_column("Progress", style="green")
    table.add_column("Source", style="dim")
    table.add_column("Updated", style="dim")

    for ckpt in checkpoints:
        status_style = {
            "in_progress": "yellow",
            "completed": "green",
            "aborted": "red",
        }.get(ckpt.status, "white")

        source = Path(ckpt.source_file).name if ckpt.source_file else "-"
        progress = f"{ckpt.percent_complete:.1f}% ({ckpt.completed_points:,}/{ckpt.total_points:,})"

        table.add_row(
            ckpt.job_id,
            f"[{status_style}]{ckpt.status}[/{status_style}]",
            progress,
            source,
            ckpt.updated_at[:19],  # Trim microseconds
        )

    console.print(table)
    console.print(f"\n[dim]Checkpoint directory: {mgr.checkpoint_dir}[/dim]")


@jobs.command("show")
@click.argument("job_id")
@click.option(
    "--checkpoint-dir",
    type=click.Path(),
    default=None,
    help="Directory for checkpoint files",
)
def jobs_show(job_id: str, checkpoint_dir: str | None) -> None:
    """Show details of a specific checkpoint."""
    mgr = CheckpointManager(checkpoint_dir)

    try:
        ckpt = mgr.load(job_id)
    except FileNotFoundError:
        console.print(f"[red]Checkpoint not found: {job_id}[/red]")
        raise click.Abort()

    console.print(f"\n[bold]Job: {ckpt.job_id}[/bold]\n")

    table = Table(show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Status", ckpt.status)
    table.add_row("Created", ckpt.created_at)
    table.add_row("Updated", ckpt.updated_at)
    table.add_row("Source file", ckpt.source_file or "-")
    table.add_row("Point cloud hash", ckpt.point_cloud_hash)
    table.add_row("Total points", f"{ckpt.total_points:,}")
    table.add_row("Completed points", f"{ckpt.completed_points:,}")
    table.add_row("Progress", f"{ckpt.percent_complete:.2f}%")
    table.add_row("Current layer", f"{ckpt.current_layer}/{ckpt.total_layers}")

    if ckpt.error_message:
        table.add_row("Error", f"[red]{ckpt.error_message}[/red]")

    if ckpt.config_snapshot:
        table.add_row("", "")
        table.add_row("[bold]Config snapshot[/bold]", "")
        for key, value in ckpt.config_snapshot.items():
            table.add_row(f"  {key}", str(value))

    console.print(table)

    if ckpt.is_resumable and ckpt.source_file:
        console.print(f"\n[cyan]Resume with: glass3d engrave {ckpt.source_file} --resume {ckpt.job_id}[/cyan]")


@jobs.command("delete")
@click.argument("job_id")
@click.option(
    "--checkpoint-dir",
    type=click.Path(),
    default=None,
    help="Directory for checkpoint files",
)
@click.option("--force", "-f", is_flag=True, help="Delete without confirmation")
def jobs_delete(job_id: str, checkpoint_dir: str | None, force: bool) -> None:
    """Delete a job checkpoint."""
    mgr = CheckpointManager(checkpoint_dir)

    try:
        ckpt = mgr.load(job_id)
    except FileNotFoundError:
        console.print(f"[red]Checkpoint not found: {job_id}[/red]")
        raise click.Abort()

    if not force:
        if ckpt.is_resumable:
            console.print(f"[yellow]Warning: Job {job_id} is resumable ({ckpt.percent_complete:.1f}% complete)[/yellow]")
        if not click.confirm(f"Delete checkpoint {job_id}?"):
            console.print("[dim]Cancelled[/dim]")
            return

    if mgr.delete(job_id):
        console.print(f"[green]Deleted checkpoint: {job_id}[/green]")
    else:
        console.print(f"[red]Failed to delete checkpoint: {job_id}[/red]")


@jobs.command("clean")
@click.option(
    "--checkpoint-dir",
    type=click.Path(),
    default=None,
    help="Directory for checkpoint files",
)
@click.option("--force", "-f", is_flag=True, help="Delete without confirmation")
def jobs_clean(checkpoint_dir: str | None, force: bool) -> None:
    """Delete all completed and aborted checkpoints."""
    mgr = CheckpointManager(checkpoint_dir)
    checkpoints = mgr.list_checkpoints()

    to_delete = [c for c in checkpoints if not c.is_resumable]

    if not to_delete:
        console.print("[dim]No completed/aborted checkpoints to clean[/dim]")
        return

    console.print(f"Found {len(to_delete)} checkpoint(s) to delete:")
    for ckpt in to_delete:
        console.print(f"  - {ckpt.job_id} ({ckpt.status})")

    if not force:
        if not click.confirm("Delete these checkpoints?"):
            console.print("[dim]Cancelled[/dim]")
            return

    deleted = 0
    for ckpt in to_delete:
        if mgr.delete(ckpt.job_id):
            deleted += 1

    console.print(f"[green]Deleted {deleted} checkpoint(s)[/green]")


# -----------------------------------------------------------------------------
# Scene commands
# -----------------------------------------------------------------------------

@main.group()
def scene() -> None:
    """Scene management commands for multi-model arrangement."""
    pass


@scene.command("create")
@click.argument("output", type=click.Path())
@click.option("--name", "-n", default="New Scene", help="Scene name")
def scene_create(output: str, name: str) -> None:
    """Create a new empty scene file.

    OUTPUT: Path for the new scene file (.g3scene)
    """
    from .scene import Scene, WorkspaceBounds

    scene_obj = Scene(
        name=name,
        workspace=WorkspaceBounds(),
    )

    output_path = Path(output)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".g3scene")

    scene_obj.save(output_path)
    console.print(f"[green]Created scene: {output_path}[/green]")
    console.print(f"Name: {name}")
    console.print(f"Workspace: {scene_obj.workspace.size[0]:.0f} x {scene_obj.workspace.size[1]:.0f} x {scene_obj.workspace.size[2]:.0f} mm")


@scene.command("add")
@click.argument("scene_file", type=click.Path(exists=True))
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--name", "-n", default=None, help="Display name for the model")
@click.option(
    "--position", "-p",
    nargs=3, type=float,
    default=(0.0, 0.0, 0.0),
    help="XYZ position in mm",
)
@click.option(
    "--rotation", "-r",
    nargs=3, type=float,
    default=(0.0, 0.0, 0.0),
    help="XYZ rotation in degrees",
)
@click.option("--scale", "-s", type=float, default=1.0, help="Scale factor")
def scene_add(
    scene_file: str,
    model_path: str,
    name: str | None,
    position: tuple[float, float, float],
    rotation: tuple[float, float, float],
    scale: float,
) -> None:
    """Add a model to a scene.

    SCENE_FILE: Path to the scene file
    MODEL_PATH: Path to the mesh file (STL/OBJ/etc)
    """
    from .scene import Scene

    scene_obj = Scene.load(scene_file)

    model = scene_obj.add_model(
        source_path=model_path,
        name=name,
        position=position,
        rotation=rotation,
        scale=scale,
    )

    scene_obj.save(scene_file)

    console.print(f"[green]Added model to scene[/green]")
    console.print(f"  ID: {model.id}")
    console.print(f"  Name: {model.name}")
    console.print(f"  Position: {position}")
    console.print(f"  Rotation: {rotation}")
    console.print(f"  Scale: {scale}")
    console.print(f"\nScene now has {len(scene_obj.models)} model(s)")


@scene.command("remove")
@click.argument("scene_file", type=click.Path(exists=True))
@click.argument("model_id")
def scene_remove(scene_file: str, model_id: str) -> None:
    """Remove a model from a scene.

    SCENE_FILE: Path to the scene file
    MODEL_ID: ID of the model to remove
    """
    from .scene import Scene

    scene_obj = Scene.load(scene_file)

    if scene_obj.remove_model(model_id):
        scene_obj.save(scene_file)
        console.print(f"[green]Removed model {model_id}[/green]")
        console.print(f"Scene now has {len(scene_obj.models)} model(s)")
    else:
        console.print(f"[red]Model {model_id} not found in scene[/red]")
        raise click.Abort()


@scene.command("info")
@click.argument("scene_file", type=click.Path(exists=True))
def scene_info(scene_file: str) -> None:
    """Show information about a scene.

    SCENE_FILE: Path to the scene file
    """
    from .scene import Scene

    scene_obj = Scene.load(scene_file)

    console.print(f"\n[bold]Scene: {scene_obj.name}[/bold]\n")

    # Workspace info
    ws = scene_obj.workspace
    console.print(f"[cyan]Workspace:[/cyan]")
    console.print(f"  X: {ws.x_range[0]:.1f} to {ws.x_range[1]:.1f} mm ({ws.size[0]:.0f} mm)")
    console.print(f"  Y: {ws.y_range[0]:.1f} to {ws.y_range[1]:.1f} mm ({ws.size[1]:.0f} mm)")
    console.print(f"  Z: {ws.z_range[0]:.1f} to {ws.z_range[1]:.1f} mm ({ws.size[2]:.0f} mm)")

    console.print(f"\n[cyan]Defaults:[/cyan]")
    console.print(f"  Strategy: {scene_obj.default_strategy}")
    console.print(f"  Point spacing: {scene_obj.default_point_spacing_mm} mm")
    console.print(f"  Layer height: {scene_obj.default_layer_height_mm} mm")

    if not scene_obj.models:
        console.print("\n[yellow]No models in scene[/yellow]")
        return

    console.print(f"\n[cyan]Models ({len(scene_obj.models)}):[/cyan]")

    table = Table()
    table.add_column("ID", style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("Position", style="green")
    table.add_column("Rotation", style="yellow")
    table.add_column("Scale", style="magenta")
    table.add_column("Source", style="dim")

    for model in scene_obj.models:
        pos = model.transform.position
        rot = model.transform.rotation
        table.add_row(
            model.id,
            model.name,
            f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})",
            f"({rot[0]:.0f}, {rot[1]:.0f}, {rot[2]:.0f})",
            f"{model.transform.scale:.2f}",
            Path(model.source_path).name,
        )

    console.print(table)

    # Validate bounds
    is_valid, errors = scene_obj.validate_bounds()
    if not is_valid:
        console.print("\n[bold red]Bounds Warnings:[/bold red]")
        for error in errors:
            console.print(f"  [red]{error}[/red]")
    else:
        console.print("\n[green]All models within workspace bounds[/green]")


@scene.command("engrave")
@click.argument("scene_file", type=click.Path(exists=True))
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Save combined point cloud to file (skip engraving)",
)
@click.option(
    "--dry-run", "-n",
    is_flag=True,
    help="Preview with red dot, don't fire laser",
)
@click.option(
    "--mock",
    is_flag=True,
    help="Use mock laser connection (for testing)",
)
@click.option(
    "--live-preview",
    is_flag=True,
    help="Show live matplotlib preview (updates per layer)",
)
@click.option(
    "--resume", "-r",
    type=str,
    default=None,
    help="Resume from checkpoint (job ID or path to checkpoint file)",
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(),
    default=None,
    help="Directory for checkpoint files (default: ./glass3d_checkpoints)",
)
@click.option(
    "--no-checkpoint",
    is_flag=True,
    help="Disable checkpoint saving (not recommended for long jobs)",
)
def scene_engrave(
    scene_file: str,
    config: str | None,
    output: str | None,
    dry_run: bool,
    mock: bool,
    live_preview: bool,
    resume: str | None,
    checkpoint_dir: str | None,
    no_checkpoint: bool,
) -> None:
    """Engrave all models in a scene.

    SCENE_FILE: Path to the scene file
    """
    from .scene import Scene

    # Load config
    if config:
        cfg = Glass3DConfig.from_file(config)
    else:
        cfg = Glass3DConfig.default()

    cfg.mock_laser = mock
    cfg.engrave.dry_run = dry_run

    # Load scene
    scene_obj = Scene.load(scene_file)

    console.print(f"\n[bold]Scene Engrave: {scene_obj.name}[/bold]\n")
    console.print(f"Models: {len(scene_obj.models)}")

    # Validate bounds
    is_valid, errors = scene_obj.validate_bounds()
    if not is_valid:
        console.print("\n[bold red]Bounds Errors:[/bold red]")
        for error in errors:
            console.print(f"  [red]{error}[/red]")
        if not click.confirm("\nProceed anyway?"):
            raise click.Abort()

    # Generate combined point cloud
    console.print("\n[cyan]Generating combined point cloud...[/cyan]")

    with console.status("Processing models..."):
        cloud = scene_obj.to_combined_point_cloud(cfg.point_cloud)

    # Show point cloud info
    cloud_stats = cloud.stats()
    table = Table(title="Combined Point Cloud")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Points", f"{cloud_stats['num_points']:,}")
    table.add_row("Layers", str(cloud_stats["num_layers"]))
    table.add_row("Size (mm)", f"{cloud_stats['size'][0]:.1f} x {cloud_stats['size'][1]:.1f} x {cloud_stats['size'][2]:.1f}")
    console.print(table)

    # Save if requested
    if output:
        output_path = Path(output)
        console.print(f"\n[cyan]Saving point cloud to {output_path}...[/cyan]")

        if output_path.suffix == ".npz":
            cloud.save_npz(str(output_path))
        else:
            cloud.save_xyz(str(output_path))

        console.print(f"[green]Saved {len(cloud):,} points[/green]")
        return

    # --- Checkpoint setup ---
    checkpoint_mgr: CheckpointManager | None = None
    resume_ckpt: CheckpointData | None = None

    if not no_checkpoint:
        checkpoint_mgr = CheckpointManager(checkpoint_dir)

        if resume:
            try:
                resume_ckpt = checkpoint_mgr.load(resume)
                valid, error = checkpoint_mgr.validate_for_resume(resume_ckpt, cloud)
                if not valid:
                    console.print(f"[bold red]Cannot resume: {error}[/bold red]")
                    raise click.Abort()

                console.print(f"\n[cyan]Resuming job {resume_ckpt.job_id}[/cyan]")
                console.print(f"  Progress: {resume_ckpt.completed_points:,}/{resume_ckpt.total_points:,} points ({resume_ckpt.percent_complete:.1f}%)")
                console.print(f"  Starting from layer {resume_ckpt.current_layer + 1}/{resume_ckpt.total_layers}")
            except FileNotFoundError:
                console.print(f"[bold red]Checkpoint not found: {resume}[/bold red]")
                raise click.Abort()

    # Estimate time (accounting for travel, delays, thermal pauses)
    est_seconds = _estimate_engrave_time(cloud, cfg)

    if resume_ckpt:
        remaining_fraction = 1.0 - (resume_ckpt.completed_points / resume_ckpt.total_points)
        est_seconds = est_seconds * remaining_fraction
        console.print(f"\n[yellow]Estimated remaining time: {est_seconds/60:.1f} minutes[/yellow]")
    elif est_seconds < 60:
        console.print(f"\n[yellow]Estimated time: {est_seconds:.0f} seconds[/yellow]")
    else:
        console.print(f"\n[yellow]Estimated time: {est_seconds/60:.1f} minutes[/yellow]")

    if not mock and not dry_run:
        if not click.confirm("\nProceed with engraving?"):
            console.print("[red]Aborted[/red]")
            return

    # Engrave
    action = "Resuming" if resume_ckpt else "Starting"
    console.print(f"\n[bold green]{'DRY RUN - ' if dry_run else ''}{action} engrave...[/bold green]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Engraving...", total=len(cloud))

        if resume_ckpt:
            progress.update(task, completed=resume_ckpt.completed_points)

        def update_progress(p: EngraveProgress) -> None:
            progress.update(
                task,
                completed=p.completed_points,
                description=f"Layer {p.current_layer + 1}/{p.total_layers}",
            )

        final_checkpoint: CheckpointData | None = None
        try:
            with LaserController(cfg) as laser:
                final_checkpoint = laser.engrave_point_cloud(
                    cloud,
                    progress_callback=update_progress,
                    dry_run=dry_run,
                    live_preview=live_preview,
                    checkpoint_manager=checkpoint_mgr,
                    resume_checkpoint=resume_ckpt,
                    source_file=str(Path(scene_file).resolve()),
                )

            console.print("\n[bold green]Engrave complete![/bold green]")
            if final_checkpoint and checkpoint_mgr:
                console.print(f"[dim]Job {final_checkpoint.job_id} finished[/dim]")

        except InterruptedError:
            console.print("\n[bold yellow]Engrave aborted[/bold yellow]")
            if checkpoint_mgr:
                resumable = checkpoint_mgr.find_resumable(str(Path(scene_file).resolve()))
                if resumable:
                    ckpt = resumable[0]
                    ckpt_path = checkpoint_mgr.checkpoint_path(ckpt.job_id)
                    console.print(f"[cyan]Checkpoint saved: {ckpt_path}[/cyan]")
                    console.print(f"[cyan]Resume with: glass3d scene engrave {scene_file} --resume {ckpt.job_id}[/cyan]")
        except Exception as e:
            console.print(f"\n[bold red]Error: {e}[/bold red]")
            if checkpoint_mgr:
                resumable = checkpoint_mgr.find_resumable(str(Path(scene_file).resolve()))
                if resumable:
                    ckpt = resumable[0]
                    console.print(f"[cyan]Checkpoint available: {ckpt.job_id} ({ckpt.percent_complete:.1f}% complete)[/cyan]")
                    console.print(f"[cyan]Resume with: glass3d scene engrave {scene_file} --resume {ckpt.job_id}[/cyan]")
            raise click.Abort()


# -----------------------------------------------------------------------------
# Calibration commands
# -----------------------------------------------------------------------------

@main.group()
def calibrate() -> None:
    """Calibration and lens correction tools."""
    pass


@calibrate.command("compare-coords")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    default=None,
    help="Config file with lens correction settings",
)
@click.option(
    "--device", "-d",
    type=click.Path(exists=True),
    default=None,
    help="LightBurn device file (.lbzip) to load settings from",
)
@click.option(
    "--grid-size", "-g",
    type=int,
    default=5,
    help="Number of grid points per axis (e.g., 5 = 5x5 = 25 points)",
)
@click.option(
    "--format", "-f",
    "output_format",
    type=click.Choice(["table", "csv"]),
    default="table",
    help="Output format",
)
def calibrate_compare_coords(
    config: str | None,
    device: str | None,
    grid_size: int,
    output_format: str,
) -> None:
    """Output transformed coordinates for comparison with LightBurn.

    Generate a grid of test points, apply lens correction, and output
    the resulting galvo coordinates. Compare these with LightBurn's
    preview coordinates to verify calibration.

    Examples:

        glass3d calibrate compare-coords --device mydevice.lbzip

        glass3d calibrate compare-coords --config config.json --grid-size 9

        glass3d calibrate compare-coords --format csv > coords.csv
    """
    from .device.calibration import compare_coordinates, format_comparison_table, format_comparison_csv
    from .device.lightburn import load_lightburn_device

    # Load configuration
    if config:
        cfg = Glass3DConfig.from_file(config)
    elif device:
        lb_device = load_lightburn_device(device)
        cfg = lb_device.to_glass3d_config()
    else:
        cfg = Glass3DConfig.default()

    # Check if lens correction is enabled
    if cfg.lens_correction is None or not cfg.lens_correction.enabled:
        console.print("[yellow]Warning: Lens correction is disabled or not configured[/yellow]")
        console.print("[dim]Results will show uncorrected coordinate transformation[/dim]\n")

    # Generate comparison data
    points = compare_coordinates(cfg, grid_size)

    # Output
    if output_format == "csv":
        output = format_comparison_csv(points)
        click.echo(output)
    else:
        console.print(f"\n[bold]Coordinate Comparison[/bold] ({grid_size}x{grid_size} grid)\n")
        console.print(f"Field size: {cfg.machine.field_size_mm[0]}x{cfg.machine.field_size_mm[1]} mm")
        if cfg.lens_correction and cfg.lens_correction.enabled:
            console.print("[green]Lens correction: enabled[/green]")
        else:
            console.print("[yellow]Lens correction: disabled[/yellow]")
        console.print()

        output = format_comparison_table(points)
        console.print(output)

        console.print("\n[dim]Compare output_galvo values with LightBurn's preview coordinates[/dim]")


if __name__ == "__main__":
    main()
