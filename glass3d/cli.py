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

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

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

        # Count anchor models
        anchor_count = sum(1 for m in scene.models if Scene.is_anchor_model(m.name))
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
            is_anchor = Scene.is_anchor_model(model.name)
            table.add_row(
                model.name,
                f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})",
                f"({rot[0]:.0f}, {rot[1]:.0f}, {rot[2]:.0f})",
                f"{model.transform.scale:.2f}",
                "[yellow]anchor (skipped)[/yellow]" if is_anchor else "",
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
    type=click.Choice(["surface", "solid", "grayscale", "contour"]),
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
@click.pass_context
def engrave(
    ctx: click.Context,
    model_path: str,
    config: str | None,
    strategy: str,
    spacing: float,
    layer_height: float,
    scale: float,
    max_size: float | None,
    dry_run: bool,
    mock: bool,
    output: str | None,
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
    cfg.point_cloud.strategy = strategy
    cfg.engrave.dry_run = dry_run

    console.print(f"\n[bold]Glass3D Engraver[/bold]\n")

    # Load model(s) and generate point cloud
    cloud, info = _load_and_generate_point_cloud(model_path, cfg, scale, max_size)
    
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
    
    # Estimate time
    points_per_second = 1000 / cfg.laser.point_dwell_ms  # Rough estimate
    est_seconds = len(cloud) / points_per_second
    console.print(f"\n[yellow]Estimated time: {est_seconds/60:.1f} minutes[/yellow]")
    
    if not mock and not dry_run:
        if not click.confirm("\nProceed with engraving?"):
            console.print("[red]Aborted[/red]")
            return
    
    # Engrave
    console.print(f"\n[bold green]{'DRY RUN - ' if dry_run else ''}Starting engrave...[/bold green]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Engraving...", total=len(cloud))
        
        def update_progress(p: EngraveProgress) -> None:
            progress.update(
                task,
                completed=p.completed_points,
                description=f"Layer {p.current_layer + 1}/{p.total_layers}",
            )
        
        try:
            with LaserController(cfg) as laser:
                laser.engrave_point_cloud(
                    cloud,
                    progress_callback=update_progress,
                    dry_run=dry_run,
                )
            
            console.print("\n[bold green]Engrave complete![/bold green]")
            
        except InterruptedError:
            console.print("\n[bold yellow]Engrave aborted[/bold yellow]")
        except Exception as e:
            console.print(f"\n[bold red]Error: {e}[/bold red]")
            raise click.Abort()


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--strategy", "-s",
    type=click.Choice(["surface", "solid", "grayscale", "contour"]),
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

    cfg = Glass3DConfig.default()
    cfg.point_cloud.point_spacing_mm = spacing
    cfg.point_cloud.strategy = strategy

    # Get workspace bounds (default or from config)
    workspace = WorkspaceBounds()

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
                color='gray', linewidth=0.8, alpha=0.5
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

        # Set axis limits to workspace bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        # Try to set equal aspect ratio
        ax.set_box_aspect([
            workspace.size[0],
            workspace.size[1],
            workspace.size[2],
        ])

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

            # Count anchor models
            anchor_count = sum(1 for m in scene.models if Scene.is_anchor_model(m.name))
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
                is_anchor = Scene.is_anchor_model(model.name)
                table.add_row(
                    model.name,
                    f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})",
                    f"({rot[0]:.1f}, {rot[1]:.1f}, {rot[2]:.1f})",
                    f"{model.transform.scale:.2f}",
                    "[yellow]anchor (skipped)[/yellow]" if is_anchor else "",
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
def scene_engrave(
    scene_file: str,
    config: str | None,
    output: str | None,
    dry_run: bool,
    mock: bool,
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

    # Estimate time
    points_per_second = 1000 / cfg.laser.point_dwell_ms
    est_seconds = len(cloud) / points_per_second
    console.print(f"\n[yellow]Estimated time: {est_seconds/60:.1f} minutes[/yellow]")

    if not mock and not dry_run:
        if not click.confirm("\nProceed with engraving?"):
            console.print("[red]Aborted[/red]")
            return

    # Engrave
    console.print(f"\n[bold green]{'DRY RUN - ' if dry_run else ''}Starting engrave...[/bold green]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Engraving...", total=len(cloud))

        def update_progress(p: EngraveProgress) -> None:
            progress.update(
                task,
                completed=p.completed_points,
                description=f"Layer {p.current_layer + 1}/{p.total_layers}",
            )

        try:
            with LaserController(cfg) as laser:
                laser.engrave_point_cloud(
                    cloud,
                    progress_callback=update_progress,
                    dry_run=dry_run,
                )

            console.print("\n[bold green]Engrave complete![/bold green]")

        except InterruptedError:
            console.print("\n[bold yellow]Engrave aborted[/bold yellow]")
        except Exception as e:
            console.print(f"\n[bold red]Error: {e}[/bold red]")
            raise click.Abort()


if __name__ == "__main__":
    main()
