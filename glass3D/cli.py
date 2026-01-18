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
    help="Scale factor for model",
)
@click.option(
    "--max-size",
    type=float,
    default=None,
    help="Scale model to fit within this size (mm)",
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
    """Engrave a 3D model into glass.
    
    MODEL_PATH: Path to STL/OBJ file to engrave
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
    
    # Load mesh
    with console.status("Loading mesh..."):
        loader = MeshLoader(model_path)
        
        # Apply transformations
        loader.center_at_origin()
        
        if max_size:
            loader.scale_to_fit(max_size)
        elif scale != 1.0:
            loader.scale(scale)
        
        loader.repair()
    
    # Show mesh info
    stats = loader.stats()
    table = Table(title="Mesh Info")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("File", Path(model_path).name)
    table.add_row("Vertices", str(stats["num_vertices"]))
    table.add_row("Faces", str(stats["num_faces"]))
    table.add_row("Size (mm)", f"{stats['size'][0]:.1f} x {stats['size'][1]:.1f} x {stats['size'][2]:.1f}")
    table.add_row("Watertight", "Yes" if stats["is_watertight"] else "No")
    console.print(table)
    
    # Generate point cloud
    console.print(f"\n[cyan]Generating point cloud ({strategy} strategy)...[/cyan]")
    
    with console.status("Generating points..."):
        strat = get_strategy(strategy)
        cloud = strat.generate(loader.mesh, cfg.point_cloud)
    
    # Sort points bottom-up for SSLE
    cloud = cloud.sort_by_z(ascending=True)
    
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
    help="Scale model to fit within this size (mm)",
)
def preview(
    model_path: str,
    strategy: str,
    spacing: float,
    max_size: float,
) -> None:
    """Preview a model's point cloud without engraving.
    
    Opens a 3D visualization of the generated points.
    """
    cfg = Glass3DConfig.default()
    cfg.point_cloud.point_spacing_mm = spacing
    cfg.point_cloud.strategy = strategy
    
    console.print(f"\n[bold]Point Cloud Preview[/bold]\n")
    
    # Load and process mesh
    with console.status("Loading mesh..."):
        loader = MeshLoader(model_path)
        loader.center_at_origin()
        loader.scale_to_fit(max_size)
    
    # Generate points
    with console.status("Generating points..."):
        strat = get_strategy(strategy)
        cloud = strat.generate(loader.mesh, cfg.point_cloud)
    
    console.print(f"Generated {len(cloud):,} points")
    console.print(f"Size: {cloud.size[0]:.1f} x {cloud.size[1]:.1f} x {cloud.size[2]:.1f} mm")
    
    # Try to visualize
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Subsample for visualization if too many points
        if len(cloud) > 10000:
            vis_cloud = cloud.random_sample(10000)
            console.print(f"[dim](Showing 10,000 of {len(cloud):,} points)[/dim]")
        else:
            vis_cloud = cloud
        
        ax.scatter(
            vis_cloud.x, vis_cloud.y, vis_cloud.z,
            s=1, alpha=0.5, c=vis_cloud.z, cmap='viridis'
        )
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'{Path(model_path).name} - {len(cloud):,} points')
        
        # Equal aspect ratio
        max_range = cloud.size.max() / 2
        mid = cloud.center
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        console.print("[yellow]matplotlib not installed - cannot show preview[/yellow]")
        console.print("Install with: pip install matplotlib")


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
def info(model_path: str) -> None:
    """Show information about a 3D model file."""
    console.print(f"\n[bold]Model Info: {Path(model_path).name}[/bold]\n")
    
    try:
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
        console.print(f"[red]Error loading model: {e}[/red]")
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


if __name__ == "__main__":
    main()
