# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Glass3D creates 3D subsurface laser engravings (SSLE) in glass/crystal blocks. It converts 3D models (STL/OBJ) into point clouds and controls BJJCZ galvo lasers via the `galvoplotter` library to create micro-fractures inside transparent materials.

## Commands

```bash
# Install in development mode
pip install -e ".[all]"

# Run tests
pytest

# Run a single test file
pytest tests/test_point_cloud.py

# Run a specific test
pytest tests/test_point_cloud.py::TestPointCloudBasics::test_create_with_points

# Type checking
mypy glass3d

# Linting
ruff check glass3d

# Format code
black glass3d

# CLI commands (after install)
glass3d info model.stl                              # View model info
glass3d preview model.stl --strategy surface        # 3D visualization
glass3d engrave model.stl --mock --dry-run          # Test without hardware
glass3d init-config -o config.json -m k9            # Generate config
glass3d test-connection --mock                      # Test laser connection
```

## Architecture

The codebase follows a pipeline: **Mesh Loading → Point Cloud Generation → Coordinate Transform → Laser Control**

### Core Components

- **`glass3D/core/config.py`**: Pydantic models for all configuration (laser params, machine params, material settings). `Glass3DConfig` is the main container with nested models like `LaserParams`, `MachineParams`, `MaterialParams`.

- **`glass3D/core/point_cloud.py`**: `PointCloud` dataclass wrapping numpy arrays. Stores Nx3 points with optional intensities and layer indices. Provides transformations (translate, scale, sort_by_z) and I/O (save_xyz, save_npz).

- **`glass3D/mesh/loader.py`**: `MeshLoader` wraps trimesh for loading STL/OBJ/PLY. Provides center_at_origin(), scale_to_fit(), repair() for mesh preparation.

- **`glass3D/mesh/pointcloud_gen.py`**: Point generation strategies. Use `get_strategy(name)` to get a strategy, then `strategy.generate(mesh, config)` to create point clouds.

- **`glass3D/laser/controller.py`**: `LaserController` wraps galvoplotter with safety checks. `CoordinateTransformer` converts mm coordinates to 16-bit galvo coordinates (0x0000-0xFFFF, center at 0x8000).

- **`glass3D/cli.py`**: Click-based CLI using Rich for output formatting.

### Coordinate Systems

1. **Model Space**: Original mesh coordinates (typically mm)
2. **Machine Space**: Physical mm, centered on work area (0,0 = galvo center)
3. **Galvo Space**: 16-bit integers (0x0000-0xFFFF), 0x8000 = center

Transformation: `CoordinateTransformer.mm_to_galvo_coords(x_mm, y_mm) → (x_galvo, y_galvo)`

## Critical Domain Knowledge

### SSLE Engraving Requirements

- **Always engrave bottom-up** (lowest Z first) - prevents laser passing through already-fractured material
- **Point spacing**: 0.05mm minimum (avoid clashing), 0.3mm maximum (avoid sparse appearance), typical 0.1-0.15mm
- **Thermal management**: Pause every 10k points to prevent glass cracking
- **Refractive index correction**: Z position must account for glass refraction (~1.5 for K9)

### galvoplotter API

Note: Package is `pip install galvoplotter` but imports as `galvo`.

```python
from galvo import GalvoController

controller = GalvoController(mock=True)  # Mock for testing
controller.set(power=50, frequency=30)

with controller.marking() as c:   # Laser fires
    c.goto(x, y)                  # Move without firing
    c.dwell(time_ms)              # Fire at current position

with controller.lighting() as c:  # Red dot preview only
    c.light(x, y)                 # Move with red light
```

## Testing Without Hardware

Always use `mock=True` when developing:

```python
config = Glass3DConfig.default()
config.mock_laser = True

# Or via CLI
glass3d engrave model.stl --mock --dry-run
```

## Adding a Point Generation Strategy

1. Create class in `glass3d/mesh/pointcloud_gen.py` implementing `generate(mesh, config) -> PointCloud`
2. Register in `STRATEGIES` dict in same file
3. Strategy names: "surface", "solid", "grayscale", "contour"
