# Glass3D

**Subsurface Laser Engraving Software for 3D Glass/Crystal Engravings**

Glass3D converts 3D models into point clouds and controls BJJCZ galvo lasers to create stunning 3D images inside glass blocks using subsurface laser engraving (SSLE).

## Features

- Load 3D models from STL, OBJ, PLY, and other formats
- Multiple point cloud generation strategies:
  - **Surface**: Points on the mesh surface (shell effect)
  - **Solid**: Fill the entire volume
  - **Grayscale**: Variable density for shading effects
  - **Contour**: Points along layer contours only
- Direct control of BJJCZ galvo lasers via `galvoplotter`
- Safety features: bounds validation, thermal management
- Progress tracking and abort capability
- Preview mode with red dot laser
- CLI and programmatic API

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/glass3d.git
cd glass3d

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[all]"
```

## Quick Start

### Command Line

```bash
# View model info
glass3d info model.stl

# Preview point cloud (no laser)
glass3d preview model.stl --strategy surface --spacing 0.1

# Test laser connection
glass3d test-connection --mock

# Dry run (red dot preview)
glass3d engrave model.stl --dry-run --mock

# Actually engrave
glass3d engrave model.stl --strategy surface --spacing 0.1 --max-size 50
```

### Python API

```python
from glass3d import Glass3DConfig, MeshLoader, get_strategy, LaserController

# Load configuration
config = Glass3DConfig.for_material("k9")
config.point_cloud.point_spacing_mm = 0.1
config.point_cloud.strategy = "surface"

# Load and prepare mesh
loader = MeshLoader("model.stl")
loader.center_at_origin()
loader.scale_to_fit(50)  # 50mm max dimension

# Generate point cloud
strategy = get_strategy("surface")
cloud = strategy.generate(loader.mesh, config.point_cloud)

# Sort bottom-up for SSLE
cloud = cloud.sort_by_z(ascending=True)

print(f"Generated {len(cloud):,} points")

# Engrave (with mock connection for testing)
config.mock_laser = True
with LaserController(config) as laser:
    laser.engrave_point_cloud(cloud, dry_run=True)
```

## Hardware Requirements

- BJJCZ galvo laser controller (tested with FBLI-B-LV7)
- Suitable laser source (fiber or UV recommended for glass)
- Z-axis control for focus depth adjustment
- Proper safety enclosure and interlocks

## Configuration

Generate a default config file:

```bash
glass3d init-config -o my_config.json -m k9
```

Edit the JSON file to match your machine parameters:

```json
{
  "laser": {
    "power": 50.0,
    "frequency": 30.0,
    "pulse_width": 4.0,
    "point_dwell_ms": 0.5
  },
  "machine": {
    "field_size_mm": [110.0, 110.0],
    "z_range_mm": [0.0, 100.0]
  },
  "material": {
    "name": "K9 Glass",
    "refractive_index": 1.5168,
    "min_point_spacing_mm": 0.05
  }
}
```

## Safety

⚠️ **LASER SAFETY IS CRITICAL** ⚠️

- Never operate without proper enclosure and interlocks
- Always wear appropriate laser safety glasses
- Verify emergency stop functionality before each session
- Start with LOW power settings and increase gradually
- Monitor for thermal buildup in glass
- Never leave running laser unattended

## How SSLE Works

Subsurface laser engraving creates images inside glass by:

1. Focusing a pulsed laser beam inside the glass material
2. Creating tiny micro-fractures (voids) at the focal point
3. These points scatter light, making them visible
4. The surface remains untouched - only internal points are created
5. Thousands to millions of points combine to form a 3D image

Key considerations:
- Engrave from bottom to top (deepest points first)
- Maintain minimum point spacing to avoid "clashing"
- Use high-quality optical glass (K9, BK7, fused silica)
- Account for refractive index when positioning focal point

## Development

See [CLAUDE.md](CLAUDE.md) for detailed development documentation.

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy glass3d

# Linting
ruff check glass3d
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [galvoplotter](https://github.com/meerk40t/galvoplotter) - Laser control library
- [trimesh](https://trimesh.org/) - Mesh processing
- [Balor](https://gitlab.com/bryce15/balor) - Original reverse engineering work
