# Glass3D - Subsurface Laser Engraving Software

## Project Overview

Glass3D is a Python application for creating 3D subsurface laser engravings (SSLE) in glass/crystal blocks. The software converts 3D models (STL/OBJ) into point clouds and controls a BJJCZ galvo laser to create micro-fractures inside transparent materials.

### How SSLE Works

1. A focused laser creates tiny micro-fractures (voids/bubbles) at precise 3D coordinates inside glass
2. These points scatter light, making the shape visible
3. The surface remains untouched - the focal point is inside the material
4. Thousands to millions of points combine to form a 3D image

### Hardware

- **Laser Controller**: BJJCZ FBLI-B-LV7 (fiber laser controller)
- **Control Library**: `galvoplotter` (Python, via PyUSB)
- **Coordinate System**: Galvo positions are 0x0000 to 0xFFFF (16-bit), with 0x8000 being center
- **Z-Axis**: Controlled separately (likely via stepper or the controller's Z output)

## Architecture

```
glass3d/
├── CLAUDE.md                 # This file
├── README.md                 # User documentation
├── pyproject.toml            # Project configuration
├── requirements.txt          # Dependencies
│
├── glass3d/
│   ├── __init__.py
│   ├── cli.py                # Command-line interface
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration management (laser settings, machine params)
│   │   ├── point_cloud.py    # PointCloud class - core data structure
│   │   └── bounds.py         # Bounding box and coordinate utilities
│   │
│   ├── mesh/
│   │   ├── __init__.py
│   │   ├── loader.py         # Load STL/OBJ files via trimesh
│   │   ├── slicer.py         # Slice mesh into layers
│   │   └── pointcloud_gen.py # Generate point clouds from meshes
│   │
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py           # Abstract base for point generation strategies
│   │   ├── surface.py        # Surface-only points (shell)
│   │   ├── solid.py          # Solid fill with configurable density
│   │   └── grayscale.py      # Variable density based on distance/normals
│   │
│   ├── laser/
│   │   ├── __init__.py
│   │   ├── controller.py     # Wrapper around galvoplotter
│   │   ├── calibration.py    # Coordinate transformation, lens correction
│   │   ├── z_axis.py         # Z-axis control (focus depth)
│   │   └── safety.py         # Safety checks, bounds validation
│   │
│   ├── path/
│   │   ├── __init__.py
│   │   ├── optimizer.py      # Path optimization (minimize travel)
│   │   ├── sorting.py        # Sort points by Z (bottom-up), then optimize XY
│   │   └── chunking.py       # Break into manageable chunks for progress
│   │
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py  # Preview point clouds (matplotlib/open3d)
│       └── export.py         # Export point clouds to various formats
│
├── tests/
│   ├── __init__.py
│   ├── test_point_cloud.py
│   ├── test_slicer.py
│   └── test_laser.py
│
├── examples/
│   ├── simple_cube.py
│   ├── from_stl.py
│   └── calibration_grid.py
│
└── configs/
    ├── default.json          # Default machine/laser settings
    └── materials/
        ├── k9_glass.json     # Settings for K9 borosilicate
        └── bk7_glass.json    # Settings for Schott BK7
```

## Key Technical Details

### Coordinate Systems

1. **Model Space**: Original STL coordinates (typically mm)
2. **Machine Space**: Physical coordinates in mm, centered on work area
3. **Galvo Space**: 16-bit integer coordinates (0x0000-0xFFFF)
   - 0x8000, 0x8000 = center of galvo field
   - Field size depends on lens (e.g., 110mm x 110mm for F-theta lens)

### Z-Axis Considerations

- For SSLE, we engrave from **bottom to top** (deepest points first)
- This prevents the laser from passing through already-fractured material
- Z position controls focal depth inside the glass
- Must account for refractive index of glass (~1.5 for K9)

### Point Spacing Guidelines

- Minimum spacing: ~0.05mm to avoid "clashing" (merged fractures)
- Maximum spacing: ~0.3mm before image appears too sparse
- Typical: 0.1-0.15mm for good detail/visibility balance
- Density can vary to create grayscale effects

### Laser Parameters (to be tuned per material)

```python
# Example starting parameters for K9 glass
laser_params = {
    "power": 50,           # Percentage (0-100)
    "frequency": 30,       # kHz - pulse frequency
    "pulse_width": 4,      # microseconds
    "speed": 1000,         # mm/s for positioning
    "q_switch_period": 50, # nanoseconds
}
```

### galvoplotter API Reference

```python
from galvoplotter import GalvoController

# Initialize
controller = GalvoController(settings_file="settings.json")

# Set laser parameters
controller.set(power=50, frequency=30)

# Marking context - laser will fire
with controller.marking() as c:
    c.goto(x, y)        # Move without firing
    c.mark(x, y)        # Move while firing
    c.dwell(time_ms)    # Fire at current position

# Lighting context - red dot preview only
with controller.lighting() as c:
    c.light(x, y)       # Move with red light on
    c.dark(x, y)        # Move with red light off

# Wait for completion
controller.wait_for_machine_idle()

# Realtime commands
controller.jog(x, y)    # Move galvos without list
controller.abort()      # Stop current operation
```

### Critical Implementation Notes

1. **Always engrave bottom-up** (lowest Z first) to avoid beam distortion
2. **Coordinate validation** - never send points outside galvo range
3. **Thermal management** - add pauses for large jobs to prevent glass cracking
4. **Progress tracking** - large point clouds can take hours
5. **Emergency stop** - must be able to abort at any time

## Development Guidelines

### Testing Without Hardware

Use `mock_connection` from galvoplotter for development:
```python
from galvoplotter import GalvoController
controller = GalvoController(mock=True)  # Uses mock connection
```

### Dependencies

```
galvoplotter>=0.2.0    # Laser control
trimesh>=4.0.0         # Mesh loading and slicing
numpy>=1.24.0          # Numerical operations
scipy>=1.10.0          # Spatial algorithms
shapely>=2.0.0         # 2D geometry (used by trimesh)
click>=8.0.0           # CLI framework
pydantic>=2.0.0        # Configuration validation
matplotlib>=3.7.0      # Visualization
open3d>=0.17.0         # 3D visualization (optional)
```

### Code Style

- Use type hints throughout
- Docstrings for all public functions (Google style)
- Unit tests for core logic
- Integration tests with mock laser

## Implementation Priority

### Phase 1: Core Infrastructure
1. [ ] Project setup (pyproject.toml, dependencies)
2. [ ] Configuration system (pydantic models)
3. [ ] PointCloud class with basic operations
4. [ ] Mesh loading via trimesh

### Phase 2: Point Cloud Generation
1. [ ] Basic slicer (mesh → layers)
2. [ ] Surface point generation strategy
3. [ ] Solid fill strategy
4. [ ] Point spacing/density controls

### Phase 3: Laser Control
1. [ ] GalvoController wrapper with safety checks
2. [ ] Coordinate transformation (model → galvo)
3. [ ] Basic engraving routine (single layer)
4. [ ] Z-axis integration

### Phase 4: Path Optimization
1. [ ] Sort points by Z (bottom-up)
2. [ ] TSP-style optimization for XY within layers
3. [ ] Chunking for progress/pause capability

### Phase 5: Polish
1. [ ] CLI interface
2. [ ] Preview/visualization
3. [ ] Progress reporting
4. [ ] Pause/resume support

## Common Tasks

### "Add a new point generation strategy"
1. Create new file in `glass3d/strategies/`
2. Inherit from `BaseStrategy` in `strategies/base.py`
3. Implement `generate(mesh, config) -> PointCloud`
4. Register in `strategies/__init__.py`

### "Adjust laser parameters"
- Edit `configs/default.json` or material-specific config
- Parameters are validated by pydantic models in `core/config.py`

### "Test with a simple shape"
```bash
python -m glass3d engrave examples/models/cube.stl --preview --mock
```

### "Calibrate the machine"
```bash
python -m glass3d calibrate --grid-size 10 --spacing 5
```

## Safety Reminders

⚠️ **LASER SAFETY IS CRITICAL** ⚠️

- Never operate without proper enclosure and interlocks
- Always wear appropriate laser safety glasses
- Verify emergency stop functionality before each session
- Start with LOW power settings and increase gradually
- Monitor for thermal buildup in glass
- Never leave running laser unattended

## Resources

- [galvoplotter GitHub](https://github.com/meerk40t/galvoplotter)
- [galvoplotter examples](https://github.com/meerk40t/galvoplotter/tree/main/examples)
- [trimesh documentation](https://trimesh.org/)
- [Original Balor project](https://gitlab.com/bryce15/balor)
