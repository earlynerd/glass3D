#!/usr/bin/env python3
"""Example: Engrave a simple cube for testing.

This script demonstrates the basic workflow for Glass3D:
1. Create or load a mesh
2. Generate a point cloud
3. Engrave (or preview) the points

Run with: python examples/simple_cube.py
"""

import numpy as np
import trimesh

from glass3d import Glass3DConfig, PointCloud, get_strategy, LaserController


def create_test_cube(size_mm: float = 20.0) -> trimesh.Trimesh:
    """Create a simple cube mesh for testing."""
    return trimesh.creation.box(extents=[size_mm, size_mm, size_mm])


def main():
    # Configuration
    config = Glass3DConfig.default()
    config.mock_laser = True  # Use mock for testing
    config.point_cloud.point_spacing_mm = 0.5  # Coarse spacing for quick test
    config.point_cloud.layer_height_mm = 0.5
    config.point_cloud.strategy = "surface"
    
    print("Glass3D - Simple Cube Example")
    print("=" * 40)
    
    # Create test geometry
    print("\n1. Creating test cube...")
    mesh = create_test_cube(20.0)  # 20mm cube
    
    # Center at origin
    mesh.vertices -= mesh.centroid
    
    print(f"   Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"   Size: {mesh.extents}")
    
    # Generate point cloud
    print(f"\n2. Generating point cloud ({config.point_cloud.strategy})...")
    strategy = get_strategy(config.point_cloud.strategy)
    cloud = strategy.generate(mesh, config.point_cloud)
    
    print(f"   Generated {len(cloud):,} points")
    print(f"   Bounds: {cloud.bounds[0]} to {cloud.bounds[1]}")
    print(f"   Layers: {cloud.num_layers}")
    
    # Sort bottom-up (critical for SSLE)
    print("\n3. Sorting points bottom-up...")
    cloud = cloud.sort_by_z(ascending=True)
    
    # Validate
    print("\n4. Validating point cloud...")
    controller = LaserController(config)
    valid, issues = controller.validate_point_cloud(cloud)
    
    if valid:
        print("   ✓ Point cloud is valid")
    else:
        print("   ✗ Issues found:")
        for issue in issues:
            print(f"     - {issue}")
    
    # Preview/engrave
    print("\n5. Engraving (mock mode)...")
    
    def progress_callback(p):
        pct = p.percent_complete
        if int(pct) % 20 == 0:  # Print every 20%
            print(f"   Progress: {pct:.0f}% ({p.completed_points}/{p.total_points})")
    
    try:
        with LaserController(config) as laser:
            laser.engrave_point_cloud(
                cloud,
                progress_callback=progress_callback,
                dry_run=True,  # Preview only
            )
        print("\n   ✓ Engrave complete!")
        
    except Exception as e:
        print(f"\n   ✗ Error: {e}")
    
    print("\n" + "=" * 40)
    print("Done! In real usage, set config.mock_laser = False")
    print("and dry_run=False to actually engrave.")


if __name__ == "__main__":
    main()
