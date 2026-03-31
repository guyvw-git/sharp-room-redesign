# tests/test_room_analyzer.py
import sys, numpy as np
sys.path.insert(0, 'scripts')
from room_analyzer import analyze_room, RoomGeometry

def test_geometry_fields():
    """RoomGeometry has all required fields."""
    g = RoomGeometry(
        floor_y=1.458, ceiling_y=-1.723,
        north_z=9.073, south_z=2.656,
        east_x=1.622, west_x=-1.853,
    )
    assert abs(g.room_height_m - 3.181) < 0.01
    assert abs(g.room_width_m - 3.475) < 0.01
    assert abs(g.room_depth_m - 6.417) < 0.01
    assert abs(g.center_x - (-0.1155)) < 0.01
    assert abs(g.center_z - 5.8645) < 0.01

def test_structure_mask_covers_boundaries(tmp_path):
    """Structure mask marks Gaussians near room boundaries as structure."""
    data = np.zeros((5, 14), dtype=np.float32)
    # Floor Gaussian (Y near floor_y=1.458)
    data[0, 1] = 1.45
    # Ceiling Gaussian
    data[1, 1] = -1.71
    # Wall Gaussian (near north wall Z=9.073)
    data[2, 2] = 9.05
    # Interior Gaussians
    data[3] = [0.0, 0.8, 6.0] + [0]*11
    data[4] = [-0.5, 0.5, 5.0] + [0]*11

    geom = RoomGeometry(floor_y=1.458, ceiling_y=-1.723,
                        north_z=9.073, south_z=2.656,
                        east_x=1.622, west_x=-1.853)
    structure, content = geom.split_structure_content(data, margin_m=0.25)
    assert structure[0]  # floor → structure
    assert structure[1]  # ceiling → structure
    assert structure[2]  # north wall → structure
    assert content[3]    # interior → content
    assert content[4]    # interior → content

def test_analyze_room_returns_geometry():
    """analyze_room reads the real PLY and returns sensible values."""
    geom = analyze_room('input/current.ply')
    assert 1.0 < geom.floor_y < 2.0,   f"floor_y={geom.floor_y}"
    assert -2.5 < geom.ceiling_y < 0.0, f"ceiling_y={geom.ceiling_y}"
    assert 2.0 < geom.room_height_m < 5.0
    assert 2.0 < geom.room_width_m < 8.0
    assert 3.0 < geom.room_depth_m < 15.0
