# scripts/furniture/screen.py
"""
Projector screen and TV screen geometry.

spec keys:
  position_x, position_z  -- center of screen on the wall
  width_m, height_m        -- screen dimensions
  wall   -- 'north'|'south'|'east'|'west'
  floor_clearance_m        -- bottom of screen above floor
  material                 -- usually 'projector_screen' or 'dark_fabric'
  n_gaussians
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from furniture.base import box_gaussians


def build_screen(spec: dict, room_geom, rng: np.random.Generator) -> np.ndarray:
    wall       = spec.get('wall', 'north')
    width_m    = spec['width_m']
    height_m   = spec['height_m']
    clearance  = spec.get('floor_clearance_m', 0.8)
    material   = spec.get('material', 'projector_screen')
    n          = spec.get('n_gaussians', 200_000)

    # Y-down: screen bottom at floor_y - clearance, center further up
    screen_bottom_y = room_geom.floor_y - clearance
    screen_center_y = screen_bottom_y - (height_m / 2.0)

    cx = spec.get('position_x', room_geom.center_x)
    cz = spec.get('position_z', _wall_z(wall, room_geom))

    frame_depth = 0.03
    frame_border = 0.06

    screen = box_gaussians(
        center=(cx, screen_center_y, cz),
        half_extents=(width_m / 2.0, height_m / 2.0, frame_depth / 2.0),
        material=material,
        n_gaussians=int(n * 0.85),
        rng=rng,
    )

    frame = box_gaussians(
        center=(cx, screen_center_y, cz - frame_depth),
        half_extents=(width_m / 2.0 + frame_border, height_m / 2.0 + frame_border, frame_border / 2.0),
        material='dark_fabric',
        n_gaussians=int(n * 0.15),
        rng=rng,
    )
    return np.concatenate([screen, frame], axis=0)


def _wall_z(wall: str, room_geom) -> float:
    wall_offset = 0.05
    if wall == 'north':
        return room_geom.north_z - wall_offset
    if wall == 'south':
        return room_geom.south_z + wall_offset
    return room_geom.center_z
