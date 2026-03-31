# scripts/furniture/storage.py
"""
Back bar shelving, acoustic panels, wall panels.
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from furniture.base import box_gaussians, cylinder_gaussians, disc_gaussians


def build_back_bar_shelving(spec: dict, room_geom, rng: np.random.Generator) -> np.ndarray:
    cx       = spec.get('position_x', room_geom.center_x)
    cz       = spec.get('position_z', room_geom.north_z - 0.20)
    width    = spec.get('width_m', 2.4)
    mat_body = spec.get('material', 'dark_walnut')
    n        = spec.get('n_gaussians', 500_000)

    floor_y   = room_geom.floor_y
    ceiling_y = room_geom.ceiling_y
    unit_h    = abs(floor_y - ceiling_y) * 0.85
    depth     = 0.35

    back_cy = floor_y - unit_h/2.0
    back = box_gaussians(
        center=(cx, back_cy, cz + depth/2.0),
        half_extents=(width/2.0, unit_h/2.0, 0.025),
        material=mat_body, n_gaussians=int(n*0.15), rng=rng,
    )

    parts = [back]
    shelf_ys = [floor_y - 0.90, floor_y - 1.50, floor_y - 2.10]
    for sy in shelf_ys:
        shelf = box_gaussians(
            center=(cx, sy, cz),
            half_extents=(width/2.0, 0.02, depth/2.0),
            material=mat_body, n_gaussians=int(n*0.08), rng=rng,
        )
        parts.append(shelf)

        n_bottles = 12
        for i in range(n_bottles):
            bx = cx - width/2.0 + 0.12 + i * (width - 0.24) / (n_bottles - 1)
            bottle_h = rng.uniform(0.22, 0.32)
            bottle = cylinder_gaussians(
                center=(bx, sy - bottle_h/2.0 - 0.02, cz - 0.05),
                radius=rng.uniform(0.02, 0.035),
                half_height=bottle_h/2.0,
                axis=1,
                material='warm_amber',
                n_gaussians=int(n*0.012),
                rng=rng,
            )
            parts.append(bottle)

    for sx in [cx - width/2.0 - 0.015, cx + width/2.0 + 0.015]:
        side = box_gaussians(
            center=(sx, back_cy, cz),
            half_extents=(0.015, unit_h/2.0, depth/2.0),
            material=mat_body, n_gaussians=int(n*0.03), rng=rng,
        )
        parts.append(side)

    return np.concatenate(parts, axis=0)


def build_acoustic_panel(spec: dict, room_geom, rng: np.random.Generator) -> np.ndarray:
    cx     = spec.get('position_x', 0.0)
    cz     = spec.get('position_z', room_geom.center_z)
    cy     = spec.get('position_y', room_geom.floor_y - 1.2)
    mat    = spec.get('material', 'dark_fabric')
    n      = spec.get('n_gaussians', 50_000)
    width  = spec.get('width_m', 0.60)
    height = spec.get('height_m', 1.20)
    depth  = spec.get('depth_m', 0.08)

    return box_gaussians(
        center=(cx, cy, cz),
        half_extents=(width/2.0, height/2.0, depth/2.0),
        material=mat, n_gaussians=n, rng=rng,
    )


def build_wall_panels(spec: dict, room_geom, rng: np.random.Generator) -> np.ndarray:
    wall   = spec.get('wall', 'north')
    mat    = spec.get('material', 'brick_warm')
    n      = spec.get('n_gaussians', 300_000)
    height = spec.get('height_m', 1.2)

    floor_y = room_geom.floor_y
    cy = floor_y - height/2.0

    if wall == 'north':
        return box_gaussians(
            center=(room_geom.center_x, cy, room_geom.north_z - 0.04),
            half_extents=(room_geom.room_width_m/2.0, height/2.0, 0.04),
            material=mat, n_gaussians=n, rng=rng,
        )
    if wall == 'east':
        return box_gaussians(
            center=(room_geom.east_x - 0.04, cy, room_geom.center_z),
            half_extents=(0.04, height/2.0, room_geom.room_depth_m/2.0),
            material=mat, n_gaussians=n, rng=rng,
        )
    if wall == 'west':
        return box_gaussians(
            center=(room_geom.west_x + 0.04, cy, room_geom.center_z),
            half_extents=(0.04, height/2.0, room_geom.room_depth_m/2.0),
            material=mat, n_gaussians=n, rng=rng,
        )
    return box_gaussians(
        center=(room_geom.center_x, cy, room_geom.south_z + 0.04),
        half_extents=(room_geom.room_width_m/2.0, height/2.0, 0.04),
        material=mat, n_gaussians=n, rng=rng,
    )
