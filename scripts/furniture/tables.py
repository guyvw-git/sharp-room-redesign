# scripts/furniture/tables.py
"""
Coffee table, bar counter, round side table.
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from furniture.base import box_gaussians, disc_gaussians, cylinder_gaussians


def build_coffee_table(spec: dict, room_geom, rng: np.random.Generator) -> np.ndarray:
    cx    = spec.get('position_x', room_geom.center_x)
    cz    = spec.get('position_z', room_geom.center_z)
    mat   = spec.get('material', 'dark_walnut')
    n     = spec.get('n_gaussians', 120_000)
    w     = spec.get('width_m', 1.20)
    d     = spec.get('depth_m', 0.60)
    h     = spec.get('height_m', 0.40)

    floor_y = room_geom.floor_y
    top_cy  = floor_y - h + 0.025
    leg_cy  = floor_y - h/2.0

    top = box_gaussians(
        center=(cx, top_cy, cz),
        half_extents=(w/2.0, 0.025, d/2.0),
        material=mat, n_gaussians=int(n*0.60), rng=rng,
    )
    legs = []
    for lx, lz in [(-w/2+0.06, -d/2+0.06), (w/2-0.06, -d/2+0.06),
                   (-w/2+0.06,  d/2-0.06), (w/2-0.06,  d/2-0.06)]:
        legs.append(cylinder_gaussians(
            center=(cx+lx, leg_cy, cz+lz),
            radius=0.025, half_height=h/2.0-0.025, axis=1,
            material=mat, n_gaussians=int(n*0.10), rng=rng,
        ))
    return np.concatenate([top] + legs, axis=0)


def build_bar_counter(spec: dict, room_geom, rng: np.random.Generator) -> np.ndarray:
    cx       = spec.get('position_x', room_geom.center_x)
    cz       = spec.get('position_z', room_geom.north_z - 0.35)
    length   = spec.get('length_m', 2.4)
    mat_body = spec.get('material', 'dark_walnut')
    mat_top  = spec.get('top_material', 'dark_marble')
    n        = spec.get('n_gaussians', 400_000)

    floor_y = room_geom.floor_y
    bar_h   = 1.10
    top_h   = 0.05
    body_h  = bar_h - top_h
    depth   = 0.58

    body_cy = floor_y - body_h/2.0
    top_cy  = floor_y - bar_h - top_h/2.0

    body = box_gaussians(
        center=(cx, body_cy, cz),
        half_extents=(length/2.0, body_h/2.0, depth/2.0),
        material=mat_body, n_gaussians=int(n*0.55), rng=rng,
    )
    top = box_gaussians(
        center=(cx, top_cy, cz),
        half_extents=(length/2.0 + 0.03, top_h/2.0, depth/2.0 + 0.02),
        material=mat_top, n_gaussians=int(n*0.30), rng=rng,
    )
    rail_y = floor_y - 0.50
    rail = cylinder_gaussians(
        center=(cx, rail_y, cz - depth/2.0 + 0.08),
        radius=0.025, half_height=length/2.0, axis=0,
        material='warm_brass', n_gaussians=int(n*0.15), rng=rng,
    )
    return np.concatenate([body, top, rail], axis=0)


def build_round_table(spec: dict, room_geom, rng: np.random.Generator) -> np.ndarray:
    cx     = spec.get('position_x', 0.0)
    cz     = spec.get('position_z', room_geom.center_z)
    mat    = spec.get('material', 'dark_walnut')
    n      = spec.get('n_gaussians', 80_000)
    radius = spec.get('radius_m', 0.40)
    h      = spec.get('height_m', 0.65)

    floor_y = room_geom.floor_y
    top_y   = floor_y - h
    post_cy = floor_y - h/2.0

    top  = disc_gaussians((cx, top_y, cz), (0,-1,0), radius, mat, int(n*0.50), rng)
    edge = cylinder_gaussians((cx, post_cy, cz), radius, 0.02, 1, mat, int(n*0.15), rng,
                               include_caps=False)
    post = cylinder_gaussians((cx, post_cy, cz), 0.04, h/2.0, 1, mat, int(n*0.20), rng)
    base = disc_gaussians((cx, floor_y, cz), (0,1,0), radius*0.5, mat, int(n*0.15), rng)
    return np.concatenate([top, edge, post, base], axis=0)
