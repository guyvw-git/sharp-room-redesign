# scripts/furniture/seating.py
"""
Sectional sofa, club chair, bar stool.
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from furniture.base import box_gaussians, disc_gaussians, cylinder_gaussians


def build_sectional_sofa(spec: dict, room_geom, rng: np.random.Generator) -> np.ndarray:
    cx    = spec.get('position_x', room_geom.center_x)
    cz    = spec.get('position_z', room_geom.center_z)
    mat   = spec.get('material', 'dark_leather')
    n     = spec.get('n_gaussians', 600_000)
    width = spec.get('width_m', 3.0)
    depth = spec.get('depth_m', 0.95)

    floor_y   = room_geom.floor_y
    seat_h    = 0.45
    back_h    = 0.85
    arm_h     = 0.65
    cushion_d = 0.55
    back_d    = 0.22

    seat_cy = floor_y - (seat_h / 2.0)
    back_cy = floor_y - (seat_h + (back_h - seat_h) / 2.0)
    arm_cy  = floor_y - (arm_h / 2.0)

    seat = box_gaussians((cx, seat_cy, cz),
                         (width/2.0, seat_h/2.0, cushion_d/2.0),
                         mat, int(n*0.35), rng)
    back_z = cz + cushion_d/2.0 + back_d/2.0
    back = box_gaussians((cx, back_cy, back_z),
                         (width/2.0, (back_h-seat_h)/2.0, back_d/2.0),
                         mat, int(n*0.30), rng)
    arm_x_l = cx - width/2.0 - 0.10
    arm_l = box_gaussians((arm_x_l, arm_cy, cz),
                          (0.10, arm_h/2.0, (cushion_d+back_d)/2.0),
                          mat, int(n*0.10), rng)
    arm_x_r = cx + width/2.0 + 0.10
    arm_r = box_gaussians((arm_x_r, arm_cy, cz),
                          (0.10, arm_h/2.0, (cushion_d+back_d)/2.0),
                          mat, int(n*0.10), rng)
    l_width = 1.20
    l_cx = cx + width/2.0 + 0.10 + l_width/2.0
    l_section = box_gaussians((l_cx, seat_cy, cz-0.20),
                              (l_width/2.0, seat_h/2.0, cushion_d/2.0),
                              mat, int(n*0.15), rng)
    return np.concatenate([seat, back, arm_l, arm_r, l_section], axis=0)


def build_club_chair(spec: dict, room_geom, rng: np.random.Generator) -> np.ndarray:
    cx  = spec.get('position_x', 0.0)
    cz  = spec.get('position_z', room_geom.center_z)
    mat = spec.get('material', 'dark_leather')
    n   = spec.get('n_gaussians', 150_000)

    floor_y = room_geom.floor_y
    seat_cy = floor_y - 0.225
    back_cy = floor_y - 0.625
    arm_cy  = floor_y - 0.325

    seat = box_gaussians((cx, seat_cy, cz),         (0.45, 0.225, 0.48), mat, int(n*0.35), rng)
    back = box_gaussians((cx, back_cy, cz+0.32),    (0.45, 0.20,  0.12), mat, int(n*0.30), rng)
    arm_l= box_gaussians((cx-0.52, arm_cy, cz+0.1), (0.07, 0.325, 0.40), mat, int(n*0.15), rng)
    arm_r= box_gaussians((cx+0.52, arm_cy, cz+0.1), (0.07, 0.325, 0.40), mat, int(n*0.15), rng)
    return np.concatenate([seat, back, arm_l, arm_r], axis=0)


def build_bar_stool(spec: dict, room_geom, rng: np.random.Generator) -> np.ndarray:
    cx       = spec.get('position_x', 0.0)
    cz       = spec.get('position_z', room_geom.center_z)
    seat_mat = spec.get('material', 'dark_leather')
    n        = spec.get('n_gaussians', 60_000)
    seat_h   = spec.get('seat_height_m', 0.75)

    floor_y = room_geom.floor_y
    seat_y  = floor_y - seat_h

    seat = disc_gaussians(
        center=(cx, seat_y, cz), normal=(0, -1, 0),
        radius=0.18, material=seat_mat, n_gaussians=int(n*0.40), rng=rng,
    )
    post = cylinder_gaussians(
        center=(cx, floor_y - seat_h/2.0, cz),
        radius=0.025, half_height=seat_h/2.0, axis=1,
        material='chrome_metal', n_gaussians=int(n*0.25), rng=rng,
    )
    footrest_y = floor_y - 0.28
    footrest_r = 0.15
    ring_parts = []
    for i in range(8):
        angle = i * np.pi / 4.0
        rx = cx + footrest_r * np.cos(angle)
        rz = cz + footrest_r * np.sin(angle)
        ring_parts.append(cylinder_gaussians(
            center=(rx, footrest_y, rz),
            radius=0.012, half_height=0.012, axis=1,
            material='chrome_metal', n_gaussians=max(100, int(n*0.04)), rng=rng,
        ))
    base = disc_gaussians(
        center=(cx, floor_y, cz), normal=(0, 1, 0),
        radius=0.22, material='chrome_metal', n_gaussians=int(n*0.15), rng=rng,
    )
    return np.concatenate([seat, post, base] + ring_parts, axis=0)
