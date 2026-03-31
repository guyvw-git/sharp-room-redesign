# scripts/room_analyzer.py
"""
Extracts real room geometry from a 3DGS PLY file.

Uses percentile-based boundary detection to find wall/floor/ceiling positions,
then classifies each Gaussian as structure (near a boundary) or content (interior).

Y-DOWN convention: floor = high positive Y, ceiling = low/negative Y.
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ply_io import read_ply


@dataclass
class RoomGeometry:
    floor_y: float    # Y-down: floor = highest Y value in room
    ceiling_y: float  # Y-down: ceiling = lowest Y value
    north_z: float    # back wall (high Z)
    south_z: float    # front wall / camera side (low Z)
    east_x: float     # right wall (high X)
    west_x: float     # left wall (low X)

    @property
    def room_height_m(self) -> float:
        return abs(self.floor_y - self.ceiling_y)

    @property
    def room_width_m(self) -> float:
        return abs(self.east_x - self.west_x)

    @property
    def room_depth_m(self) -> float:
        return abs(self.north_z - self.south_z)

    @property
    def center_x(self) -> float:
        return (self.east_x + self.west_x) / 2.0

    @property
    def center_z(self) -> float:
        return (self.north_z + self.south_z) / 2.0

    def split_structure_content(
        self, data: np.ndarray, margin_m: float = 0.25
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split Gaussians into structure (near room boundaries) and content (interior).
        Returns (structure_mask, content_mask) — boolean arrays of shape (N,).
        """
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        near_floor   = y > (self.floor_y   - margin_m)
        near_ceiling = y < (self.ceiling_y + margin_m)
        near_north   = z > (self.north_z   - margin_m)
        near_south   = z < (self.south_z   + margin_m)
        near_east    = x > (self.east_x    - margin_m)
        near_west    = x < (self.west_x    + margin_m)

        structure_mask = (
            near_floor | near_ceiling |
            near_north | near_south  |
            near_east  | near_west
        )
        content_mask = ~structure_mask
        return structure_mask, content_mask

    def to_dict(self) -> dict:
        return {
            "floor_y": self.floor_y,
            "ceiling_y": self.ceiling_y,
            "north_z": self.north_z,
            "south_z": self.south_z,
            "east_x": self.east_x,
            "west_x": self.west_x,
            "room_height_m": round(self.room_height_m, 3),
            "room_width_m": round(self.room_width_m, 3),
            "room_depth_m": round(self.room_depth_m, 3),
            "center_x": round(self.center_x, 3),
            "center_z": round(self.center_z, 3),
        }


def analyze_room(ply_path: str, structure_margin_m: float = 0.25) -> RoomGeometry:
    """
    Read a PLY and extract room geometry using percentile-based boundary detection.
    Uses P8/P92 percentiles for walls and P10/P90 for floor/ceiling.
    """
    data, _, _, _ = read_ply(ply_path)
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    return RoomGeometry(
        floor_y   = float(np.percentile(y, 90)),
        ceiling_y = float(np.percentile(y, 10)),
        north_z   = float(np.percentile(z, 92)),
        south_z   = float(np.percentile(z, 8)),
        east_x    = float(np.percentile(x, 92)),
        west_x    = float(np.percentile(x, 8)),
    )


if __name__ == '__main__':
    import json
    path = sys.argv[1] if len(sys.argv) > 1 else 'input/current.ply'
    geom = analyze_room(path)
    print(json.dumps(geom.to_dict(), indent=2))
