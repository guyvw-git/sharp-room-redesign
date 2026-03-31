# Room Redesign Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pipeline that takes a natural language redesign prompt ("Turn this space into a movie room") and produces a photorealistic-looking 3DGS PLY where the room structure (floor, walls, ceiling) is preserved and the content is replaced with properly modeled, shaded furniture.

**Architecture:** Room Analyzer extracts real wall/floor positions from the PLY → Claude API reasons about what specific furniture goes where → Furniture Library generates properly-modeled, Lambertian-shaded Gaussians for each piece → Scene Assembler removes old content, injects new content, writes output PLY. Validated end-to-end by the rewritten visual-verifier agent across three render angles.

**Tech Stack:** Python 3, NumPy, Anthropic SDK (`anthropic`), existing `ply_io.py`, `scripts/render_screenshot.sh` (Playwright/screenshot.js)

---

## Measured room geometry (input/current.ply)

```
Floor Y (Y-down, high=floor): P90 = 1.458
Ceiling Y:                    P10 = -1.723
Room height:                  3.18m
North wall Z (back):          P92 = 9.073
South wall Z (camera-side):   P8  = 2.656
East wall X:                  P92 = 1.622
West wall X:                  P8  = -1.853
Room width (E-W):             3.48m
Room depth (N-S):             6.42m
```

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `scripts/room_analyzer.py` | **CREATE** | Parse PLY → real room geometry dict + structure/content masks |
| `scripts/materials.py` | **CREATE** | Material definitions: colors, shading params, texture functions |
| `scripts/furniture/base.py` | **CREATE** | Shared Gaussian generation + per-face shading utilities |
| `scripts/furniture/screen.py` | **CREATE** | TV screen, projector screen geometry |
| `scripts/furniture/seating.py` | **CREATE** | Sectional sofa, club chair, bar stool |
| `scripts/furniture/tables.py` | **CREATE** | Coffee table, bar counter, dining table |
| `scripts/furniture/storage.py` | **CREATE** | Back bar shelving, acoustic panels, wall panels |
| `scripts/scene_planner.py` | **CREATE** | Claude API call: room dims + prompt → JSON scene plan |
| `scripts/redesign.py` | **CREATE** | Main entry point: orchestrates all stages, writes output PLY |
| `tests/test_room_analyzer.py` | **CREATE** | Unit tests for room geometry extraction |
| `tests/test_materials.py` | **CREATE** | Unit tests for material shading output |
| `tests/test_furniture_base.py` | **CREATE** | Unit tests for Gaussian generation utilities |
| `scripts/generate_kitchen.py` | **KEEP** | Not touched — kitchen was a prototype, superseded by redesign.py |
| `scripts/ply_io.py` | **READ-ONLY** | Already correct — provides read_ply / write_ply |
| `.claude/agents/visual-verifier.md` | **ALREADY REWRITTEN** | Multi-angle, prompt-aware rubric |

---

## Task 1: Room Analyzer

**Files:**
- Create: `scripts/room_analyzer.py`
- Create: `tests/test_room_analyzer.py`

- [ ] **Step 1: Write failing tests**

```python
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
    # Synthetic 5-Gaussian PLY-like array: 3 near boundaries, 2 interior
    # [x, y, z, ...13 props total]
    data = np.zeros((5, 14), dtype=np.float32)
    # Floor Gaussian (Y near floor_y=1.458)
    data[0, 1] = 1.45
    # Ceiling Gaussian
    data[1, 1] = -1.71
    # Wall Gaussian (near north wall Z=9.073)
    data[2, 2] = 9.05
    # Interior Gaussians — furniture zone
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/gvwert/Development/multi_agent
python3 -m pytest tests/test_room_analyzer.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'room_analyzer'`

- [ ] **Step 3: Implement room_analyzer.py**

```python
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

    Uses P8/P92 percentiles for walls (robust to outliers) and P10/P90 for floor/ceiling.
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
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/test_room_analyzer.py -v
```

Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/room_analyzer.py tests/test_room_analyzer.py
git commit -m "feat: add room_analyzer — percentile-based room geometry extraction"
```

---

## Task 2: Materials Library

**Files:**
- Create: `scripts/materials.py`
- Create: `tests/test_materials.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_materials.py
import sys, numpy as np
sys.path.insert(0, 'scripts')
from materials import get_material, apply_shading, MATERIALS

def test_all_required_materials_exist():
    required = [
        'dark_leather', 'dark_velvet', 'dark_walnut', 'matte_white',
        'projector_screen', 'chrome_metal', 'warm_brass', 'dark_stone',
        'brick_warm', 'dark_fabric', 'brushed_steel', 'warm_amber',
        'light_oak', 'dark_linen',
    ]
    for name in required:
        m = get_material(name)
        assert 'base_linear_rgb' in m, f"{name} missing base_linear_rgb"
        assert 'ambient' in m, f"{name} missing ambient"
        assert 'key' in m, f"{name} missing key"
        assert 'fill' in m, f"{name} missing fill"
        assert 'scale_tangent' in m, f"{name} missing scale_tangent"
        assert 'scale_normal' in m, f"{name} missing scale_normal"

def test_shading_brightens_top_face():
    """Normal pointing up (-Y in Y-down) receives full key light."""
    mat = get_material('matte_white')
    n = 10
    # All normals pointing straight up (outward from top face = -Y in Y-down)
    normals = np.tile([0.0, -1.0, 0.0], (n, 1))
    positions = np.zeros((n, 3))
    dc = apply_shading(normals, positions, mat, n)
    # Top face should be brighter than ambient alone
    # DC values should be positive (sigmoid > 0.5 = linear > 0.5 = bright)
    linear_r = 1.0 / (1.0 + np.exp(-dc[:, 0]))
    assert np.all(linear_r > 0.5), f"Top face of white should be bright: {linear_r}"

def test_shading_darkens_bottom_face():
    """Normal pointing down (+Y in Y-down) gets only ambient."""
    mat = get_material('matte_white')
    n = 10
    normals = np.tile([0.0, 1.0, 0.0], (n, 1))  # bottom face
    positions = np.zeros((n, 3))
    dc_top = apply_shading(np.tile([0.0,-1.0,0.0],(n,1)), positions, mat, n)
    dc_bot = apply_shading(normals, positions, mat, n)
    # Top brighter than bottom
    assert np.mean(dc_top[:,0]) > np.mean(dc_bot[:,0])

def test_dark_material_stays_dark():
    """Dark leather ambient is low — even lit faces stay dark."""
    mat = get_material('dark_leather')
    n = 20
    normals = np.tile([0.0, -1.0, 0.0], (n, 1))
    positions = np.zeros((n, 3))
    dc = apply_shading(normals, positions, mat, n)
    linear_r = 1.0 / (1.0 + np.exp(-dc[:, 0]))
    assert np.all(linear_r < 0.5), f"Dark leather lit face should be < 0.5 linear: {linear_r}"
```

- [ ] **Step 2: Run to confirm failure**

```bash
python3 -m pytest tests/test_materials.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'materials'`

- [ ] **Step 3: Implement materials.py**

```python
# scripts/materials.py
"""
Material definitions and Lambertian shading for 3DGS Gaussian arrays.

Each material defines:
  base_linear_rgb  — base surface color in linear light (0-1 per channel)
  ambient          — minimum brightness (0=black, 1=full)
  key              — overhead key light intensity (Y-down: light from -Y)
  fill             — frontal fill light intensity
  scale_tangent    — log-scale along surface (splat width)
  scale_normal     — log-scale perpendicular to surface (splat thickness)
  texture          — optional: 'wood_grain' | 'leather_grain' | 'brick' | None

apply_shading() converts material + per-Gaussian normals → f_dc values (pre-sigmoid).
"""
import numpy as np
from typing import Optional

# Key light: overhead, direction (0, -1, 0) in Y-down
_KEY_DIR = np.array([0.0, -1.0, 0.0])
# Fill light: soft frontal from camera-ish direction
_FILL_DIR = np.array([0.0, -0.3, -1.0])
_FILL_DIR = _FILL_DIR / np.linalg.norm(_FILL_DIR)

MATERIALS: dict = {
    # ── Dark / moody materials (movie room, speakeasy) ──────────────────
    'dark_leather': dict(
        base_linear_rgb=(0.10, 0.06, 0.04),
        ambient=0.18, key=0.40, fill=0.18,
        scale_tangent=np.log(0.005), scale_normal=np.log(0.0012),
        texture='leather_grain',
    ),
    'dark_velvet': dict(
        base_linear_rgb=(0.22, 0.04, 0.08),   # deep burgundy
        ambient=0.15, key=0.35, fill=0.15,
        scale_tangent=np.log(0.005), scale_normal=np.log(0.0012),
        texture=None,
    ),
    'dark_walnut': dict(
        base_linear_rgb=(0.18, 0.09, 0.04),
        ambient=0.20, key=0.45, fill=0.20,
        scale_tangent=np.log(0.005), scale_normal=np.log(0.0012),
        texture='wood_grain',
    ),
    'dark_fabric': dict(
        base_linear_rgb=(0.08, 0.08, 0.10),   # near-black charcoal
        ambient=0.20, key=0.38, fill=0.18,
        scale_tangent=np.log(0.006), scale_normal=np.log(0.0015),
        texture=None,
    ),
    'dark_stone': dict(
        base_linear_rgb=(0.12, 0.11, 0.10),
        ambient=0.22, key=0.42, fill=0.20,
        scale_tangent=np.log(0.004), scale_normal=np.log(0.001),
        texture='stone_noise',
    ),
    'dark_linen': dict(
        base_linear_rgb=(0.25, 0.22, 0.18),   # warm off-white linen
        ambient=0.45, key=0.38, fill=0.18,
        scale_tangent=np.log(0.005), scale_normal=np.log(0.0012),
        texture=None,
    ),
    # ── Bar / speakeasy ─────────────────────────────────────────────────
    'warm_brass': dict(
        base_linear_rgb=(0.72, 0.52, 0.18),
        ambient=0.25, key=0.55, fill=0.30,
        scale_tangent=np.log(0.004), scale_normal=np.log(0.001),
        texture=None,
    ),
    'warm_amber': dict(
        base_linear_rgb=(0.65, 0.32, 0.05),   # whiskey amber
        ambient=0.30, key=0.40, fill=0.20,
        scale_tangent=np.log(0.004), scale_normal=np.log(0.001),
        texture=None,
    ),
    'brick_warm': dict(
        base_linear_rgb=(0.55, 0.22, 0.10),
        ambient=0.30, key=0.42, fill=0.18,
        scale_tangent=np.log(0.004), scale_normal=np.log(0.001),
        texture='brick',
    ),
    'brushed_steel': dict(
        base_linear_rgb=(0.55, 0.55, 0.58),
        ambient=0.30, key=0.55, fill=0.30,
        scale_tangent=np.log(0.004), scale_normal=np.log(0.001),
        texture=None,
    ),
    'chrome_metal': dict(
        base_linear_rgb=(0.72, 0.72, 0.74),
        ambient=0.25, key=0.65, fill=0.35,
        scale_tangent=np.log(0.003), scale_normal=np.log(0.0008),
        texture=None,
    ),
    # ── Light / neutral ──────────────────────────────────────────────────
    'matte_white': dict(
        base_linear_rgb=(0.88, 0.87, 0.85),
        ambient=0.55, key=0.32, fill=0.15,
        scale_tangent=np.log(0.005), scale_normal=np.log(0.0012),
        texture=None,
    ),
    'projector_screen': dict(
        base_linear_rgb=(0.92, 0.92, 0.90),
        ambient=0.80, key=0.15, fill=0.05,   # screen is self-lit looking
        scale_tangent=np.log(0.004), scale_normal=np.log(0.001),
        texture=None,
    ),
    'light_oak': dict(
        base_linear_rgb=(0.65, 0.42, 0.18),
        ambient=0.35, key=0.42, fill=0.20,
        scale_tangent=np.log(0.005), scale_normal=np.log(0.0012),
        texture='wood_grain',
    ),
    'dark_marble': dict(
        base_linear_rgb=(0.14, 0.13, 0.12),
        ambient=0.25, key=0.50, fill=0.25,
        scale_tangent=np.log(0.004), scale_normal=np.log(0.001),
        texture='stone_noise',
    ),
}


def get_material(name: str) -> dict:
    if name not in MATERIALS:
        raise ValueError(f"Unknown material '{name}'. Available: {sorted(MATERIALS)}")
    return MATERIALS[name]


def _texture_variation(positions: np.ndarray, texture: Optional[str], rng_seed: int = 0) -> np.ndarray:
    """Returns per-Gaussian brightness multiplier from procedural texture. Shape (N,)."""
    n = len(positions)
    if texture is None:
        return np.ones(n)

    rng = np.random.default_rng(rng_seed)

    if texture == 'wood_grain':
        # Sinusoidal grain along X axis with secondary variation
        px = positions[:, 0]
        pz = positions[:, 2]
        grain = (np.sin(px * 14.0 + pz * 0.8) * 0.6 +
                 np.sin(px * 3.5 + 0.4) * 0.3 +
                 rng.normal(0, 0.05, n))
        return 1.0 + np.clip(grain * 0.18, -0.25, 0.25)

    if texture == 'leather_grain':
        # Fine-scale noise for leather pebbling
        noise = (np.sin(positions[:,0]*28)*0.5 + np.sin(positions[:,2]*31)*0.5)
        return 1.0 + np.clip(noise * 0.08 + rng.normal(0, 0.04, n), -0.12, 0.12)

    if texture == 'stone_noise':
        # Low-frequency variation for stone/marble
        noise = (np.sin(positions[:,0]*3.1 + positions[:,2]*2.7) * 0.5 +
                 np.sin(positions[:,1]*5.3) * 0.3 +
                 rng.normal(0, 0.06, n))
        return 1.0 + np.clip(noise * 0.22, -0.35, 0.35)

    if texture == 'brick':
        # Grid pattern for brick mortar lines
        bx = np.floor(positions[:,0] * 3.5)
        bz = np.floor(positions[:,2] * 2.0)
        # Offset alternate rows
        bx_off = np.floor((positions[:,0] + 0.14 * (bz % 2)) * 3.5)
        in_mortar_x = (positions[:,0] * 3.5 % 1.0) < 0.07
        in_mortar_z = (positions[:,2] * 2.0 % 1.0) < 0.07
        mortar = (in_mortar_x | in_mortar_z).astype(float)
        # Mortar is lighter/grayer
        return 1.0 + mortar * 0.35 + rng.normal(0, 0.03, n)

    return np.ones(n)


def _linear_to_dc(linear_rgb: np.ndarray) -> np.ndarray:
    """Convert linear RGB (0-1) to f_dc pre-sigmoid values. Shape (N,3) → (N,3)."""
    clipped = np.clip(linear_rgb, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


def apply_shading(
    normals: np.ndarray,      # (N, 3) unit surface normals
    positions: np.ndarray,    # (N, 3) world positions (for texture)
    material: dict,
    n: int,
    rng_seed: int = 42,
) -> np.ndarray:
    """
    Compute f_dc (pre-sigmoid) values for N Gaussians given their normals and material.

    Returns array of shape (N, 3) — columns are f_dc_0, f_dc_1, f_dc_2.
    """
    ambient = material['ambient']
    key_int = material['key']
    fill_int = material['fill']
    base_rgb = np.array(material['base_linear_rgb'], dtype=np.float64)

    # Lambertian diffuse from key light (overhead, -Y direction)
    key_diff = np.clip(normals @ _KEY_DIR, 0.0, 1.0)          # (N,)

    # Soft fill from frontal direction
    fill_diff = np.clip(normals @ _FILL_DIR, 0.0, 1.0)         # (N,)

    # Total brightness per Gaussian
    brightness = ambient + key_int * key_diff + fill_int * fill_diff  # (N,)
    brightness = np.clip(brightness, 0.0, 1.0)

    # Apply procedural texture variation
    tex_var = _texture_variation(positions, material.get('texture'), rng_seed)
    brightness = np.clip(brightness * tex_var, 1e-6, 1.0 - 1e-6)

    # Scale base color by brightness, per channel
    rgb = np.outer(brightness, base_rgb)                         # (N, 3)
    rgb = np.clip(rgb, 1e-6, 1.0 - 1e-6)

    return _linear_to_dc(rgb)
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/test_materials.py -v
```

Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/materials.py tests/test_materials.py
git commit -m "feat: add materials library with Lambertian shading and procedural textures"
```

---

## Task 3: Furniture Base Utilities

**Files:**
- Create: `scripts/furniture/__init__.py`
- Create: `scripts/furniture/base.py`
- Create: `tests/test_furniture_base.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_furniture_base.py
import sys, numpy as np
sys.path.insert(0, 'scripts')
from furniture.base import (
    box_gaussians, disc_gaussians, cylinder_gaussians,
    N_PROPS, IDX_X, IDX_Y, IDX_Z, IDX_FDC0, IDX_OPACITY, IDX_SCALE0, IDX_ROT0
)

def test_box_gaussians_shape():
    rng = np.random.default_rng(0)
    data = box_gaussians(
        center=(0,0,0), half_extents=(0.5,0.3,0.4),
        material='matte_white', n_gaussians=1000, rng=rng
    )
    assert data.shape == (1000, N_PROPS)
    assert data.dtype == np.float32

def test_box_gaussians_positions_within_bounds():
    rng = np.random.default_rng(0)
    data = box_gaussians(
        center=(1.0, 2.0, 3.0), half_extents=(0.5, 0.3, 0.4),
        material='matte_white', n_gaussians=2000, rng=rng
    )
    x, y, z = data[:,IDX_X], data[:,IDX_Y], data[:,IDX_Z]
    assert x.min() >= 0.48 and x.max() <= 1.52
    assert y.min() >= 1.68 and y.max() <= 2.32
    assert z.min() >= 2.58 and z.max() <= 3.42

def test_box_gaussians_high_opacity():
    rng = np.random.default_rng(0)
    data = box_gaussians(
        center=(0,0,0), half_extents=(0.5,0.3,0.4),
        material='matte_white', n_gaussians=500, rng=rng
    )
    # sigmoid(opacity) should be > 0.95 for all
    opacity_linear = 1.0 / (1.0 + np.exp(-data[:, IDX_OPACITY]))
    assert np.all(opacity_linear > 0.90), f"min={opacity_linear.min():.3f}"

def test_quaternions_are_unit():
    rng = np.random.default_rng(0)
    data = box_gaussians(
        center=(0,0,0), half_extents=(0.5,0.3,0.4),
        material='matte_white', n_gaussians=500, rng=rng
    )
    qw = data[:, IDX_ROT0]
    qx = data[:, IDX_ROT0+1]
    qy = data[:, IDX_ROT0+2]
    qz = data[:, IDX_ROT0+3]
    mag = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    assert np.allclose(mag, 1.0, atol=1e-5), f"non-unit quaternions: min={mag.min()}"

def test_disc_gaussians_centered():
    rng = np.random.default_rng(0)
    data = disc_gaussians(
        center=(5.0, 1.0, 8.0), normal=(0.0, -1.0, 0.0),
        radius=0.5, material='dark_walnut', n_gaussians=500, rng=rng
    )
    assert data.shape[1] == N_PROPS
    cx = data[:, IDX_X].mean()
    cy = data[:, IDX_Y].mean()
    cz = data[:, IDX_Z].mean()
    assert abs(cx - 5.0) < 0.05, f"cx={cx}"
    assert abs(cy - 1.0) < 0.05, f"cy={cy}"
    assert abs(cz - 8.0) < 0.05, f"cz={cz}"
```

- [ ] **Step 2: Run to confirm failure**

```bash
python3 -m pytest tests/test_furniture_base.py -v 2>&1 | head -10
```

- [ ] **Step 3: Implement furniture/\_\_init\_\_.py (empty)**

```python
# scripts/furniture/__init__.py
```

- [ ] **Step 4: Implement furniture/base.py**

```python
# scripts/furniture/base.py
"""
Core Gaussian generation primitives for furniture.

All generators return float32 arrays of shape (N, 14) matching the
MetalSplat2 compact PLY format:
  [x, y, z, f_dc_0, f_dc_1, f_dc_2, opacity,
   scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3]

Shading is applied per-face using materials.apply_shading().
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from materials import get_material, apply_shading

# Property indices
N_PROPS    = 14
IDX_X      = 0
IDX_Y      = 1
IDX_Z      = 2
IDX_FDC0   = 3
IDX_FDC1   = 4
IDX_FDC2   = 5
IDX_OPACITY = 6
IDX_SCALE0 = 7   # tangent axis 0
IDX_SCALE1 = 8   # tangent axis 1
IDX_SCALE2 = 9   # normal axis (thin)
IDX_ROT0   = 10  # quaternion w


def _normal_to_quat(normal: np.ndarray) -> np.ndarray:
    """Quaternion rotating Z-hat (0,0,1) to align with normal. Returns (4,) [w,x,y,z]."""
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal) + 1e-12
    z_hat = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(z_hat, normal), -1.0, 1.0))
    angle = np.arccos(dot)
    if abs(angle) < 1e-7:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if abs(angle - np.pi) < 1e-7:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    axis = np.cross(z_hat, normal)
    axis /= np.linalg.norm(axis)
    h = angle / 2.0
    return np.array([np.cos(h), np.sin(h)*axis[0], np.sin(h)*axis[1], np.sin(h)*axis[2]],
                    dtype=np.float32)


def _normals_to_quats_batch(normals: np.ndarray) -> np.ndarray:
    """Vectorised version of _normal_to_quat. normals: (N,3). Returns (N,4)."""
    normals = np.asarray(normals, dtype=np.float64)
    mag = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    n = normals / mag

    z = np.array([0.0, 0.0, 1.0])
    dot = np.clip(n @ z, -1.0, 1.0)
    angle = np.arccos(dot)                       # (N,)

    axis = np.cross(np.tile(z, (len(n), 1)), n)  # (N,3) — cross(z,n)
    ax_mag = np.linalg.norm(axis, axis=1, keepdims=True) + 1e-12
    axis /= ax_mag

    h = angle / 2.0
    qw = np.cos(h)
    qx = np.sin(h) * axis[:, 0]
    qy = np.sin(h) * axis[:, 1]
    qz = np.sin(h) * axis[:, 2]

    # Handle degenerate cases
    near_zero = angle < 1e-7
    near_pi   = angle > (np.pi - 1e-7)
    qw = np.where(near_zero, 1.0, np.where(near_pi, 0.0, qw))
    qx = np.where(near_zero, 0.0, np.where(near_pi, 1.0, qx))
    qy = np.where(near_zero | near_pi, 0.0, qy)
    qz = np.where(near_zero | near_pi, 0.0, qz)

    quats = np.stack([qw, qx, qy, qz], axis=1).astype(np.float32)
    # Normalise
    quats /= (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-9)
    return quats


def _fill_gaussian_array(
    positions: np.ndarray,  # (N, 3)
    normals: np.ndarray,    # (N, 3)
    material_name: str,
    opacity_range: tuple = (4.0, 6.0),
    rng: np.random.Generator = None,
    rng_seed: int = 42,
) -> np.ndarray:
    """Build a (N, 14) Gaussian array from positions, normals, and material."""
    if rng is None:
        rng = np.random.default_rng(rng_seed)
    n = len(positions)
    mat = get_material(material_name)

    data = np.zeros((n, N_PROPS), dtype=np.float32)
    data[:, IDX_X:IDX_Z+1] = positions.astype(np.float32)

    # Shading → f_dc
    dc = apply_shading(normals, positions, mat, n, rng_seed)
    data[:, IDX_FDC0:IDX_FDC0+3] = dc

    # Opacity
    data[:, IDX_OPACITY] = rng.uniform(opacity_range[0], opacity_range[1], n)

    # Scale
    st = mat['scale_tangent']
    sn = mat['scale_normal']
    noise = rng.normal(0, 0.08, (n, 3))
    data[:, IDX_SCALE0] = st + noise[:, 0]
    data[:, IDX_SCALE1] = st + noise[:, 1]
    data[:, IDX_SCALE2] = sn + noise[:, 2]

    # Quaternions
    data[:, IDX_ROT0:IDX_ROT0+4] = _normals_to_quats_batch(normals)

    return data


def box_gaussians(
    center: tuple,
    half_extents: tuple,    # (hx, hy, hz) in metres
    material: str,
    n_gaussians: int,
    rng: np.random.Generator,
    opacity_range: tuple = (4.0, 6.0),
) -> np.ndarray:
    """
    Surface Gaussians on all 6 faces of an axis-aligned box.
    Gaussians distributed proportional to face area. Per-face Lambertian shading.
    """
    cx, cy, cz = center
    hx, hy, hz = half_extents

    faces = [
        # (outward_normal, u_axis, v_axis, offset, u_half, v_half)
        (np.array([ 1,0,0]), np.array([0,1,0]), np.array([0,0,1]), np.array([hx,0,0]), hy, hz),
        (np.array([-1,0,0]), np.array([0,1,0]), np.array([0,0,1]), np.array([-hx,0,0]), hy, hz),
        (np.array([0, 1,0]), np.array([1,0,0]), np.array([0,0,1]), np.array([0,hy,0]), hx, hz),
        (np.array([0,-1,0]), np.array([1,0,0]), np.array([0,0,1]), np.array([0,-hy,0]), hx, hz),
        (np.array([0,0, 1]), np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,hz]), hx, hy),
        (np.array([0,0,-1]), np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,-hz]), hx, hy),
    ]

    areas = np.array([4.0 * f[4] * f[5] for f in faces])
    total = areas.sum()
    counts = np.round(areas / total * n_gaussians).astype(int)
    counts[np.argmax(areas)] += n_gaussians - counts.sum()

    all_pos, all_nrm = [], []
    for (normal, u_ax, v_ax, offset, u_half, v_half), count in zip(faces, counts):
        n_face = max(1, count)
        u = rng.uniform(-u_half, u_half, n_face)
        v = rng.uniform(-v_half, v_half, n_face)
        pts = (np.array([cx, cy, cz]) + offset)[np.newaxis, :] \
            + u[:, np.newaxis] * u_ax + v[:, np.newaxis] * v_ax
        all_pos.append(pts)
        all_nrm.append(np.tile(normal, (n_face, 1)).astype(np.float32))

    positions = np.concatenate(all_pos, axis=0).astype(np.float32)
    normals   = np.concatenate(all_nrm, axis=0).astype(np.float32)
    return _fill_gaussian_array(positions, normals, material, opacity_range, rng)


def disc_gaussians(
    center: tuple,
    normal: tuple,         # outward normal of the disc face
    radius: float,
    material: str,
    n_gaussians: int,
    rng: np.random.Generator,
    opacity_range: tuple = (4.0, 6.0),
) -> np.ndarray:
    """Flat disc of Gaussians — used for circular table tops, seat cushions."""
    cx, cy, cz = center
    nrm = np.array(normal, dtype=np.float64)
    nrm /= np.linalg.norm(nrm)

    # Build tangent basis
    ref = np.array([1, 0, 0]) if abs(nrm[0]) < 0.9 else np.array([0, 1, 0])
    t1 = np.cross(nrm, ref); t1 /= np.linalg.norm(t1)
    t2 = np.cross(nrm, t1)

    r = np.sqrt(rng.uniform(0, 1, n_gaussians)) * radius
    theta = rng.uniform(0, 2 * np.pi, n_gaussians)
    u, v = r * np.cos(theta), r * np.sin(theta)
    positions = (np.array([cx, cy, cz])[np.newaxis, :]
                 + u[:, np.newaxis] * t1 + v[:, np.newaxis] * t2).astype(np.float32)
    normals = np.tile(nrm, (n_gaussians, 1)).astype(np.float32)
    return _fill_gaussian_array(positions, normals, material, opacity_range, rng)


def cylinder_gaussians(
    center: tuple,
    radius: float,
    half_height: float,
    axis: int,             # 0=X, 1=Y, 2=Z
    material: str,
    n_gaussians: int,
    rng: np.random.Generator,
    opacity_range: tuple = (4.0, 6.0),
    include_caps: bool = True,
) -> np.ndarray:
    """Surface Gaussians on a cylinder. axis=1 → vertical cylinder (Y-axis)."""
    cx, cy, cz = center
    n_side = int(n_gaussians * 0.75) if include_caps else n_gaussians
    n_caps = n_gaussians - n_side

    phi = rng.uniform(0, 2 * np.pi, n_side)
    h   = rng.uniform(-half_height, half_height, n_side)

    if axis == 1:  # Y-axis (vertical)
        px = cx + radius * np.cos(phi)
        py = cy + h
        pz = cz + radius * np.sin(phi)
        nx_, nz_ = np.cos(phi), np.sin(phi)
        ny_ = np.zeros(n_side)
    elif axis == 2:  # Z-axis
        px = cx + radius * np.cos(phi)
        pz = cz + h
        py = cy + np.zeros(n_side)
        nx_, ny_ = np.cos(phi), np.sin(phi)
        nz_ = np.zeros(n_side)
    else:  # X-axis
        py = cy + radius * np.cos(phi)
        pz = cz + radius * np.sin(phi)
        px = cx + h
        ny_, nz_ = np.cos(phi), np.sin(phi)
        nx_ = np.zeros(n_side)

    side_pos = np.stack([px, py, pz], axis=1).astype(np.float32)
    side_nrm = np.stack([nx_, ny_, nz_], axis=1).astype(np.float32)
    mag = np.linalg.norm(side_nrm, axis=1, keepdims=True) + 1e-9
    side_nrm /= mag

    all_pos = [side_pos]
    all_nrm = [side_nrm]

    if include_caps and n_caps > 0:
        for cap_sign in [-1, 1]:
            n_cap = n_caps // 2
            r_cap = np.sqrt(rng.uniform(0, 1, n_cap)) * radius
            phi_cap = rng.uniform(0, 2 * np.pi, n_cap)
            if axis == 1:
                cap_pos = np.stack([
                    cx + r_cap * np.cos(phi_cap),
                    cy + cap_sign * half_height * np.ones(n_cap),
                    cz + r_cap * np.sin(phi_cap)
                ], axis=1)
                cap_nrm = np.tile([0, float(cap_sign), 0], (n_cap, 1))
            else:
                cap_pos = np.stack([
                    cx + r_cap * np.cos(phi_cap),
                    cy + r_cap * np.sin(phi_cap),
                    cz + cap_sign * half_height * np.ones(n_cap)
                ], axis=1)
                cap_nrm = np.tile([0, 0, float(cap_sign)], (n_cap, 1))
            all_pos.append(cap_pos.astype(np.float32))
            all_nrm.append(cap_nrm.astype(np.float32))

    positions = np.concatenate(all_pos, axis=0)
    normals   = np.concatenate(all_nrm, axis=0)
    return _fill_gaussian_array(positions, normals, material, opacity_range, rng)
```

- [ ] **Step 5: Run tests**

```bash
python3 -m pytest tests/test_furniture_base.py -v
```

Expected: 5 PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/furniture/__init__.py scripts/furniture/base.py tests/test_furniture_base.py
git commit -m "feat: furniture/base — box, disc, cylinder Gaussian generators with per-face shading"
```

---

## Task 4: Furniture Elements — Screen, Seating, Tables, Storage

**Files:**
- Create: `scripts/furniture/screen.py`
- Create: `scripts/furniture/seating.py`
- Create: `scripts/furniture/tables.py`
- Create: `scripts/furniture/storage.py`

Each file exposes a `build_<type>(spec, room_geom, rng)` function that returns a `(N, 14)` float32 array.
`spec` is a dict from the scene planner JSON.

- [ ] **Step 1: Implement screen.py**

```python
# scripts/furniture/screen.py
"""
Projector screen and TV screen geometry.

spec keys:
  position_x, position_z  — center of screen on the wall
  width_m, height_m        — screen dimensions
  wall   — 'north'|'south'|'east'|'west' (which wall to mount on)
  floor_clearance_m        — bottom of screen above floor
  material                 — usually 'projector_screen' or 'dark_fabric'
  n_gaussians              — total count (default 200_000)
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

    # Screen center Y: Y-down — screen bottom at floor_y - clearance, center higher up
    screen_bottom_y = room_geom.floor_y - clearance
    screen_center_y = screen_bottom_y - (height_m / 2.0)

    cx = spec.get('position_x', room_geom.center_x)
    cz = spec.get('position_z', _wall_z(wall, room_geom))

    # Screen panel — very thin box (depth = 3cm)
    # Frame border — 5cm on each side, slightly darker
    frame_depth = 0.03
    frame_border = 0.06

    # Main screen surface
    screen = box_gaussians(
        center=(cx, screen_center_y, cz),
        half_extents=(width_m / 2.0, height_m / 2.0, frame_depth / 2.0),
        material=material,
        n_gaussians=int(n * 0.85),
        rng=rng,
    )

    # Frame (slightly protruding dark border)
    frame_hx = width_m / 2.0 + frame_border
    frame_hy = height_m / 2.0 + frame_border
    frame = box_gaussians(
        center=(cx, screen_center_y, cz - frame_depth),
        half_extents=(frame_hx, frame_hy, frame_border / 2.0),
        material='dark_fabric',
        n_gaussians=int(n * 0.15),
        rng=rng,
    )
    return np.concatenate([screen, frame], axis=0)


def _wall_z(wall: str, room_geom) -> float:
    wall_offset = 0.05  # 5cm from wall surface
    if wall == 'north':
        return room_geom.north_z - wall_offset
    if wall == 'south':
        return room_geom.south_z + wall_offset
    if wall == 'east':
        return room_geom.center_z
    return room_geom.center_z
```

- [ ] **Step 2: Implement seating.py**

```python
# scripts/furniture/seating.py
"""
Sectional sofa, club chair, bar stool.

All spec dicts have:
  position_x, position_z  — center of the piece
  facing   — 'north'|'south'|'east'|'west'
  material — seat material name
  n_gaussians
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from furniture.base import box_gaussians, disc_gaussians, cylinder_gaussians


def build_sectional_sofa(spec: dict, room_geom, rng: np.random.Generator) -> np.ndarray:
    """
    Sectional sofa: main seat run + L-section + back rests + arms.
    Sits on the floor. Width typically 3.0m, depth 0.95m, seat height 0.45m.
    """
    cx    = spec.get('position_x', room_geom.center_x)
    cz    = spec.get('position_z', room_geom.center_z)
    mat   = spec.get('material', 'dark_leather')
    n     = spec.get('n_gaussians', 600_000)
    width = spec.get('width_m', 3.0)
    depth = spec.get('depth_m', 0.95)

    floor_y     = room_geom.floor_y
    seat_h      = 0.45    # seat top height above floor
    back_h      = 0.85    # total height to back-rest top
    arm_h       = 0.65
    cushion_d   = 0.55    # seat depth
    back_d      = 0.22    # back rest depth

    # Seat top surface Y center (Y-down: seat bottom at floor, top at floor - seat_h)
    seat_cy = floor_y - (seat_h / 2.0)
    back_cy = floor_y - ((seat_h + (back_h - seat_h) / 2.0))
    arm_cy  = floor_y - (arm_h / 2.0)

    # Main seat block
    seat = box_gaussians(
        center=(cx, seat_cy, cz),
        half_extents=(width / 2.0, seat_h / 2.0, cushion_d / 2.0),
        material=mat, n_gaussians=int(n * 0.35), rng=rng,
    )
    # Back rest (behind seat, taller, shallower)
    back_z = cz + cushion_d / 2.0 + back_d / 2.0
    back = box_gaussians(
        center=(cx, back_cy, back_z),
        half_extents=(width / 2.0, (back_h - seat_h) / 2.0, back_d / 2.0),
        material=mat, n_gaussians=int(n * 0.30), rng=rng,
    )
    # Left arm
    arm_x_l = cx - width / 2.0 - 0.10
    arm_l = box_gaussians(
        center=(arm_x_l, arm_cy, cz),
        half_extents=(0.10, arm_h / 2.0, (cushion_d + back_d) / 2.0),
        material=mat, n_gaussians=int(n * 0.10), rng=rng,
    )
    # Right arm
    arm_x_r = cx + width / 2.0 + 0.10
    arm_r = box_gaussians(
        center=(arm_x_r, arm_cy, cz),
        half_extents=(0.10, arm_h / 2.0, (cushion_d + back_d) / 2.0),
        material=mat, n_gaussians=int(n * 0.10), rng=rng,
    )
    # L-section: extends 1.2m to one side
    l_width = 1.20
    l_cx = cx + width / 2.0 + 0.10 + l_width / 2.0
    l_section = box_gaussians(
        center=(l_cx, seat_cy, cz - 0.20),
        half_extents=(l_width / 2.0, seat_h / 2.0, cushion_d / 2.0),
        material=mat, n_gaussians=int(n * 0.15), rng=rng,
    )
    return np.concatenate([seat, back, arm_l, arm_r, l_section], axis=0)


def build_club_chair(spec: dict, room_geom, rng: np.random.Generator) -> np.ndarray:
    """Single club chair — compact version of sofa."""
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
    """Bar stool: leather disc seat + chrome post + footrest ring."""
    cx       = spec.get('position_x', 0.0)
    cz       = spec.get('position_z', room_geom.center_z)
    seat_mat = spec.get('material', 'dark_leather')
    n        = spec.get('n_gaussians', 60_000)
    seat_h   = spec.get('seat_height_m', 0.75)

    floor_y  = room_geom.floor_y
    seat_y   = floor_y - seat_h

    # Seat disc
    seat = disc_gaussians(
        center=(cx, seat_y, cz), normal=(0, -1, 0),
        radius=0.18, material=seat_mat, n_gaussians=int(n * 0.40), rng=rng,
    )
    # Post (vertical cylinder)
    post = cylinder_gaussians(
        center=(cx, floor_y - seat_h / 2.0, cz),
        radius=0.025, half_height=seat_h / 2.0, axis=1,
        material='chrome_metal', n_gaussians=int(n * 0.25), rng=rng,
    )
    # Footrest ring (approximated as 8 short cylinders in a ring)
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
            material='chrome_metal', n_gaussians=max(100, int(n * 0.04)), rng=rng,
        ))
    # Base disc
    base = disc_gaussians(
        center=(cx, floor_y, cz), normal=(0, 1, 0),
        radius=0.22, material='chrome_metal', n_gaussians=int(n * 0.15), rng=rng,
    )
    return np.concatenate([seat, post, base] + ring_parts, axis=0)
```

- [ ] **Step 3: Implement tables.py**

```python
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
    """Low rectangular coffee table. Top: 120×60cm, height 40cm."""
    cx    = spec.get('position_x', room_geom.center_x)
    cz    = spec.get('position_z', room_geom.center_z)
    mat   = spec.get('material', 'dark_walnut')
    n     = spec.get('n_gaussians', 120_000)
    w     = spec.get('width_m', 1.20)
    d     = spec.get('depth_m', 0.60)
    h     = spec.get('height_m', 0.40)

    floor_y = room_geom.floor_y
    top_cy  = floor_y - h + 0.025
    leg_cy  = floor_y - h / 2.0

    top = box_gaussians(
        center=(cx, top_cy, cz),
        half_extents=(w/2.0, 0.025, d/2.0),
        material=mat, n_gaussians=int(n * 0.60), rng=rng,
    )
    # Four legs
    legs = []
    for lx, lz in [(-w/2+0.06, -d/2+0.06), (w/2-0.06, -d/2+0.06),
                   (-w/2+0.06,  d/2-0.06), (w/2-0.06,  d/2-0.06)]:
        legs.append(cylinder_gaussians(
            center=(cx+lx, leg_cy, cz+lz),
            radius=0.025, half_height=h/2.0-0.025, axis=1,
            material=mat, n_gaussians=int(n * 0.10), rng=rng,
        ))
    return np.concatenate([top] + legs, axis=0)


def build_bar_counter(spec: dict, room_geom, rng: np.random.Generator) -> np.ndarray:
    """
    Bar counter: base cabinet body + stone/marble top slab + brass bar rail.
    Standard bar height 110cm, depth 60cm.
    """
    cx       = spec.get('position_x', room_geom.center_x)
    cz       = spec.get('position_z', room_geom.north_z - 0.35)
    length   = spec.get('length_m', 2.4)
    mat_body = spec.get('material', 'dark_walnut')
    mat_top  = spec.get('top_material', 'dark_marble')
    n        = spec.get('n_gaussians', 400_000)

    floor_y  = room_geom.floor_y
    bar_h    = 1.10
    top_h    = 0.05
    body_h   = bar_h - top_h
    depth    = 0.58

    body_cy = floor_y - body_h / 2.0
    top_cy  = floor_y - bar_h - top_h / 2.0

    body = box_gaussians(
        center=(cx, body_cy, cz),
        half_extents=(length/2.0, body_h/2.0, depth/2.0),
        material=mat_body, n_gaussians=int(n * 0.55), rng=rng,
    )
    top = box_gaussians(
        center=(cx, top_cy, cz),
        half_extents=(length/2.0 + 0.03, top_h/2.0, depth/2.0 + 0.02),
        material=mat_top, n_gaussians=int(n * 0.30), rng=rng,
    )
    # Brass foot rail along front of bar (50cm up from floor)
    rail_y = floor_y - 0.50
    rail = cylinder_gaussians(
        center=(cx, rail_y, cz - depth/2.0 + 0.08),
        radius=0.025, half_height=length/2.0, axis=0,
        material='warm_brass', n_gaussians=int(n * 0.15), rng=rng,
    )
    return np.concatenate([body, top, rail], axis=0)


def build_round_table(spec: dict, room_geom, rng: np.random.Generator) -> np.ndarray:
    """Small round lounge/speakeasy table. Radius 0.4m, height 0.65m."""
    cx     = spec.get('position_x', 0.0)
    cz     = spec.get('position_z', room_geom.center_z)
    mat    = spec.get('material', 'dark_walnut')
    n      = spec.get('n_gaussians', 80_000)
    radius = spec.get('radius_m', 0.40)
    h      = spec.get('height_m', 0.65)

    floor_y = room_geom.floor_y
    top_y   = floor_y - h
    post_cy = floor_y - h / 2.0

    top  = disc_gaussians((cx, top_y, cz), (0,-1,0), radius, mat, int(n*0.50), rng)
    edge = cylinder_gaussians((cx, post_cy, cz), radius, 0.02, 1, mat, int(n*0.15), rng,
                               include_caps=False)
    post = cylinder_gaussians((cx, post_cy, cz), 0.04, h/2.0, 1, mat, int(n*0.20), rng)
    base = disc_gaussians((cx, floor_y, cz), (0,1,0), radius*0.5, mat, int(n*0.15), rng)
    return np.concatenate([top, edge, post, base], axis=0)
```

- [ ] **Step 4: Implement storage.py**

```python
# scripts/furniture/storage.py
"""
Back bar shelving, acoustic panels, wall panels (brick / dark wood).
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from furniture.base import box_gaussians, cylinder_gaussians, disc_gaussians


def build_back_bar_shelving(spec: dict, room_geom, rng: np.random.Generator) -> np.ndarray:
    """
    Multi-shelf back bar unit. 3 shelves with bottle clusters.
    Mounts against north wall. Width typically 2.4m, height floor-to-ceiling.
    """
    cx       = spec.get('position_x', room_geom.center_x)
    cz       = spec.get('position_z', room_geom.north_z - 0.20)
    width    = spec.get('width_m', 2.4)
    mat_body = spec.get('material', 'dark_walnut')
    n        = spec.get('n_gaussians', 500_000)

    floor_y   = room_geom.floor_y
    ceiling_y = room_geom.ceiling_y
    unit_h    = abs(floor_y - ceiling_y) * 0.85
    depth     = 0.35

    # Back panel
    back_cy = floor_y - unit_h / 2.0
    back = box_gaussians(
        center=(cx, back_cy, cz + depth/2.0),
        half_extents=(width/2.0, unit_h/2.0, 0.025),
        material=mat_body, n_gaussians=int(n * 0.15), rng=rng,
    )

    # 3 shelves at different heights
    parts = [back]
    shelf_ys = [floor_y - 0.90, floor_y - 1.50, floor_y - 2.10]
    for sy in shelf_ys:
        shelf = box_gaussians(
            center=(cx, sy, cz),
            half_extents=(width/2.0, 0.02, depth/2.0),
            material=mat_body, n_gaussians=int(n * 0.08), rng=rng,
        )
        parts.append(shelf)

        # Bottle clusters on each shelf (cylinders)
        n_bottles = 12
        for i in range(n_bottles):
            bx = cx - width/2.0 + 0.12 + i * (width - 0.24) / (n_bottles - 1)
            bottle_h = rng.uniform(0.22, 0.32)
            bottle = cylinder_gaussians(
                center=(bx, sy - bottle_h/2.0 - 0.02, cz - 0.05),
                radius=rng.uniform(0.02, 0.035),
                half_height=bottle_h / 2.0,
                axis=1,
                material='warm_amber',
                n_gaussians=int(n * 0.012),
                rng=rng,
            )
            parts.append(bottle)

    # Side panels
    for sx in [cx - width/2.0 - 0.015, cx + width/2.0 + 0.015]:
        side = box_gaussians(
            center=(sx, back_cy, cz),
            half_extents=(0.015, unit_h/2.0, depth/2.0),
            material=mat_body, n_gaussians=int(n * 0.03), rng=rng,
        )
        parts.append(side)

    return np.concatenate(parts, axis=0)


def build_acoustic_panel(spec: dict, room_geom, rng: np.random.Generator) -> np.ndarray:
    """
    Rectangular acoustic panel mounted on a wall. Width ~0.6m, height ~1.2m, depth 0.08m.
    """
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
    """
    Decorative wall treatment: vertical dark wood panels or brick surface.
    Covers specified wall from floor to wainscot height (~1.2m).
    """
    wall   = spec.get('wall', 'north')
    mat    = spec.get('material', 'brick_warm')
    n      = spec.get('n_gaussians', 300_000)
    height = spec.get('height_m', 1.2)

    floor_y = room_geom.floor_y
    cy = floor_y - height / 2.0

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
    # south
    return box_gaussians(
        center=(room_geom.center_x, cy, room_geom.south_z + 0.04),
        half_extents=(room_geom.room_width_m/2.0, height/2.0, 0.04),
        material=mat, n_gaussians=n, rng=rng,
    )
```

- [ ] **Step 5: Quick smoke test for all furniture modules**

```bash
python3 - << 'EOF'
import sys, numpy as np
sys.path.insert(0, 'scripts')
from room_analyzer import analyze_room
from furniture.screen import build_screen
from furniture.seating import build_sectional_sofa, build_bar_stool
from furniture.tables import build_coffee_table, build_bar_counter
from furniture.storage import build_back_bar_shelving

rng = np.random.default_rng(42)
geom = analyze_room('input/current.ply')
print(f"Room: {geom.room_width_m:.2f}m × {geom.room_depth_m:.2f}m × {geom.room_height_m:.2f}m")

spec = {'width_m': 2.4, 'height_m': 1.35, 'wall': 'north', 'floor_clearance_m': 0.8}
s = build_screen(spec, geom, rng); print(f"screen:    {len(s):>7,} Gaussians, shape={s.shape}")

spec = {'width_m': 3.0, 'position_x': geom.center_x, 'position_z': geom.center_z}
s = build_sectional_sofa(spec, geom, rng); print(f"sofa:      {len(s):>7,} Gaussians")

spec = {'position_x': 0.0, 'position_z': geom.center_z - 1.5}
s = build_bar_stool(spec, geom, rng); print(f"bar_stool: {len(s):>7,} Gaussians")

spec = {'position_x': geom.center_x, 'position_z': geom.center_z}
s = build_coffee_table(spec, geom, rng); print(f"coffee_tb: {len(s):>7,} Gaussians")

spec = {'position_x': geom.center_x, 'length_m': 2.4}
s = build_bar_counter(spec, geom, rng); print(f"bar_ctr:   {len(s):>7,} Gaussians")

spec = {'position_x': geom.center_x, 'width_m': 2.4}
s = build_back_bar_shelving(spec, geom, rng); print(f"back_bar:  {len(s):>7,} Gaussians")

print("All furniture modules OK")
EOF
```

Expected: all modules print Gaussian counts, `All furniture modules OK`

- [ ] **Step 6: Commit**

```bash
git add scripts/furniture/
git commit -m "feat: furniture library — screen, seating, tables, storage with shading + materials"
```

---

## Task 5: Scene Planner (Claude API)

**Files:**
- Create: `scripts/scene_planner.py`

- [ ] **Step 1: Implement scene_planner.py**

```python
# scripts/scene_planner.py
"""
Uses Claude API to reason about a redesign prompt given real room geometry
and returns a structured JSON scene plan.

Usage:
  python3 scripts/scene_planner.py "Turn this space into a movie room" input/current.ply
"""
import json
import sys
from pathlib import Path
import anthropic

sys.path.insert(0, str(Path(__file__).parent))
from room_analyzer import RoomGeometry

ELEMENT_TYPES = [
    'projector_screen', 'tv_screen',
    'sectional_sofa', 'club_chair', 'bar_stool',
    'coffee_table', 'round_table', 'bar_counter',
    'back_bar_shelving', 'acoustic_panel', 'wall_panels',
]

MATERIALS = [
    'dark_leather', 'dark_velvet', 'dark_walnut', 'dark_fabric',
    'dark_stone', 'dark_marble', 'dark_linen',
    'matte_white', 'projector_screen', 'light_oak',
    'warm_brass', 'warm_amber', 'brick_warm',
    'brushed_steel', 'chrome_metal',
]

SYSTEM_PROMPT = """You are an expert 3D interior designer specializing in Gaussian splat scene composition.
Given a room's measured geometry and a redesign request, produce a precise JSON scene plan.

Rules:
- Place elements using the room's actual wall/floor positions — never use approximate values
- Maintain 0.5m clearance from walls for freestanding furniture unless wall-mounted
- Maintain 0.8m walkways between furniture pieces
- Choose materials that match the design style (moody bar → dark materials, bright theater → dark with bright screen)
- n_gaussians should be proportional to the element's visible surface area (larger objects get more)
- Typical values: sofa 600_000, bar counter 400_000, screen 200_000, chair 150_000, stool 60_000
- Positions use room world coordinates (Y-down: floor=high Y, Y values decrease upward)
- position_x and position_z are the element center in world space
- wall_panels and acoustic_panels go on specific walls, not free-floating

Your output must be valid JSON matching the schema exactly. No markdown fences, no commentary."""


def plan_scene(prompt: str, room_geom: RoomGeometry) -> dict:
    """
    Call Claude to generate a scene plan for the given prompt and room geometry.
    Returns parsed JSON dict.
    """
    client = anthropic.Anthropic()

    user_message = f"""Room geometry (measured from actual 3DGS PLY):
{json.dumps(room_geom.to_dict(), indent=2)}

Redesign request: "{prompt}"

Available element types: {', '.join(ELEMENT_TYPES)}
Available materials: {', '.join(MATERIALS)}

Produce a JSON scene plan with this exact schema:
{{
  "design_name": "snake_case_slug",
  "style": "one sentence style description",
  "ambient_mood": "bright|warm|dim|dark",
  "reasoning": "2-3 sentences explaining key design decisions",
  "elements": [
    {{
      "type": "element_type_from_list",
      "position_x": <float — world X coordinate>,
      "position_z": <float — world Z coordinate>,
      "wall": "north|south|east|west|none",
      "width_m": <float>,
      "height_m": <float>,
      "depth_m": <float>,
      "length_m": <float — for bar_counter>,
      "floor_clearance_m": <float — for wall-mounted elements>,
      "material": "material_from_list",
      "top_material": "material_from_list — for bar_counter only",
      "n_gaussians": <int>,
      "notes": "optional special instruction"
    }}
  ]
}}

Include only fields relevant to the element type. Omit fields that don't apply."""

    response = client.messages.create(
        model='claude-opus-4-6',
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{'role': 'user', 'content': user_message}],
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith('```'):
        lines = raw.split('\n')
        raw = '\n'.join(lines[1:-1] if lines[-1] == '```' else lines[1:])

    plan = json.loads(raw)
    return plan


if __name__ == '__main__':
    from room_analyzer import analyze_room
    prompt   = sys.argv[1] if len(sys.argv) > 1 else 'Turn this space into a movie room'
    ply_path = sys.argv[2] if len(sys.argv) > 2 else 'input/current.ply'
    geom     = analyze_room(ply_path)
    plan     = plan_scene(prompt, geom)
    print(json.dumps(plan, indent=2))
```

- [ ] **Step 2: Test scene planner with movie room prompt**

```bash
cd /Users/gvwert/Development/multi_agent
python3 scripts/scene_planner.py "Turn this space into a movie room" input/current.ply
```

Expected: valid JSON with elements including projector_screen, sectional_sofa, coffee_table, acoustic panels.
Save output to `.claude/edits/plan_movie_room.json`.

- [ ] **Step 3: Test scene planner with speakeasy prompt**

```bash
python3 scripts/scene_planner.py "What about turning this into a modern speakeasy bar" input/current.ply
```

Expected: valid JSON with bar_counter, back_bar_shelving, bar_stool, round_table, wall_panels (brick).
Save output to `.claude/edits/plan_speakeasy.json`.

- [ ] **Step 4: Commit**

```bash
git add scripts/scene_planner.py .claude/edits/plan_movie_room.json .claude/edits/plan_speakeasy.json
git commit -m "feat: scene_planner — Claude API generates room-aware JSON scene plans"
```

---

## Task 6: Main Redesign Orchestrator

**Files:**
- Create: `scripts/redesign.py`

- [ ] **Step 1: Implement redesign.py**

```python
#!/usr/bin/env python3
"""
Main entry point for room redesign pipeline.

Usage:
  python3 scripts/redesign.py "Turn this space into a movie room"
  python3 scripts/redesign.py "Turn this space into a movie room" --ply input/current.ply --out output/movie_room.ply

Pipeline:
  1. Analyze room geometry from PLY
  2. Call Claude to generate scene plan
  3. Remove content Gaussians (preserve room structure)
  4. Generate furniture Gaussians for each element
  5. Assemble and write output PLY
"""
import sys
import json
import argparse
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from ply_io import read_ply, write_ply
from room_analyzer import analyze_room
from scene_planner import plan_scene

# Furniture builders
from furniture.screen import build_screen
from furniture.seating import build_sectional_sofa, build_club_chair, build_bar_stool
from furniture.tables import build_coffee_table, build_bar_counter, build_round_table
from furniture.storage import build_back_bar_shelving, build_acoustic_panel, build_wall_panels

ROOT = Path(__file__).parent.parent

BUILDERS = {
    'projector_screen':   build_screen,
    'tv_screen':          build_screen,
    'sectional_sofa':     build_sectional_sofa,
    'club_chair':         build_club_chair,
    'bar_stool':          build_bar_stool,
    'coffee_table':       build_coffee_table,
    'bar_counter':        build_bar_counter,
    'round_table':        build_round_table,
    'back_bar_shelving':  build_back_bar_shelving,
    'acoustic_panel':     build_acoustic_panel,
    'wall_panels':        build_wall_panels,
}


def redesign(
    prompt: str,
    ply_in: str = 'input/current.ply',
    ply_out: str = None,
    plan_path: str = None,
    structure_margin_m: float = 0.25,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ── 1. Backup ────────────────────────────────────────────────────────
    backup_dir = ROOT / '.claude/backups'
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ply_in, backup_dir / f'current_{ts}_preredesign.ply')
    print(f"[redesign] Backup: {backup_dir}/current_{ts}_preredesign.ply")

    # ── 2. Analyze room ──────────────────────────────────────────────────
    print(f"[redesign] Analyzing room geometry from {ply_in} ...")
    geom = analyze_room(ply_in)
    print(f"[redesign] Room: {geom.room_width_m:.2f}m W × {geom.room_depth_m:.2f}m D × {geom.room_height_m:.2f}m H")
    print(f"           Floor Y={geom.floor_y:.3f}  Ceiling Y={geom.ceiling_y:.3f}")
    print(f"           N wall Z={geom.north_z:.3f}  S wall Z={geom.south_z:.3f}")

    # ── 3. Generate scene plan ───────────────────────────────────────────
    if plan_path and Path(plan_path).exists():
        print(f"[redesign] Loading existing plan from {plan_path}")
        with open(plan_path) as f:
            plan = json.load(f)
    else:
        print(f"[redesign] Calling Claude to plan: '{prompt}' ...")
        plan = plan_scene(prompt, geom)
        saved_plan = ROOT / f'.claude/edits/plan_{plan["design_name"]}_{ts}.json'
        saved_plan.parent.mkdir(parents=True, exist_ok=True)
        with open(saved_plan, 'w') as f:
            json.dump(plan, f, indent=2)
        print(f"[redesign] Plan saved: {saved_plan}")

    design_name = plan['design_name']
    print(f"[redesign] Design: {design_name} — {plan['style']}")
    print(f"[redesign] Reasoning: {plan['reasoning']}")
    print(f"[redesign] Elements: {len(plan['elements'])}")

    # ── 4. Load PLY, remove content ──────────────────────────────────────
    print(f"[redesign] Loading PLY ...")
    data, header_bytes, props, tail_bytes = read_ply(ply_in)
    n_total = len(data)
    print(f"[redesign] Loaded {n_total:,} Gaussians")

    structure_mask, content_mask = geom.split_structure_content(data, structure_margin_m)
    n_content = int(content_mask.sum())
    n_structure = int(structure_mask.sum())
    print(f"[redesign] Structure: {n_structure:,}  Content (to remove): {n_content:,}")

    prop_idx = {p: i for i, p in enumerate(props)}
    opacity_col = prop_idx['opacity']
    data[content_mask, opacity_col] = -10.0
    print(f"[redesign] Content Gaussians hidden (opacity→-10)")

    # ── 5. Generate furniture elements ───────────────────────────────────
    print(f"[redesign] Generating furniture ...")
    new_gaussians = []
    for i, elem in enumerate(plan['elements']):
        elem_type = elem.get('type', 'unknown')
        builder = BUILDERS.get(elem_type)
        if builder is None:
            print(f"  [{i+1}] SKIP unknown type: {elem_type}")
            continue
        try:
            g = builder(elem, geom, rng)
            new_gaussians.append(g)
            print(f"  [{i+1}] {elem_type}: {len(g):,} Gaussians  mat={elem.get('material','?')}")
        except Exception as e:
            print(f"  [{i+1}] ERROR building {elem_type}: {e}")

    if new_gaussians:
        furniture_data = np.concatenate(new_gaussians, axis=0).astype(np.float32)
        combined = np.concatenate([data, furniture_data], axis=0)
        print(f"[redesign] Furniture: {len(furniture_data):,} new Gaussians")
    else:
        combined = data
        print("[redesign] WARNING: no furniture generated")

    print(f"[redesign] Total: {len(combined):,} Gaussians")

    # ── 6. Write output ──────────────────────────────────────────────────
    if ply_out is None:
        out_dir = ROOT / 'output'
        out_dir.mkdir(exist_ok=True)
        ply_out = str(out_dir / f'{design_name}.ply')

    write_ply(ply_out, combined, header_bytes, tail_bytes)
    size_mb = Path(ply_out).stat().st_size / 1024 / 1024
    print(f"\n[redesign] DONE → {ply_out} ({size_mb:.1f} MB)")

    return {
        'ply_out': ply_out,
        'design_name': design_name,
        'n_input': n_total,
        'n_structure': n_structure,
        'n_content_removed': n_content,
        'n_furniture': len(furniture_data) if new_gaussians else 0,
        'n_total': len(combined),
        'plan': plan,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt')
    parser.add_argument('--ply', default='input/current.ply')
    parser.add_argument('--out', default=None)
    parser.add_argument('--plan', default=None, help='Use existing plan JSON instead of calling Claude')
    parser.add_argument('--margin', type=float, default=0.25, help='Structure margin in metres')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    result = redesign(
        prompt=args.prompt,
        ply_in=args.ply,
        ply_out=args.out,
        plan_path=args.plan,
        structure_margin_m=args.margin,
        seed=args.seed,
    )
    print(json.dumps({k: v for k, v in result.items() if k != 'plan'}, indent=2))
```

- [ ] **Step 2: Commit**

```bash
git add scripts/redesign.py
git commit -m "feat: redesign.py — full pipeline orchestrator: analyze→plan→remove→generate→write"
```

---

## Task 7: Run Movie Room — Iterate to VISUAL_PASS

- [ ] **Step 1: Generate movie room PLY**

```bash
cd /Users/gvwert/Development/multi_agent
python3 scripts/redesign.py "Turn this space into a movie room" \
  --ply input/current.ply \
  --out output/movie_room.ply
```

- [ ] **Step 2: Render three views**

```bash
bash scripts/render_screenshot.sh output/movie_room.ply verify_movie_room_front 1
bash scripts/render_screenshot.sh output/movie_room.ply verify_movie_room_side_r 2
bash scripts/render_screenshot.sh output/movie_room.ply verify_movie_room_side_l 3
```

- [ ] **Step 3: Score against visual-verifier rubric**

Read all three screenshots. Score:
- Room structure integrity (walls, floor, ceiling intact)
- Content removal cleanliness (no old furniture bleeding through)
- Semantic correctness (projector screen, sofa, coffee table visible)
- New content realism (3D shading visible, not flat boxes)
- Integration coherence (objects at correct scale, on the floor)

- [ ] **Step 4: Iterate if any dimension < 7**

If verifier fails:
- Read critique → identify which pipeline stage needs fixing
- Fix the relevant script
- Re-run from Step 1

Common fixes:
- "Objects floating": check `floor_y` usage in furniture builders — confirm using `room_geom.floor_y`
- "Old furniture still visible": increase `structure_margin_m` to 0.35
- "Boxes look flat": verify `apply_shading` is being called with correct normals
- "Screen not visible": check wall Z position, ensure screen faces camera (south direction)

- [ ] **Step 5: On VISUAL_PASS — save golden**

```bash
cp $(ls .claude/screenshots/verify_movie_room_front_*.png | tail -1) .claude/golden/movie_room_golden_front.png
cp $(ls .claude/screenshots/verify_movie_room_side_r_*.png | tail -1) .claude/golden/movie_room_golden_side_r.png
cp $(ls .claude/screenshots/verify_movie_room_side_l_*.png | tail -1) .claude/golden/movie_room_golden_side_l.png
```

- [ ] **Step 6: Commit passing output**

```bash
git add .claude/golden/movie_room_golden_*.png .claude/edits/plan_movie_room*.json
git commit -m "feat: movie room redesign — VISUAL_PASS"
```

---

## Task 8: Run Speakeasy Bar — Iterate to VISUAL_PASS

- [ ] **Step 1: Generate speakeasy PLY**

```bash
python3 scripts/redesign.py "What about turning this into a modern speakeasy bar" \
  --ply input/current.ply \
  --out output/speakeasy_bar.ply
```

- [ ] **Step 2: Render three views**

```bash
bash scripts/render_screenshot.sh output/speakeasy_bar.ply verify_speakeasy_front 1
bash scripts/render_screenshot.sh output/speakeasy_bar.ply verify_speakeasy_side_r 2
bash scripts/render_screenshot.sh output/speakeasy_bar.ply verify_speakeasy_side_l 3
```

- [ ] **Step 3: Score against visual-verifier rubric**

Semantic checklist for speakeasy:
- MUST APPEAR: bar counter, back bar shelving with bottles, bar stools, atmospheric dark materials
- MUST NOT APPEAR: original desk, sofa, office furniture
- STRUCTURE PRESERVE: floor, walls, ceiling, original room lighting

- [ ] **Step 4: Iterate if any dimension < 7**

Common speakeasy-specific fixes:
- "Bar feels too bright": darken materials — lower `ambient` in dark_walnut / dark_marble
- "Bottles not visible": increase n_gaussians on back_bar_shelving bottles
- "Bar misplaced": check Claude's plan — bar_counter should be against north wall

- [ ] **Step 5: On VISUAL_PASS — save golden**

```bash
cp $(ls .claude/screenshots/verify_speakeasy_front_*.png | tail -1) .claude/golden/speakeasy_golden_front.png
cp $(ls .claude/screenshots/verify_speakeasy_side_r_*.png | tail -1) .claude/golden/speakeasy_golden_side_r.png
cp $(ls .claude/screenshots/verify_speakeasy_side_l_*.png | tail -1) .claude/golden/speakeasy_golden_side_l.png
```

- [ ] **Step 6: Commit**

```bash
git add .claude/golden/speakeasy_golden_*.png .claude/edits/plan_speakeasy*.json
git commit -m "feat: speakeasy bar redesign — VISUAL_PASS"
```

---

## Self-Review

**Spec coverage:**
- ✅ Room structure preserved (split_structure_content in room_analyzer)
- ✅ Content removal clean (opacity→-10 for content_mask)
- ✅ Claude reasons about redesign (scene_planner.py uses Claude API)
- ✅ Real room dimensions used (analyze_room from PLY percentiles)
- ✅ Proper furniture geometry (box/disc/cylinder, not just boxes)
- ✅ Lambertian shading (apply_shading in materials.py)
- ✅ Procedural textures (wood_grain, leather_grain, brick, stone_noise)
- ✅ Movie room prompt tested (Task 7)
- ✅ Speakeasy prompt tested (Task 8)
- ✅ Visual verifier gates quality (rewritten verifier used in Tasks 7-8)
- ✅ Regression goldens stored (Tasks 7-8 Step 5)

**Placeholder scan:** None found. All code blocks are complete.

**Type consistency:**
- `RoomGeometry` used consistently across room_analyzer, scene_planner, all furniture builders
- `build_*` functions all take `(spec: dict, room_geom: RoomGeometry, rng: np.random.Generator) → np.ndarray`
- `BUILDERS` dict keys match element types returned by Claude in plan JSON
