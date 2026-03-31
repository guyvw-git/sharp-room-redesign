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

apply_shading() converts material + per-Gaussian normals -> f_dc values (pre-sigmoid).
"""
import numpy as np
from typing import Optional

# Key light: overhead, direction (0, -1, 0) in Y-down
_KEY_DIR = np.array([0.0, -1.0, 0.0])
# Fill light: soft frontal from camera-ish direction
_FILL_DIR = np.array([0.0, -0.3, -1.0])
_FILL_DIR = _FILL_DIR / np.linalg.norm(_FILL_DIR)

MATERIALS: dict = {
    # Dark / moody materials (movie room, speakeasy)
    'dark_leather': dict(
        base_linear_rgb=(0.10, 0.06, 0.04),
        ambient=0.18, key=0.40, fill=0.18,
        scale_tangent=np.log(0.005), scale_normal=np.log(0.0012),
        texture='leather_grain',
    ),
    'dark_velvet': dict(
        base_linear_rgb=(0.22, 0.04, 0.08),
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
        base_linear_rgb=(0.08, 0.08, 0.10),
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
        base_linear_rgb=(0.25, 0.22, 0.18),
        ambient=0.45, key=0.38, fill=0.18,
        scale_tangent=np.log(0.005), scale_normal=np.log(0.0012),
        texture=None,
    ),
    # Bar / speakeasy
    'warm_brass': dict(
        base_linear_rgb=(0.72, 0.52, 0.18),
        ambient=0.25, key=0.55, fill=0.30,
        scale_tangent=np.log(0.004), scale_normal=np.log(0.001),
        texture=None,
    ),
    'warm_amber': dict(
        base_linear_rgb=(0.65, 0.32, 0.05),
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
    # Light / neutral
    'matte_white': dict(
        base_linear_rgb=(0.88, 0.87, 0.85),
        ambient=0.55, key=0.32, fill=0.15,
        scale_tangent=np.log(0.005), scale_normal=np.log(0.0012),
        texture=None,
    ),
    'projector_screen': dict(
        base_linear_rgb=(0.92, 0.92, 0.90),
        ambient=0.80, key=0.15, fill=0.05,
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
        px = positions[:, 0]
        pz = positions[:, 2]
        grain = (np.sin(px * 14.0 + pz * 0.8) * 0.6 +
                 np.sin(px * 3.5 + 0.4) * 0.3 +
                 rng.normal(0, 0.05, n))
        return 1.0 + np.clip(grain * 0.18, -0.25, 0.25)

    if texture == 'leather_grain':
        noise = (np.sin(positions[:,0]*28)*0.5 + np.sin(positions[:,2]*31)*0.5)
        return 1.0 + np.clip(noise * 0.08 + rng.normal(0, 0.04, n), -0.12, 0.12)

    if texture == 'stone_noise':
        noise = (np.sin(positions[:,0]*3.1 + positions[:,2]*2.7) * 0.5 +
                 np.sin(positions[:,1]*5.3) * 0.3 +
                 rng.normal(0, 0.06, n))
        return 1.0 + np.clip(noise * 0.22, -0.35, 0.35)

    if texture == 'brick':
        bz = np.floor(positions[:,2] * 2.0)
        in_mortar_x = (positions[:,0] * 3.5 % 1.0) < 0.07
        in_mortar_z = (positions[:,2] * 2.0 % 1.0) < 0.07
        mortar = (in_mortar_x | in_mortar_z).astype(float)
        return 1.0 + mortar * 0.35 + rng.normal(0, 0.03, n)

    return np.ones(n)


def _linear_to_dc(linear_rgb: np.ndarray) -> np.ndarray:
    """Convert linear RGB (0-1) to f_dc pre-sigmoid values. Shape (N,3) -> (N,3)."""
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
    Returns array of shape (N, 3).
    """
    ambient = material['ambient']
    key_int = material['key']
    fill_int = material['fill']
    base_rgb = np.array(material['base_linear_rgb'], dtype=np.float64)

    key_diff = np.clip(normals @ _KEY_DIR, 0.0, 1.0)
    fill_diff = np.clip(normals @ _FILL_DIR, 0.0, 1.0)

    brightness = ambient + key_int * key_diff + fill_int * fill_diff
    brightness = np.clip(brightness, 0.0, 1.0)

    tex_var = _texture_variation(positions, material.get('texture'), rng_seed)
    brightness = np.clip(brightness * tex_var, 1e-6, 1.0 - 1e-6)

    rgb = np.outer(brightness, base_rgb)
    rgb = np.clip(rgb, 1e-6, 1.0 - 1e-6)

    return _linear_to_dc(rgb)
