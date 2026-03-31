# scripts/furniture/base.py
"""
Core Gaussian generation primitives for furniture.

All generators return float32 arrays of shape (N, 14) matching MetalSplat2 compact PLY:
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


def _normals_to_quats_batch(normals: np.ndarray) -> np.ndarray:
    """Vectorised quaternion from Z-hat to each normal. normals: (N,3). Returns (N,4)."""
    normals = np.asarray(normals, dtype=np.float64)
    mag = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    n = normals / mag

    z = np.array([0.0, 0.0, 1.0])
    dot = np.clip(n @ z, -1.0, 1.0)
    angle = np.arccos(dot)

    axis = np.cross(np.tile(z, (len(n), 1)), n)
    ax_mag = np.linalg.norm(axis, axis=1, keepdims=True) + 1e-12
    axis /= ax_mag

    h = angle / 2.0
    qw = np.cos(h)
    qx = np.sin(h) * axis[:, 0]
    qy = np.sin(h) * axis[:, 1]
    qz = np.sin(h) * axis[:, 2]

    near_zero = angle < 1e-7
    near_pi   = angle > (np.pi - 1e-7)
    qw = np.where(near_zero, 1.0, np.where(near_pi, 0.0, qw))
    qx = np.where(near_zero, 0.0, np.where(near_pi, 1.0, qx))
    qy = np.where(near_zero | near_pi, 0.0, qy)
    qz = np.where(near_zero | near_pi, 0.0, qz)

    quats = np.stack([qw, qx, qy, qz], axis=1)  # float64
    quats /= (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-12)
    return quats.astype(np.float32)


def _fill_gaussian_array(
    positions: np.ndarray,
    normals: np.ndarray,
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

    dc = apply_shading(normals, positions, mat, n, rng_seed)
    data[:, IDX_FDC0:IDX_FDC0+3] = dc

    data[:, IDX_OPACITY] = rng.uniform(opacity_range[0], opacity_range[1], n)

    st = mat['scale_tangent']
    sn = mat['scale_normal']
    noise = rng.normal(0, 0.08, (n, 3))
    data[:, IDX_SCALE0] = st + noise[:, 0]
    data[:, IDX_SCALE1] = st + noise[:, 1]
    data[:, IDX_SCALE2] = sn + noise[:, 2]

    data[:, IDX_ROT0:IDX_ROT0+4] = _normals_to_quats_batch(normals)

    return data


def box_gaussians(
    center: tuple,
    half_extents: tuple,
    material: str,
    n_gaussians: int,
    rng: np.random.Generator,
    opacity_range: tuple = (4.0, 6.0),
) -> np.ndarray:
    """Surface Gaussians on all 6 faces of an axis-aligned box, area-weighted."""
    cx, cy, cz = center
    hx, hy, hz = half_extents

    faces = [
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
    normal: tuple,
    radius: float,
    material: str,
    n_gaussians: int,
    rng: np.random.Generator,
    opacity_range: tuple = (4.0, 6.0),
) -> np.ndarray:
    """Flat disc of Gaussians — circular table tops, seat cushions."""
    cx, cy, cz = center
    nrm = np.array(normal, dtype=np.float64)
    nrm /= np.linalg.norm(nrm)

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
    axis: int,
    material: str,
    n_gaussians: int,
    rng: np.random.Generator,
    opacity_range: tuple = (4.0, 6.0),
    include_caps: bool = True,
) -> np.ndarray:
    """Surface Gaussians on a cylinder. axis=1 -> vertical (Y-axis)."""
    cx, cy, cz = center
    n_side = int(n_gaussians * 0.75) if include_caps else n_gaussians
    n_caps = n_gaussians - n_side

    phi = rng.uniform(0, 2 * np.pi, n_side)
    h   = rng.uniform(-half_height, half_height, n_side)

    if axis == 1:
        px = cx + radius * np.cos(phi)
        py = cy + h
        pz = cz + radius * np.sin(phi)
        nx_, nz_ = np.cos(phi), np.sin(phi)
        ny_ = np.zeros(n_side)
    elif axis == 2:
        px = cx + radius * np.cos(phi)
        pz = cz + h
        py = cy + np.zeros(n_side)
        nx_, ny_ = np.cos(phi), np.sin(phi)
        nz_ = np.zeros(n_side)
    else:
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
