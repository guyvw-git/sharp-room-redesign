"""generate_kitchen.py — IKEA-style kitchen scene replacement for 3DGS PLY files.

Replaces existing room furniture Gaussians with a complete IKEA METOD-style kitchen
using surface-only Gaussian primitives (disc-shaped, flat along face normals).

Coordinate system: Y-DOWN
  - Floor = HIGH positive Y (~1.425)
  - Ceiling = LOW negative Y (~-1.647)
  - Left wall = negative X
  - Right wall = positive X
  - Near (camera) = low Z
  - Far (back wall) = high Z

Usage:
    python3 scripts/generate_kitchen.py [ply_in] [ply_out]
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from ply_io import read_ply, write_ply


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FLOOR_Y = 1.425      # P85 from scene analysis
CEILING_Y = -1.647   # P15 from scene analysis

# Property column indices matching scene format:
# [x, y, z, f_dc_0, f_dc_1, f_dc_2, opacity, scale_0, scale_1, scale_2,
#  rot_0, rot_1, rot_2, rot_3]
IDX_X = 0
IDX_Y = 1
IDX_Z = 2
IDX_FDC0 = 3
IDX_FDC1 = 4
IDX_FDC2 = 5
IDX_OPACITY = 6
IDX_SCALE0 = 7   # tangent axis 0
IDX_SCALE1 = 8   # tangent axis 1
IDX_SCALE2 = 9   # normal axis (thin)
IDX_ROT0 = 10    # quaternion w
IDX_ROT1 = 11    # quaternion x
IDX_ROT2 = 12    # quaternion y
IDX_ROT3 = 13    # quaternion z

N_PROPS = 14

# Surface Gaussian scale parameters (log-scale)
SCALE_TANGENT = np.log(0.008)    # ~-4.828, wide along surface
SCALE_NORMAL  = np.log(0.002)    # ~-6.215, thin perpendicular to surface

# IKEA METOD color palette (f_dc pre-sigmoid values)
COLOR_CABINET    = np.array([1.8,  1.8,  1.8 ], dtype=np.float32)   # white
COLOR_COUNTERTOP = np.array([0.8,  0.75, 0.65], dtype=np.float32)   # gray-beige
COLOR_HANDLE     = np.array([-1.0, -1.0, -1.0], dtype=np.float32)   # dark
COLOR_FRIDGE     = np.array([1.6,  1.65, 1.7 ], dtype=np.float32)   # cool white
COLOR_SINK       = np.array([0.5,  0.5,  0.55], dtype=np.float32)   # steel gray


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def axis_angle_to_quat(axis, angle):
    """Convert axis-angle to quaternion (w, x, y, z)."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    axis = axis / norm
    half = angle * 0.5
    return np.array(
        [np.cos(half),
         np.sin(half) * axis[0],
         np.sin(half) * axis[1],
         np.sin(half) * axis[2]],
        dtype=np.float32
    )


def normal_to_quat(normal):
    """Return quaternion that rotates Z-axis (0,0,1) to align with 'normal'.

    This sets the Gaussian disc so its thin (normal) dimension points along
    'normal', making it flat against the surface.
    """
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    z_hat = np.array([0.0, 0.0, 1.0])
    dot = np.clip(np.dot(z_hat, normal), -1.0, 1.0)
    angle = np.arccos(dot)
    if abs(angle) < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if abs(angle - np.pi) < 1e-6:
        # Flip: any perpendicular axis works
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    axis = np.cross(z_hat, normal)
    return axis_angle_to_quat(axis, angle)


# ---------------------------------------------------------------------------
# Surface Gaussian generators
# ---------------------------------------------------------------------------

def _face_brightness(normal, ambient=0.65):
    """Compute face brightness using three-light Lambertian shading.

    Kitchen lights are bright — ambient is high (well-lit room).
    Y-DOWN convention:
      Ambient:              0.65 (bright room)
      Key light (overhead): direction (0, -1, 0), intensity 0.25
      Fill light (front):   direction normalized(0, -0.3, -1), intensity 0.15
    """
    normal = np.asarray(normal, dtype=np.float64)
    # Key: overhead ceiling lights
    key_dir = np.array([0.0, -1.0, 0.0])
    key = max(0.0, float(np.dot(normal, key_dir))) * 0.25
    # Fill: soft frontal fill
    fill_dir = np.array([0.0, -0.3, -1.0])
    fill_dir /= np.linalg.norm(fill_dir)
    fill = max(0.0, float(np.dot(normal, fill_dir))) * 0.15
    return min(1.0, ambient + key + fill)


def _shade_dc(color_rgb, brightness):
    """Scale f_dc pre-sigmoid values by brightness factor via linear color space."""
    out = np.zeros(3, dtype=np.float32)
    for i in range(3):
        linear = 1.0 / (1.0 + np.exp(-float(color_rgb[i])))
        shaded = np.clip(linear * brightness, 1e-6, 1.0 - 1e-6)
        out[i] = np.log(shaded / (1.0 - shaded))
    return out


def surface_gaussians_box(center, half_extents, color_rgb, n_gaussians, rng,
                          opacity_range=(3.0, 5.0)):
    """Generate surface-only Gaussians on all 6 faces of an axis-aligned box.

    Parameters
    ----------
    center : array-like, shape (3,)  — (x, y, z) center of the box
    half_extents : array-like, shape (3,)  — half-widths along x, y, z
    color_rgb : array-like, shape (3,)  — f_dc pre-sigmoid color values
    n_gaussians : int  — total number of Gaussians (distributed across faces)
    rng : np.random.Generator
    opacity_range : tuple — (min, max) for uniform opacity sampling

    Returns
    -------
    np.ndarray, shape (n_gaussians, 14), dtype float32
    """
    center = np.asarray(center, dtype=np.float64)
    hx, hy, hz = half_extents

    # Define 6 faces: (normal_dir, u_axis, v_axis, face_center_offset, u_half, v_half)
    faces = [
        # +X face (right)
        (np.array([ 1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
         np.array([ hx, 0, 0]), hy, hz),
        # -X face (left)
        (np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
         np.array([-hx, 0, 0]), hy, hz),
        # +Y face (down in Y-down = bottom/floor-facing)
        (np.array([ 0, 1, 0]), np.array([1, 0, 0]), np.array([0, 0, 1]),
         np.array([0,  hy, 0]), hx, hz),
        # -Y face (up in Y-down = top/ceiling-facing)
        (np.array([ 0,-1, 0]), np.array([1, 0, 0]), np.array([0, 0, 1]),
         np.array([0, -hy, 0]), hx, hz),
        # +Z face (back)
        (np.array([ 0, 0, 1]), np.array([1, 0, 0]), np.array([0, 1, 0]),
         np.array([0, 0,  hz]), hx, hy),
        # -Z face (front)
        (np.array([ 0, 0,-1]), np.array([1, 0, 0]), np.array([0, 1, 0]),
         np.array([0, 0, -hz]), hx, hy),
    ]

    # Distribute Gaussians proportional to face area
    areas = []
    for (normal, u_ax, v_ax, offset, u_half, v_half) in faces:
        areas.append(4.0 * u_half * v_half)
    areas = np.array(areas)
    total_area = areas.sum()
    counts = np.round(areas / total_area * n_gaussians).astype(int)
    # Adjust rounding to hit exactly n_gaussians
    diff = n_gaussians - counts.sum()
    if diff > 0:
        counts[np.argmax(areas)] += diff
    elif diff < 0:
        counts[np.argmax(areas)] += diff  # subtract

    all_gaussians = []

    for i, (normal, u_ax, v_ax, offset, u_half, v_half) in enumerate(faces):
        n_face = max(1, counts[i])
        # Sample random UV on face
        u = rng.uniform(-u_half, u_half, n_face)
        v = rng.uniform(-v_half, v_half, n_face)

        # 3D positions
        positions = (center + offset)[np.newaxis, :] + \
                    u[:, np.newaxis] * u_ax[np.newaxis, :] + \
                    v[:, np.newaxis] * v_ax[np.newaxis, :]

        # Quaternion aligning Gaussian Z-axis with face normal
        q = normal_to_quat(normal)  # shape (4,) — (w, x, y, z)

        # Apply per-face Lambertian shading for 3D depth cues
        brightness = _face_brightness(normal)
        shaded_color = _shade_dc(color_rgb, brightness)

        # Build Gaussian array
        g = np.zeros((n_face, N_PROPS), dtype=np.float32)
        g[:, IDX_X]      = positions[:, 0]
        g[:, IDX_Y]      = positions[:, 1]
        g[:, IDX_Z]      = positions[:, 2]
        g[:, IDX_FDC0]   = shaded_color[0]
        g[:, IDX_FDC1]   = shaded_color[1]
        g[:, IDX_FDC2]   = shaded_color[2]
        g[:, IDX_OPACITY] = rng.uniform(opacity_range[0], opacity_range[1], n_face).astype(np.float32)
        g[:, IDX_SCALE0] = SCALE_TANGENT
        g[:, IDX_SCALE1] = SCALE_TANGENT
        g[:, IDX_SCALE2] = SCALE_NORMAL
        g[:, IDX_ROT0]   = q[0]
        g[:, IDX_ROT1]   = q[1]
        g[:, IDX_ROT2]   = q[2]
        g[:, IDX_ROT3]   = q[3]

        all_gaussians.append(g)

    return np.concatenate(all_gaussians, axis=0)


def surface_gaussians_cylinder(center, radius, height, color_rgb, n_gaussians, rng,
                                opacity_range=(3.0, 5.0)):
    """Generate surface-only Gaussians on the curved wall and caps of a Y-axis cylinder.

    The cylinder axis runs along Y (vertical in Y-down). This is used for the
    sink basin, oriented as a downward-opening bowl approximation.

    Parameters
    ----------
    center : (x, y, z) center of the cylinder
    radius : float — radius in XZ plane
    height : float — full height along Y
    color_rgb : array-like, shape (3,)
    n_gaussians : int
    rng : np.random.Generator
    """
    center = np.asarray(center, dtype=np.float64)
    half_h = height * 0.5

    # Distribute: side wall vs top/bottom caps
    side_area = 2.0 * np.pi * radius * height
    cap_area  = np.pi * radius ** 2
    total_area = side_area + 2.0 * cap_area
    n_side   = max(1, int(round(n_gaussians * side_area / total_area)))
    n_cap    = max(1, int(round(n_gaussians * cap_area / total_area)))
    n_cap_top = n_cap
    n_cap_bot = n_gaussians - n_side - n_cap_top
    if n_cap_bot < 1:
        n_cap_bot = 1

    all_gaussians = []

    # --- Curved side wall ---
    theta = rng.uniform(0, 2 * np.pi, n_side)
    y_off = rng.uniform(-half_h, half_h, n_side)

    px = center[0] + radius * np.cos(theta)
    py = center[1] + y_off
    pz = center[2] + radius * np.sin(theta)

    g_side = np.zeros((n_side, N_PROPS), dtype=np.float32)
    g_side[:, IDX_X]       = px
    g_side[:, IDX_Y]       = py
    g_side[:, IDX_Z]       = pz
    g_side[:, IDX_FDC0]    = color_rgb[0]
    g_side[:, IDX_FDC1]    = color_rgb[1]
    g_side[:, IDX_FDC2]    = color_rgb[2]
    g_side[:, IDX_OPACITY] = rng.uniform(opacity_range[0], opacity_range[1], n_side).astype(np.float32)
    g_side[:, IDX_SCALE0]  = SCALE_TANGENT
    g_side[:, IDX_SCALE1]  = SCALE_TANGENT
    g_side[:, IDX_SCALE2]  = SCALE_NORMAL
    # Per-Gaussian quaternion: outward radial normal for each point
    for j in range(n_side):
        normal_xz = np.array([np.cos(theta[j]), 0.0, np.sin(theta[j])])
        q = normal_to_quat(normal_xz)
        g_side[j, IDX_ROT0] = q[0]
        g_side[j, IDX_ROT1] = q[1]
        g_side[j, IDX_ROT2] = q[2]
        g_side[j, IDX_ROT3] = q[3]

    all_gaussians.append(g_side)

    # --- Top cap (normal = -Y, i.e. upward in Y-down) ---
    r_top = np.sqrt(rng.uniform(0, 1, n_cap_top)) * radius
    t_top = rng.uniform(0, 2 * np.pi, n_cap_top)
    g_top = np.zeros((n_cap_top, N_PROPS), dtype=np.float32)
    g_top[:, IDX_X]       = center[0] + r_top * np.cos(t_top)
    g_top[:, IDX_Y]       = center[1] - half_h
    g_top[:, IDX_Z]       = center[2] + r_top * np.sin(t_top)
    g_top[:, IDX_FDC0]    = color_rgb[0]
    g_top[:, IDX_FDC1]    = color_rgb[1]
    g_top[:, IDX_FDC2]    = color_rgb[2]
    g_top[:, IDX_OPACITY] = rng.uniform(opacity_range[0], opacity_range[1], n_cap_top).astype(np.float32)
    g_top[:, IDX_SCALE0]  = SCALE_TANGENT
    g_top[:, IDX_SCALE1]  = SCALE_TANGENT
    g_top[:, IDX_SCALE2]  = SCALE_NORMAL
    q_top = normal_to_quat(np.array([0.0, -1.0, 0.0]))
    g_top[:, IDX_ROT0]    = q_top[0]
    g_top[:, IDX_ROT1]    = q_top[1]
    g_top[:, IDX_ROT2]    = q_top[2]
    g_top[:, IDX_ROT3]    = q_top[3]
    all_gaussians.append(g_top)

    # --- Bottom cap (normal = +Y, facing floor in Y-down) ---
    if n_cap_bot > 0:
        r_bot = np.sqrt(rng.uniform(0, 1, n_cap_bot)) * radius
        t_bot = rng.uniform(0, 2 * np.pi, n_cap_bot)
        g_bot = np.zeros((n_cap_bot, N_PROPS), dtype=np.float32)
        g_bot[:, IDX_X]       = center[0] + r_bot * np.cos(t_bot)
        g_bot[:, IDX_Y]       = center[1] + half_h
        g_bot[:, IDX_Z]       = center[2] + r_bot * np.sin(t_bot)
        g_bot[:, IDX_FDC0]    = color_rgb[0]
        g_bot[:, IDX_FDC1]    = color_rgb[1]
        g_bot[:, IDX_FDC2]    = color_rgb[2]
        g_bot[:, IDX_OPACITY] = rng.uniform(opacity_range[0], opacity_range[1], n_cap_bot).astype(np.float32)
        g_bot[:, IDX_SCALE0]  = SCALE_TANGENT
        g_bot[:, IDX_SCALE1]  = SCALE_TANGENT
        g_bot[:, IDX_SCALE2]  = SCALE_NORMAL
        q_bot = normal_to_quat(np.array([0.0, 1.0, 0.0]))
        g_bot[:, IDX_ROT0]    = q_bot[0]
        g_bot[:, IDX_ROT1]    = q_bot[1]
        g_bot[:, IDX_ROT2]    = q_bot[2]
        g_bot[:, IDX_ROT3]    = q_bot[3]
        all_gaussians.append(g_bot)

    return np.concatenate(all_gaussians, axis=0)


# ---------------------------------------------------------------------------
# Kitchen element builders
# ---------------------------------------------------------------------------

def make_base_cabinets(n_per_element, rng):
    """Back wall base cabinet run: X -2.5 to 1.5, Z 11.1 to 11.9."""
    cx = (-2.5 + 1.5) / 2.0          # -0.5
    cy = FLOOR_Y - 0.6                # mid-height of cabinet
    cz = 11.5
    hx = (1.5 - (-2.5)) / 2.0        # 2.0
    hy = 0.6                          # half-height = 1.2 / 2
    hz = 0.4                          # half-depth = 0.8 / 2
    return surface_gaussians_box(
        center=(cx, cy, cz),
        half_extents=(hx, hy, hz),
        color_rgb=COLOR_CABINET,
        n_gaussians=n_per_element,
        rng=rng
    )


def make_upper_cabinets(n_per_element, rng):
    """Back wall upper cabinet run: X -2.5 to 1.5, mounted high."""
    cx = (-2.5 + 1.5) / 2.0          # -0.5
    # upper cabinets Y: floor_y - 1.8 to floor_y - 2.8  → center = floor_y - 2.3
    cy = FLOOR_Y - 2.3
    cz = 11.2
    hx = 2.0
    hy = 0.5                          # half-height = 1.0 / 2
    hz = 0.3
    return surface_gaussians_box(
        center=(cx, cy, cz),
        half_extents=(hx, hy, hz),
        color_rgb=COLOR_CABINET,
        n_gaussians=n_per_element,
        rng=rng
    )


def make_countertop(n_per_element, rng):
    """Thin slab countertop at floor_y - 1.2, spanning full base cabinet run."""
    cx = (-2.5 + 1.5) / 2.0          # -0.5
    cy = FLOOR_Y - 1.2                # top surface
    cz = (11.0 + 11.6) / 2.0         # 11.3
    hx = 2.0
    hy = 0.03                         # very thin slab
    hz = 0.3
    return surface_gaussians_box(
        center=(cx, cy, cz),
        half_extents=(hx, hy, hz),
        color_rgb=COLOR_COUNTERTOP,
        n_gaussians=n_per_element,
        rng=rng
    )


def make_side_cabinets(n_per_element, rng):
    """Left wall cabinet run: X ~-2.6, Z 4.0 to 10.0."""
    cx = -2.6
    cy = FLOOR_Y - 0.6
    cz = (4.0 + 10.0) / 2.0          # 7.0
    hx = 0.3                          # depth from wall
    hy = 0.6
    hz = (10.0 - 4.0) / 2.0          # 3.0
    base = surface_gaussians_box(
        center=(cx, cy, cz),
        half_extents=(hx, hy, hz),
        color_rgb=COLOR_CABINET,
        n_gaussians=n_per_element,
        rng=rng
    )

    # Side upper cabinets
    cy_upper = FLOOR_Y - 2.3
    upper = surface_gaussians_box(
        center=(cx, cy_upper, cz),
        half_extents=(hx, 0.5, hz),
        color_rgb=COLOR_CABINET,
        n_gaussians=n_per_element // 2,
        rng=rng
    )
    return np.concatenate([base, upper], axis=0)


def make_fridge(n_per_element, rng):
    """Standalone fridge: X 1.5 to 2.0, Z 10.5 to 11.5, Y full height."""
    cx = (1.5 + 2.0) / 2.0           # 1.75
    # fridge height: floor_y to floor_y - 2.4 → center = floor_y - 1.2
    cy = FLOOR_Y - 1.2
    cz = (10.5 + 11.5) / 2.0         # 11.0
    hx = 0.25
    hy = 1.2
    hz = 0.5
    return surface_gaussians_box(
        center=(cx, cy, cz),
        half_extents=(hx, hy, hz),
        color_rgb=COLOR_FRIDGE,
        n_gaussians=n_per_element,
        rng=rng
    )


def make_sink(n_per_element, rng):
    """Sink basin inset at countertop level, centered at X=0.0, back wall."""
    center = (0.0, FLOOR_Y - 1.25, 11.2)
    radius = 0.22
    height = 0.18
    return surface_gaussians_cylinder(
        center=center,
        radius=radius,
        height=height,
        color_rgb=COLOR_SINK,
        n_gaussians=n_per_element,
        rng=rng
    )


def make_island(n_per_element, rng):
    """Optional kitchen island near center of room."""
    center = (-0.5, FLOOR_Y - 0.45, 7.0)
    hx = 0.5    # half of 1.0
    hy = 0.45   # half of 0.9 (base + slight countertop)
    hz = 0.9    # half of 1.8
    body = surface_gaussians_box(
        center=center,
        half_extents=(hx, hy, hz),
        color_rgb=COLOR_CABINET,
        n_gaussians=n_per_element,
        rng=rng
    )
    # Island countertop
    top_center = (-0.5, FLOOR_Y - 0.9 + 0.02, 7.0)
    top = surface_gaussians_box(
        center=top_center,
        half_extents=(hx + 0.05, 0.025, hz + 0.05),
        color_rgb=COLOR_COUNTERTOP,
        n_gaussians=n_per_element // 4,
        rng=rng
    )
    return np.concatenate([body, top], axis=0)


def make_handles(n_per_element, rng):
    """Dark bar handles on base cabinet fronts (back wall, Z ~ 11.1)."""
    # Place handles as thin boxes along the front face of base cabinets
    handle_y = FLOOR_Y - 0.9   # mid-height of base cabinet
    handle_z = 11.1 - 0.01     # just in front of cabinet face
    handles = []
    # 4 equally-spaced handles along X: -2.0, -1.0, 0.0, 1.0
    for hx_center in [-2.0, -1.0, 0.0, 1.0]:
        h = surface_gaussians_box(
            center=(hx_center, handle_y, handle_z),
            half_extents=(0.06, 0.015, 0.015),
            color_rgb=COLOR_HANDLE,
            n_gaussians=max(50, n_per_element // 40),
            rng=rng
        )
        handles.append(h)
    # Upper cabinet handles
    handle_y_upper = FLOOR_Y - 1.85
    for hx_center in [-2.0, -1.0, 0.0, 1.0]:
        h = surface_gaussians_box(
            center=(hx_center, handle_y_upper, 10.92),
            half_extents=(0.06, 0.015, 0.015),
            color_rgb=COLOR_HANDLE,
            n_gaussians=max(50, n_per_element // 40),
            rng=rng
        )
        handles.append(h)
    return np.concatenate(handles, axis=0)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_kitchen(ply_in, ply_out, n_per_element=80000):
    """Generate IKEA-style kitchen, replacing furniture in the existing PLY.

    Parameters
    ----------
    ply_in : str   — path to input PLY
    ply_out : str  — path to output PLY
    n_per_element : int — Gaussians per kitchen element (reduce for speed tests)
    """
    print(f"[generate_kitchen] Loading {ply_in} ...")
    data, header_bytes, props, tail_bytes = read_ply(ply_in)
    print(f"  Loaded {len(data):,} Gaussians, {len(props)} properties")

    # ------------------------------------------------------------------
    # 1. Remove furniture Gaussians in the mid-scene furniture zone
    # ------------------------------------------------------------------
    print("[generate_kitchen] Masking furniture Gaussians ...")

    x = data[:, IDX_X]
    y = data[:, IDX_Y]
    z = data[:, IDX_Z]

    x_min = x.min()
    x_max = x.max()

    furniture_mask = (
        (y >= FLOOR_Y - 1.8) & (y <= FLOOR_Y - 0.2) &  # furniture height zone
        (x >= x_min + 0.3) & (x <= x_max - 0.3) &       # exclude extreme wall Gaussians
        (z >= 3.0) & (z <= 12.0)                          # furniture depth zone
    )

    n_removed = furniture_mask.sum()
    print(f"  Setting opacity=-10 for {n_removed:,} furniture Gaussians")
    data[furniture_mask, IDX_OPACITY] = -10.0

    # ------------------------------------------------------------------
    # 2. Generate all kitchen elements
    # ------------------------------------------------------------------
    print("[generate_kitchen] Generating kitchen elements ...")
    rng = np.random.default_rng(42)

    elements = []

    print("  - Base cabinets (back wall) ...")
    elements.append(make_base_cabinets(n_per_element, rng))

    print("  - Upper cabinets (back wall) ...")
    elements.append(make_upper_cabinets(n_per_element, rng))

    print("  - Countertop ...")
    elements.append(make_countertop(n_per_element // 2, rng))

    print("  - Side cabinets (left wall) ...")
    elements.append(make_side_cabinets(n_per_element, rng))

    print("  - Fridge ...")
    elements.append(make_fridge(n_per_element // 2, rng))

    print("  - Sink basin ...")
    elements.append(make_sink(max(500, n_per_element // 8), rng))

    print("  - Island ...")
    elements.append(make_island(n_per_element // 2, rng))

    print("  - Cabinet handles ...")
    elements.append(make_handles(n_per_element, rng))

    kitchen_data = np.concatenate(elements, axis=0)
    print(f"  Generated {len(kitchen_data):,} kitchen Gaussians")

    # ------------------------------------------------------------------
    # 3. Concatenate modified scene + kitchen elements
    # ------------------------------------------------------------------
    combined = np.concatenate([data, kitchen_data], axis=0)
    print(f"[generate_kitchen] Total Gaussians: {len(combined):,}")

    # ------------------------------------------------------------------
    # 4. Write output PLY
    # ------------------------------------------------------------------
    print(f"[generate_kitchen] Writing {ply_out} ...")
    write_ply(ply_out, combined, header_bytes, tail_bytes)
    print(f"[generate_kitchen] Done. Output: {ply_out}")

    return {
        'n_input': len(data),
        'n_removed': int(n_removed),
        'n_kitchen': len(kitchen_data),
        'n_total': len(combined),
        'ply_out': ply_out,
    }


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    ply_in  = sys.argv[1] if len(sys.argv) > 1 else 'input/current.ply'
    ply_out = sys.argv[2] if len(sys.argv) > 2 else 'output/test4_kitchen.ply'

    # Quick speed test with reduced Gaussian count
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 80000

    result = generate_kitchen(ply_in, ply_out, n_per_element=n)
    print("\nSummary:")
    for k, v in result.items():
        print(f"  {k}: {v}")
