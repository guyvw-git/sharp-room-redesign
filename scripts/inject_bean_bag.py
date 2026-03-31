#!/usr/bin/env python3
"""
Test 3: Replace chair with a red bean bag chair.

Uses surface_gaussians_sphere from generate_surface_ply.py to produce
proper disc-shaped, surface-coherent Gaussians — not random interior blobs.

Usage:
  python3 scripts/inject_bean_bag.py [ply_in] [ply_out] [segments.json]
"""
import sys, json, os, shutil
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from ply_io import read_ply, write_ply
from generate_surface_ply import surface_gaussians_sphere

ROOT = Path(__file__).parent.parent

# Vivid red in SH DC space (pre-sigmoid):
#   sigmoid(2.2) ≈ 0.90 → bright red channel
#   sigmoid(-2.2) ≈ 0.10 → dark green/blue channels
RED_DC = [2.2, -2.2, -2.2]


def apply_lambertian_shading(bean_data, center, radii):
    """
    Apply fake Lambertian shading to bean bag Gaussians for 3D depth cues.

    Light source is above in Y-DOWN convention (light_dir = (0,-1,0)).
    Top normals (-Y) face the light → bright. Side/bottom → darker.
    Also adds a subtle fill light from the front (+Z direction).
    """
    cx, cy, cz = center
    rx, ry, rz = radii

    # Positions relative to center
    px = bean_data[:, 0] - cx
    py = bean_data[:, 1] - cy
    pz = bean_data[:, 2] - cz

    # Outward normal of ellipsoid surface (unnormalized)
    nx = px / (rx * rx)
    ny = py / (ry * ry)
    nz = pz / (rz * rz)
    mag = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-9
    nx /= mag; ny /= mag; nz /= mag

    # Y-down: light from above = (0, -1, 0)
    # diffuse = max(0, dot(n, light)) = max(0, -ny)
    diffuse_top   = np.clip(-ny, 0, 1)

    # Fill light from front-right to reveal silhouette: direction (-0.5, -0.3, -0.8)
    fill_dir = np.array([-0.5, -0.3, -0.8])
    fill_dir /= np.linalg.norm(fill_dir)
    diffuse_fill  = np.clip(nx * fill_dir[0] + ny * fill_dir[1] + nz * fill_dir[2], 0, 1)

    # Combine: strong key light from above, soft fill from front
    brightness = 0.30 + 0.55 * diffuse_top + 0.15 * diffuse_fill

    # Apply to f_dc_0/1/2 (columns 3,4,5)
    # Shift pre-sigmoid values by log(brightness) to scale apparent color
    # Since sigmoid is nonlinear, approximate: multiply by brightness in linear space
    # then convert back via inverse sigmoid
    for ch in range(3):
        col = 3 + ch
        # Convert pre-sigmoid → linear color, scale, back to pre-sigmoid
        linear = 1.0 / (1.0 + np.exp(-bean_data[:, col].astype(np.float64)))
        linear_shaded = np.clip(linear * brightness, 1e-6, 1 - 1e-6)
        bean_data[:, col] = np.log(linear_shaded / (1.0 - linear_shaded)).astype(np.float32)

    return bean_data


def inject_bean_bag(ply_in, ply_out, segments_path=None, n_gaussians=500_000, seed=42):
    """
    Remove chair Gaussians, inject a high-quality surface bean bag with Lambertian shading.

    n_gaussians: 500K gives dense coverage for a large bean bag object.
    """
    rng = np.random.default_rng(seed)

    # Backup
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = ROOT / f'.claude/backups/current_{ts}_test3.ply'
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ply_in, backup)
    print(f"Backup: {backup}")

    # Load segments
    seg_file = segments_path or str(ROOT / '.claude/segments/latest_segments.json')
    with open(seg_file) as f:
        segs = json.load(f)

    if 'chair' not in segs:
        print("ERROR: 'chair' segment not found")
        sys.exit(1)

    chair_seg = segs['chair']
    chair_indices = np.array(chair_seg['indices'], dtype=np.int64)
    centroid = chair_seg['centroid']   # already floor-adjusted
    scale_override = chair_seg.get('scale_override', [1.0, 0.85, 1.0])
    floor_y = chair_seg.get('floor_y', 1.425)

    print(f"Chair indices: {len(chair_indices):,}")
    print(f"Bean bag centroid: {centroid}")
    print(f"Scale override: {scale_override}")

    # Load PLY
    data, header_bytes, props, tail_bytes = read_ply(ply_in)
    n_verts, n_props = data.shape
    pi = {p: i for i, p in enumerate(props)}
    print(f"Loaded {n_verts:,} Gaussians, {n_props} properties")

    # Step 1: Zero out chair opacity
    opacity_col = pi['opacity']
    data[chair_indices, opacity_col] = -10.0
    print(f"Removed {len(chair_indices):,} chair Gaussians (opacity → -10)")

    # Step 2: Compute bean bag radii
    # Bean bag is a large rounded cushion, sits on floor
    # Use larger scale factors than before so it's clearly visible and chair-sized
    rx = scale_override[0] * 0.62   # horizontal width radius (~0.75m for scale 1.2)
    ry = scale_override[1] * 0.55   # vertical radius — tall enough to look 3D
    rz = scale_override[2] * 0.62   # horizontal depth radius

    # Bean bag centroid Y: place so bottom of bag touches floor (Y-down)
    bb_cx = centroid[0]
    bb_cy = floor_y - ry   # sit on floor
    bb_cz = centroid[2]

    print(f"\nBean bag parameters:")
    print(f"  Center: ({bb_cx:.3f}, {bb_cy:.3f}, {bb_cz:.3f})")
    print(f"  Radii: rx={rx:.3f}, ry={ry:.3f}, rz={rz:.3f}")
    print(f"  Floor Y: {floor_y:.3f}, bottom of bag at Y={bb_cy+ry:.3f}")
    print(f"  Generating {n_gaussians:,} surface Gaussians...")

    # Step 3: Generate proper surface bean bag
    bean_data = surface_gaussians_sphere(
        center=(bb_cx, bb_cy, bb_cz),
        radii=(rx, ry, rz),
        color_dc=RED_DC,
        n_gaussians=n_gaussians,
        rng=rng,
        scale_tangent=-3.8,   # exp(-3.8)≈0.022m — larger splats for smoother surface
        scale_normal=-6.5,    # exp(-6.5)≈0.0015m — very thin disc
        opacity_range=(4.0, 6.0),  # high opacity for solid appearance
    )

    # Step 4: Apply Lambertian shading for 3D depth cues
    print("  Applying Lambertian shading (key + fill lights)...")
    bean_data = apply_lambertian_shading(bean_data, (bb_cx, bb_cy, bb_cz), (rx, ry, rz))

    # Step 5: Append to scene data
    new_data = np.concatenate([data, bean_data.astype(np.float32)], axis=0)
    new_n = len(new_data)

    # Step 5: Write output (write_ply updates vertex count in header)
    os.makedirs(os.path.dirname(os.path.abspath(ply_out)), exist_ok=True)
    write_ply(ply_out, new_data, header_bytes, tail_bytes)

    out_mb = os.path.getsize(ply_out) / 1024 / 1024
    print(f"\nPLY_INJECT_DONE")
    print(f"  Output: {ply_out} ({out_mb:.1f} MB)")
    print(f"  Original:    {n_verts:,} Gaussians")
    print(f"  Chair removed: {len(chair_indices):,}")
    print(f"  Bean bag added: {len(bean_data):,}")
    print(f"  Total: {new_n:,}")


if __name__ == '__main__':
    ply_in  = sys.argv[1] if len(sys.argv) > 1 else 'input/current.ply'
    ply_out = sys.argv[2] if len(sys.argv) > 2 else 'output/test3_red_beanbag.ply'
    segs    = sys.argv[3] if len(sys.argv) > 3 else None
    inject_bean_bag(ply_in, ply_out, segs)
