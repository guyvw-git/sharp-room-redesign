#!/usr/bin/env python3
"""
Apply a delta spec JSON to a PLY file.
Implements the ply-editor.md logic.

Usage:
  python3 scripts/apply_delta.py <ply_in> <delta_spec.json> <ply_out> [segments.json]
"""
import sys
import json
import os
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from ply_io import read_ply, write_ply

ROOT = Path(__file__).parent.parent

PALETTES = {
    "tile_white":    [0.8,  0.8,  0.8],
    "tile_gray":     [0.5,  0.5,  0.5],
    "brick_red":     [0.6,  0.2,  0.1],
    "brick_gray":    [0.45, 0.42, 0.4],
    "wood_oak":      [0.55, 0.35, 0.15],
    "wood_dark":     [0.25, 0.15, 0.08],
    "concrete":      [0.4,  0.4,  0.38],
    "plaster_white": [0.88, 0.87, 0.85],
    "red":           [0.8,  0.05, 0.05],
    "bean_bag_red":  [0.75, 0.08, 0.08],
}


def apply_delta(ply_in, spec_path, ply_out, segments_path=None):
    # Step 1: Backup
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = ROOT / '.claude/backups'
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f'current_{ts}.ply'
    import shutil
    shutil.copy2(ply_in, backup_path)
    print(f"Backup: {backup_path}")

    # Step 2: Load spec
    with open(spec_path) as f:
        spec = json.load(f)
    print(f"Strategy: {spec.get('strategy')}")
    print(f"Target: {spec.get('target_indices', 'ALL')}")
    print(f"Operations: {len(spec.get('operations', []))}")

    # Step 3: Load PLY
    data, header_bytes, props, tail_bytes = read_ply(ply_in)
    n_verts, n_props = data.shape
    prop_idx = {p: i for i, p in enumerate(props)}
    print(f"Loaded {n_verts:,} Gaussians, {n_props} properties")

    # Load segments
    segments = {}
    seg_file = segments_path or str(ROOT / '.claude/segments/latest_segments.json')
    if os.path.exists(seg_file):
        with open(seg_file) as f:
            segments = json.load(f)

    # Resolve target indices
    target_str = spec.get('target_indices', 'ALL')

    if target_str == 'ALL':
        target_idx = np.arange(n_verts)

    elif target_str.startswith('from_segments:'):
        region = target_str.split(':', 1)[1]
        if region not in segments:
            print(f"ERROR: region '{region}' not found in segments (found: {list(segments.keys())})")
            sys.exit(1)
        target_idx = np.array(segments[region]['indices'], dtype=np.int64)

    elif target_str == 'spatial_outliers':
        from scipy.spatial import KDTree
        params = spec.get('outlier_params', {})
        k = params.get('k_neighbors', 10)
        threshold_pct = params.get('density_threshold_percentile', 2)
        xi = prop_idx.get('x', 0)
        yi = prop_idx.get('y', 1)
        zi = prop_idx.get('z', 2)
        xyz = data[:, [xi, yi, zi]]
        tree = KDTree(xyz)
        dists, _ = tree.query(xyz, k=k + 1)
        avg_dist = dists[:, 1:].mean(axis=1)
        threshold = np.percentile(avg_dist, 100 - threshold_pct)
        target_idx = np.where(avg_dist > threshold)[0]
        print(f"Outlier detection: {len(target_idx):,} Gaussians flagged (threshold={threshold:.4f})")

    else:
        print(f"ERROR: unknown target_indices format: '{target_str}'")
        sys.exit(1)

    print(f"Editing {len(target_idx):,} Gaussians ({100*len(target_idx)/n_verts:.1f}% of total)")

    # Apply operations
    warnings = []
    for op in spec.get('operations', []):
        prop_name = op['property']
        operation = op['op']

        # Handle wildcard f_rest_*
        if prop_name.endswith('*'):
            prefix = prop_name[:-1]
            affected = [p for p in props if p.startswith(prefix)]
        else:
            affected = [prop_name] if prop_name in prop_idx else []

        if not affected:
            w = f"property '{prop_name}' not found in PLY, skipping"
            print(f"WARNING: {w}")
            warnings.append(w)
            continue

        for p in affected:
            col = prop_idx[p]

            if operation == 'set':
                data[target_idx, col] = float(op['value'])

            elif operation == 'set_from_palette':
                palette_name = op['palette']
                if palette_name not in PALETTES:
                    w = f"palette '{palette_name}' not found"
                    print(f"WARNING: {w}")
                    warnings.append(w)
                    continue
                blend = float(op.get('blend', 0.85))
                dc_map = {'f_dc_0': 0, 'f_dc_1': 1, 'f_dc_2': 2}
                if p in dc_map:
                    target_val = PALETTES[palette_name][dc_map[p]]
                    original = data[target_idx, col]
                    data[target_idx, col] = original * (1 - blend) + target_val * blend

            elif operation == 'scale':
                data[target_idx, col] *= float(op['factor'])

            elif operation == 'multiply':
                channel_map = {'f_dc_0': 'r_factor', 'f_dc_1': 'g_factor', 'f_dc_2': 'b_factor'}
                if p in channel_map:
                    factor = float(op.get(channel_map[p], 1.0))
                    data[target_idx, col] *= factor

            elif operation == 'clamp':
                lo = float(op.get('min', -float('inf')))
                hi = float(op.get('max', float('inf')))
                data[target_idx, col] = np.clip(data[target_idx, col], lo, hi)

            else:
                w = f"unknown operation '{operation}'"
                print(f"WARNING: {w}")
                warnings.append(w)

        preview = affected[:3]
        suffix = '...' if len(affected) > 3 else ''
        print(f"  Applied '{operation}' to {len(affected)} prop(s): {preview}{suffix}")

    # Write output
    os.makedirs(os.path.dirname(ply_out) or '.', exist_ok=True)
    write_ply(ply_out, data, header_bytes, tail_bytes)

    in_size = os.path.getsize(ply_in)
    out_size = os.path.getsize(ply_out)
    size_match = "PASS" if in_size == out_size else "NOTE (sizes differ)"
    print(f"\nWritten: {ply_out}")
    print(f"Input size:  {in_size/1024/1024:.1f} MB")
    print(f"Output size: {out_size/1024/1024:.1f} MB")
    print(f"Size check: {size_match}")
    print(f"Gaussians modified: {len(target_idx):,} of {n_verts:,}")
    if warnings:
        print(f"Warnings: {warnings}")

    return {
        'status': 'PLY_EDIT_DONE',
        'output_path': ply_out,
        'gaussians_modified': len(target_idx),
        'gaussians_total': n_verts,
        'backup_path': str(backup_path),
        'size_check': size_match,
        'warnings': warnings
    }


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: apply_delta.py <ply_in> <spec.json> <ply_out> [segments.json]")
        sys.exit(1)
    ply_in = sys.argv[1]
    spec = sys.argv[2]
    ply_out = sys.argv[3]
    segs = sys.argv[4] if len(sys.argv) > 4 else None
    result = apply_delta(ply_in, spec, ply_out, segs)
    print(f"\n{result['status']}")
