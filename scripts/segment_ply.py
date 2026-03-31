#!/usr/bin/env python3
"""
Segment Gaussians in a PLY file into spatial regions.
Outputs latest_segments.json with region → index lists.

Usage:
  python3 scripts/segment_ply.py [ply_path] [target_region]

Target region options:
  floor, ceiling, walls, floaters, furniture, chair, all
"""
import sys
import json
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ply_io import read_positions

ROOT = Path(__file__).parent.parent
SEG_OUT = ROOT / '.claude/segments/latest_segments.json'


def detect_floaters(xyz, k_neighbors=10, threshold_pct=98):
    """Detect spatially isolated Gaussians (floaters) using KD-tree density."""
    from scipy.spatial import KDTree
    tree = KDTree(xyz)
    dists, _ = tree.query(xyz, k=k_neighbors + 1)
    avg_dist = dists[:, 1:].mean(axis=1)
    threshold = np.percentile(avg_dist, threshold_pct)
    return np.where(avg_dist > threshold)[0]


def segment_scene(ply_path, target_hint=None):
    xyz, props, n_verts = read_positions(ply_path)
    print(f"Loaded {n_verts:,} Gaussians")

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    print(f"X range: {x.min():.3f} to {x.max():.3f}")
    print(f"Y range: {y.min():.3f} to {y.max():.3f}")
    print(f"Z range: {z.min():.3f} to {z.max():.3f}")

    # In 3DGS, Y is typically the vertical axis (up = negative Y in many scenes)
    # Floor = lowest percentile Y values
    y_p05 = np.percentile(y, 5)
    y_p10 = np.percentile(y, 10)
    y_p20 = np.percentile(y, 20)
    y_p80 = np.percentile(y, 80)
    y_p90 = np.percentile(y, 90)
    y_p95 = np.percentile(y, 95)

    segments = {}

    # Floor: bottom 15% of Y — but cap at density to avoid capturing scene center
    floor_thresh = np.percentile(y, 15)
    # Also spatial constraint: floor should be roughly centered in X, Z
    floor_mask = (y <= floor_thresh)
    floor_indices = np.where(floor_mask)[0].tolist()
    segments['floor'] = {
        'indices': floor_indices,
        'count': len(floor_indices),
        'method': 'percentile_y_bottom_15',
        'y_range': [float(y[floor_mask].min()), float(y[floor_mask].max())],
        'confidence': 'medium'
    }

    # Ceiling: top 10% of Y
    ceiling_thresh = np.percentile(y, 90)
    ceiling_mask = (y >= ceiling_thresh)
    ceiling_indices = np.where(ceiling_mask)[0].tolist()
    segments['ceiling'] = {
        'indices': ceiling_indices,
        'count': len(ceiling_indices),
        'method': 'percentile_y_top_10',
        'y_range': [float(y[ceiling_mask].min()), float(y[ceiling_mask].max())],
        'confidence': 'medium'
    }

    # Walls: extreme X or Z values (outermost 10%)
    x_lo = np.percentile(x, 8)
    x_hi = np.percentile(x, 92)
    z_lo = np.percentile(z, 8)
    z_hi = np.percentile(z, 92)
    wall_mask = (x <= x_lo) | (x >= x_hi) | (z <= z_lo) | (z >= z_hi)
    # Exclude floor and ceiling from walls
    wall_mask = wall_mask & ~floor_mask & ~ceiling_mask
    wall_indices = np.where(wall_mask)[0].tolist()
    segments['walls'] = {
        'indices': wall_indices,
        'count': len(wall_indices),
        'method': 'spatial_perimeter',
        'confidence': 'low'
    }

    # Middle region: furniture / objects (center of scene, middle Y range)
    mid_y_lo = np.percentile(y, 25)
    mid_y_hi = np.percentile(y, 75)
    mid_x_lo = np.percentile(x, 20)
    mid_x_hi = np.percentile(x, 80)
    mid_z_lo = np.percentile(z, 20)
    mid_z_hi = np.percentile(z, 80)
    furniture_mask = (
        (y >= mid_y_lo) & (y <= mid_y_hi) &
        (x >= mid_x_lo) & (x <= mid_x_hi) &
        (z >= mid_z_lo) & (z <= mid_z_hi)
    )
    furniture_indices = np.where(furniture_mask)[0].tolist()
    segments['furniture'] = {
        'indices': furniture_indices,
        'count': len(furniture_indices),
        'method': 'spatial_center_band',
        'confidence': 'low',
        'note': 'broad region — includes all objects in scene center'
    }

    # Floaters: spatially isolated Gaussians
    print("Detecting floaters...")
    floater_indices = detect_floaters(xyz, k_neighbors=10, threshold_pct=98).tolist()
    segments['floaters'] = {
        'indices': floater_indices,
        'count': len(floater_indices),
        'method': 'local_density_knn10_p98',
        'confidence': 'high',
        'note': 'Gaussians with avg KNN distance above 98th percentile'
    }

    # Chair detection: look for a discrete cluster in the furniture region
    # Use DBSCAN clustering on the middle-Y, middle-XZ region
    if target_hint in ('chair', 'bean_bag', 'furniture'):
        print("Running DBSCAN to find chair cluster...")
        from sklearn.cluster import DBSCAN
        mid_pts = xyz[furniture_mask]
        mid_orig_idx = np.where(furniture_mask)[0]

        # Normalize to unit cube for clustering
        if len(mid_pts) > 1000:
            # Downsample for speed
            sample_idx = np.random.choice(len(mid_pts), min(50000, len(mid_pts)), replace=False)
            sample_pts = mid_pts[sample_idx]
            sample_orig = mid_orig_idx[sample_idx]
        else:
            sample_pts = mid_pts
            sample_orig = mid_orig_idx

        pts_norm = (sample_pts - sample_pts.mean(0)) / (sample_pts.std(0) + 1e-6)
        db = DBSCAN(eps=0.5, min_samples=20, n_jobs=-1).fit(pts_norm)
        labels = db.labels_

        # Find largest cluster(s) that aren't noise
        unique_labels = set(labels) - {-1}
        if unique_labels:
            # Sort clusters by size
            cluster_sizes = [(l, (labels == l).sum()) for l in unique_labels]
            cluster_sizes.sort(key=lambda x: x[1], reverse=True)

            # The chair is likely a mid-sized cluster, not the floor-touching mass
            # Find clusters with centroid above floor level
            cluster_info = []
            for label, size in cluster_sizes[:10]:
                mask_c = labels == label
                centroid = sample_pts[mask_c].mean(0)
                cluster_info.append({
                    'label': int(label),
                    'size': int(size),
                    'centroid': centroid.tolist(),
                    'y_centroid': float(centroid[1])
                })

            print(f"Found {len(cluster_info)} clusters (top 10 shown)")
            for ci in cluster_info[:5]:
                print(f"  Cluster {ci['label']}: {ci['size']} pts, centroid_y={ci['y_centroid']:.3f}")

            # Heuristic: chair is a cluster with centroid in middle Y range and reasonable size
            y_mid = float(np.median(y))
            chair_candidates = [
                ci for ci in cluster_info
                if ci['size'] > 100
                and abs(ci['y_centroid'] - y_mid) < 0.5 * (y.max() - y.min())
            ]

            if chair_candidates:
                # Take the largest chair candidate
                best = chair_candidates[0]
                chair_mask_sample = labels == best['label']
                chair_orig_idx = sample_orig[chair_mask_sample].tolist()

                # Extend to full PLY using proximity to cluster centroid
                centroid = np.array(best['centroid'])
                chair_pts = sample_pts[chair_mask_sample]
                radius = np.percentile(np.linalg.norm(chair_pts - centroid, axis=1), 90)
                radius *= 1.3  # expand a bit

                # Find all Gaussians within this radius
                full_dists = np.linalg.norm(xyz - centroid, axis=1)
                all_chair_idx = np.where(full_dists <= radius)[0].tolist()

                segments['chair'] = {
                    'indices': all_chair_idx,
                    'count': len(all_chair_idx),
                    'method': 'dbscan_cluster',
                    'centroid': centroid.tolist(),
                    'radius': float(radius),
                    'confidence': 'medium',
                    'bbox': {
                        'x_min': float(xyz[all_chair_idx, 0].min()),
                        'x_max': float(xyz[all_chair_idx, 0].max()),
                        'y_min': float(xyz[all_chair_idx, 1].min()),
                        'y_max': float(xyz[all_chair_idx, 1].max()),
                        'z_min': float(xyz[all_chair_idx, 2].min()),
                        'z_max': float(xyz[all_chair_idx, 2].max()),
                    }
                }
            else:
                print("WARNING: no clear chair cluster found — using furniture region")
                segments['chair'] = segments['furniture'].copy()
                segments['chair']['note'] = 'fallback: used furniture region — no distinct chair cluster detected'
        else:
            print("WARNING: DBSCAN found no clusters in furniture region")
            segments['chair'] = segments['furniture'].copy()

    SEG_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SEG_OUT, 'w') as f:
        json.dump(segments, f)

    print(f"\nSegments saved: {SEG_OUT}")
    for name, seg in segments.items():
        print(f"  {name}: {seg['count']:,} Gaussians ({seg['confidence']} confidence)")

    return segments


if __name__ == '__main__':
    ply = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / 'input/current.ply')
    hint = sys.argv[2] if len(sys.argv) > 2 else None
    segment_scene(ply, target_hint=hint)
