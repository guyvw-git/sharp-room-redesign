#!/usr/bin/env python3
"""
Main entry point for room redesign pipeline.

Usage:
  python3 scripts/redesign.py "Turn this space into a movie room"
  python3 scripts/redesign.py "Turn this space into a movie room" --ply input/current.ply --out output/movie_room.ply
  python3 scripts/redesign.py "..." --plan .claude/edits/plan_movie_room.json  # skip Claude call
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

    # 1. Backup
    backup_dir = ROOT / '.claude/backups'
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ply_in, backup_dir / f'current_{ts}_preredesign.ply')
    print(f"[redesign] Backup: {backup_dir}/current_{ts}_preredesign.ply")

    # 2. Analyze room
    print(f"[redesign] Analyzing room geometry from {ply_in} ...")
    geom = analyze_room(ply_in)
    print(f"[redesign] Room: {geom.room_width_m:.2f}m W x {geom.room_depth_m:.2f}m D x {geom.room_height_m:.2f}m H")
    print(f"           Floor Y={geom.floor_y:.3f}  Ceiling Y={geom.ceiling_y:.3f}")
    print(f"           N wall Z={geom.north_z:.3f}  S wall Z={geom.south_z:.3f}")

    # 3. Generate scene plan
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

    # 4. Load PLY, remove content
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
    print(f"[redesign] Content Gaussians hidden (opacity->-10)")

    # 5. Generate furniture elements
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
        n_furniture = len(furniture_data)
        print(f"[redesign] Furniture: {n_furniture:,} new Gaussians")
    else:
        combined = data
        n_furniture = 0
        print("[redesign] WARNING: no furniture generated")

    print(f"[redesign] Total: {len(combined):,} Gaussians")

    # 6. Write output
    if ply_out is None:
        out_dir = ROOT / 'output'
        out_dir.mkdir(exist_ok=True)
        ply_out = str(out_dir / f'{design_name}.ply')

    write_ply(ply_out, combined, header_bytes, tail_bytes)
    size_mb = Path(ply_out).stat().st_size / 1024 / 1024
    print(f"\n[redesign] DONE -> {ply_out} ({size_mb:.1f} MB)")

    return {
        'ply_out': ply_out,
        'design_name': design_name,
        'n_input': n_total,
        'n_structure': n_structure,
        'n_content_removed': n_content,
        'n_furniture': n_furniture,
        'n_total': len(combined),
        'plan': plan,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt')
    parser.add_argument('--ply', default='input/current.ply')
    parser.add_argument('--out', default=None)
    parser.add_argument('--plan', default=None, help='Use existing plan JSON instead of calling Claude')
    parser.add_argument('--margin', type=float, default=0.25)
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
