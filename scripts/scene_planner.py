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
- Choose materials that match the design style (moody bar -> dark materials, bright theater -> dark with bright screen)
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
      "position_x": <float>,
      "position_z": <float>,
      "wall": "north|south|east|west|none",
      "width_m": <float>,
      "height_m": <float>,
      "depth_m": <float>,
      "length_m": <float>,
      "floor_clearance_m": <float>,
      "material": "material_from_list",
      "top_material": "material_from_list",
      "n_gaussians": <int>,
      "notes": "optional"
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
