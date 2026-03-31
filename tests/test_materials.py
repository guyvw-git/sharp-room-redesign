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
    normals = np.tile([0.0, -1.0, 0.0], (n, 1))
    positions = np.zeros((n, 3))
    dc = apply_shading(normals, positions, mat, n)
    linear_r = 1.0 / (1.0 + np.exp(-dc[:, 0]))
    assert np.all(linear_r > 0.5), f"Top face of white should be bright: {linear_r}"

def test_shading_darkens_bottom_face():
    """Normal pointing down (+Y in Y-down) gets only ambient."""
    mat = get_material('matte_white')
    n = 10
    normals = np.tile([0.0, 1.0, 0.0], (n, 1))
    positions = np.zeros((n, 3))
    dc_top = apply_shading(np.tile([0.0,-1.0,0.0],(n,1)), positions, mat, n)
    dc_bot = apply_shading(normals, positions, mat, n)
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
