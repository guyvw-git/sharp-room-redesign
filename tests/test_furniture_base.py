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
