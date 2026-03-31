"""
Surface-coherent 3DGS primitive generator.

Produces Gaussians that are:
  - On the SURFACE only (not interior)
  - Disc-shaped: flat perpendicular to surface normal (like real 3DGS reconstructions)
  - Dense enough to form a solid-looking object
  - High opacity, properly oriented quaternions

Property order matches the scene PLY:
  x, y, z, f_dc_0, f_dc_1, f_dc_2, opacity,
  scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3
"""
import numpy as np

# Scene property order (compact MetalSplat2 format — no normals, no SH rest)
PROPS = ['x','y','z','f_dc_0','f_dc_1','f_dc_2','opacity',
         'scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']


def _normals_to_quaternions(normals):
    """
    Compute quaternions that rotate (0,0,1) to align with each normal.
    normals: (N, 3) array, unit vectors
    returns: (N, 4) quaternions [w, x, y, z]
    """
    # z_hat = (0, 0, 1)
    # axis = cross(z_hat, n) = (0*nz - 1*ny, 1*nx - 0*nz, 0*ny - 0*nx) = (-ny, nx, 0)
    nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]
    ax = -ny
    ay = nx
    az = np.zeros_like(nx)

    sin_angle = np.sqrt(ax**2 + ay**2)  # |cross| = sin(angle)
    cos_angle = np.clip(nz, -1.0, 1.0)  # dot(z_hat, n) = nz = cos(angle)

    half = np.arccos(np.abs(cos_angle)) / 2  # half-angle magnitude
    # Sign of angle: if nz < 0, we need > 90° rotation
    half_angle = np.where(cos_angle >= 0, half, np.pi/2 + (np.pi/2 - half))

    # Degenerate cases
    degenerate = sin_angle < 1e-6
    # n ≈ +z_hat → identity quaternion
    # n ≈ -z_hat → 180° rotation around x
    qw = np.where(degenerate, np.where(nz > 0, 1.0, 0.0), np.cos(half_angle))
    qx = np.where(degenerate, np.where(nz > 0, 0.0, 1.0),
                  np.sin(half_angle) * ax / (sin_angle + 1e-9))
    qy = np.where(degenerate, 0.0,
                  np.sin(half_angle) * ay / (sin_angle + 1e-9))
    qz = np.zeros_like(qw)

    # Normalize
    mag = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2) + 1e-9
    return np.stack([qw/mag, qx/mag, qy/mag, qz/mag], axis=1)


def surface_gaussians_sphere(center, radii, color_dc, n_gaussians,
                              rng, scale_tangent=-4.5, scale_normal=-6.0,
                              opacity_range=(3.0, 5.5)):
    """
    Surface Gaussians on an ellipsoid (bean bag shape).

    center:        (3,) world position
    radii:         (3,) [rx, ry, rz] ellipsoid radii
    color_dc:      (3,) f_dc values [R, G, B] (pre-sigmoid)
    n_gaussians:   number of surface points
    scale_tangent: log-scale along surface (wider — blends between neighbours)
    scale_normal:  log-scale perpendicular to surface (very thin disc)
    """
    cx, cy, cz = center
    rx, ry, rz = radii

    # Sample unit sphere uniformly
    phi = rng.uniform(0, 2*np.pi, n_gaussians)
    cos_theta = rng.uniform(-1, 1, n_gaussians)
    sin_theta = np.sqrt(np.maximum(0, 1 - cos_theta**2))

    # Unit sphere surface points
    ux = sin_theta * np.cos(phi)
    uy = cos_theta
    uz = sin_theta * np.sin(phi)

    # Scale to ellipsoid surface
    px = cx + ux * rx
    py = cy + uy * ry
    pz = cz + uz * rz

    # Surface normal: gradient of ellipsoid F = (x/rx)²+(y/ry)²+(z/rz)²-1
    # Normal ∝ (2(x-cx)/rx², 2(y-cy)/ry², 2(z-cz)/rz²)
    nx_raw = ux / rx
    ny_raw = uy / ry
    nz_raw = uz / rz
    norm = np.sqrt(nx_raw**2 + ny_raw**2 + nz_raw**2) + 1e-9
    normals = np.stack([nx_raw/norm, ny_raw/norm, nz_raw/norm], axis=1)

    # Quaternions aligning Gaussian disc to surface
    quats = _normals_to_quaternions(normals)

    n = n_gaussians
    data = np.zeros((n, 14), dtype=np.float32)

    # Positions
    data[:, 0] = px
    data[:, 1] = py
    data[:, 2] = pz

    # Color (f_dc_0/1/2) with small per-Gaussian variation
    for i, dc in enumerate(color_dc):
        noise = rng.normal(0, 0.05, n)
        data[:, 3+i] = dc + noise

    # Opacity — high (very visible)
    data[:, 6] = rng.uniform(*opacity_range, n)

    # Scale — disc shaped: tangent directions wide, normal direction thin
    # Add small random variation so adjacent splats don't all look identical
    noise_s = rng.normal(0, 0.15, (n, 3))
    data[:, 7] = scale_tangent + noise_s[:, 0]   # scale_0 (tangent)
    data[:, 8] = scale_tangent + noise_s[:, 1]   # scale_1 (tangent)
    data[:, 9] = scale_normal  + noise_s[:, 2]   # scale_2 (normal — thin)

    # Rotation — aligned to surface normal
    data[:, 10] = quats[:, 0]  # rot_0 (w)
    data[:, 11] = quats[:, 1]  # rot_1 (x)
    data[:, 12] = quats[:, 2]  # rot_2 (y)
    data[:, 13] = quats[:, 3]  # rot_3 (z)

    return data


def surface_gaussians_box(center, half_extents, color_dc, n_gaussians,
                           rng, scale_tangent=-4.6, scale_normal=-6.2,
                           opacity_range=(3.5, 5.5)):
    """
    Surface Gaussians on a rectangular box (for cabinets, countertops, etc).

    center:       (3,) world position
    half_extents: (3,) [hx, hy, hz] half-sizes in each axis
    color_dc:     (3,) f_dc values for the element color
    """
    cx, cy, cz = center
    hx, hy, hz = half_extents

    # 6 faces: define by normal direction and the 2 tangent extents
    faces = [
        (np.array([1., 0., 0.]),  np.array([0.,1.,0.]), np.array([0.,0.,1.]), hx, hy, hz),  # +X
        (np.array([-1.,0., 0.]),  np.array([0.,1.,0.]), np.array([0.,0.,1.]), hx, hy, hz),  # -X
        (np.array([0., 1., 0.]),  np.array([1.,0.,0.]), np.array([0.,0.,1.]), hy, hx, hz),  # +Y
        (np.array([0.,-1., 0.]),  np.array([1.,0.,0.]), np.array([0.,0.,1.]), hy, hx, hz),  # -Y
        (np.array([0., 0., 1.]),  np.array([1.,0.,0.]), np.array([0.,1.,0.]), hz, hx, hy),  # +Z
        (np.array([0., 0.,-1.]),  np.array([1.,0.,0.]), np.array([0.,1.,0.]), hz, hx, hy),  # -Z
    ]

    # Distribute Gaussians proportional to face area
    areas = []
    for normal, t1, t2, hn, ht1, ht2 in faces:
        areas.append(4 * ht1 * ht2)   # face area = 2*ht1 × 2*ht2
    total_area = sum(areas)
    counts = [max(10, int(n_gaussians * a / total_area)) for a in areas]

    all_pts = []
    all_normals = []

    for (normal, t1, t2, hn, ht1, ht2), count in zip(faces, counts):
        # Sample 2D UV on face
        u = rng.uniform(-ht1, ht1, count)
        v = rng.uniform(-ht2, ht2, count)

        # 3D position: center + normal*hn + u*t1 + v*t2
        pts = (np.array([cx, cy, cz]) +
               normal * hn +
               np.outer(u, t1) +
               np.outer(v, t2))
        normals_face = np.tile(normal, (count, 1))

        all_pts.append(pts)
        all_normals.append(normals_face)

    pts = np.concatenate(all_pts, axis=0)
    normals = np.concatenate(all_normals, axis=0)

    # Shuffle
    idx = rng.permutation(len(pts))
    pts = pts[idx]
    normals = normals[idx]

    n = len(pts)
    quats = _normals_to_quaternions(normals)

    data = np.zeros((n, 14), dtype=np.float32)
    data[:, :3] = pts.astype(np.float32)

    for i, dc in enumerate(color_dc):
        noise = rng.normal(0, 0.04, n)
        data[:, 3+i] = dc + noise

    data[:, 6] = rng.uniform(*opacity_range, n)

    noise_s = rng.normal(0, 0.12, (n, 3))
    data[:, 7] = scale_tangent + noise_s[:, 0]
    data[:, 8] = scale_tangent + noise_s[:, 1]
    data[:, 9] = scale_normal  + noise_s[:, 2]

    data[:, 10] = quats[:, 0]
    data[:, 11] = quats[:, 1]
    data[:, 12] = quats[:, 2]
    data[:, 13] = quats[:, 3]

    return data


def surface_gaussians_cylinder(center, radius, half_height, axis,
                                color_dc, n_gaussians, rng,
                                scale_tangent=-4.7, scale_normal=-6.3,
                                opacity_range=(3.0, 5.0)):
    """Surface Gaussians on a cylinder (sink basin, handles, etc)."""
    cx, cy, cz = center
    # axis: 0=X, 1=Y, 2=Z axis of cylinder

    n_side = int(n_gaussians * 0.75)
    n_caps = n_gaussians - n_side

    pts = []
    normals_list = []

    # Side surface
    phi = rng.uniform(0, 2*np.pi, n_side)
    h   = rng.uniform(-half_height, half_height, n_side)
    if axis == 1:  # Y axis cylinder (Y-down, for basin)
        px = cx + radius * np.cos(phi)
        py = cy + h
        pz = cz + radius * np.sin(phi)
        nx_ = np.cos(phi); ny_ = np.zeros(n_side); nz_ = np.sin(phi)
    elif axis == 2:  # Z axis
        px = cx + radius * np.cos(phi)
        py = cy + h
        pz = cz + np.zeros(n_side)
        nx_ = np.cos(phi); ny_ = h*0; nz_ = np.zeros(n_side)
    else:  # X axis
        px = cx + np.zeros(n_side)
        py = cy + radius * np.cos(phi)
        pz = cz + radius * np.sin(phi)
        nx_ = np.zeros(n_side); ny_ = np.cos(phi); nz_ = np.sin(phi)

    pts.append(np.stack([px, py, pz], axis=1))
    n_arr = np.stack([nx_, ny_, nz_], axis=1)
    mag = np.linalg.norm(n_arr, axis=1, keepdims=True) + 1e-9
    normals_list.append(n_arr / mag)

    # End caps (simple discs)
    for cap_sign in [-1, 1]:
        n_cap = n_caps // 2
        r_cap = np.sqrt(rng.uniform(0, 1, n_cap)) * radius
        phi_cap = rng.uniform(0, 2*np.pi, n_cap)
        if axis == 1:
            px_c = cx + r_cap * np.cos(phi_cap)
            py_c = cy + cap_sign * half_height * np.ones(n_cap)
            pz_c = cz + r_cap * np.sin(phi_cap)
            cap_normal = np.array([0, float(cap_sign), 0])
        else:
            px_c = cx + r_cap * np.cos(phi_cap)
            py_c = cy + r_cap * np.sin(phi_cap)
            pz_c = cz + cap_sign * half_height * np.ones(n_cap)
            cap_normal = np.array([0, 0, float(cap_sign)])
        pts.append(np.stack([px_c, py_c, pz_c], axis=1))
        normals_list.append(np.tile(cap_normal, (n_cap, 1)))

    pts = np.concatenate(pts, axis=0).astype(np.float32)
    normals = np.concatenate(normals_list, axis=0)
    quats = _normals_to_quaternions(normals)

    n = len(pts)
    data = np.zeros((n, 14), dtype=np.float32)
    data[:, :3] = pts

    for i, dc in enumerate(color_dc):
        data[:, 3+i] = dc + rng.normal(0, 0.04, n)

    data[:, 6] = rng.uniform(*opacity_range, n)

    noise_s = rng.normal(0, 0.12, (n, 3))
    data[:, 7] = scale_tangent + noise_s[:, 0]
    data[:, 8] = scale_tangent + noise_s[:, 1]
    data[:, 9] = scale_normal  + noise_s[:, 2]

    data[:, 10] = quats[:, 0]
    data[:, 11] = quats[:, 1]
    data[:, 12] = quats[:, 2]
    data[:, 13] = quats[:, 3]

    return data
