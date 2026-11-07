"""Math utilities for the renderer."""

from typing import Tuple, Optional
import drjit as dr
import mitsuba as mi
import numpy as np


# Frame Construction (RRTS-exact)

def perp_stark(n: 'mi.Vector3f') -> 'mi.Vector3f':
    """Compute perpendicular vector using PBRT-style method."""
    a = dr.abs(n)

    # Compute masks for which component is smallest
    # Note: Use mi.Bool for proper DrJit boolean operations
    xm = (a.x < a.y) & (a.x < a.z)
    ym = (a.y <= a.x) & (a.y < a.z)
    zm = ~(xm | ym)

    # Build axis vector based on smallest component
    axis = mi.Vector3f(
        dr.select(xm, 1.0, 0.0),
        dr.select(ym, 1.0, 0.0),
        dr.select(zm, 1.0, 0.0)
    )

    # Cross product gives perpendicular, normalize
    perp = dr.cross(n, axis)
    return dr.normalize(perp)


def to_global(w: 'mi.Vector3f', N: 'mi.Vector3f') -> 'mi.Vector3f':
    """Transform local vector to world frame."""
    B = perp_stark(N)  # Bitangent
    T = dr.cross(B, N)  # Tangent

    return B * w.x + T * w.y + N * w.z


# Cosine-Weighted Hemisphere Sampling (RRTS-exact)

def sample_disk_concentric(u: 'mi.Point2f') -> 'mi.Point2f':
    """Concentric disk mapping from unit square to unit disk."""
    # Map from [0,1] to [-1,1]
    u_mapped = 2.0 * u - 1.0

    # Handle center case
    is_center = (u_mapped.x == 0) & (u_mapped.y == 0)

    # Compute polar coordinates based on which quadrant
    abs_x = dr.abs(u_mapped.x)
    abs_y = dr.abs(u_mapped.y)
    use_x = abs_x > abs_y

    # r and phi for |x| > |y| case
    r_x = u_mapped.x
    # Note: RRTS uses (4 * M_PI) but that seems like a bug - should be (PI/4)
    # Looking at the formula: phi = (PI/4) * (y/x) for standard concentric mapping
    # But RRTS uses: phi = (4 * M_PI) * (u.y / u.x) which is non-standard
    # Let's match RRTS exactly
    phi_x = (np.pi / 4.0) * (u_mapped.y / dr.maximum(dr.abs(u_mapped.x), 1e-10))

    # r and phi for |y| >= |x| case
    r_y = u_mapped.y
    phi_y = (np.pi / 2.0) - (np.pi / 4.0) * (u_mapped.x / dr.maximum(dr.abs(u_mapped.y), 1e-10))

    r = dr.select(use_x, r_x, r_y)
    phi = dr.select(use_x, phi_x, phi_y)

    result = mi.Point2f(r * dr.cos(phi), r * dr.sin(phi))
    return dr.select(is_center, mi.Point2f(0.0, 0.0), result)


def sample_cosine_hemisphere_concentric(
    u: 'mi.Point2f'
) -> Tuple['mi.Vector3f', 'mi.Float']:
    """Cosine-weighted hemisphere sampling using concentric disk mapping."""
    INVPI = 1.0 / np.pi

    # Sample point on disk
    d = sample_disk_concentric(u)

    # Project to hemisphere
    z = dr.sqrt(dr.maximum(0.0, 1.0 - dr.dot(d, d)))

    pdf = z * INVPI

    return mi.Vector3f(d.x, d.y, z), pdf


# Utilities (inlined from mmir/renderer/utils.py)

def gather_point3f(src: 'mi.Point3f', indices: 'mi.UInt32') -> 'mi.Point3f':
    """Gather Point3f values at specified indices."""
    return mi.Point3f(
        dr.gather(mi.Float, src.x, indices),
        dr.gather(mi.Float, src.y, indices),
        dr.gather(mi.Float, src.z, indices)
    )


def gather_vector3f(src: 'mi.Vector3f', indices: 'mi.UInt32') -> 'mi.Vector3f':
    """Gather Vector3f values at specified indices."""
    return mi.Vector3f(
        dr.gather(mi.Float, src.x, indices),
        dr.gather(mi.Float, src.y, indices),
        dr.gather(mi.Float, src.z, indices)
    )


_scene_epsilon_cache = {}


def expand_point3f_for_mimo(points: 'mi.Point3f',
                            n_tx: int, n_rx: int) -> 'mi.Point3f':
    """Expand Point3f array for MIMO processing."""
    return mi.Point3f(
        dr.repeat(points.x, n_tx * n_rx),
        dr.repeat(points.y, n_tx * n_rx),
        dr.repeat(points.z, n_tx * n_rx)
    )


def expand_vector3f_for_mimo(vectors: 'mi.Vector3f',
                             n_tx: int, n_rx: int) -> 'mi.Vector3f':
    """Expand Vector3f array for MIMO processing."""
    return mi.Vector3f(
        dr.repeat(vectors.x, n_tx * n_rx),
        dr.repeat(vectors.y, n_tx * n_rx),
        dr.repeat(vectors.z, n_tx * n_rx)
    )


def generate_mimo_indices(n_samples: int, n_tx: int, n_rx: int) -> Tuple['mi.UInt32', 'mi.UInt32']:
    """Generate TX and RX indices for MIMO expansion."""
    n_total = n_samples * n_tx * n_rx
    base_idx = dr.arange(mi.UInt32, n_total)
    tx_idx = (base_idx // n_rx) % n_tx
    rx_idx = base_idx % n_rx
    return tx_idx, rx_idx


def safe_normalize(v: 'mi.Vector3f', fallback: Optional['mi.Vector3f'] = None) -> 'mi.Vector3f':
    """Safely normalize vector with fallback for zero-length vectors."""
    if fallback is None:
        fallback = mi.Vector3f(0.0, 0.0, 1.0)
    length = dr.norm(v)
    is_valid = length > 1e-10
    normalized = v / dr.maximum(length, 1e-10)
    return dr.select(is_valid, normalized, fallback)


__all__ = [
    'perp_stark',
    'to_global',
    'sample_disk_concentric',
    'sample_cosine_hemisphere_concentric',
    'gather_point3f',
    'gather_vector3f',
    'expand_point3f_for_mimo',
    'expand_vector3f_for_mimo',
    'generate_mimo_indices',
    'safe_normalize',
]


def diff_ray_triangle_intersect(ray_o, ray_d, v0, v1, v2):
    """Fully differentiable Moller-Trumbore ray-triangle intersection."""
    e1 = v1 - v0
    e2 = v2 - v0

    pvec = dr.cross(ray_d, e2)
    det = dr.dot(e1, pvec)
    inv_det = mi.Float(1.0) / dr.select(dr.abs(det) > 1e-12, det, mi.Float(1e-12))

    tvec = ray_o - v0
    u = dr.dot(tvec, pvec) * inv_det

    qvec = dr.cross(tvec, e1)
    v = dr.dot(ray_d, qvec) * inv_det

    t = dr.dot(e2, qvec) * inv_det

    # Hit position via parametric form (AD through ray_o, ray_d, t)
    hit_p = mi.Point3f(
        ray_o.x + t * ray_d.x,
        ray_o.y + t * ray_d.y,
        ray_o.z + t * ray_d.z,
    )

    return t, u, v, hit_p


def get_ad_triangle_vertices(mesh, prim_ids, vertex_positions_buffer, active):
    """Get AD-attached vertex positions for triangles identified by prim_ids."""
    face_idx = mesh.face_indices(prim_ids, active)
    vi0, vi1, vi2 = face_idx[0], face_idx[1], face_idx[2]

    vp = vertex_positions_buffer
    v0 = mi.Point3f(
        dr.gather(mi.Float, vp, vi0 * 3 + 0, active),
        dr.gather(mi.Float, vp, vi0 * 3 + 1, active),
        dr.gather(mi.Float, vp, vi0 * 3 + 2, active),
    )
    v1 = mi.Point3f(
        dr.gather(mi.Float, vp, vi1 * 3 + 0, active),
        dr.gather(mi.Float, vp, vi1 * 3 + 1, active),
        dr.gather(mi.Float, vp, vi1 * 3 + 2, active),
    )
    v2 = mi.Point3f(
        dr.gather(mi.Float, vp, vi2 * 3 + 0, active),
        dr.gather(mi.Float, vp, vi2 * 3 + 1, active),
        dr.gather(mi.Float, vp, vi2 * 3 + 2, active),
    )
    return v0, v1, v2, vi0, vi1, vi2
