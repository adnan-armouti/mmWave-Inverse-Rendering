"""BSDF mixture importance sampling for multi-lobe evaluation."""

import numpy as np
import drjit as dr
import mitsuba as mi


def _sample_cosine_hemisphere_drjit(normal, n_paths, seed):
    """Sample cosine-weighted directions in the hemisphere defined by `normal`."""
    # Generate quasi-random samples using PCG
    sampler = mi.load_dict({'type': 'independent'})
    sampler.seed(seed, n_paths)
    sample1 = sampler.next_1d()
    sample2 = sampler.next_1d()

    # Cosine-weighted hemisphere sampling in local frame
    cos_theta = dr.sqrt(dr.maximum(mi.Float(1.0) - sample1, mi.Float(1e-20)))
    sin_theta = dr.sqrt(dr.maximum(sample1, mi.Float(1e-20)))
    phi = mi.Float(2.0 * np.pi) * sample2
    local_x = sin_theta * dr.cos(phi)
    local_y = sin_theta * dr.sin(phi)
    local_z = cos_theta

    # Build tangent frame from normal (Frisvad method)
    # Handle degenerate normals
    n = normal
    abs_nz = dr.abs(n.z)

    # When n.z is not close to -1
    denom = mi.Float(1.0) + n.z
    denom_safe = dr.select(denom > 1e-6, denom, mi.Float(1e-6))
    a = mi.Float(-1.0) / denom_safe
    b = n.x * n.y * a
    t1_x = mi.Float(1.0) + n.x * n.x * a
    t1_y = b
    t1_z = -n.x
    t2_x = b
    t2_y = mi.Float(1.0) + n.y * n.y * a
    t2_z = -n.y

    # Fallback for n.z close to -1
    use_fallback = abs_nz > 0.9999
    t1_x = dr.select(use_fallback, mi.Float(0.0), t1_x)
    t1_y = dr.select(use_fallback, mi.Float(-1.0), t1_y)
    t1_z = dr.select(use_fallback, mi.Float(0.0), t1_z)
    t2_x = dr.select(use_fallback, mi.Float(-1.0), t2_x)
    t2_y = dr.select(use_fallback, mi.Float(0.0), t2_y)
    t2_z = dr.select(use_fallback, mi.Float(0.0), t2_z)

    # Transform to world space
    world_x = local_x * t1_x + local_y * t2_x + local_z * n.x
    world_y = local_x * t1_y + local_y * t2_y + local_z * n.y
    world_z = local_x * t1_z + local_y * t2_z + local_z * n.z

    sampled_dir = mi.Vector3f(world_x, world_y, world_z)
    # Normalize for safety
    d_len = dr.maximum(dr.norm(sampled_dir), mi.Float(1e-10))
    sampled_dir = mi.Vector3f(sampled_dir.x / d_len, sampled_dir.y / d_len, sampled_dir.z / d_len)

    pdf = cos_theta / mi.Float(np.pi)

    return sampled_dir, pdf, sampler
