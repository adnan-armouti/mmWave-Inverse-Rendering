"""GPU-accelerated voxelization for LiDAR point clouds."""


try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None


def check_cupy_available():
    """Check if CuPy is available and raise informative error if not."""
    if not HAS_CUPY:
        raise ImportError(
            "CuPy is required for GPU acceleration. "
            "Install with: pip install cupy-cuda11x (adjust for your CUDA version)"
        )


def skew_symmetric_matrix_gpu(v: cp.ndarray) -> cp.ndarray:
    """Create skew-symmetric matrix from a 3D vector on GPU."""
    K = cp.zeros((3, 3), dtype=cp.float32)
    K[0, 1] = -v[2]
    K[0, 2] = v[1]
    K[1, 0] = v[2]
    K[1, 2] = -v[0]
    K[2, 0] = -v[1]
    K[2, 1] = v[0]
    return K


def compute_rotation_matrix_gpu(boresight: cp.ndarray, target: cp.ndarray) -> cp.ndarray:
    """Compute rotation matrix to align boresight with target direction."""
    # Normalize vectors
    boresight = boresight / cp.linalg.norm(boresight)
    target = target / cp.linalg.norm(target)

    dot = cp.dot(boresight, target)
    dot_val = float(dot.get())  # Transfer to CPU for comparison

    # Nearly parallel - identity matrix
    if dot_val > 0.9999:
        return cp.eye(3, dtype=cp.float32)

    # General case: Rodrigues rotation formula
    axis = cp.cross(boresight, target)
    axis = axis / cp.linalg.norm(axis)
    angle = cp.arccos(cp.clip(dot, -1.0, 1.0))

    K = skew_symmetric_matrix_gpu(axis)
    R = cp.eye(3, dtype=cp.float32) + cp.sin(angle) * K + (1 - cp.cos(angle)) * (K @ K)
    return R


def voxelize_lidar_gpu(
    pts_gpu: cp.ndarray,
    intensities_gpu: cp.ndarray,
    origin: cp.ndarray,
    boresight: cp.ndarray,
    az_edges: cp.ndarray,
    el_edges: cp.ndarray,
    r_edges: cp.ndarray,
    antenna_weights: cp.ndarray = None,
    near_field_m: float = 1.5,
) -> cp.ndarray:
    """GPU-accelerated voxelization of LiDAR point cloud to RAE tensor."""
    check_cupy_available()

    pts_gpu.shape[0]
    num_az = len(az_edges) - 1
    num_el = len(el_edges) - 1
    num_r = len(r_edges) - 1

    # 1. Translate to radar origin frame
    pts_local = pts_gpu - origin

    # 2. Compute rotation to align boresight with +Y
    default_bore = cp.array([0.0, 1.0, 0.0], dtype=cp.float32)
    R_inv = compute_rotation_matrix_gpu(boresight, default_bore)

    # 3. Rotate points
    pts_radar = (R_inv @ pts_local.T).T  # (N, 3)

    # 4. Compute spherical coordinates
    x = pts_radar[:, 0]
    y = pts_radar[:, 1]
    z = pts_radar[:, 2]
    r = cp.linalg.norm(pts_radar, axis=1)

    # 5. Near-field filter
    valid_mask = r >= near_field_m

    # 6. Spherical coordinates (matching radar FFT convention with -x for azimuth)
    az = cp.arctan2(-x, y)  # azimuth: -x/y (flipped to match radar)
    el = cp.arcsin(cp.clip(z / cp.clip(r, 1e-12, None), -1.0, 1.0))

    # 7. Bin indices using searchsorted
    az_idx = cp.searchsorted(az_edges, az, side='right') - 1
    el_idx = cp.searchsorted(el_edges, el, side='right') - 1
    r_idx = cp.searchsorted(r_edges, r, side='right') - 1

    # 8. Valid bin mask
    valid_bins = (
        valid_mask &
        (az_idx >= 0) & (az_idx < num_az) &
        (el_idx >= 0) & (el_idx < num_el) &
        (r_idx >= 0) & (r_idx < num_r)
    )

    # 9. Filter to valid points
    az_idx_valid = az_idx[valid_bins]
    el_idx_valid = el_idx[valid_bins]
    r_idx_valid = r_idx[valid_bins]
    intensities_valid = intensities_gpu[valid_bins]

    # 10. Compute linear indices for scatter_add
    linear_idx = az_idx_valid * (num_el * num_r) + el_idx_valid * num_r + r_idx_valid

    # 11. Histogram using scatter_add (CuPy's cupyx.scatter_add or bincount)
    rae_flat = cp.zeros(num_az * num_el * num_r, dtype=cp.float32)

    # Use bincount with weights for histogram (faster than scatter_add for this case)
    # bincount requires integer indices and handles duplicates by summing weights
    rae_flat = cp.bincount(
        linear_idx.astype(cp.int64),
        weights=intensities_valid,
        minlength=num_az * num_el * num_r
    ).astype(cp.float32)

    rae_tensor = rae_flat.reshape(num_az, num_el, num_r)

    # 12. Apply antenna weights if provided
    if antenna_weights is not None:
        # Broadcast along range dimension
        rae_tensor = rae_tensor * antenna_weights[:, :, cp.newaxis]

    return rae_tensor


def rae_to_ra_map_gpu(rae_tensor: cp.ndarray) -> cp.ndarray:
    """Collapse RAE tensor along elevation axis to get RA map."""
    # Sum along elevation axis (axis=-2 for arbitrary batch dimensions)
    return cp.sum(rae_tensor, axis=-2)
