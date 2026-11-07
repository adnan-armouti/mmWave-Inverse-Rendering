"""LiDAR-based 4-DOF alignment of the cascaded radar.

The LiDAR point cloud is voxelized in the radar's range-azimuth-elevation grid, and range,
azimuth and two rotations are searched (GPU grid search, then L-BFGS-B) for the best match with
the measured range-azimuth map.
"""

import json
import os
import time
from typing import Any, Dict, Optional, Tuple

import cupy as cp
import mitsuba as mi
import numpy as np
import open3d as o3d
from scipy.optimize import minimize

if mi.variant() is None:
    mi.set_variant('cuda_ad_rgb')

from mmir.data.ra_utils import adc_to_ra_image, compute_range_res_from_cfg, load_target
from mmir.evaluation.utils.single_view_proc import _centers_to_edges, make_angle_grids_np
from mmir.evaluation.utils.single_view_viz import load_config_positions_and_boresight
from mmir.losses.loss_utils import minmax_normalize_numpy
from .gpu_utils.batch_eval_gpu import apply_transformations_to_boresight_origin, coarse_grid_search_gpu
from .gpu_utils.metrics_gpu import compute_alignment_metrics_gpu, minmax_normalize_gpu
from .gpu_utils.voxelize_gpu import rae_to_ra_map_gpu, voxelize_lidar_gpu


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """Load point cloud from PLY or NPY file."""
    path = str(path)

    if path.endswith('.ply'):
        return o3d.io.read_point_cloud(path)

    elif path.endswith('.npy'):
        data = np.load(path)

        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, :3].astype(np.float64))

        ncols = data.shape[1]

        if ncols == 7:
            # x, y, z, n_x, n_y, n_z, intensity - use column 6 (intensity)
            # This is our LiDAR format: points + normals + intensity
            intensity = data[:, 6]  # Last column is intensity
            print("[load_point_cloud] 7-column format detected: using column 6 as intensity")
            print(f"  Intensity range: [{intensity.min():.4f}, {intensity.max():.4f}]")
            # Normalize intensity to [0, 1]
            if intensity.max() > intensity.min():
                intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min())
            else:
                intensity_norm = np.ones_like(intensity)
            colors = np.stack([intensity_norm, intensity_norm, intensity_norm], axis=1)
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        elif ncols == 6:
            # x, y, z, r, g, b OR x, y, z, n_x, n_y, n_z
            # Check if values are in typical color range vs normal range
            vals = data[:, 3:6]
            if vals.min() >= 0 and vals.max() <= 255:
                # Likely RGB colors
                colors = vals
                if colors.max() > 1.0:
                    colors = colors / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
            else:
                # Likely normals (can be negative) - use uniform intensity
                print("[load_point_cloud] 6-column format with negative values: assuming normals, using uniform intensity")
                intensity_norm = np.ones((data.shape[0],), dtype=np.float64)
                colors = np.stack([intensity_norm, intensity_norm, intensity_norm], axis=1)
                pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    else:
        raise ValueError(f"Unsupported file format: {path}. Expected .ply or .npy")


# Default antenna pattern path
DEFAULT_ANTENNA_PATTERN_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "assets", "antenna_pattern", "MMWCAS", "tx1_76.npy"
)


def load_antenna_pattern(pattern_path: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load antenna beam pattern from NPY file."""
    if pattern_path is None:
        pattern_path = DEFAULT_ANTENNA_PATTERN_PATH

    pattern = np.load(pattern_path)

    # Pattern has 361 samples from -180 to +180 degrees
    angles_deg = np.linspace(-180, 180, pattern.shape[0])
    # Column 1 = azimuth (wider beam, ~136° 0dB width)
    # Column 0 = elevation (narrower beam, ~41° 0dB width)
    az_gain_db = pattern[:, 1]
    el_gain_db = pattern[:, 0]

    return angles_deg, az_gain_db, el_gain_db


def interpolate_antenna_gain(angles_deg: np.ndarray, gain_db: np.ndarray,
                              query_angles_deg: np.ndarray) -> np.ndarray:
    """Interpolate antenna gain for query angles."""
    # Wrap query angles to [-180, 180]
    query_wrapped = np.mod(query_angles_deg + 180, 360) - 180

    return np.interp(query_wrapped, angles_deg, gain_db)


def load_radar_config(config_path: str) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Load radar config - wraps existing load_config_positions_and_boresight."""
    tx_m, rx_m, cfg_center, cfg_bore = load_config_positions_and_boresight(config_path)

    # Build params dict (same pattern as single_view_viz.py:213-240)
    # Note: num_az_bins and num_el_bins are 127 (not 128) because radar FFT
    # drops the first bin (DC bin) before fft shift, resulting in 128-1=127 bins.
    # This ensures center azimuth bin (63) aligns with boresight direction.
    params = {
        'range_resolution': compute_range_res_from_cfg(config_path),
        'num_adc': 256,  # Default, override from config if available
        'num_az_bins': 127,  # 128-1: drop DC bin to match radar FFT output
        'num_el_bins': 127,  # 128-1: drop DC bin to match radar FFT output
    }

    # Load numAdcSamples from config if available
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    if 'numAdcSamples' in cfg:
        params['num_adc'] = int(cfg['numAdcSamples'])

    return params, cfg_center, cfg_bore


def radar_gt_to_ra_map(gt_adc_path: str) -> np.ndarray:
    """Load radar ground truth ADC and convert to magnitude RA map."""
    adc_data, _ = load_target(gt_adc_path, device='cpu', normalize=False)
    ra_map = adc_to_ra_image(adc_data)
    return ra_map.numpy()  # Shape (127, 256)


def create_aligned_config(
    config_path: str,
    delta_range_m: float,
    delta_azimuth_deg: float,
    rotation_elev_deg: float,
    rotation_azim_deg: float,
    output_path: Optional[str] = None
) -> str:
    """Create a new radar config file with updated antenna positions and boresights based on
    optimized alignment parameters.
    """
    # Load original config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Extract antenna positions (in meters)
    tx_pos = np.array([t['pos_mm'] for t in config['tx_array']], dtype=float) / 1000.0
    rx_pos = np.array([r['pos_mm'] for r in config['rx_array']], dtype=float) / 1000.0
    all_pos = np.vstack([tx_pos, rx_pos])
    n_tx = tx_pos.shape[0]
    rx_pos.shape[0]

    # Compute board center (geometric mean of all antennas)
    board_center = np.mean(all_pos, axis=0)

    # Extract original boresight (normalized)
    base_boresight = np.array(config['tx_array'][0]['boresight'], dtype=float)
    base_boresight = _norm(base_boresight)

    # 1a. Rotation around elevation axis (Z-axis, azimuth sweep)
    rot_elev_rad = np.radians(rotation_elev_deg)
    rot_vec_elev = np.array([0.0, 0.0, rot_elev_rad])
    R_elev = _axis_angle_to_matrix(rot_vec_elev)

    # Apply first rotation to boresight
    boresight_after_elev = R_elev @ base_boresight
    boresight_after_elev = _norm(boresight_after_elev)

    # 1b. Rotation around azimuth axis (board X-axis, elevation tilt)
    # X-axis is perpendicular to boresight in horizontal plane
    z_world = np.array([0.0, 0.0, 1.0])
    x_board = _norm(np.cross(boresight_after_elev, z_world))
    if np.linalg.norm(x_board) < 1e-6:
        x_board = np.array([1.0, 0.0, 0.0])

    rot_azim_rad = np.radians(rotation_azim_deg)
    rot_vec_azim = x_board * rot_azim_rad
    R_azim = _axis_angle_to_matrix(rot_vec_azim)

    # Combined rotation matrix
    R_combined = R_azim @ R_elev

    # Final rotated boresight
    new_boresight = R_combined @ base_boresight
    new_boresight = _norm(new_boresight)

    # Antenna positions relative to board center
    rel_pos = all_pos - board_center

    # Rotate relative positions
    rel_pos_rotated = (R_combined @ rel_pos.T).T  # (N, 3)

    # Range translation: along new boresight direction
    range_offset = new_boresight * delta_range_m

    # Azimuth translation: perpendicular to boresight in horizontal plane
    # Convert degrees to meters using arc length at reference range
    reference_range_m = 10.0
    delta_azimuth_m = reference_range_m * np.radians(delta_azimuth_deg)

    # New X-axis after rotation
    x_new = _norm(np.cross(new_boresight, z_world))
    if np.linalg.norm(x_new) < 1e-6:
        x_new = np.array([1.0, 0.0, 0.0])

    azimuth_offset = x_new * delta_azimuth_m

    # Total translation
    translation = range_offset + azimuth_offset

    # STEP 4: Apply translation to board center, reconstruct antenna positions

    new_board_center = board_center + translation
    new_all_pos = rel_pos_rotated + new_board_center

    # Split back into TX and RX
    new_tx_pos = new_all_pos[:n_tx]  # (n_tx, 3)
    new_rx_pos = new_all_pos[n_tx:]  # (n_rx, 3)

    # STEP 5: Update config with new positions and boresights

    aligned_config = json.loads(json.dumps(config))  # Deep copy

    # Update TX array
    for i, tx in enumerate(aligned_config['tx_array']):
        tx['pos_mm'] = (new_tx_pos[i] * 1000.0).tolist()
        tx['boresight'] = new_boresight.tolist()

    # Update RX array
    for i, rx in enumerate(aligned_config['rx_array']):
        rx['pos_mm'] = (new_rx_pos[i] * 1000.0).tolist()
        rx['boresight'] = new_boresight.tolist()

    # STEP 6: Save aligned config

    with open(output_path, 'w') as f:
        json.dump(aligned_config, f, indent=2)

    # Print summary
    print(f"\n[Aligned Config] Created: {output_path}")
    print(f"  Original boresight: [{base_boresight[0]:.4f}, {base_boresight[1]:.4f}, {base_boresight[2]:.4f}]")
    print(f"  New boresight:      [{new_boresight[0]:.4f}, {new_boresight[1]:.4f}, {new_boresight[2]:.4f}]")
    print(f"  Board center shift: [{translation[0]*1000:.2f}, {translation[1]*1000:.2f}, {translation[2]*1000:.2f}] mm")

    return output_path


def _norm(v, eps: float = 1e-9):
    """L2-normalise `v` along its last dimension."""
    return v / (np.linalg.norm(v) + eps)


def _axis_angle_to_matrix(w):
    """Rodrigues' formula - converts a 3-vector axis-angle `w` (rad) to a 3x3 rotation matrix. Works
    with numpy arrays.
    """
    theta = np.linalg.norm(w)
    if theta < 1e-9:
        return np.eye(3)

    k = w / theta  # unit axis
    kx, ky, kz = k[0], k[1], k[2]

    # Skew-symmetric matrix
    K = np.array([
        [0, -kz,  ky],
        [kz,  0, -kx],
        [-ky, kx,  0]
    ])

    eye = np.eye(3)
    return eye + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def prepare_gpu_data(
    lidar_pcd: o3d.geometry.PointCloud,
    ra_radar: np.ndarray,
    params: Dict[str, Any],
    antenna_pattern_path: str = None
) -> Tuple:
    """Transfer data to GPU and prepare grid edges."""
    # Get points and intensities
    pts = np.asarray(lidar_pcd.points, dtype=np.float32)

    # Get intensities from colors (grayscale)
    if lidar_pcd.has_colors():
        colors = np.asarray(lidar_pcd.colors, dtype=np.float32)
        intensities = colors[:, 0]  # R channel (grayscale)
    else:
        intensities = np.ones(pts.shape[0], dtype=np.float32)

    # Transfer to GPU
    pts_gpu = cp.asarray(pts)
    intensities_gpu = cp.asarray(intensities)

    # Normalize radar RA and transfer
    ra_radar_norm = minmax_normalize_numpy(ra_radar).astype(np.float32)
    ra_radar_gpu = cp.asarray(ra_radar_norm)

    # Generate grid edges
    fft_size_az = int(params['num_az_bins']) + 1
    fft_size_el = int(params['num_el_bins']) + 1
    az_cent, el_cent = make_angle_grids_np(fft_size_az, fft_size_el)
    az_edges = _centers_to_edges(az_cent, low_clip=-np.pi/2, high_clip=np.pi/2).astype(np.float32)
    el_edges = _centers_to_edges(el_cent, low_clip=-np.pi/2, high_clip=np.pi/2).astype(np.float32)
    r_edges = (np.arange(int(params['num_adc']) + 1, dtype=np.float32) * float(params['range_resolution']))

    az_edges_gpu = cp.asarray(az_edges)
    el_edges_gpu = cp.asarray(el_edges)
    r_edges_gpu = cp.asarray(r_edges)

    # Compute antenna weights
    angles_deg, az_gain_db, el_gain_db = load_antenna_pattern(antenna_pattern_path)

    az_deg = np.degrees(az_cent)
    el_deg = np.degrees(el_cent)

    az_gain_interp = interpolate_antenna_gain(angles_deg, az_gain_db, az_deg)
    el_gain_interp = interpolate_antenna_gain(angles_deg, el_gain_db, el_deg)

    az_gain_linear = 10 ** (az_gain_interp / 10)
    el_gain_linear = 10 ** (el_gain_interp / 10)

    weights = np.outer(az_gain_linear, el_gain_linear)
    weights = (weights / weights.max()).astype(np.float32)

    antenna_weights_gpu = cp.asarray(weights)

    return (pts_gpu, intensities_gpu, ra_radar_gpu,
            az_edges_gpu, el_edges_gpu, r_edges_gpu, antenna_weights_gpu)


def evaluate_alignment_gpu(
    pts_gpu,
    intensities_gpu,
    ra_radar_gpu,
    base_boresight: np.ndarray,
    base_origin: np.ndarray,
    delta_range_m: float,
    delta_azimuth_deg: float,
    rotation_elev_deg: float,
    rotation_azim_deg: float,
    az_edges_gpu,
    el_edges_gpu,
    r_edges_gpu,
    antenna_weights_gpu,
    near_field_m: float = 1.5,
) -> Dict[str, float]:
    """Evaluate alignment for a single parameter set on GPU."""
    # Compute transformed boresight and origin
    boresight, origin = apply_transformations_to_boresight_origin(
        base_boresight, base_origin,
        delta_range_m, delta_azimuth_deg,
        rotation_elev_deg, rotation_azim_deg
    )

    boresight_gpu = cp.asarray(boresight, dtype=cp.float32)
    origin_gpu = cp.asarray(origin, dtype=cp.float32)

    # Voxelize
    rae_tensor = voxelize_lidar_gpu(
        pts_gpu, intensities_gpu,
        origin_gpu, boresight_gpu,
        az_edges_gpu, el_edges_gpu, r_edges_gpu,
        antenna_weights_gpu, near_field_m
    )

    # Collapse to RA
    ra_lidar = rae_to_ra_map_gpu(rae_tensor)

    # Normalize
    ra_lidar_norm = minmax_normalize_gpu(ra_lidar)

    # Compute metrics
    return compute_alignment_metrics_gpu(ra_lidar_norm, ra_radar_gpu)


def optimize_alignment_gpu(
    lidar_pcd: o3d.geometry.PointCloud,
    ra_radar: np.ndarray,
    params: Dict[str, Any],
    base_boresight: np.ndarray,
    base_origin: np.ndarray,
    range_search_m: Tuple[float, float],
    azimuth_search_deg: Tuple[float, float],
    rotation_elev_search_deg: Tuple[float, float],
    rotation_azim_search_deg: Tuple[float, float],
    metric: str = 'correlation',
    coarse_steps: int = 7,
    near_field_m: float = 1.5,
    batch_size: int = 100,
    verbose: bool = True
) -> Dict[str, Any]:
    """GPU-accelerated coarse-to-fine alignment optimization."""
    # Prepare GPU data

    (pts_gpu, intensities_gpu, ra_radar_gpu,
     az_edges_gpu, el_edges_gpu, r_edges_gpu,
     antenna_weights_gpu) = prepare_gpu_data(lidar_pcd, ra_radar, params)

    # Phase 1: GPU coarse grid search
    t1 = time.time()

    best_params, best_value, grid_results = coarse_grid_search_gpu(
        pts_gpu, intensities_gpu, ra_radar_gpu,
        base_boresight, base_origin,
        range_search_m, azimuth_search_deg,
        rotation_elev_search_deg, rotation_azim_search_deg,
        az_edges_gpu, el_edges_gpu, r_edges_gpu,
        antenna_weights_gpu, near_field_m,
        metric, coarse_steps, batch_size, verbose
    )

    if verbose:
        print(f"[GPU] Phase 1 complete in {time.time() - t1:.2f}s")

    # Phase 2: Fine optimization (unconstrained, matching original)
    maximize = metric in ['correlation', 'ssim', 'iou']

    if verbose:
        print("Phase 2: Fine optimization (Nelder-Mead, unconstrained)...")

    t2 = time.time()

    def objective(x):
        metrics = evaluate_alignment_gpu(
            pts_gpu, intensities_gpu, ra_radar_gpu,
            base_boresight, base_origin,
            x[0], x[1], x[2], x[3],
            az_edges_gpu, el_edges_gpu, r_edges_gpu,
            antenna_weights_gpu, near_field_m
        )
        value = metrics[metric]
        return -value if maximize else value

    # Use unconstrained Nelder-Mead (matching original lidar_radar_ra_alignment.py)
    result = minimize(
        objective,
        x0=best_params,
        method='Nelder-Mead',
        options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.0001}
    )

    if verbose:
        print(f"[GPU] Phase 2 complete in {time.time() - t2:.2f}s")

    optimal_params = {
        'delta_range_m': float(result.x[0]),
        'delta_azimuth_deg': float(result.x[1]),
        'rotation_elev_deg': float(result.x[2]),
        'rotation_azim_deg': float(result.x[3])
    }

    # Compute final metrics
    best_metrics = evaluate_alignment_gpu(
        pts_gpu, intensities_gpu, ra_radar_gpu,
        base_boresight, base_origin,
        optimal_params['delta_range_m'],
        optimal_params['delta_azimuth_deg'],
        optimal_params['rotation_elev_deg'],
        optimal_params['rotation_azim_deg'],
        az_edges_gpu, el_edges_gpu, r_edges_gpu,
        antenna_weights_gpu, near_field_m
    )

    # Compute initial metrics
    if verbose:
        print("\n" + "=" * 60)
        print("DEBUG: Initial evaluation (no transformation)")
        print("=" * 60)

    initial_metrics = evaluate_alignment_gpu(
        pts_gpu, intensities_gpu, ra_radar_gpu,
        base_boresight, base_origin,
        0.0, 0.0, 0.0, 0.0,
        az_edges_gpu, el_edges_gpu, r_edges_gpu,
        antenna_weights_gpu, near_field_m
    )

    if verbose:
        print(f"  Optimization complete. Final {metric}: {best_metrics[metric]:.4f}")

    # Synchronize GPU before returning
    cp.cuda.Stream.null.synchronize()

    return {
        'optimal_params': optimal_params,
        'best_metrics': best_metrics,
        'initial_metrics': initial_metrics,
        'grid_search_results': grid_results
    }
