import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

try:
    import open3d as o3d
except Exception:
    o3d = None


def extract_board_frame_from_config(config_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract orthonormal board frame [azimuth, range, elevation] from dense antenna config."""
    # Load config
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    # Extract antenna positions (in meters)
    tx_pos = np.array([t['pos_mm'] for t in cfg['tx_array']], dtype=float) / 1000.0
    rx_pos = np.array([r['pos_mm'] for r in cfg['rx_array']], dtype=float) / 1000.0
    all_pos = np.vstack([tx_pos, rx_pos])

    # Compute board center (geometric center of all antennas)
    center = np.mean(all_pos, axis=0)

    # Extract boresight from config (normal to board plane)
    boresight = np.array(cfg['tx_array'][0]['boresight'], dtype=float)
    boresight /= np.linalg.norm(boresight)

    # Project all antenna positions onto board plane
    # Remove component along boresight to get in-plane vectors
    rel_pos = all_pos - center  # Relative to center
    plane_proj = rel_pos - np.outer(np.dot(rel_pos, boresight), boresight)

    # PCA on projected positions to find principal directions
    # This gives us the axes of maximum variance (horizontal and vertical on board)
    cov = plane_proj.T @ plane_proj
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (largest variance first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Principal directions on board plane
    v_principal1 = eigenvectors[:, 0]  # Largest variance
    v_principal2 = eigenvectors[:, 1]  # Second largest variance

    # Normalize
    v_principal1 /= np.linalg.norm(v_principal1)
    v_principal2 /= np.linalg.norm(v_principal2)

    # Determine which principal component corresponds to horizontal vs vertical
    # TX antennas form vertical columns, RX form horizontal rows
    # Check which direction has more TX variation
    tx_rel = tx_pos - center
    tx_proj = tx_rel - np.outer(np.dot(tx_rel, boresight), boresight)

    # Variance of TX along each principal direction
    tx_var1 = np.var(np.dot(tx_proj, v_principal1))
    tx_var2 = np.var(np.dot(tx_proj, v_principal2))

    # TX columns are vertical, so larger TX variance -> vertical axis
    v1_elevation = v_principal2  # Vertical (elevation)
    v2_azimuth = v_principal1    # Horizontal (azimuth)

    # Ensure right-handed coordinate system: v2 x boresight = v1
    # If not, flip v2
    cross_check = np.cross(v2_azimuth, boresight)
    if np.dot(cross_check, v1_elevation) < 0:
        v2_azimuth = -v2_azimuth

    return v2_azimuth, boresight, v1_elevation


from .single_view_proc import (
    # primitives
    to_tensor,
    get_hann,
    txrx_to_vx_chirps_dense_gpu_batched,
    make_angle_grids_np,
    _centers_to_edges,
    build_point_cloud,
    load_point_cloud,
    get_pcl_intensities,
    create_colored_sphere,
    create_axis_arrow,
    _compute_range_resolution_from_config,
)

# Import RA utils for cartesian conversion and image saving


def load_config_positions_and_boresight(config_path: Optional[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if config_path is None:
        return None, None, None, None
    try:
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        tx_m = np.array([t['pos_mm'] for t in cfg['tx_array']], dtype=float) / 1000.0
        rx_m = np.array([r['pos_mm'] for r in cfg['rx_array']], dtype=float) / 1000.0
        center = np.mean(np.vstack([tx_m, rx_m]), axis=0)
        bore = None
        if 'boresight' in cfg['tx_array'][0]:
            bore = np.array(cfg['tx_array'][0]['boresight'], dtype=float)
            n = float(np.linalg.norm(bore))
            bore = bore / n
        return tx_m, rx_m, center, bore
    except Exception:
        return None, None, None, None


def load_runtime_config(explicit_config_path: Optional[str], output_directory: Path) -> Dict[str, Any]:
    params = {
        'range_resolution': 0.117,
        'num_adc': 256,
        'num_ant': 100,
        'num_az_bins': 128,
        'num_el_bins': 128,
    }
    cfg0 = None
    if explicit_config_path is not None:
        p = Path(explicit_config_path)
        if p.exists():
            cfg0 = p
    if cfg0 is None:
        return params
    try:
        dr = _compute_range_resolution_from_config(cfg0)
        if dr > 0:
            params['range_resolution'] = float(dr)
        with open(cfg0, 'r') as f:
            cfg = json.load(f)
        if 'numAdcSamples' in cfg:
            params['num_adc'] = int(cfg['numAdcSamples'])
        n_tx = len(cfg.get('tx_array', []))
        n_rx = len(cfg.get('rx_array', []))
        if n_tx > 0 and n_tx == n_rx:
            params['num_ant'] = int(n_tx)
        return params
    except Exception:
        return params


def compute_rae_cube(adc_file: Path, params: Dict[str, Any], device: torch.device) -> np.ndarray:
    adc = np.load(str(adc_file))
    adc_c = adc[:, :, :, 0] + 1j * adc[:, :, :, 1]
    adc_c = np.expand_dims(adc_c, axis=0)
    adc_c = np.transpose(adc_c, (0, 2, 1, 3))
    adc_t = to_tensor(adc_c, device=device, dtype=torch.complex64)
    vx = txrx_to_vx_chirps_dense_gpu_batched(adc_t, num_ant=int(params['num_ant']))
    h_range = get_hann(int(params['num_adc']), vx.device)
    vx = vx * h_range.view(1, 1, 1, -1)
    vx = torch.fft.fft(vx, n=int(params['num_adc']), dim=-1)
    vol = vx
    # Swap FFT dimensions: dim 1 for azimuth, dim 2 for elevation
    h_az = get_hann(vol.shape[1], vx.device)  # Changed from shape[2] to shape[1]
    vol = vol * h_az.view(1, -1, 1, 1)         # Changed view pattern
    vol = torch.fft.ifftshift(vol, dim=1)      # Changed from dim=2 to dim=1
    vol = torch.fft.fft(vol, n=int(params['num_az_bins']), dim=1)  # Changed from dim=2 to dim=1
    vol = vol[:, 1:, :, :]                     # Changed from [:, :, 1:, :] to [:, 1:, :, :]
    vol = torch.fft.fftshift(vol, dim=1)       # Changed from dim=2 to dim=1
    h_el = get_hann(vol.shape[2], vx.device)  # Changed from shape[1] to shape[2]
    vol = vol * h_el.view(1, 1, -1, 1)         # Changed view pattern
    vol = torch.fft.ifftshift(vol, dim=2)      # Changed from dim=1 to dim=2
    vol = torch.fft.fft(vol, n=int(params['num_el_bins']), dim=2)  # Changed from dim=1 to dim=2
    vol = vol[:, :, 1:, :]                     # Changed from [:, 1:, :, :] to [:, :, 1:, :]
    vol = torch.fft.fftshift(vol, dim=2)       # Changed from dim=1 to dim=2
    mag = torch.abs(vol[0]).to(torch.float32).cpu().numpy()  # (Az, El, R) - swapped due to FFT dimension swap
    rae = mag  # Already in (Az, El, R) format, no transpose needed
    return rae


def collapse_to_ra(rae: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ra_map = np.sum(rae, axis=1)
    num_az = int(params['num_az_bins'])
    t = np.arange(-num_az // 2 + 1, num_az // 2) * (2.0 / num_az)
    az_deg = np.degrees(np.arcsin(t))[::-1]
    r_m = np.arange(int(params['num_adc'])) * float(params['range_resolution'])
    return ra_map, az_deg, r_m


def radar_points_from_rae(rae: np.ndarray, params: Dict[str, Any], near_field_m: float, percentile: Optional[float]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    int(params['num_adc'])
    num_az_full = int(params['num_az_bins'])
    num_el_full = int(params['num_el_bins'])
    t_az = np.arange(-num_az_full//2 + 1, num_az_full//2) * (2.0 / num_az_full)
    t_el = np.arange(-num_el_full//2 + 1, num_el_full//2) * (2.0 / num_el_full)
    az_angles = np.arcsin(np.clip(t_az, -1.0 + 1e-6, 1.0 - 1e-6)).astype(np.float32)
    el_angles = np.arcsin(np.clip(t_el, -1.0 + 1e-6, 1.0 - 1e-6)).astype(np.float32)
    mag = rae  # (Az, El, R)
    Rg, Ag, Eg = np.meshgrid(np.arange(mag.shape[2]), np.arange(mag.shape[0]), np.arange(mag.shape[1]), indexing='ij')
    r_vals = (Rg.ravel().astype(np.float32)) * float(params['range_resolution'])
    az_vals = az_angles[Ag.ravel()]
    el_vals = el_angles[Eg.ravel()]
    x =  r_vals * np.cos(el_vals) * np.sin(az_vals)  # Fixed: removed negative sign
    y =  r_vals * np.cos(el_vals) * np.cos(az_vals)
    z =  r_vals * np.sin(el_vals)                     # Fixed: removed negative sign
    intensity = mag.transpose(2, 0, 1).ravel().astype(np.float32)
    rng = np.sqrt(x*x + y*y + z*z)
    keep = rng >= float(near_field_m)
    x = x[keep]; y = y[keep]; z = z[keep]; intensity = intensity[keep]
    keys = None
    if percentile is not None:
        p = float(np.clip(percentile, 0.0, 100.0))
        thr = float(np.percentile(intensity, p))
        keep2 = intensity >= thr if np.any(intensity >= thr) else intensity >= intensity.max() if intensity.size else np.array([], dtype=bool)
        x = x[keep2]; y = y[keep2]; z = z[keep2]; intensity = intensity[keep2]
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    return pts, intensity, keys


def load_and_align_lidar(lidar_pcl_path: Optional[str], cfg_center: Optional[np.ndarray]) -> Optional[o3d.geometry.PointCloud]:
    if o3d is None or lidar_pcl_path is None:
        return None
    pcd = load_point_cloud(lidar_pcl_path)
    if pcd is None or (not pcd.has_points()):
        return None
    # LiDAR is already in world coordinates (matching mesh frame), so do NOT translate
    # The voxelization code will handle coordinate transforms as needed
    return pcd


def voxelize_lidar_to_radar_grid(
    pcd: o3d.geometry.PointCloud,
    params: Dict[str, Any],
    near_field_m: float,
    percentile: float,
    cfg_bore: Optional[np.ndarray] = None,
    agg_method: str = 'median',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    az_cent, el_cent = make_angle_grids_np(int(params['num_az_bins']), int(params['num_el_bins']))
    az_edges = _centers_to_edges(az_cent, low_clip=-np.pi/2, high_clip=np.pi/2).astype(np.float32)
    el_edges = _centers_to_edges(el_cent, low_clip=-np.pi/2, high_clip=np.pi/2).astype(np.float32)
    r_edges = (np.arange(int(params['num_adc']) + 1, dtype=np.float32) * float(params['range_resolution']))
    pts_world = np.asarray(pcd.points, dtype=np.float32)
    # Apply inverse boresight to bring LiDAR into radar default (+Y) frame before binning
    if cfg_bore is not None:
        try:
            a = np.array([0.0, 1.0, 0.0], dtype=float)
            b = cfg_bore.astype(float)
            dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
            if abs(dot - 1.0) < 1e-8:
                R_inv = np.eye(3)
            elif abs(dot + 1.0) < 1e-8:
                axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                axis = axis - np.dot(axis, a) * a
                axis = axis / (np.linalg.norm(axis) + 1e-12)
                ux, uy, uz = axis
                c = -1.0; s = 0.0
                R_forward = np.array([
                    [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
                    [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),     uy*uz*(1-c) - ux*s],
                    [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s,  c + uz*uz*(1-c)]
                ], dtype=float)
                R_inv = R_forward.T
            else:
                axis = np.cross(a, b)
                axis = axis / (np.linalg.norm(axis) + 1e-12)
                ang = np.arccos(dot)
                ux, uy, uz = axis
                c = np.cos(ang); s = np.sin(ang)
                R_forward = np.array([
                    [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
                    [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),     uy*uz*(1-c) - ux*s],
                    [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s,  c + uz*uz*(1-c)]
                ], dtype=float)
                R_inv = R_forward.T
            pts_world = (pts_world @ R_inv.T).astype(np.float32)
        except Exception:
            pass
    intens = get_pcl_intensities(pcd).astype(np.float32)
    # Compute spherical in sensor default coordinates (assuming already centered)
    x = pts_world[:, 0]; y = pts_world[:, 1]; z = pts_world[:, 2]
    r = np.linalg.norm(pts_world, axis=1)
    keep = (r >= float(near_field_m)) & np.isfinite(r)
    x = x[keep]; y = y[keep]; z = z[keep]; r = r[keep]; intens = intens[keep]
    if x.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    az = np.arctan2(x, y)  # Fixed: removed negative sign to match spherical-to-Cartesian conversion
    el = np.arcsin(np.clip(z / np.clip(r, 1e-12, None), -1.0, 1.0))  # Fixed: removed negative sign
    az_idx = np.searchsorted(az_edges, az, side='right') - 1
    el_idx = np.searchsorted(el_edges, el, side='right') - 1
    r_idx  = np.searchsorted(r_edges,  r,  side='right') - 1
    A = az_cent.size; E = el_cent.size; Rb = int(params['num_adc'])
    # Strict validity mask (match 09_v2): discard out-of-range indices
    valid = (az_idx >= 0) & (az_idx < A) & (el_idx >= 0) & (el_idx < E) & (r_idx >= 0) & (r_idx < Rb)
    az_idx = az_idx[valid]; el_idx = el_idx[valid]; r_idx = r_idx[valid]; intens = intens[valid]
    if intens.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    keys = (az_idx.astype(np.int64) * E + el_idx.astype(np.int64)) * Rb + r_idx.astype(np.int64)
    order = np.argsort(keys)
    k_sorted = keys[order]
    v_sorted = intens[order]
    boundaries = np.where(np.diff(k_sorted) != 0)[0] + 1
    groups = np.split(v_sorted, boundaries)
    # Aggregate each voxel's intensities per requested method (default: median)
    agg_vals = np.array([float(np.median(g)) for g in groups], dtype=np.float32)
    unique_keys = k_sorted[np.r_[0, boundaries]]
    az_u = (unique_keys // (E * Rb)).astype(np.int32)
    rem = unique_keys % (E * Rb)
    el_u = (rem // Rb).astype(np.int32)
    r_u = (rem % Rb).astype(np.int32)
    az_c = az_cent[az_u].astype(np.float32)
    el_c = el_cent[el_u].astype(np.float32)
    r_c = (r_u.astype(np.float32) * float(params['range_resolution'])).astype(np.float32)
    x_s =  r_c * np.cos(el_c) * np.sin(az_c)  # Fixed: removed negative sign
    y_s =  r_c * np.cos(el_c) * np.cos(az_c)
    z_s =  r_c * np.sin(el_c)                  # Fixed: removed negative sign
    centers = np.stack([x_s, y_s, z_s], axis=1).astype(np.float32)
    return centers, agg_vals, unique_keys.astype(np.int64)


def load_mesh(path: Optional[str]) -> Optional['o3d.geometry.TriangleMesh']:
    if o3d is None or path is None:
        return None
    try:
        mesh_o3d = o3d.io.read_triangle_mesh(path)
        if mesh_o3d is not None and mesh_o3d.has_triangles():
            mesh_o3d.compute_vertex_normals()
            return mesh_o3d
    except Exception:
        return None
    return None


def run_single_view(
    adc_file: str,
    config_path: Optional[str],
    lidar_pcl_path: Optional[str],
    lidar_intensity_pcl_path: Optional[str],
    mesh_path: Optional[str],
    output_directory: Path,
    device: torch.device,
    near_field_m: float = 1.5,
    radar_threshold_percentile: Optional[float] = None,
    lidar_voxelize: bool = True,
    lidar_voxel_percentile: float = 0.0,
    lidar_color_by_intensity: bool = True,
    mesh_filter_enabled: bool = False,
    mesh_filter_eps_m: float = 0.05,
    mesh_filter_samples: int = 200000,
    wireframe_enabled: bool = False,
    wf_range_step: int = 16,
    wf_grid_step_az: int = 8,
    wf_grid_step_el: int = 8,
    radar_shift_to_config: bool = False,
    radar_orient_to_config: bool = False,
    show_radar: bool = True,
    show_intensity_voxelized: bool = False,
    show_intensity_points: bool = False,
    show_scene: bool = True,
    lidar_agg_method: str = 'median',
    save_ra_path: Optional[str] = None,
) -> Dict[str, Any]:
    # Load config-derived parameters
    params = load_runtime_config(config_path, output_directory)
    # Compute RAE cube and RA collapse
    rae = compute_rae_cube(Path(adc_file), params, device)

    # Load config geometry
    tx_m, rx_m, cfg_center, cfg_bore = load_config_positions_and_boresight(config_path)

    # Extract board frame azimuth direction and apply azimuth flip if needed
    # This ensures consistency between RA images and 3D visualizations
    if radar_orient_to_config and cfg_bore is not None and config_path is not None:
        try:
            board_azimuth, _, _ = extract_board_frame_from_config(config_path)

            # Flip RAE along azimuth axis when board azimuth points in +X direction
            # This compensates for the PCA sign ambiguity and ensures correct orientation
            # for both RA collapse and 3D point extraction
            if board_azimuth[0] >= 0:
                print(f"\n[RAE Orientation] Board azimuth: {board_azimuth}")
                print(f"[RAE Orientation] Flipping RAE along azimuth axis (azimuth[0]={board_azimuth[0]:.3f} >= 0)")
                rae = np.flip(rae, axis=0).copy()  # Flip along azimuth axis
            else:
                print(f"\n[RAE Orientation] Board azimuth: {board_azimuth}")
                print(f"[RAE Orientation] No RAE flip needed (azimuth[0]={board_azimuth[0]:.3f} < 0)")
        except Exception as e:
            print(f"[WARNING] Failed to extract board frame for RAE orientation: {e}")

    # Collapse RAE to RA map
    ra_map, az_deg, r_m = collapse_to_ra(rae, params)

    # Build radar point cloud
    radar_pts, radar_vals, _ = radar_points_from_rae(
        rae, params, near_field_m=near_field_m, percentile=radar_threshold_percentile
    )
    # Optional orientation to config boresight then translation to config center
    if (radar_orient_to_config or radar_shift_to_config) and radar_pts.shape[0] > 0:
        try:
            if radar_orient_to_config and cfg_bore is not None and config_path is not None:
                # Extract full board frame from antenna positions (PCA-based)
                # This recovers the exact horizontal/vertical axes from the dense array layout
                print("\n" + "="*80)
                print("ALIGNING RADAR TO BOARD FRAME (PCA-based)")
                print("="*80)

                try:
                    v2_azimuth, y_range, v1_elevation = extract_board_frame_from_config(config_path)

                    print(f"Board horizontal (azimuth): {v2_azimuth}")
                    print(f"Board boresight (range):    {y_range}")
                    print(f"Board vertical (elevation): {v1_elevation}")

                    # Build rotation matrix: columns are target frame axes
                    # Radar frame: X=azimuth, Y=range, Z=elevation
                    # Board frame: v2=azimuth, y_range=range, v1=elevation
                    R_align = np.column_stack([v2_azimuth, y_range, v1_elevation])

                    # Verify orthonormality (should be 1.0 for valid rotation matrix)
                    det = np.linalg.det(R_align)
                    print(f"Rotation matrix determinant: {det:.6f} (expect 1.0)")

                    if abs(det - 1.0) > 0.01:
                        print("  WARNING: Determinant far from 1.0, frame may not be orthonormal!")

                    # Apply rotation to radar points
                    radar_pts = (radar_pts @ R_align.T).astype(np.float32)
                    print("[PASS] Radar points rotated to board frame")
                    print("="*80)

                except Exception as e:
                    print(f"[ERROR] Failed to extract board frame: {e}")
                    import traceback
                    traceback.print_exc()
                    print("Falling back to identity rotation (no alignment)")
                    print("="*80)
            if radar_shift_to_config and cfg_center is not None:
                radar_pts = radar_pts + cfg_center.reshape(1, 3).astype(np.float32)
        except Exception:
            pass
    # Always create radar_pcd for artifacts (show_radar only controls visualization)
    radar_pcd = None
    if o3d is not None and radar_pts.shape[0] > 0:
        radar_pcd = build_point_cloud(
            radar_pts[:, 0], radar_pts[:, 1], radar_pts[:, 2], radar_vals,
            percentile=0.0 if radar_threshold_percentile is None else 0.0,
            color_by_intensity=True
        )

    # Load mesh (optional)
    mesh_o3d = load_mesh(mesh_path)

    # LiDAR processing
    lidar_geom = None
    lidar_keys_eval = np.zeros((0,), dtype=np.int64)
    lidar_vals_agg_eval = np.zeros((0,), dtype=np.float32)
    lidar_pcd_raw_eval = None
    # Intensity-specific LiDAR artifacts (single-frame intensity PCL)
    lidar_intensity_pcd_raw_eval = None
    lidar_intensity_pcd_voxelized = None
    lidar_intensity_keys_eval = np.zeros((0,), dtype=np.int64)
    lidar_intensity_vals_agg_eval = np.zeros((0,), dtype=np.float32)
    if lidar_pcl_path is not None and o3d is not None:
        print(f"[LiDAR] Voxelize enabled (agg={lidar_agg_method}, viz_percentile={lidar_voxel_percentile}, near_field={near_field_m}m)")
        # For voxelization, align by subtracting center before spherical (the upstream function assumes centered)
        # We load raw then subtract center for computation, then return world centers
        pcd = load_and_align_lidar(lidar_pcl_path, cfg_center)
        if pcd is not None and pcd.has_points():
            # Print AABB of aggregated LiDAR immediately after loading
            try:
                pts_np = np.asarray(pcd.points)
                bb_min = pts_np.min(axis=0)
                bb_max = pts_np.max(axis=0)
                print(f"[LiDAR] Aggregated PCL AABB min={bb_min.tolist()} max={bb_max.tolist()}")
            except Exception:
                pass
            # Save aligned raw LiDAR for evaluation even when voxelizing
            lidar_pcd_raw_eval = o3d.geometry.PointCloud(pcd)
            try:
                print(f"[LiDAR] Raw points loaded: {len(pcd.points)}")
            except Exception:
                pass
            try:
                pcd_cent = o3d.geometry.PointCloud(pcd)
                pcd_cent.translate((-cfg_center).tolist())
            except Exception:
                pcd_cent = pcd
            centers, agg_vals, keys = voxelize_lidar_to_radar_grid(
                pcd_cent, params, near_field_m=near_field_m, percentile=float(lidar_voxel_percentile), cfg_bore=cfg_bore, agg_method=str(lidar_agg_method)
            )
            print(f"[LiDAR] Voxel centers: {centers.shape[0]}")
            # Orient voxelized LiDAR to config boresight to match radar/wireframe (match 09_v2)
            if centers.shape[0] > 0 and cfg_bore is not None:
                try:
                    a = np.array([0.0, 1.0, 0.0], dtype=float)
                    b = cfg_bore.astype(float)
                    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
                    if abs(dot - 1.0) < 1e-8:
                        R_align = np.eye(3)
                    elif abs(dot + 1.0) < 1e-8:
                        axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                        axis = axis - np.dot(axis, a) * a
                        axis = axis / (np.linalg.norm(axis) + 1e-12)
                        ux, uy, uz = axis
                        c = -1.0; s = 0.0
                        R_align = np.array([
                            [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
                            [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),     uy*uz*(1-c) - ux*s],
                            [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s,  c + uz*uz*(1-c)]
                        ], dtype=float)
                    else:
                        axis = np.cross(a, b)
                        axis = axis / (np.linalg.norm(axis) + 1e-12)
                        ang = np.arccos(dot)
                        ux, uy, uz = axis
                        c = np.cos(ang); s = np.sin(ang)
                        R_align = np.array([
                            [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
                            [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),     uy*uz*(1-c) - ux*s],
                            [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s,  c + uz*uz*(1-c)]
                        ], dtype=float)
                    centers = (centers @ R_align.T).astype(np.float32)
                except Exception:
                    pass
            if cfg_center is not None and centers.shape[0] > 0:
                centers = centers + cfg_center.reshape(1, 3).astype(np.float32)
            if centers.shape[0] > 0:
                try:
                    print(f"[LiDAR] Rendering voxelized LiDAR (first center) {centers[0].tolist()} with {agg_vals.shape[0]} intensities")
                except Exception:
                    pass
                lidar_geom = build_point_cloud(
                    centers[:, 0], centers[:, 1], centers[:, 2], agg_vals,
                    # Apply visualization percentile like 09_v2
                    percentile=float(lidar_voxel_percentile), color_by_intensity=True
                )
                # Eval artifacts for LiDAR voxelized
                lidar_keys_eval = keys
                lidar_vals_agg_eval = agg_vals
            else:
                print("[LiDAR] No centers after voxelization (nothing to render)")

    wf_ls = None

    # Assemble scene geometries
    geoms: List[Any] = []
    if lidar_geom is not None:
        geoms.append(lidar_geom)
    # Optional visualization of intensity LiDAR artifacts
    try:
        print(f"[viz] Assembled geoms: radar={'yes' if radar_pcd is not None else 'no'}, lidar_vox={'yes' if lidar_geom is not None else 'no'}, intensity_raw={'yes' if (show_intensity_points and lidar_intensity_pcd_raw_eval is not None) else 'no'}, intensity_vox={'yes' if (show_intensity_voxelized and lidar_intensity_pcd_voxelized is not None) else 'no'}")
    except Exception:
        pass
    if mesh_o3d is not None:
        try:
            mesh_o3d.paint_uniform_color([0.7, 0.7, 0.7])
        except Exception:
            pass
        geoms.append(mesh_o3d)
    # Optional config overlay (TX/RX spheres and axes)
    if o3d is not None and tx_m is not None and rx_m is not None:
        try:
            for pos in tx_m:
                geoms.append(create_colored_sphere(pos, radius=0.02, color_rgba_255=[255, 165, 0, 255]))
            for pos in rx_m:
                geoms.append(create_colored_sphere(pos, radius=0.02, color_rgba_255=[0, 255, 0, 255]))
            if cfg_center is not None:
                axis_len = 0.5
                geoms.append(create_axis_arrow(cfg_center, cfg_center + np.array([axis_len, 0, 0]), [255, 0, 0]))
                geoms.append(create_axis_arrow(cfg_center, cfg_center + np.array([0, axis_len, 0]), [0, 255, 0]))
                geoms.append(create_axis_arrow(cfg_center, cfg_center + np.array([0, 0, axis_len]), [0, 0, 255]))
                # boresight in yellow if present
                if cfg_bore is not None:
                    geoms.append(create_axis_arrow(cfg_center, cfg_center + axis_len * cfg_bore, [255, 255, 0]))
        except Exception:
            pass

    # Assemble evaluation artifacts (radar keys/vals aggregated; LiDAR keys/vals if voxelized)
    # Compute radar voxel keys using the SAME binning as LiDAR (edges from centers) for alignment
    num_az_full = int(params['num_az_bins'])
    num_el_full = int(params['num_el_bins'])
    t_az = np.arange(-num_az_full // 2 + 1, num_az_full // 2) * (2.0 / num_az_full)
    t_el = np.arange(-num_el_full // 2 + 1, num_el_full // 2) * (2.0 / num_el_full)
    az_angles = np.arcsin(np.clip(t_az, -1.0 + 1e-6, 1.0 - 1e-6)).astype(np.float32)
    el_angles = np.arcsin(np.clip(t_el, -1.0 + 1e-6, 1.0 - 1e-6)).astype(np.float32)
    mag_eval = rae  # (Az, El, R)
    Rg, Ag, Eg = np.meshgrid(np.arange(mag_eval.shape[2]), np.arange(mag_eval.shape[0]), np.arange(mag_eval.shape[1]), indexing='ij')
    r_vals = (Rg.ravel().astype(np.float32)) * float(params['range_resolution'])
    az_vals = az_angles[Ag.ravel()]
    el_vals = el_angles[Eg.ravel()]
    x =  r_vals * np.cos(el_vals) * np.sin(az_vals)  # Fixed: removed negative sign
    y =  r_vals * np.cos(el_vals) * np.cos(az_vals)
    z =  r_vals * np.sin(el_vals)                     # Fixed: removed negative sign
    intensity = mag_eval.transpose(2, 0, 1).ravel().astype(np.float32)
    rng = np.sqrt(x*x + y*y + z*z)
    keep_nf = rng >= float(near_field_m)
    x = x[keep_nf]; y = y[keep_nf]; z = z[keep_nf]; intensity = intensity[keep_nf]
    # Build BOTH thresholded and un-thresholded radar intensity sets (for fair evaluation)
    thr = float(np.percentile(intensity, float(radar_threshold_percentile)))
    keep_thr = intensity >= thr if np.any(intensity >= thr) else intensity >= intensity.max()
    intensity_thr = intensity[keep_thr]
    x_thr = x[keep_thr]; y_thr = y[keep_thr]; z_thr = z[keep_thr]
    # Bin with same edges as LiDAR
    az_cent, el_cent = make_angle_grids_np(num_az_full, num_el_full)
    az_edges = _centers_to_edges(az_cent, low_clip=-np.pi/2, high_clip=np.pi/2).astype(np.float32)
    el_edges = _centers_to_edges(el_cent, low_clip=-np.pi/2, high_clip=np.pi/2).astype(np.float32)
    r_edges = (np.arange(int(params['num_adc']) + 1, dtype=np.float32) * float(params['range_resolution']))
    # Keys for thresholded subset
    r = np.sqrt(x_thr*x_thr + y_thr*y_thr + z_thr*z_thr)
    az = np.arctan2(-x_thr, y_thr)
    el = np.arcsin(np.clip(-z_thr / np.clip(r, 1e-12, None), -1.0, 1.0))
    az_idx = np.searchsorted(az_edges, az, side='right') - 1
    el_idx = np.searchsorted(el_edges, el, side='right') - 1
    r_idx  = np.searchsorted(r_edges,  r,  side='right') - 1
    A = az_cent.size; E = el_cent.size; Rb = int(params['num_adc'])
    valid = (az_idx >= 0) & (az_idx < A) & (el_idx >= 0) & (el_idx < E) & (r_idx >= 0) & (r_idx < Rb)
    az_idx = az_idx[valid]; el_idx = el_idx[valid]; r_idx = r_idx[valid]; intensity_thr = intensity_thr[valid]
    radar_keys = (az_idx.astype(np.int64) * E + el_idx.astype(np.int64)) * Rb + r_idx.astype(np.int64)
    order_r = np.argsort(radar_keys)
    k_sorted_r = radar_keys[order_r]
    v_sorted_r = intensity_thr[order_r]
    boundaries_r = np.where(np.diff(k_sorted_r) != 0)[0] + 1
    groups_r = np.split(v_sorted_r, boundaries_r)
    radar_vals_agg = np.array([float(np.percentile(g, 95.0)) for g in groups_r], dtype=np.float32)
    radar_keys_uniq = k_sorted_r[np.r_[0, boundaries_r]].astype(np.int64)

    # Also compute un-thresholded radar keys/vals ONLY at LiDAR voxel keys to reduce workload
    try:
        target_keys = None
        if 'lidar_intensity_keys_eval' in locals() and lidar_intensity_keys_eval is not None and lidar_intensity_keys_eval.size > 0:
            target_keys = lidar_intensity_keys_eval.astype(np.int64)
            print(f"[eval] Using intensity LiDAR keys for radar pairing: {int(target_keys.size)} keys")
        elif 'lidar_keys_eval' in locals() and lidar_keys_eval is not None and lidar_keys_eval.size > 0:
            target_keys = lidar_keys_eval.astype(np.int64)
            print(f"[eval] Using aggregated LiDAR voxel keys for radar pairing: {int(target_keys.size)} keys")
        else:
            target_keys = np.zeros((0,), dtype=np.int64)
            print("[eval] No LiDAR voxel keys available for radar pairing; defaulting to empty")
        az_idx_f = (target_keys // (E * Rb)).astype(np.int64)
        rem_f = (target_keys % (E * Rb)).astype(np.int64)
        el_idx_f = (rem_f // Rb).astype(np.int64)
        r_idx_f = (rem_f % Rb).astype(np.int64)
        # Strict validity
        valid_f = (az_idx_f >= 0) & (az_idx_f < A) & (el_idx_f >= 0) & (el_idx_f < E) & (r_idx_f >= 0) & (r_idx_f < Rb)
        az_idx_f = az_idx_f[valid_f]; el_idx_f = el_idx_f[valid_f]; r_idx_f = r_idx_f[valid_f]
        radar_keys_uniq_full = target_keys[valid_f].astype(np.int64)
        # Direct lookup from RAE magnitude (Az, El, R)
        try:
            radar_vals_agg_full = rae[az_idx_f, el_idx_f, r_idx_f].astype(np.float32)
        except Exception:
            # Fallback via transposed mag if needed
            radar_vals_agg_full = rae.astype(np.float32)[az_idx_f, el_idx_f, r_idx_f]
        # Build radar XYZ centers corresponding to these keys (apply same orientation/translation)
        try:
            az_cent_f, el_cent_f = make_angle_grids_np(num_az_full, num_el_full)
            az_c = az_cent_f[az_idx_f].astype(np.float32)
            el_c = el_cent_f[el_idx_f].astype(np.float32)
            r_c = (r_idx_f.astype(np.float32) * float(params['range_resolution'])).astype(np.float32)
            x_r =  r_c * np.cos(el_c) * np.sin(az_c)  # Fixed: removed negative sign
            y_r =  r_c * np.cos(el_c) * np.cos(az_c)
            z_r =  r_c * np.sin(el_c)                  # Fixed: removed negative sign
            centers_r = np.stack([x_r, y_r, z_r], axis=1).astype(np.float32)
            # Forward align to config boresight and translate to config center to match LiDAR world frame
            if cfg_bore is not None and centers_r.shape[0] > 0:
                try:
                    a = np.array([0.0, 1.0, 0.0], dtype=float)
                    b = cfg_bore.astype(float)
                    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
                    if abs(dot - 1.0) < 1e-8:
                        R_align_r = np.eye(3)
                    elif abs(dot + 1.0) < 1e-8:
                        axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                        axis = axis - np.dot(axis, a) * a
                        axis = axis / (np.linalg.norm(axis) + 1e-12)
                        ux, uy, uz = axis
                        c = -1.0; s = 0.0
                        R_align_r = np.array([
                            [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
                            [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),     uy*uz*(1-c) - ux*s],
                            [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s,  c + uz*uz*(1-c)]
                        ], dtype=float)
                    else:
                        axis = np.cross(a, b)
                        axis = axis / (np.linalg.norm(axis) + 1e-12)
                        ang = np.arccos(dot)
                        ux, uy, uz = axis
                        c = np.cos(ang); s = np.sin(ang)
                        R_align_r = np.array([
                            [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
                            [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),     uy*uz*(1-c) - ux*s],
                            [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s,  c + uz*uz*(1-c)]
                        ], dtype=float)
                    centers_r = (centers_r @ R_align_r.T).astype(np.float32)
                except Exception:
                    pass
            if cfg_center is not None and centers_r.shape[0] > 0:
                centers_r = centers_r + cfg_center.reshape(1, 3).astype(np.float32)
            paired_radar_xyz_for_viz = centers_r
            paired_radar_vals_for_viz = radar_vals_agg_full
        except Exception:
            paired_radar_xyz_for_viz = np.zeros((0,3), dtype=np.float32)
            paired_radar_vals_for_viz = np.zeros((0,), dtype=np.float32)
    except Exception as e:
        print(f"[warn] Failed to build paired radar intensities: {e}")
        radar_keys_uniq_full = np.zeros((0,), dtype=np.int64)
        radar_vals_agg_full = np.zeros((0,), dtype=np.float32)
        paired_radar_xyz_for_viz = np.zeros((0,3), dtype=np.float32)
        paired_radar_vals_for_viz = np.zeros((0,), dtype=np.float32)

    return {
        'radar_pcd': radar_pcd if (o3d is None or radar_pcd is not None) else o3d.geometry.PointCloud(),
        'lidar_pcd_raw': lidar_pcd_raw_eval if (o3d is None or lidar_pcd_raw_eval is not None) else o3d.geometry.PointCloud(),
        'lidar_pcd_voxelized': lidar_geom if (o3d is None or lidar_geom is not None) and lidar_voxelize else (o3d.geometry.PointCloud() if o3d is not None else None),
        'radar_keys': radar_keys_uniq,
        'lidar_keys': lidar_keys_eval,
        'radar_vals_agg': radar_vals_agg,
        'radar_keys_full': radar_keys_uniq_full,
        'radar_vals_agg_full': radar_vals_agg_full,
        'lidar_vals_agg': lidar_vals_agg_eval,
        # Intensity LiDAR artifacts
        'lidar_intensity_pcd_raw': lidar_intensity_pcd_raw_eval if (o3d is None or lidar_intensity_pcd_raw_eval is not None) else o3d.geometry.PointCloud(),
        'lidar_intensity_pcd_voxelized': lidar_intensity_pcd_voxelized if (o3d is None or lidar_intensity_pcd_voxelized is not None) else o3d.geometry.PointCloud(),
        'lidar_intensity_keys': lidar_intensity_keys_eval,
        'lidar_intensity_vals_agg': lidar_intensity_vals_agg_eval,
        'mesh': mesh_o3d,
        'params': params,
        # Paired visualization arrays (world coordinates)
        'paired_lidar_xyz': np.zeros((0, 3), dtype=np.float32),
        'paired_lidar_vals': np.zeros((0,), dtype=np.float32),
        'paired_radar_xyz': paired_radar_xyz_for_viz,
        'paired_radar_vals': paired_radar_vals_for_viz,
    }
