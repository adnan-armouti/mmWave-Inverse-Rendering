import json
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better figure management
from pathlib import Path
from typing import Dict, Tuple
import open3d as o3d
from .dense_array import generate_uniform_virtual_antennas
# Cached Hann windows by (length, device)
_HANN_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}

def get_hann(n: int, device: torch.device) -> torch.Tensor:
    key = (int(n), str(device))
    t = _HANN_CACHE.get(key)
    if t is None:
        t = torch.tensor(np.hanning(n), device=device, dtype=torch.float32)
        _HANN_CACHE[key] = t
    return t

def to_tensor(x, device: torch.device, dtype=torch.float32, ):
    """Convert numpy array to torch tensor on the correct device."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def txrx_to_vx_chirps_dense_gpu_batched(adc_batch_t: torch.Tensor, num_ant: int = 50) -> torch.Tensor:
    """Vectorized TX-RX -> virtual array mapping for a batch on GPU. adc_batch_t: (B, N_Rx, N_Tx,
    N_ADC) complex64 returns: (B, Vy, Vx, N_ADC) complex64
    """
    virtual_antennas, tx_locs, rx_locs = generate_uniform_virtual_antennas(num_ant)
    vx_x = virtual_antennas[:, 0]
    vx_y = virtual_antennas[:, 1]
    min_x, max_x = vx_x.min(), vx_x.max()
    min_y, max_y = vx_y.min(), vx_y.max()
    x_off = int(-min_x)
    y_off = int(-min_y)
    Vx = int(max_x - min_x + 1)
    Vy = int(max_y - min_y + 1)

    # Build pair indices
    pair_rx = []
    pair_tx = []
    pair_lin = []
    for rx_id, (rx_x, rx_y) in enumerate(rx_locs):
        for tx_id, (tx_x, tx_y) in enumerate(tx_locs):
            vx_xi = int(rx_x + tx_x + x_off)
            vx_yi = int(rx_y + tx_y + y_off)
            lin = vx_yi * Vx + vx_xi
            pair_rx.append(rx_id)
            pair_tx.append(tx_id)
            pair_lin.append(lin)

    dev = adc_batch_t.device
    pair_rx = torch.tensor(pair_rx, device=dev, dtype=torch.long)
    pair_tx = torch.tensor(pair_tx, device=dev, dtype=torch.long)
    pair_lin = torch.tensor(pair_lin, device=dev, dtype=torch.long)
    counts = torch.bincount(pair_lin, minlength=Vx * Vy).to(device=dev)
    counts = counts.clamp_min_(1)

    B, NR, NT, NADC = adc_batch_t.shape
    adc_pairs = adc_batch_t[:, pair_rx, pair_tx, :]  # (B, P, NADC)
    out_real = torch.zeros((B, Vx * Vy, NADC), device=dev, dtype=torch.float32)
    out_imag = torch.zeros_like(out_real)
    lin_idx = pair_lin.view(1, -1, 1).expand(B, -1, NADC)
    out_real = out_real.scatter_add(1, lin_idx, adc_pairs.real)
    out_imag = out_imag.scatter_add(1, lin_idx, adc_pairs.imag)
    cnts = counts.view(1, -1, 1)
    out_real = out_real / cnts
    out_imag = out_imag / cnts
    out = torch.complex(out_real, out_imag).view(B, Vy, Vx, NADC)
    return out


# Verbatim helper from 05_visualize_voxels.py
def build_point_cloud(x: np.ndarray,
                      y: np.ndarray,
                      z: np.ndarray,
                      values: np.ndarray | None,
                      percentile: float,
                      color_by_intensity: bool) -> o3d.geometry.PointCloud:
    """Create an Open3D PointCloud from arrays, applying percentile filtering and coloring."""
    # Filter by percentile
    if values is not None and values.size == x.size and values.size > 0:
        p = float(np.clip(percentile, 0.0, 100.0))
        thr = float(np.percentile(values, p))
        keep = values >= thr
        x, y, z = x[keep], y[keep], z[keep]
        values = values[keep]

    pts = np.column_stack([x, y, z])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    if color_by_intensity and values is not None and values.size == x.size:
        from matplotlib import cm
        vmin = float(values.min()) if values.size > 0 else 0.0
        vmax = float(values.max()) if values.size > 0 else 1.0
        if vmax <= vmin:
            vals_n = np.ones_like(values, dtype=np.float32)
        else:
            vals_n = np.clip((values - vmin) / (vmax - vmin + 1e-12), 0, 1)
        cols = cm.get_cmap('plasma')(vals_n)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(cols)

    return pcd


# LiDAR point cloud helper (from 09_viz_lidar_mesh_pcl.py)
def load_point_cloud(pcl_path: str) -> o3d.geometry.PointCloud | None:
    if o3d is None:
        print("Error: open3d is required to visualize geometries. pip install open3d")
        return None
    try:
        import os as _os
        ext = _os.path.splitext(pcl_path)[1].lower()

        # Handle .npy files separately
        if ext == '.npy':
            data = np.load(pcl_path)
            if data.shape[1] < 3:
                print(f"[warn] .npy file has insufficient columns: {data.shape}")
                return None
            # Extract XYZ (first 3 columns)
            points = data[:, :3].astype(np.float64)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            # If there's a 4th column, treat it as intensity
            if data.shape[1] >= 4:
                intensity = data[:, 3].astype(np.float32)
                try:
                    setattr(pcd, 'intensities', intensity)
                    print(f"[loader] Loaded .npy with intensity: n={len(points)}, intensity range=[{float(intensity.min()):.6f}, {float(intensity.max()):.6f}]")
                except Exception:
                    print(f"[loader] Loaded .npy: n={len(points)} (intensity cannot be attached as attribute, skipping)")
            else:
                print(f"[loader] Loaded .npy: n={len(points)} (no intensity)")
            # Assign default color if no colors
            if not pcd.has_colors():
                pcd.paint_uniform_color([0.1, 0.8, 1.0])
            return pcd

        # For other formats, use Open3D's loader
    except Exception as e:
        print(f"[warn] Failed to load point cloud '{pcl_path}': {e}")
        return None


# Extract per-point intensity from an Open3D point cloud
def get_pcl_intensities(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    n = len(pcd.points)
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    # If Open3D exposes intensities (not always present), use them
    # Fallback: derive grayscale intensity from colors if present
    if pcd.has_colors():
        cols = np.asarray(pcd.colors, dtype=np.float32)
        if cols.ndim == 2 and cols.shape[0] == n and cols.shape[1] >= 3:
            r, g, b = cols[:, 0], cols[:, 1], cols[:, 2]
            gray = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)
            try:
                if gray.size > 0:
                    print(f"[loader] get_pcl_intensities: using color->gray; min/max/mean={float(gray.min()):.6f}/{float(gray.max()):.6f}/{float(gray.mean()):.6f}")
            except Exception:
                pass
            return gray
    # Default: ones


# Helpers analogous to 11_lidar_visualization_new.py, adapted for Open3D


def create_colored_sphere(center: np.ndarray, radius: float, color_rgba_255: list):
    sph = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius))
    sph.compute_vertex_normals()
    # Normalize 0-255 to 0-1 if needed
    if len(color_rgba_255) >= 3 and max(color_rgba_255[:3]) > 1.0:
        col = [c / 255.0 for c in color_rgba_255[:3]]
    else:
        col = color_rgba_255[:3]
    sph.paint_uniform_color(col)
    sph.translate(np.asarray(center, dtype=float).tolist())
    return sph


def create_axis_arrow(start: np.ndarray, end: np.ndarray, color_rgba_255: list):
    pts = o3d.utility.Vector3dVector(np.vstack([np.asarray(start, dtype=float), np.asarray(end, dtype=float)]))
    lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
    ls = o3d.geometry.LineSet(points=pts, lines=lines)
    if len(color_rgba_255) >= 3 and max(color_rgba_255[:3]) > 1.0:
        col = [color_rgba_255[0] / 255.0, color_rgba_255[1] / 255.0, color_rgba_255[2] / 255.0]
    else:
        col = color_rgba_255[:3]
    ls.colors = o3d.utility.Vector3dVector(np.array([col], dtype=float))
    return ls


def _compute_range_resolution_from_config(cfg_path: Path) -> float:
    """Compute range resolution DeltaR = c / (2 * S * (rampEndTime - adcStartTime))."""
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    c = 299_792_458.0
    slope_hz_per_s = float(cfg["freqSlope"])  # Hz/s
    if slope_hz_per_s <= 0:
        raise ValueError(f"Invalid freqSlope in config: {slope_hz_per_s}")
    if "rampEndTime" not in cfg or "adcStartTime" not in cfg:
        raise KeyError("Config must contain 'rampEndTime' and 'adcStartTime' to compute DeltaR = c/(2*B)")
    T_chirp = float(cfg["rampEndTime"]) - float(cfg["adcStartTime"])  # s
    if T_chirp <= 0:
        raise ValueError(f"Invalid chirp timing: rampEndTime - adcStartTime = {T_chirp}")
    bandwidth = slope_hz_per_s * T_chirp      # Hz
    dr = c / (2.0 * bandwidth)
    return float(dr)


# RA-style angle grids (DC-removed centers) as used for wireframes/radar
def make_angle_grids_np(az_bins_full: int, el_bins_full: int):
    eps = 1e-6
    t_az = np.arange(-az_bins_full // 2 + 1, az_bins_full // 2, dtype=np.float64) * (2.0 / float(az_bins_full))
    t_el = np.arange(-el_bins_full // 2 + 1, el_bins_full // 2, dtype=np.float64) * (2.0 / float(el_bins_full))
    t_az = np.clip(t_az, -1.0 + eps, 1.0 - eps)
    t_el = np.clip(t_el, -1.0 + eps, 1.0 - eps)
    return np.arcsin(t_az), np.arcsin(t_el)

def _centers_to_edges(centers: np.ndarray, low_clip: float, high_clip: float) -> np.ndarray:
    centers = centers.astype(np.float64)
    if centers.size == 1:
        width = 1e-3
        return np.array([centers[0] - width, centers[0] + width], dtype=np.float64)
    diffs = np.diff(centers)
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = centers[:-1] + 0.5 * diffs
    edges[0] = centers[0] - 0.5 * diffs[0]
    edges[-1] = centers[-1] + 0.5 * diffs[-1]
    edges[0] = max(edges[0], low_clip)
    edges[-1] = min(edges[-1], high_clip)
    return edges
