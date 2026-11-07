#!/usr/bin/env python3
"""Generate a uniform virtual antenna array (using the provided function EXACTLY), plot the 2D
layout, map TX/RX onto the old board plane (orthogonal to boresight) with HALF-WAVELENGTH
spacing, translate the new board so its geometric center matches the old board's center, export
an updated config JSON with new TX/RX elements (keeping all other settings), and visualize old vs
new configs in 3D.
"""

import json
import os
import numpy as np

# EXACT function (do not modify)
def generate_uniform_virtual_antennas(num_antennas, spacing=1.0):
    """Generate uniform virtual antenna positions based on a single antenna count. Args:
    num_antennas: Total number of transmitters and receivers (must be even) num_tx = num_rx =
    num_antennas spacing: Spacing between virtual antenna positions (default: 1.0) Returns:
    tuple: (virtual_antennas, tx_locations, rx_locations) - virtual_antennas: set of (x, y)
    tuples - tx_locations: list of (x, y) tuples for transmitters - rx_locations: list of (x, y)
    tuples for receivers Raises: AssertionError: If num_antennas is not even
    """

    # Step 1: Assert that input is an even number
    assert num_antennas % 2 == 0, f"Number of antennas ({num_antennas}) must be an even number"

    # Set num_tx = num_rx = num_antennas
    total_tx = num_antennas
    total_rx = num_antennas

    # Step 2: Divide by 2 (receivers form two rows and transmitters form two columns)
    tx_per_column = total_tx // 2
    rx_per_row = total_rx // 2

    # Step 3: Establish transmitter and receiver locations
    tx_locations = []
    rx_locations = []

    # Transmitter positions (two columns: left and right)
    # Left column: x=0.5, y from 1 to tx_per_column
    # Right column: x=tx_per_column+0.5, y from 1 to tx_per_column
    for y in (np.arange(-(tx_per_column-1)/2, (tx_per_column)/2, 1)):
        # Left column
        tx_locations.append((-(tx_per_column/2), y))
        # Right column (positioned to create uniform spacing)
        tx_locations.append(((tx_per_column/2), y))

    # Receiver positions (two rows: bottom and top)
    # Bottom row: y=0, x from 0.5 to rx_per_row-0.5
    # Top row: y=rx_per_row, x from 0.5 to rx_per_row-0.5
    for x in (np.arange(-(rx_per_row-1)/2, (rx_per_row)/2, 1)):
        # Bottom row
        rx_locations.append((x, -(tx_per_column/2)))
        # Top row (positioned to create uniform spacing)
        rx_locations.append((x, (tx_per_column/2)))

    # Step 4: Map to uniformly spaced virtual antenna positions
    virtual_antennas = []

    for rx_loc in rx_locations:
        for tx_loc in tx_locations:
            vx_x = rx_loc[0] + tx_loc[0]
            vx_y = rx_loc[1] + tx_loc[1]
            virtual_antennas.append((vx_x, vx_y))

    return np.array(virtual_antennas), np.array(tx_locations), np.array(rx_locations)

# Utilities

def calculate_wavelength(carrier_frequency_hz: float) -> float:
    c = 299792458.0
    return c / carrier_frequency_hz

def load_config(path: str):
    with open(path, 'r') as f:
        return json.load(f)

def get_geometric_center_and_boresight(cfg: dict):
    tx = np.array([t['pos_mm'] for t in cfg['tx_array']]) / 1000.0
    rx = np.array([r['pos_mm'] for r in cfg['rx_array']]) / 1000.0
    all_pos = np.vstack([tx, rx])
    center = np.mean(all_pos, axis=0)
    # Assume common boresight for all
    boresight = np.array(cfg['tx_array'][0]['boresight'], dtype=float)
    boresight /= np.linalg.norm(boresight)
    return center, boresight

def plane_basis_from_normal(n: np.ndarray):
    # Find a vector least aligned with n
    axes = [np.array([1.0,0.0,0.0]), np.array([0.0,1.0,0.0]), np.array([0.0,0.0,1.0])]
    dots = [abs(np.dot(n, a)) for a in axes]
    v1 = axes[int(np.argmin(dots))]
    # Project to plane and normalize
    v1 = v1 - np.dot(v1, n) * n
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(n, v1)
    v2 /= np.linalg.norm(v2)
    return v1, v2

def map_2d_to_plane(points_2d, center3d_m: np.ndarray, basis_v1: np.ndarray, basis_v2: np.ndarray, scale_m: float):
    mapped = []
    for (x, y) in points_2d:
        # Fix the orientation: swap basis vectors to preserve intended layout
        # TX column (vertical) should map to Y direction, RX row (horizontal) should map to X direction
        mapped.append(center3d_m + scale_m * (y * basis_v1 + x * basis_v2))
    return np.array(mapped)

# Callable API

def generate_dense_config(cascaded_config_path: str, output_path: str,
                          num_antennas: int = 100, verbose: bool = True) -> str:
    """Generate a dense 100×100 virtual array config from a cascaded config."""
    cfg = load_config(cascaded_config_path)
    center_m, boresight_n = get_geometric_center_and_boresight(cfg)
    v1, v2 = plane_basis_from_normal(boresight_n)

    wavelength_m = calculate_wavelength(cfg['carrierFrequency'])
    half_wavelength_m = 0.5 * wavelength_m

    _, tx_locations, rx_locations = generate_uniform_virtual_antennas(num_antennas)

    tx_3d = map_2d_to_plane(tx_locations, center_m, v1, v2, half_wavelength_m)
    rx_3d = map_2d_to_plane(rx_locations, center_m, v1, v2, half_wavelength_m)

    # Translate so new board center matches old
    new_center = np.mean(np.vstack([tx_3d, rx_3d]), axis=0)
    delta = center_m - new_center
    tx_3d = tx_3d + delta
    rx_3d = rx_3d + delta

    # Build output config (deep copy, replace arrays)
    new_cfg = json.loads(json.dumps(cfg))

    new_cfg['tx_array'] = [
        {
            "name": f"TX{i+1}",
            "pos_mm": (pos_m * 1000.0).tolist(),
            "boresight": boresight_n.tolist(),
            "radius": 1.0,
            "polarization": "V",
        }
        for i, pos_m in enumerate(tx_3d)
    ]
    new_cfg['rx_array'] = [
        {
            "name": f"RX{i+1}",
            "pos_mm": (pos_m * 1000.0).tolist(),
            "boresight": boresight_n.tolist(),
            "power_dBm": 10.0,
            "polarization": "V",
        }
        for i, pos_m in enumerate(rx_3d)
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(new_cfg, f, indent=2)

    if verbose:
        print(f"[generate_dense_config] Saved dense {num_antennas}×{num_antennas} config")
        print(f"  Input: {cascaded_config_path}")
        print(f"  Output: {output_path}")
        print(f"  λ/2 spacing: {half_wavelength_m*1000:.3f} mm")

    return output_path
