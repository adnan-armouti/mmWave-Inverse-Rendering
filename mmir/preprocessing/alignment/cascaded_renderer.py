#!/usr/bin/env python3
"""Renderer-Based 2DOF Alignment (range + azimuth only, NO boresight rotation)."""

import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

import drjit as dr
dr.set_flag(dr.JitFlag.VCallOptimize, False)
dr.set_flag(dr.JitFlag.LoopOptimize, False)

import os
import json
import time
import numpy as np
import torch
from scipy.optimize import minimize

from mmir.data.ra_utils import adc_to_ra_image, ra_polar_to_cartesian

# repository root (alignment → preprocessing → mmir → repository)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TX_PATTERN = os.path.join(PROJECT_ROOT, "assets", "antenna_pattern", "MMWCAS", "tx1_76.npy")
RX_PATTERN = os.path.join(PROJECT_ROOT, "assets", "antenna_pattern", "MMWCAS", "rx1_76.npy")

SCENES = [
    ("seq_0_frame_135", 135),
    ("seq_0_frame_390", 390),
    ("seq_1_frame_185", 185),
    ("seq_1_frame_438", 438),
    ("seq_2_frame_105", 105),
    ("seq_2_frame_160", 160),
    ("seq_2_frame_300", 300),
]


# Geometry helpers (from renderer_alignment_refinement.py)

def _norm(v, eps=1e-9):
    return v / (np.linalg.norm(v) + eps)


def apply_2dof_to_config(base_config, delta_range_m, delta_azimuth_deg):
    """Apply 2DOF transform (range + azimuth) with ZERO rotation."""
    tx_pos = np.array([t["pos_mm"] for t in base_config["tx_array"]], dtype=float) / 1000.0
    rx_pos = np.array([r["pos_mm"] for r in base_config["rx_array"]], dtype=float) / 1000.0
    all_pos = np.vstack([tx_pos, rx_pos])
    n_tx = len(tx_pos)

    board_center = np.mean(all_pos, axis=0)
    boresight = _norm(np.array(base_config["tx_array"][0]["boresight"], dtype=float))

    # No rotation — use original boresight and positions directly
    rel_pos = all_pos - board_center

    # Translation only
    range_offset = boresight * delta_range_m
    reference_range_m = 10.0
    delta_azimuth_m = reference_range_m * np.radians(delta_azimuth_deg)
    z_world = np.array([0.0, 0.0, 1.0])
    x_board = _norm(np.cross(boresight, z_world))
    if np.linalg.norm(x_board) < 1e-6:
        x_board = np.array([1.0, 0.0, 0.0])
    azimuth_offset = x_board * delta_azimuth_m
    translation = range_offset + azimuth_offset

    new_board_center = board_center + translation
    new_all_pos = rel_pos + new_board_center

    new_tx_pos_mm = new_all_pos[:n_tx] * 1000.0
    new_rx_pos_mm = new_all_pos[n_tx:] * 1000.0

    return new_tx_pos_mm, new_rx_pos_mm, boresight


def update_renderer_antennas(renderer, tx_pos_mm, rx_pos_mm, boresight):
    """Update renderer antenna positions and boresights in-place."""
    scene_ctx = renderer.scene_ctx
    n_tx = tx_pos_mm.shape[0]
    n_rx = rx_pos_mm.shape[0]

    tx_pos_m = tx_pos_mm / 1000.0
    rx_pos_m = rx_pos_mm / 1000.0

    scene_ctx.tx_array.positions = mi.Point3f(
        mi.Float(tx_pos_m[:, 0].tolist()),
        mi.Float(tx_pos_m[:, 1].tolist()),
        mi.Float(tx_pos_m[:, 2].tolist()),
    )
    scene_ctx.tx_array.orientations = mi.Vector3f(
        mi.Float([boresight[0]] * n_tx),
        mi.Float([boresight[1]] * n_tx),
        mi.Float([boresight[2]] * n_tx),
    )
    scene_ctx.rx_array.positions = mi.Point3f(
        mi.Float(rx_pos_m[:, 0].tolist()),
        mi.Float(rx_pos_m[:, 1].tolist()),
        mi.Float(rx_pos_m[:, 2].tolist()),
    )
    scene_ctx.rx_array.orientations = mi.Vector3f(
        mi.Float([boresight[0]] * n_rx),
        mi.Float([boresight[1]] * n_rx),
        mi.Float([boresight[2]] * n_rx),
    )


# RA processing helpers

def minmax_normalize(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-30:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def adc_to_ra_cart(adc_ri, range_res):
    adc_t = torch.from_numpy(adc_ri).float()
    ra_polar = adc_to_ra_image(adc_t).numpy()
    return ra_polar_to_cartesian(ra_polar, range_res)


def compute_cart_corr(ra1_cart, ra2_cart):
    n1 = minmax_normalize(ra1_cart)
    n2 = minmax_normalize(ra2_cart)
    return float(np.corrcoef(n1.ravel(), n2.ravel())[0, 1])


def load_gt_ra_cart(gt_adc_path, range_res):
    gt_raw = np.load(gt_adc_path)
    if gt_raw.ndim == 4:
        gt_chirp0 = gt_raw[0].transpose(1, 0, 2)
    elif gt_raw.ndim == 3:
        gt_chirp0 = gt_raw
    else:
        raise ValueError(f"Unexpected GT shape: {gt_raw.shape}")
    gt_ri = np.stack([gt_chirp0.real, gt_chirp0.imag], axis=-1).astype(np.float32)
    return adc_to_ra_cart(gt_ri, range_res)


# Core: render, grid search, refine

def render_and_evaluate(renderer, range_res, ra_gt_cart, seed=42):
    """Render with current antenna config, return cart_corr."""
    result = renderer.render(seed=seed)
    adc_ri = result.adc_result.get_ri_array()
    ra_rend_cart = adc_to_ra_cart(adc_ri, range_res)
    cc = compute_cart_corr(ra_gt_cart, ra_rend_cart)
    del result, adc_ri
    dr.sync_thread()
    return cc


def grid_search_2dof(renderer, base_config, range_res, ra_gt_cart,
                     range_min=-2.0, range_max=2.0, range_step=0.25,
                     az_min=-10.0, az_max=10.0, az_step=1.0,
                     seed=42):
    """Grid search over range and azimuth with zero rotation."""
    range_grid = np.arange(range_min, range_max + range_step / 2, range_step)
    az_grid = np.arange(az_min, az_max + az_step / 2, az_step)
    total = len(range_grid) * len(az_grid)

    print(f"  Grid search: {len(range_grid)} range x {len(az_grid)} azimuth = {total} evaluations")

    best_cc = -999.0
    best_dr = 0.0
    best_da = 0.0
    all_results = []
    count = 0
    t0 = time.time()

    for dr_m in range_grid:
        for da_deg in az_grid:
            count += 1
            tx_mm, rx_mm, bs = apply_2dof_to_config(base_config, dr_m, da_deg)
            update_renderer_antennas(renderer, tx_mm, rx_mm, bs)
            cc = render_and_evaluate(renderer, range_res, ra_gt_cart, seed=seed)
            all_results.append((float(dr_m), float(da_deg), float(cc)))

            if cc > best_cc:
                best_cc = cc
                best_dr = dr_m
                best_da = da_deg

            if count % 50 == 0 or count == total:
                elapsed = time.time() - t0
                rate = count / elapsed
                eta = (total - count) / rate if rate > 0 else 0
                print(f"    [{count}/{total}] best CC={best_cc:.4f} "
                      f"(range={best_dr:.2f}m, az={best_da:.1f}°) "
                      f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

    print(f"  Grid search complete in {time.time()-t0:.1f}s")
    print(f"  Best: CC={best_cc:.4f} at range={best_dr:.3f}m, azimuth={best_da:.2f}°")

    return [best_dr, best_da], best_cc, all_results


def refine_2dof(renderer, base_config, initial_2dof, range_res, ra_gt_cart,
                max_evals=30, seed=42):
    """Fine-tune 2DOF params with bounded Nelder-Mead."""
    eval_count = [0]
    best_cc = [-999.0]
    best_params = [list(initial_2dof)]

    def objective(x):
        eval_count[0] += 1
        tx_mm, rx_mm, bs = apply_2dof_to_config(base_config, x[0], x[1])
        update_renderer_antennas(renderer, tx_mm, rx_mm, bs)
        cc = render_and_evaluate(renderer, range_res, ra_gt_cart, seed=seed)
        if cc > best_cc[0]:
            best_cc[0] = cc
            best_params[0] = [float(x[0]), float(x[1])]
        print(f"    refine {eval_count[0]:3d}: CC={cc:.4f}  "
              f"range={x[0]:.3f}m az={x[1]:.2f}°")
        return -cc

    result = minimize(
        objective,
        x0=initial_2dof,
        method="Nelder-Mead",
        options={
            "maxiter": max_evals,
            "maxfev": max_evals + 5,
            "xatol": 0.02,
            "fatol": 0.001,
            "adaptive": True,
        },
    )

    return best_params[0], best_cc[0], eval_count[0]


def save_aligned_config(base_config, delta_range_m, delta_azimuth_deg, output_path):
    """Apply 2DOF params (zero rotation) and save aligned config."""
    tx_mm, rx_mm, boresight = apply_2dof_to_config(
        base_config, delta_range_m, delta_azimuth_deg
    )
    config = json.loads(json.dumps(base_config))
    for i, tx in enumerate(config["tx_array"]):
        tx["pos_mm"] = tx_mm[i].tolist()
        tx["boresight"] = boresight.tolist()
    for i, rx in enumerate(config["rx_array"]):
        rx["pos_mm"] = rx_mm[i].tolist()
        rx["boresight"] = boresight.tolist()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    return output_path
