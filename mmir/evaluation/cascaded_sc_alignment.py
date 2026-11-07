#!/usr/bin/env python3
"""Cascaded-Mediated Single-Chip Radar Alignment"""

# Must set Mitsuba variant before mmir.sensor/renderer imports mi.Vector3f
import mitsuba as mi
if mi.variant() is None:
    mi.set_variant('cuda_ad_rgb')

import os
import json
import time
import numpy as np
from typing import Tuple, Optional

from scipy.stats import pearsonr

from mmir.evaluation.utils.single_view_viz import (
    load_config_positions_and_boresight,
)
from mmir.data.ra_utils import adc_to_ra_image_single_chip
from mmir.data.ra_utils import compute_range_res_from_cfg


# Stage 1: Board-frame transform transfer

def _extract_all_positions(config_path: str) -> np.ndarray:
    """Extract all antenna positions (TX + RX) from a config, in meters."""
    tx_m, rx_m, _, _ = load_config_positions_and_boresight(config_path)
    return np.vstack([tx_m, rx_m])  # (N, 3)


def compute_board_transform(
    casc_orig_config: str,
    casc_aligned_config: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the rigid-body transform from original → aligned cascaded config."""
    P_orig = _extract_all_positions(casc_orig_config)     # (N, 3)
    P_aligned = _extract_all_positions(casc_aligned_config)  # (N, 3)

    assert P_orig.shape == P_aligned.shape, (
        f"Position count mismatch: {P_orig.shape} vs {P_aligned.shape}"
    )

    # Centroids
    c1 = P_orig.mean(axis=0)
    c2 = P_aligned.mean(axis=0)

    # Cross-covariance matrix
    H = (P_orig - c1).T @ (P_aligned - c2)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Ensure proper rotation (det = +1)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    # Translation
    t = c2 - R @ c1

    return R, t


def transfer_alignment_to_sc(
    casc_orig_config: str,
    casc_aligned_config: str,
    sc_orig_config: str,
    output_path: Optional[str] = None,
) -> str:
    """Stage 1: Transfer cascaded alignment to the single-chip config."""
    R, t = compute_board_transform(casc_orig_config, casc_aligned_config)

    # Load original SC config
    with open(sc_orig_config, 'r') as f:
        sc_cfg = json.load(f)

    # Transform each antenna position (stored in mm, work in meters)
    for arr_key in ('tx_array', 'rx_array'):
        for ant in sc_cfg[arr_key]:
            pos_m = np.array(ant['pos_mm'], dtype=np.float64) / 1000.0
            pos_aligned = R @ pos_m + t
            ant['pos_mm'] = (pos_aligned * 1000.0).tolist()

            # Transform boresight direction
            if 'boresight' in ant:
                bore = np.array(ant['boresight'], dtype=np.float64)
                bore_aligned = R @ bore
                bore_aligned /= np.linalg.norm(bore_aligned)
                ant['boresight'] = bore_aligned.tolist()

    # Save

    with open(output_path, 'w') as f:
        json.dump(sc_cfg, f, indent=2)

    return output_path


# Stage 2: Render-based grid search refinement

def render_alignment_grid_search(
    base_config: str,
    gt_adc_path: str,
    mesh_file: str,
    training_dir: str,
    output_config_path: str,
    range_search_m: float = 0.5,
    azimuth_search_deg: float = 3.0,
    rotation_search_deg: float = 2.0,
    n_steps: int = 5,
    n_hits_per_rx: int = 2000,
    skip_range_bins: int = 5,
    metric: str = "cart_corr",
    verbose: bool = True,
) -> dict:
    """Grid search over small perturbations to maximize rendered↔GT RA correlation."""
    import io
    import contextlib
    import tempfile
    from mmir.preprocessing.alignment.lidar_alignment import create_aligned_config
    from mmir.evaluation.renderer_wrapper import RendererWrapper

    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sc_pattern = os.path.join(_PROJECT_ROOT, 'assets', 'antenna_pattern', 'IWR1443', 'pattern_76.npy')

    # Load and beamform GT ADC (once)
    gt_raw = np.load(gt_adc_path)
    gt_complex = gt_raw[0].transpose(1, 0, 2)  # (3, 4, 128) complex
    adc_gt = np.stack([gt_complex.real, gt_complex.imag], axis=-1).astype(np.float32)
    ra_gt = adc_to_ra_image_single_chip(adc_gt)
    ra_gt_crop = ra_gt[:, skip_range_bins:]

    def _to_db(x, floor=-40.0):
        x_n = x / max(x.max(), 1e-30)
        return np.maximum(20.0 * np.log10(np.maximum(x_n, 1e-10)), floor)

    ra_gt_db = _to_db(ra_gt_crop)

    # Pre-compute GT cartesian RA for cart_corr metric
    use_cart_corr = (metric == "cart_corr")
    if use_cart_corr:
        from mmir.data.ra_utils import ra_polar_to_cartesian
        range_res = compute_range_res_from_cfg(base_config)
        ra_gt_cart = ra_polar_to_cartesian(ra_gt_crop, range_res)
        # Normalize GT cartesian for correlation
        gt_cart_flat = ra_gt_cart.flatten().astype(np.float64)
        gt_cart_flat_normed = gt_cart_flat / max(np.max(np.abs(gt_cart_flat)), 1e-30)

    # Build renderer once
    if verbose:
        print("  Building renderer (one-time setup)...")
    wrapper = RendererWrapper.from_training_dir(
        mesh_file=mesh_file,
        config_file=base_config,
        training_dir=training_dir,
        tx_pattern_file=sc_pattern,
        rx_pattern_file=sc_pattern,
        verbose=verbose,
    )
    wrapper._render_config.n_hits_per_rx = n_hits_per_rx
    wrapper.renderer.config = wrapper._render_config
    wrapper.renderer.sampler.n_hits_per_rx = n_hits_per_rx
    wrapper.load_all_learned_params(training_dir)
    wrapper.verbose = False  # suppress per-render prints

    # Generate search grid
    range_offsets = np.linspace(-range_search_m, range_search_m, n_steps)
    azimuth_offsets = np.linspace(-azimuth_search_deg, azimuth_search_deg, n_steps)
    rotation_offsets = np.linspace(-rotation_search_deg, rotation_search_deg, n_steps)

    total = len(range_offsets) * len(azimuth_offsets) * len(rotation_offsets)
    metric_label = "cart_corr" if use_cart_corr else "corr_dB"
    if verbose:
        print(f"\n  Grid search: {n_steps}³ = {total} evaluations (metric: {metric_label})")
        print(f"    Range:    ±{range_search_m}m")
        print(f"    Azimuth:  ±{azimuth_search_deg}°")
        print(f"    Rotation: ±{rotation_search_deg}°")
        print(f"    n_hits_per_rx: {n_hits_per_rx}")

    best_corr = -np.inf
    best_params = None
    all_results = []
    t_start = time.time()
    idx = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for dr_val in range_offsets:
            for da_val in azimuth_offsets:
                for drot_val in rotation_offsets:
                    idx += 1

                    # Create perturbed config from base
                    temp_path = os.path.join(tmpdir, f"cfg_{idx}.json")
                    with contextlib.redirect_stdout(io.StringIO()):
                        create_aligned_config(
                            base_config, dr_val, da_val, drot_val, 0.0,
                            output_path=temp_path,
                        )

                    # Update antenna positions (no mesh rebuild)
                    with open(temp_path) as f:
                        perturbed_cfg = json.load(f)
                    wrapper.update_antenna_config(perturbed_cfg)

                    # Render
                    adc = wrapper.render_forward(seed=42)

                    # Beamform → RA
                    ra = adc_to_ra_image_single_chip(adc)
                    ra_crop = ra[:, skip_range_bins:]

                    # Compute metrics
                    ra_db = _to_db(ra_crop)
                    corr_db, _ = pearsonr(ra_db.flatten(), ra_gt_db.flatten())

                    ra_cart = ra_polar_to_cartesian(ra_crop, range_res)
                    cart_flat = ra_cart.flatten().astype(np.float64)
                    cart_flat_normed = cart_flat / max(np.max(np.abs(cart_flat)), 1e-30)
                    cart_corr, _ = pearsonr(cart_flat_normed, gt_cart_flat_normed)
                    opt_metric = cart_corr

                    is_best = opt_metric > best_corr
                    entry = {
                        'delta_range_m': float(dr_val),
                        'delta_azimuth_deg': float(da_val),
                        'rotation_elev_deg': float(drot_val),
                        'corr_dB': float(corr_db),
                    }
                    if cart_corr is not None:
                        entry['cart_corr'] = float(cart_corr)
                    all_results.append(entry)

                    if is_best:
                        best_corr = opt_metric
                        best_params = entry.copy()

                    if verbose:
                        elapsed = time.time() - t_start
                        eta = elapsed / idx * (total - idx) if idx > 0 else 0
                        tag = " ***" if is_best else ""
                        m_val = f"cart_corr={cart_corr:.4f}" if use_cart_corr else f"corr_dB={corr_db:.4f}"
                        print(f"  [{idx:3d}/{total}] Δr={dr_val:+.3f}m Δaz={da_val:+.2f}° "
                              f"Δrot={drot_val:+.2f}° → {m_val}"
                              f"  ({elapsed:.0f}s, ETA {eta:.0f}s){tag}")

    # Save best config
    is_identity = (
        best_params['delta_range_m'] == 0.0
        and best_params['delta_azimuth_deg'] == 0.0
        and best_params['rotation_elev_deg'] == 0.0
    )
    import io as _io
    import contextlib as _ctx
    with _ctx.redirect_stdout(_io.StringIO()):
        from mmir.preprocessing.alignment.lidar_alignment import create_aligned_config as _cac
        _cac(
            base_config,
            best_params['delta_range_m'],
            best_params['delta_azimuth_deg'],
            best_params['rotation_elev_deg'],
            0.0,
            output_path=output_config_path,
        )

    wrapper.cleanup()

    total_time = time.time() - t_start

    # Find baseline (0,0,0) correlation
    base_corr_db = None
    base_metric = None
    for r in all_results:
        if (r['delta_range_m'] == 0.0
                and r['delta_azimuth_deg'] == 0.0
                and r['rotation_elev_deg'] == 0.0):
            base_corr_db = r['corr_dB']
            base_metric = r.get('cart_corr', r['corr_dB'])
            break

    if verbose:
        print(f"\n  Grid search complete in {total_time:.1f}s (metric: {metric_label})")
        if base_metric is not None:
            print(f"  Stage 1 baseline {metric_label}: {base_metric:.4f}")
        print(f"  Best {metric_label}: {best_corr:.4f}")
        if base_metric is not None:
            print(f"  Improvement: {best_corr - base_metric:+.4f}")
        print(f"  Best params: Δr={best_params['delta_range_m']:+.3f}m, "
              f"Δaz={best_params['delta_azimuth_deg']:+.2f}°, "
              f"Δrot={best_params['rotation_elev_deg']:+.2f}°")
        print(f"  Saved: {output_config_path}")

    return {
        'base_corr_dB': base_corr_db,
        'best_corr_dB': float(best_corr),
        'best_params': best_params,
        'all_results': all_results,
        'total_time_s': total_time,
        'output_config_path': output_config_path,
    }


# Multi-resolution grid search

def multi_resolution_grid_search(
    base_config: str,
    gt_adc_path: str,
    mesh_file: str,
    training_dir: str,
    output_config_path: str,
    # Coarse pass parameters
    coarse_range_m: float = 2.0,
    coarse_azimuth_deg: float = 10.0,
    coarse_rotation_deg: float = 5.0,
    coarse_steps: int = 11,
    # Fine pass parameters
    fine_range_m: float = 0.4,
    fine_azimuth_deg: float = 2.0,
    fine_rotation_deg: float = 1.0,
    fine_steps: int = 9,
    n_hits_per_rx: int = 2000,
    skip_range_bins: int = 5,
    metric: str = "cart_corr",
    verbose: bool = True,
) -> dict:
    """Two-pass grid search: coarse sweep over wide range, then fine refinement."""
    import tempfile
    t_start = time.time()

    if verbose:
        total_evals = coarse_steps ** 3 + fine_steps ** 3
        print(f"\n  Multi-resolution alignment: {total_evals} total evaluations")
        print(f"    Pass 1 (coarse): {coarse_steps}³ = {coarse_steps**3} evals, "
              f"±{coarse_range_m}m / ±{coarse_azimuth_deg}° / ±{coarse_rotation_deg}°")
        print(f"    Pass 2 (fine):   {fine_steps}³ = {fine_steps**3} evals, "
              f"±{fine_range_m}m / ±{fine_azimuth_deg}° / ±{fine_rotation_deg}°")

    # Pass 1: Coarse sweep
    with tempfile.TemporaryDirectory() as tmpdir:
        coarse_output = os.path.join(tmpdir, "coarse_best.json")

        if verbose:
            print("\n  === Pass 1: Coarse Sweep ===")

        coarse_result = render_alignment_grid_search(
            base_config=base_config,
            gt_adc_path=gt_adc_path,
            mesh_file=mesh_file,
            training_dir=training_dir,
            output_config_path=coarse_output,
            range_search_m=coarse_range_m,
            azimuth_search_deg=coarse_azimuth_deg,
            rotation_search_deg=coarse_rotation_deg,
            n_steps=coarse_steps,
            n_hits_per_rx=n_hits_per_rx,
            skip_range_bins=skip_range_bins,
            metric=metric,
            verbose=verbose,
        )

        if verbose:
            bp = coarse_result['best_params']
            print(f"\n  Coarse best: {metric}={coarse_result['best_corr_dB']:.4f}")
            print(f"    Δr={bp['delta_range_m']:+.3f}m, "
                  f"Δaz={bp['delta_azimuth_deg']:+.2f}°, "
                  f"Δrot={bp['rotation_elev_deg']:+.2f}°")

        # Pass 2: Fine refinement
        if verbose:
            print("\n  === Pass 2: Fine Refinement ===")

        fine_result = render_alignment_grid_search(
            base_config=coarse_output,
            gt_adc_path=gt_adc_path,
            mesh_file=mesh_file,
            training_dir=training_dir,
            output_config_path=output_config_path,
            range_search_m=fine_range_m,
            azimuth_search_deg=fine_azimuth_deg,
            rotation_search_deg=fine_rotation_deg,
            n_steps=fine_steps,
            n_hits_per_rx=n_hits_per_rx,
            skip_range_bins=skip_range_bins,
            metric=metric,
            verbose=verbose,
        )

    total_time = time.time() - t_start

    if verbose:
        print(f"\n  Multi-resolution complete in {total_time:.1f}s")
        print(f"  Coarse best: {coarse_result['best_corr_dB']:.4f}")
        print(f"  Fine best:   {fine_result['best_corr_dB']:.4f}")
        if coarse_result.get('base_corr_dB') is not None:
            print(f"  Total improvement: {fine_result['best_corr_dB'] - coarse_result['base_corr_dB']:+.4f}")
        print(f"  Saved: {output_config_path}")

    return {
        'coarse_result': coarse_result,
        'fine_result': fine_result,
        'best_corr_dB': fine_result['best_corr_dB'],
        'best_params': fine_result['best_params'],
        'total_time_s': total_time,
        'output_config_path': output_config_path,
    }
