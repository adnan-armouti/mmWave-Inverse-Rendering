"""The training loop."""

import os
import json
import time
import math
import numpy as np
from dataclasses import asdict

import torch

from tqdm import tqdm

from mmir.data.ra_utils import adc_to_ra_image
from mmir.data.ra_utils import compute_range_res_from_cfg

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    lpips = None
from mmir.training.config import TrainingConfig
from mmir.training.forward import (
    clip_gradients, create_renderer, extract_adc_for_metrics, extract_ra_from_adc,
    run_backward_and_extract_gradients, run_end_to_end_forward, run_error_map_initialization,
)
from mmir.training.metrics import compute_adc_metrics, compute_all_metrics, compute_correlation_metrics, evaluate_against_targets
from mmir.training.parameters import (
    ParameterManager, SimpleAdam, _extract_pattern_data, save_final_results,
)
from mmir.training.visualization import plot_training_dashboard


def get_lr_scale(iteration: int, config: TrainingConfig) -> float:
    """Compute LR multiplier: linear warmup -> constant."""
    if not config.enable_lr_schedule:
        return 1.0

    warmup = min(config.lr_warmup_iters, max(3, config.num_iterations // 10))
    warmup_factor = config.lr_warmup_factor

    if iteration < warmup:
        return warmup_factor + (1.0 - warmup_factor) * (iteration / max(warmup, 1))
    else:
        return 1.0


def update_frozen_params_for_stage(config: TrainingConfig, iteration: int):
    """Update frozen_params based on training stage."""
    if not config.enable_staged_training:
        return

    if iteration < config.stage1_iterations:
        # Stage 1: pose + patterns only
        config.frozen_params = ['materials', 'normals', 'vertex_positions']
        if iteration == 0:
            tqdm.write(f"  [Stage 1] Training pose+patterns for {config.stage1_iterations} iterations")
    else:
        # Stage 2: materials + normals + geometry
        config.frozen_params = ['pose_rotation', 'pose_translation']
        if iteration == config.stage1_iterations:
            tqdm.write("  [Stage 2] Training materials+normals+geometry")


def train(config: TrainingConfig):
    """Main training loop."""

    # Setup
    os.makedirs(config.output_dir, exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(config.output_dir, 'config.json'), 'w') as f:
        json.dump(asdict(config), f, indent=2, default=str)

    print("=" * 80)
    print("mmIR Training")
    print("=" * 80)

    # Create renderer
    print("Creating renderer...")
    renderer = create_renderer(config)

    # Load GT ADC
    print(f"Loading GT ADC from {config.gt_adc_file}")
    gt_adc_np = np.load(config.gt_adc_file)
    if gt_adc_np.ndim == 4:
        # Multi-chirp: use first chirp for single-chirp training
        gt_adc_single = gt_adc_np[0]
    elif gt_adc_np.ndim == 3:
        gt_adc_single = gt_adc_np
    else:
        raise ValueError(f"Unexpected GT shape: {gt_adc_np.shape}")

    # Convert to torch (NR, NT, K) complex -> (NT, NR, K, 2) real/imag
    gt_complex = gt_adc_single  # (NR, NT, K) complex
    gt_real_imag = np.stack([gt_complex.real, gt_complex.imag], axis=-1)  # (NR, NT, K, 2)
    # Transpose to (NT, NR, K, 2) for compute_ra_loss_with_gradients
    gt_adc_torch = torch.from_numpy(gt_real_imag.transpose(1, 0, 2, 3).astype(np.float32))

    # Compute GT RA
    ra_gt = adc_to_ra_image(gt_adc_torch).detach().cpu().numpy()

    # Compute range resolution
    range_res = compute_range_res_from_cfg(config.config_file)

    # Load multi-chirp GT if needed
    gt_all_chirps = None

    # Create parameter manager
    print("Creating parameter manager...")
    pm = ParameterManager(renderer, config, config.scene_file)

    NT, NR, K = pm.NT, pm.NR, pm.K
    print(f"  NT={NT}, NR={NR}, K={K}, n_opt_params={pm.n_opt_params}")

    # ITU initialization
    if config.initial_material:
        pm.initialize_from_itu(config.initial_material)

    # Error-map initialization
    if config.enable_error_map_init:
        run_error_map_initialization(pm, ra_gt, range_res)

    # Create optimizer (before checkpoint so we can restore Adam state)
    optimizer = SimpleAdam(pm, config)

    # Load checkpoint
    start_iter = 0

    # LPIPS model
    lpips_model = None

    # History tracking - comprehensive metric suite
    history = {
        'iteration': [], 'total_loss': [], 'loss_reg': [], 'adc_mag_loss': [],
        'polar_corr': [], 'cart_corr': [], 'mse': [], 'lr_multiplier': [],
        # Multi-chirp loss components
        'mc_adc_mag': [], 'mc_ra_mag': [], 'mc_phase': [],
        # Physics params
        'eps_real_mean': [], 'eps_imag_mean': [], 'sigma_h_mean': [],
        'l_c_mean': [], 'tau_mean': [], 'thickness_mean': [],
        # ADC metrics (log magnitude)
        'adc_log_mag_l1': [], 'adc_log_mag_mse': [], 'adc_log_mag_rmse': [],
        'adc_log_mag_mae': [], 'adc_log_mag_pearson': [],
        # ADC metrics (relative phase)
        'adc_relphase_l1': [], 'adc_relphase_mse': [], 'adc_relphase_rmse': [],
        'adc_relphase_mae': [], 'adc_relphase_pearson': [],
        # ADC metrics (minmax magnitude)
        'adc_minmax_mag_l1': [], 'adc_minmax_mag_mse': [], 'adc_minmax_mag_rmse': [],
        'adc_minmax_mag_mae': [], 'adc_minmax_mag_pearson': [],
        # RA metrics (with alignment / from loss)
        'ra_mse': [], 'ra_rmse': [], 'ra_psnr': [], 'ra_ssim': [], 'ra_lpips': [],
        # RA metrics (without alignment / unaligned)
        'ra_unaligned_mse': [], 'ra_unaligned_rmse': [], 'ra_unaligned_psnr': [],
        'ra_unaligned_ssim': [], 'ra_unaligned_lpips': [],
        # Cartesian RA metrics (computed every iteration)
        'cart_psnr': [], 'cart_ssim': [], 'cart_mse': [], 'cart_rmse': [],
    }

    best_loss = float('inf')
    best_raw_params = pm.raw_params.copy()
    best_normal_params = pm.normal_params_np.copy() if pm.normal_params_np is not None else None
    best_pose_params = pm.pose_params_np.copy() if pm.pose_params_np is not None else None
    best_vertex_offset = pm.vertex_offset_np.copy() if pm.vertex_offset_np is not None else None
    best_pattern_data = _extract_pattern_data(pm)
    best_metrics_snapshot = {}
    best_ra_rendered = None
    best_iteration = 0

    print(f"\nStarting training: {config.num_iterations} iterations")
    print(f"  LEARN: MAT={config.LEARN_MAT} POSE={config.LEARN_POSE} "
          f"NRM={config.LEARN_NRM} VTX={config.LEARN_VTX} PAT={config.LEARN_PAT}")
    print(f"  Loss: RA_weight={config.loss_config.ra_mag_weight}, "
          f"ADC_weight={config.loss_config.adc_mag_weight}")
    print(f"  Multi-chirp: {config.loss_config.use_multi_chirp_loss}")
    print()

    # Pre-training: save GT figures and precompute GT Cartesian RA
    from mmir.data.ra_utils import ra_polar_to_cartesian, save_ra_cartesian_png
    ra_gt_cart = ra_polar_to_cartesian(ra_gt, range_res)
    np.save(os.path.join(config.output_dir, 'ra_gt_cart.npy'), ra_gt_cart)
    for scale in ('dB', 'linear'):
        save_ra_cartesian_png(
            ra_gt_cart,
            os.path.join(config.output_dir, f'gt_ra_{scale}.png'),
            range_res=range_res, scale=scale, title=f'GT ({scale})')
    print(f"  Saved GT figures + ra_gt_cart.npy to {config.output_dir}")

    # Training Loop
    pbar = tqdm(range(start_iter, config.num_iterations), desc="Training", ncols=120)
    for iteration in pbar:
        t_start = time.time()

        # [A] Staged Training Update
        update_frozen_params_for_stage(config, iteration)

        # [B] LR Scheduling
        lr_scale = get_lr_scale(iteration, config)

        # [D+E] Forward pass (E2E or two-phase depending on config)
        t_forward = time.time()
        try:
            (adc_real, adc_imag, raw_drjit, pose_dr, normal_dr,
             vertex_offset_dr, pattern_loaders_dr) = run_end_to_end_forward(pm, iteration)
        except Exception as e:
            tqdm.write(f"  Warning: Forward pass failed at iteration {iteration}: {e}")
            continue
        t_forward = time.time() - t_forward

        # [F] Extract RA for metrics (suspend grad)
        ra_rendered = extract_ra_from_adc(adc_real, adc_imag, NT, NR, K)

        # [G] Backward pass & gradient extraction
        t_backward = time.time()
        gradients, loss_dict = run_backward_and_extract_gradients(
            adc_real, adc_imag, gt_adc_torch, config.loss_config,
            raw_drjit, pose_dr, normal_dr, vertex_offset_dr, pattern_loaders_dr,
            config, NT, NR, K,
            gt_all_chirps=gt_all_chirps,
        )
        t_backward = time.time() - t_backward

        # [H] Gradient clipping
        gradients = clip_gradients(gradients, config)

        # [H2] Track per-group gradient RMS (before optimizer step)
        grad_rms = {}
        for gname, gval in gradients.items():
            if isinstance(gval, np.ndarray):
                grad_rms[gname] = float(np.sqrt(np.mean(gval ** 2)))
            elif isinstance(gval, dict):
                # Pattern gradients: dict of arrays
                rms_vals = [float(np.sqrt(np.mean(v ** 2))) for v in gval.values() if isinstance(v, np.ndarray)]
                grad_rms[gname] = float(np.mean(rms_vals)) if rms_vals else 0.0

        # [I] Optimizer step
        optimizer.step(gradients, lr_scale=lr_scale)

        # [J] Post-step constraints
        pm.post_step_constraints()

        # [K] Compute metrics
        polar_corr, mse, cart_corr = compute_correlation_metrics(ra_rendered, ra_gt, range_res)
        total_loss = loss_dict.get('total', 0.0)

        # [L] Track history
        history['iteration'].append(iteration)
        history['total_loss'].append(total_loss)
        history['polar_corr'].append(polar_corr)
        history['cart_corr'].append(cart_corr)
        history['mse'].append(mse)
        history['lr_multiplier'].append(lr_scale)
        history['loss_reg'].append(loss_dict.get('loss_reg', 0.0))
        history['adc_mag_loss'].append(loss_dict.get('adc_mag_loss', None))
        history['mc_adc_mag'].append(loss_dict.get('mc_adc_mag', None))
        history['mc_ra_mag'].append(loss_dict.get('mc_ra_mag', None))
        history['mc_phase'].append(loss_dict.get('mc_phase', None))
        # Per-group gradient RMS
        for gname in ['materials', 'pose_rotation', 'pose_translation',
                       'normals', 'vertex_positions', 'patterns']:
            key = f'grad_rms_{gname}'
            if key not in history:
                history[key] = []
            history[key].append(grad_rms.get(gname, None))
        # Parameter state tracking
        if 'pose_translation_norm' not in history:
            history['pose_translation_norm'] = []
            history['pose_rotation_norm'] = []
            history['vertex_offset_rms'] = []
        history['pose_translation_norm'].append(float(np.linalg.norm(pm.pose_params_np[3:])))
        history['pose_rotation_norm'].append(float(np.linalg.norm(pm.pose_params_np[:3])))
        history['vertex_offset_rms'].append(
            float(np.sqrt(np.mean(pm.vertex_offset_np ** 2))) if pm.vertex_offset_np is not None else 0.0)
        phys_means = pm.get_physics_param_means()
        for k, v in phys_means.items():
            history[k].append(v)

        # [M] Track best (by lowest total_loss)
        if total_loss < best_loss:
            best_loss = total_loss
            best_raw_params = pm.raw_params.copy()
            best_ra_rendered = ra_rendered.copy()
            best_iteration = iteration
            if pm.normal_params_np is not None:
                best_normal_params = pm.normal_params_np.copy()
            if pm.pose_params_np is not None:
                best_pose_params = pm.pose_params_np.copy()
            best_pattern_data = _extract_pattern_data(pm)

        # [N] tqdm progress bar
        t_total = time.time() - t_start
        pbar.set_postfix({
            'Loss': f'{total_loss:.4e}',
            'Corr': f'{cart_corr:.4f}',
            'Fwd': f'{t_forward:.1f}s',
            'Bwd': f'{t_backward:.1f}s',
            'Tot': f'{t_total:.1f}s',
        })

        # [O] Detailed logging with full metrics suite
        if (iteration + 1) % config.log_interval == 0:
            # RA metrics (unaligned - separate min-max normalization) + Cartesian RA metrics
            ra_rendered_torch = torch.from_numpy(ra_rendered).float()
            ra_gt_torch = torch.from_numpy(ra_gt).float()
            ra_metrics = compute_all_metrics(ra_rendered_torch, ra_gt_torch, lpips_model,
                                             range_res=range_res, ra_gt_cart=ra_gt_cart)
            evaluate_against_targets(ra_metrics)

            # Track RA metrics (unaligned)
            history['ra_unaligned_mse'].append(ra_metrics.get('ra_mse', None))
            history['ra_unaligned_rmse'].append(ra_metrics.get('ra_rmse', None))
            history['ra_unaligned_psnr'].append(ra_metrics.get('ra_psnr', None))
            history['ra_unaligned_ssim'].append(ra_metrics.get('ra_ssim', None))
            history['ra_unaligned_lpips'].append(ra_metrics.get('ra_lpips', None))

            # Track Cartesian RA metrics
            history['cart_psnr'].append(ra_metrics.get('cart_psnr', None))
            history['cart_ssim'].append(ra_metrics.get('cart_ssim', None))
            history['cart_mse'].append(ra_metrics.get('cart_mse', None))
            history['cart_rmse'].append(ra_metrics.get('cart_rmse', None))

            # Per-epoch rendered .png saves (linear + dB)
            ra_rend_cart = ra_metrics.get('_ra_rend_cart')
            if ra_rend_cart is not None:
                viz_dir = os.path.join(config.output_dir, 'visualizations')
                os.makedirs(viz_dir, exist_ok=True)
                for scale in ('dB', 'linear'):
                    save_ra_cartesian_png(
                        ra_rend_cart,
                        os.path.join(viz_dir, f'rendered_ra_{scale}_iter_{iteration+1:04d}.png'),
                        range_res=range_res, scale=scale,
                        title=f'Rendered ({scale}, iter={iteration+1}, corr={cart_corr:.4f})')

            # Update best_metrics_snapshot if this was the best iteration
            if iteration == best_iteration:
                best_metrics_snapshot = {
                    'cart_corr': cart_corr,
                    'cart_psnr': ra_metrics.get('cart_psnr'),
                    'cart_ssim': ra_metrics.get('cart_ssim'),
                    'cart_mse': ra_metrics.get('cart_mse'),
                    'cart_rmse': ra_metrics.get('cart_rmse'),
                    'polar_corr': polar_corr,
                    'total_loss': total_loss,
                    'best_iteration': iteration,
                }

            # ADC metrics (30+ metrics with log/minmax normalization + phase)
            adc_rendered = extract_adc_for_metrics(adc_real, adc_imag, NT, NR, K)
            adc_metrics = compute_adc_metrics(adc_rendered, gt_adc_single)

            # Track ADC log magnitude metrics
            history['adc_log_mag_l1'].append(adc_metrics.get('adc_log_mag_l1', None))
            history['adc_log_mag_mse'].append(adc_metrics.get('adc_log_mag_mse', None))
            history['adc_log_mag_rmse'].append(adc_metrics.get('adc_log_mag_rmse', None))
            history['adc_log_mag_mae'].append(adc_metrics.get('adc_log_mag_mae', None))
            history['adc_log_mag_pearson'].append(adc_metrics.get('adc_log_mag_pearson', None))

            # Track ADC phase metrics
            history['adc_relphase_l1'].append(adc_metrics.get('adc_relphase_l1', None))
            history['adc_relphase_mse'].append(adc_metrics.get('adc_relphase_mse', None))
            history['adc_relphase_rmse'].append(adc_metrics.get('adc_relphase_rmse', None))
            history['adc_relphase_mae'].append(adc_metrics.get('adc_relphase_mae', None))
            history['adc_relphase_pearson'].append(adc_metrics.get('adc_relphase_pearson', None))

            # Track ADC minmax magnitude metrics
            history['adc_minmax_mag_l1'].append(adc_metrics.get('adc_minmax_mag_l1', None))
            history['adc_minmax_mag_mse'].append(adc_metrics.get('adc_minmax_mag_mse', None))
            history['adc_minmax_mag_rmse'].append(adc_metrics.get('adc_minmax_mag_rmse', None))
            history['adc_minmax_mag_mae'].append(adc_metrics.get('adc_minmax_mag_mae', None))
            history['adc_minmax_mag_pearson'].append(adc_metrics.get('adc_minmax_mag_pearson', None))

            # Track RA metrics from loss (with alignment)
            history['ra_mse'].append(loss_dict.get('ra_l2', ra_metrics.get('ra_mse', None)))
            history['ra_rmse'].append(math.sqrt(loss_dict['ra_l2']) if 'ra_l2' in loss_dict else ra_metrics.get('ra_rmse', None))
            history['ra_psnr'].append(ra_metrics.get('ra_psnr', None))
            history['ra_ssim'].append(ra_metrics.get('ra_ssim', None))
            history['ra_lpips'].append(ra_metrics.get('ra_lpips', None))

            tqdm.write(
                f"  [{iteration+1}/{config.num_iterations}] "
                f"corr={cart_corr:.4f} mse={mse:.2e} "
                f"psnr={ra_metrics.get('ra_psnr', 0):.1f}dB "
                f"ssim={ra_metrics.get('ra_ssim', 0):.3f} "
                f"adc_log_l1={adc_metrics.get('adc_log_mag_l1', 0):.4f} "
                f"phase_l1={adc_metrics.get('adc_relphase_l1', 0):.4f} "
                f"({t_forward:.1f}s fwd, {t_backward:.1f}s bwd, {t_total:.1f}s total)"
            )

            # Write live metrics
            live_path = os.path.join(config.output_dir, 'normalized_metrics_live.txt')
            with open(live_path, 'a') as f:
                f.write(f"iter={iteration+1} corr={cart_corr:.4f} mse={mse:.2e} "
                        f"psnr={ra_metrics.get('ra_psnr', 0):.2f} "
                        f"ssim={ra_metrics.get('ra_ssim', 0):.4f} "
                        f"lpips={ra_metrics.get('ra_lpips', float('nan')):.4f} "
                        f"adc_log_l1={adc_metrics.get('adc_log_mag_l1', float('nan')):.4f} "
                        f"adc_phase_l1={adc_metrics.get('adc_relphase_l1', float('nan')):.4f}\n")
        else:
            # Pad metric histories with None on non-log iterations
            for key in ['ra_unaligned_mse', 'ra_unaligned_rmse', 'ra_unaligned_psnr',
                        'ra_unaligned_ssim', 'ra_unaligned_lpips',
                        'adc_log_mag_l1', 'adc_log_mag_mse', 'adc_log_mag_rmse',
                        'adc_log_mag_mae', 'adc_log_mag_pearson',
                        'adc_relphase_l1', 'adc_relphase_mse', 'adc_relphase_rmse',
                        'adc_relphase_mae', 'adc_relphase_pearson',
                        'adc_minmax_mag_l1', 'adc_minmax_mag_mse', 'adc_minmax_mag_rmse',
                        'adc_minmax_mag_mae', 'adc_minmax_mag_pearson',
                        'ra_mse', 'ra_rmse', 'ra_psnr', 'ra_ssim', 'ra_lpips',
                        'cart_psnr', 'cart_ssim', 'cart_mse', 'cart_rmse']:
                history[key].append(None)

    # Final
    save_final_results(pm, best_raw_params, history, ra_gt, config.output_dir,
                       best_normal_params=best_normal_params,
                       best_pose_params=best_pose_params,
                       best_vertex_offset=best_vertex_offset,
                       best_pattern_data=best_pattern_data)
    plot_training_dashboard(history, config.output_dir)

    # Save best-iteration artifacts
    if best_ra_rendered is not None:
        ra_best_cart = ra_polar_to_cartesian(best_ra_rendered, range_res)
        np.save(os.path.join(config.output_dir, 'ra_rendered_cart.npy'), ra_best_cart)
        for scale in ('dB', 'linear'):
            save_ra_cartesian_png(
                ra_best_cart,
                os.path.join(config.output_dir, f'rendered_ra_{scale}.png'),
                range_res=range_res, scale=scale,
                title=f'Best Rendered ({scale}, iter={best_iteration+1})')

    # Save best_metrics.json
    best_metrics_out = dict(best_metrics_snapshot)
    best_metrics_out.setdefault('best_iteration', best_iteration)
    best_metrics_out.setdefault('best_loss', best_loss)
    with open(os.path.join(config.output_dir, 'best_metrics.json'), 'w') as f:
        json.dump(best_metrics_out, f, indent=2)

    print("\nTraining complete!")
    print(f"  Best loss: {best_loss:.4e} at iteration {best_iteration+1}")
    if best_metrics_snapshot:
        print(f"  Best cart_corr: {best_metrics_snapshot.get('cart_corr', 'N/A')}")
    print(f"  Results saved to: {config.output_dir}")

    return history
