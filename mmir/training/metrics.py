"""Range-azimuth and ADC metrics reported during training."""

import math
import numpy as np
from typing import Dict, Tuple

import torch

from mmir.data.ra_utils import ra_polar_to_cartesian
from mmir.losses.loss_utils import (
    compute_psnr, compute_ssim,
    minmax_normalize_numpy, minmax_normalize_torch,
)

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    lpips = None


def compute_all_metrics(rendered_ra: torch.Tensor, gt_ra: torch.Tensor,
                        lpips_model=None, range_res: float = None,
                        ra_gt_cart: np.ndarray = None) -> Dict[str, float]:
    """Compute all RA metrics with SEPARATE min-max normalization."""
    metrics = {}
    rendered_ra = rendered_ra.detach().float()
    gt_ra = gt_ra.detach().float()

    if rendered_ra.dim() > 2:
        rendered_ra = rendered_ra.squeeze()
    if gt_ra.dim() > 2:
        gt_ra = gt_ra.squeeze()

    rendered_ra_norm = minmax_normalize_torch(rendered_ra)
    gt_ra_norm = minmax_normalize_torch(gt_ra)

    metrics['ra_mse'] = torch.mean((rendered_ra_norm - gt_ra_norm)**2).item()
    metrics['ra_rmse'] = math.sqrt(metrics['ra_mse'])
    metrics['ra_ssim'] = compute_ssim(rendered_ra_norm, gt_ra_norm)
    metrics['ra_psnr'] = compute_psnr(rendered_ra_norm, gt_ra_norm)

    metrics['ra_lpips'] = float('nan')

    # Cartesian RA metrics
    if range_res is not None or ra_gt_cart is not None:
        from mmir.data.ra_utils import ra_polar_to_cartesian, compute_cartesian_ra_metrics
        rendered_ra_np = rendered_ra.detach().cpu().numpy()
        ra_rend_cart = ra_polar_to_cartesian(rendered_ra_np, range_res)
        cart_metrics = compute_cartesian_ra_metrics(ra_rend_cart, ra_gt_cart)
        metrics.update(cart_metrics)
        metrics['_ra_rend_cart'] = ra_rend_cart  # cached for reuse

    return metrics


def evaluate_against_targets(metrics: Dict[str, float]) -> Dict:
    """Check if metrics meet publication targets."""
    targets = {
        'ra_mse': {'threshold': 1e-4, 'op': 'less', 'target_str': '< 1e-4'},
        'ra_rmse': {'threshold': 0.2, 'op': 'less', 'target_str': '< 0.2'},
        'ra_psnr': {'threshold': 25.0, 'op': 'greater', 'target_str': '> 25 dB'},
        'ra_ssim': {'threshold': 0.4, 'op': 'greater', 'target_str': '> 0.4'},
        'ra_lpips': {'threshold': 0.4, 'op': 'less', 'target_str': '< 0.4'},
    }

    evaluation = {}
    passing_count = 0

    for metric_name, target_info in targets.items():
        value = metrics.get(metric_name, float('nan'))
        threshold = target_info['threshold']

        if not math.isfinite(value):
            evaluation[metric_name] = {
                'value': value, 'target': target_info['target_str'],
                'threshold': threshold, 'passes': False, 'status': '?'
            }
            continue

        passes = (value < threshold) if target_info['op'] == 'less' else (value > threshold)
        evaluation[metric_name] = {
            'value': value, 'target': target_info['target_str'],
            'threshold': threshold, 'passes': passes,
            'status': '[OK]' if passes else '[WARNING]'
        }
        if passes:
            passing_count += 1

    evaluation['passing_count'] = passing_count
    evaluation['all_pass'] = (passing_count == 5)
    return evaluation


def compute_correlation_metrics(ra_rendered: np.ndarray, ra_gt: np.ndarray,
                                range_res: float = 0.0) -> Tuple[float, float, float]:
    """Compute polar correlation, MSE, and Cartesian correlation."""
    ra_r_norm = minmax_normalize_numpy(ra_rendered)
    ra_g_norm = minmax_normalize_numpy(ra_gt)

    polar_corr = np.corrcoef(ra_r_norm.flatten(), ra_g_norm.flatten())[0, 1]
    if np.isnan(polar_corr):
        polar_corr = 0.0

    mse = np.mean((ra_r_norm - ra_g_norm) ** 2)

    cart_corr = 0.0
    if range_res > 0:
        ra_r_cart = ra_polar_to_cartesian(ra_rendered, range_res)
        ra_g_cart = ra_polar_to_cartesian(ra_gt, range_res)
        ra_r_cart_norm = minmax_normalize_numpy(ra_r_cart)
        ra_g_cart_norm = minmax_normalize_numpy(ra_g_cart)
        cart_corr = np.corrcoef(ra_r_cart_norm.flatten(), ra_g_cart_norm.flatten())[0, 1]
        if np.isnan(cart_corr):
            cart_corr = 0.0

    return polar_corr, mse, cart_corr


def compute_adc_metrics(
    rendered_adc_complex: np.ndarray,
    gt_adc_complex: np.ndarray,
    adc_use_log: bool = False,
    log_epsilon: float = 1e-6,
    log_scale: float = 1.0,
    top_percentile: float = 0.9,
    phase_metrics_type: str = "unit_phasor"
) -> Dict[str, float]:
    """Compute reconstruction metrics on complex-valued ADC data."""
    metrics = {}
    eps = 1e-10

    rendered_mag = np.abs(rendered_adc_complex)
    gt_mag = np.abs(gt_adc_complex)

    # Log normalization
    rendered_mag_log = np.log(log_epsilon + log_scale * rendered_mag)
    gt_mag_log = np.log(log_epsilon + log_scale * gt_mag)
    # Min-max normalization
    rendered_mag_minmax = minmax_normalize_numpy(rendered_mag)
    gt_mag_minmax = minmax_normalize_numpy(gt_mag)

    # Log-normalized ADC magnitude metrics
    mag_diff_log = rendered_mag_log - gt_mag_log
    metrics['adc_log_mag_l1'] = float(np.mean(np.abs(mag_diff_log)))
    metrics['adc_log_mag_l2'] = float(np.sum(mag_diff_log ** 2))
    metrics['adc_log_mag_mse'] = float(np.mean(mag_diff_log ** 2))
    metrics['adc_log_mag_rmse'] = float(np.sqrt(metrics['adc_log_mag_mse']))
    metrics['adc_log_mag_mae'] = metrics['adc_log_mag_l1']
    r_log_flat = rendered_mag_log.flatten()
    g_log_flat = gt_mag_log.flatten()
    if np.std(r_log_flat) < eps or np.std(g_log_flat) < eps:
        metrics['adc_log_mag_pearson'] = 0.0
    else:
        metrics['adc_log_mag_pearson'] = float(np.corrcoef(r_log_flat, g_log_flat)[0, 1])

    # Min-max normalized ADC magnitude metrics
    mag_diff_mm = rendered_mag_minmax - gt_mag_minmax
    metrics['adc_minmax_mag_l1'] = float(np.mean(np.abs(mag_diff_mm)))
    metrics['adc_minmax_mag_l2'] = float(np.sum(mag_diff_mm ** 2))
    metrics['adc_minmax_mag_mse'] = float(np.mean(mag_diff_mm ** 2))
    metrics['adc_minmax_mag_rmse'] = float(np.sqrt(metrics['adc_minmax_mag_mse']))
    metrics['adc_minmax_mag_mae'] = metrics['adc_minmax_mag_l1']
    r_mm_flat = rendered_mag_minmax.flatten()
    g_mm_flat = gt_mag_minmax.flatten()
    if np.std(r_mm_flat) < eps or np.std(g_mm_flat) < eps:
        metrics['adc_minmax_mag_pearson'] = 0.0
    else:
        metrics['adc_minmax_mag_pearson'] = float(np.corrcoef(r_mm_flat, g_mm_flat)[0, 1])

    # Legacy metrics (based on adc_use_log flag)
    mag_diff = mag_diff_log if adc_use_log else mag_diff_mm
    r_norm_flat = r_log_flat if adc_use_log else r_mm_flat
    g_norm_flat = g_log_flat if adc_use_log else g_mm_flat

    metrics['adc_l1'] = float(np.mean(np.abs(mag_diff)))
    metrics['adc_l2'] = float(np.sum(mag_diff ** 2))
    metrics['adc_mse'] = float(np.mean(mag_diff ** 2))
    metrics['adc_rmse'] = float(np.sqrt(metrics['adc_mse']))
    metrics['adc_mae'] = metrics['adc_l1']
    if np.std(r_norm_flat) < eps or np.std(g_norm_flat) < eps:
        metrics['adc_pearson'] = 0.0
    else:
        metrics['adc_pearson'] = float(np.corrcoef(r_norm_flat, g_norm_flat)[0, 1])
    for s in ['l1', 'l2', 'mse', 'rmse', 'mae', 'pearson']:
        metrics[f'adc_mag_{s}'] = metrics[f'adc_{s}']

    # ADC Phase Metrics
    NT, NR, K = rendered_adc_complex.shape
    mag_threshold = np.percentile(gt_mag.flatten(), top_percentile * 100)
    valid_mask = gt_mag >= mag_threshold

    if np.sum(valid_mask) > 10:
        rendered_phase = np.angle(rendered_adc_complex)
        gt_phase = np.angle(gt_adc_complex)
        phase_diff = rendered_phase - gt_phase
        unit_phasor_loss = 2.0 * (1.0 - np.cos(phase_diff))
        metrics['adc_phase_l1'] = float(np.mean(np.abs(unit_phasor_loss[valid_mask])))
        metrics['adc_phase_mse'] = float(np.mean(unit_phasor_loss[valid_mask]))
        metrics['adc_phase_rmse'] = float(np.sqrt(metrics['adc_phase_mse']))
        metrics['adc_phase_mae'] = float(np.mean(unit_phasor_loss[valid_mask]))
        metrics['adc_phase_l2'] = float(np.sum(unit_phasor_loss[valid_mask]))
        cos_r = np.cos(rendered_phase[valid_mask]).flatten()
        cos_g = np.cos(gt_phase[valid_mask]).flatten()
        if np.std(cos_r) < eps or np.std(cos_g) < eps:
            metrics['adc_phase_pearson'] = 0.0
        else:
            metrics['adc_phase_pearson'] = float(np.corrcoef(cos_r, cos_g)[0, 1])
    else:
        for s in ['l1', 'l2', 'mse', 'rmse', 'mae', 'pearson']:
            metrics[f'adc_phase_{s}'] = float('nan')

    # Phase aliases for dashboard compatibility
    for pfx in ['adc_relphase', 'adc_log_phase', 'adc_minmax_phase']:
        for s in ['l1', 'l2', 'mse', 'rmse', 'mae', 'pearson']:
            metrics[f'{pfx}_{s}'] = metrics[f'adc_phase_{s}']

    # Virtual array phase coherence
    metrics['va_phase_coherence'] = _compute_va_phase_coherence(
        rendered_adc_complex, gt_adc_complex, top_percentile)

    return metrics


def _compute_va_phase_coherence(rendered_complex, gt_complex, top_percentile=0.9):
    """Virtual Array Phase Coherence metric."""
    NT, NR, K = rendered_complex.shape
    N_vx = NT * NR
    eps = 1e-10

    r = rendered_complex.reshape(N_vx, K)
    g = gt_complex.reshape(N_vx, K)

    gt_energy = np.sum(np.abs(g) ** 2, axis=0)
    thresh = np.percentile(gt_energy, top_percentile * 100)
    valid = gt_energy >= thresh

    if np.sum(valid) < 2:
        return float('nan')

    r_hat = r / (np.abs(r) + eps)
    g_hat = g / (np.abs(g) + eps)

    dot = np.sum(r_hat[:, valid] * np.conj(g_hat[:, valid]), axis=0)
    rho = np.abs(dot) / N_vx

    return float(np.mean(rho))
