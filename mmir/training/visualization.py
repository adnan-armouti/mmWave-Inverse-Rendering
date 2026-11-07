"""Training visualizations: RA comparisons, BSDF components and the dashboard."""

import os
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict






def plot_training_dashboard(history: Dict, output_dir: str):
    """Plot comprehensive training dashboard (8 rows x 5 columns)."""
    if not history.get('iteration'):
        return

    iters = history['iteration']
    fig, axes = plt.subplots(8, 5, figsize=(25, 32))
    fig.suptitle('Training Metrics: Loss Components and Reconstruction Quality',
                 fontsize=16, fontweight='bold')

    def safe_plot(ax, key, title, ylabel, color='b', log_scale=False):
        """Helper to safely plot a metric if available."""
        if key in history and len(history[key]) > 0:
            vals = history[key]
            valid_pairs = [(it, v) for it, v in zip(iters, vals)
                          if v is not None and not (isinstance(v, float) and math.isnan(v))]
            if len(valid_pairs) > 0:
                x_vals, valid_vals = zip(*valid_pairs)
                marker_size = 2 if len(valid_vals) > 100 else 3
                ax.plot(x_vals, valid_vals, color=color, linewidth=1.5,
                        marker='o', markersize=marker_size)
                ax.set_xlabel('Iteration', fontsize=9)
                ax.set_ylabel(ylabel, fontsize=9)
                ax.set_title(title, fontsize=10, fontweight='bold')
                if log_scale:
                    ax.set_yscale('log')
                ax.grid(True, alpha=0.3)
                return True
        ax.set_title(f'{title} (no data)', fontsize=10, fontweight='bold', color='gray')
        ax.set_xlabel('Iteration', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)
        return False

    # Row 0: Total Loss (span 2) + Multi-Chirp Components (span 3)
    ax_total = plt.subplot2grid((7, 5), (0, 0), colspan=2, fig=fig)
    if 'total_loss' in history and history['total_loss']:
        ax_total.plot(iters, history['total_loss'], 'b-', linewidth=2, label='Total Loss')
    ax_total.set_xlabel('Iteration', fontsize=10)
    ax_total.set_ylabel('Loss', fontsize=10)
    ax_total.set_title('Total Loss', fontsize=12, fontweight='bold')
    ax_total.grid(True, alpha=0.3)
    ax_total.legend()

    ax_mc = plt.subplot2grid((7, 5), (0, 2), colspan=3, fig=fig)
    for key, color, label in [('mc_adc_mag', 'tab:blue', 'ADC Mag'),
                               ('mc_ra_mag', 'tab:green', 'RA Mag'),
                               ('mc_phase', 'tab:red', 'Phase')]:
        if key in history and history[key]:
            [(it, v) for it, v in zip(iters, history[key]) if v is not None]
    ax_mc.set_title('Multi-Chirp (not enabled)', fontsize=12, fontweight='bold', color='gray')
    ax_mc.set_xlabel('Iteration', fontsize=10)
    ax_mc.grid(True, alpha=0.3)

    for col in range(5):
        axes[0, col].axis('off')

    # Row 1: Regularization (span 2) + LR Schedule (span 3)
    ax_reg = plt.subplot2grid((7, 5), (1, 0), colspan=2, fig=fig)
    if 'loss_reg' in history and history['loss_reg']:
        ax_reg.plot(iters, history['loss_reg'], 'r-', linewidth=2, label='Reg Loss')
    if 'adc_mag_loss' in history and history['adc_mag_loss']:
        [(it, v) for it, v in zip(iters, history['adc_mag_loss']) if v is not None and v > 0]
    ax_reg.set_xlabel('Iteration', fontsize=10)
    ax_reg.set_ylabel('Loss', fontsize=10)
    ax_reg.set_title('Loss Components', fontsize=12, fontweight='bold')
    ax_reg.grid(True, alpha=0.3)
    ax_reg.legend()

    ax_lr = plt.subplot2grid((7, 5), (1, 2), colspan=3, fig=fig)
    ax_lr.plot(iters, history['lr_multiplier'], 'purple', linewidth=2)
    ax_lr.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax_lr.set_xlabel('Iteration', fontsize=10)
    ax_lr.grid(True, alpha=0.3)

    for col in range(5):
        axes[1, col].axis('off')

    # Row 2: Log ADC Magnitude metrics
    safe_plot(axes[2, 0], 'adc_log_mag_l1', 'Log ADC Mag L1', 'L1', color='tab:blue')
    safe_plot(axes[2, 1], 'adc_log_mag_mse', 'Log ADC Mag MSE', 'MSE', color='tab:green', log_scale=True)
    safe_plot(axes[2, 2], 'adc_log_mag_rmse', 'Log ADC Mag RMSE', 'RMSE', color='tab:orange')
    safe_plot(axes[2, 3], 'adc_log_mag_mae', 'Log ADC Mag MAE', 'MAE', color='tab:red')
    safe_plot(axes[2, 4], 'adc_log_mag_pearson', 'Log ADC Mag Pearson', 'Pearson r', color='tab:brown')

    # Row 3: ADC Phase metrics
    safe_plot(axes[3, 0], 'adc_relphase_l1', 'ADC Phase L1', 'L1', color='tab:purple')
    safe_plot(axes[3, 1], 'adc_relphase_mse', 'ADC Phase MSE', 'MSE', color='tab:cyan', log_scale=True)
    safe_plot(axes[3, 2], 'adc_relphase_rmse', 'ADC Phase RMSE', 'RMSE', color='tab:pink')
    safe_plot(axes[3, 3], 'adc_relphase_mae', 'ADC Phase MAE', 'MAE', color='tab:olive')
    safe_plot(axes[3, 4], 'adc_relphase_pearson', 'ADC Phase Pearson', 'Pearson r', color='tab:brown')

    # Row 4: MinMax ADC Magnitude metrics
    safe_plot(axes[4, 0], 'adc_minmax_mag_l1', 'MinMax ADC L1', 'L1', color='tab:blue')
    safe_plot(axes[4, 1], 'adc_minmax_mag_mse', 'MinMax ADC MSE', 'MSE', color='tab:green', log_scale=True)
    safe_plot(axes[4, 2], 'adc_minmax_mag_rmse', 'MinMax ADC RMSE', 'RMSE', color='tab:orange')
    safe_plot(axes[4, 3], 'adc_minmax_mag_mae', 'MinMax ADC MAE', 'MAE', color='tab:red')
    safe_plot(axes[4, 4], 'adc_minmax_mag_pearson', 'MinMax ADC Pearson', 'Pearson r', color='tab:brown')

    # Row 5: RA metrics with alignment
    safe_plot(axes[5, 0], 'ra_mse', 'RA MSE (aligned)', 'MSE', color='tab:pink', log_scale=True)
    safe_plot(axes[5, 1], 'ra_rmse', 'RA RMSE (aligned)', 'RMSE', color='tab:olive')
    safe_plot(axes[5, 2], 'ra_psnr', 'RA PSNR (aligned)', 'PSNR (dB)', color='tab:cyan')
    safe_plot(axes[5, 3], 'ra_ssim', 'RA SSIM (aligned)', 'SSIM', color='orange')
    safe_plot(axes[5, 4], 'ra_lpips', 'RA LPIPS (aligned)', 'LPIPS', color='tab:gray')

    # Row 6: RA metrics without alignment
    safe_plot(axes[6, 0], 'ra_unaligned_mse', 'RA MSE (unaligned)', 'MSE', color='tab:pink', log_scale=True)
    safe_plot(axes[6, 1], 'ra_unaligned_rmse', 'RA RMSE (unaligned)', 'RMSE', color='tab:olive')
    safe_plot(axes[6, 2], 'ra_unaligned_psnr', 'RA PSNR (unaligned)', 'PSNR (dB)', color='tab:cyan')
    safe_plot(axes[6, 3], 'ra_unaligned_ssim', 'RA SSIM (unaligned)', 'SSIM', color='orange')
    safe_plot(axes[6, 4], 'ra_unaligned_lpips', 'RA LPIPS (unaligned)', 'LPIPS', color='tab:gray')

    # Row 7: Cartesian RA metrics
    safe_plot(axes[7, 0], 'cart_mse', 'Cart RA MSE', 'MSE', color='tab:pink', log_scale=True)
    safe_plot(axes[7, 1], 'cart_rmse', 'Cart RA RMSE', 'RMSE', color='tab:olive')
    safe_plot(axes[7, 2], 'cart_psnr', 'Cart RA PSNR', 'PSNR (dB)', color='tab:cyan')
    safe_plot(axes[7, 3], 'cart_ssim', 'Cart RA SSIM', 'SSIM', color='orange')
    safe_plot(axes[7, 4], 'cart_corr', 'Cart RA Corr', 'Correlation', color='tab:blue')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
