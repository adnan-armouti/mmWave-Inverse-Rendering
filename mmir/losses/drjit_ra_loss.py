"""DrJit-compatible RA loss with gradient flow through PyTorch FFT operations."""

import torch
import numpy as np
import drjit as dr
import mitsuba as mi
from typing import Tuple, Optional, Dict

from mmir.data.ra_utils import adc_to_ra_complex


def compute_ra_loss_with_gradients(
    rendered_adc_real: mi.Float,
    rendered_adc_imag: mi.Float,
    gt_adc_full: torch.Tensor,
    NT: int,
    NR: int,
    K: int,
    normalize: bool = True,
    weight: float = 1.0,
    ssim_weight: float = 0.0,
    lpips_weight: float = 0.0,
    lpips_model = None,
    ra_use_log: bool = False,
    ra_loss_type: str = "l2",
    log_epsilon: float = 1e-6,
    log_scale: float = 1.0,
    # ADC phase loss parameters
    adc_phase_weight: float = 0.0,
    adc_phase_loss_type: str = "none",
    phase_magnitude_weighting: bool = True,
    phase_magnitude_power: float = 0.5,
    phase_reference_channel: int = 0,
    phase_use_gt_magnitude: bool = True,
    phase_magnitude_clip_max: float = 0.0,
    # Normalization mode
    use_joint_normalization: bool = False  # If True, both normalized by GT max (preserves scale for gain learning)
) -> Tuple[mi.Float, mi.Float, mi.Float, Optional[Dict[str, float]]]:
    """Compute RA and ADC phase losses with gradient flow back to DrJit tensors."""
    # Step 1: Extract current values (without breaking gradient graph)
    with dr.suspend_grad():
        adc_real_np = np.array(rendered_adc_real).reshape(NT, NR, K)
        adc_imag_np = np.array(rendered_adc_imag).reshape(NT, NR, K)

    # Step 2: Create PyTorch tensors with gradient tracking
    adc_real_torch = torch.from_numpy(adc_real_np.copy()).float()
    adc_imag_torch = torch.from_numpy(adc_imag_np.copy()).float()

    adc_real_torch.requires_grad = True
    adc_imag_torch.requires_grad = True

    # Step 3: Compute complex RA using PyTorch FFT (SHARED across all loss terms)
    rendered_adc_full = torch.stack([adc_real_torch, adc_imag_torch], dim=-1)

    ra_rendered_complex = adc_to_ra_complex(rendered_adc_full)  # Complex tensor (127, 256)
    ra_gt_complex = adc_to_ra_complex(gt_adc_full)  # Complex tensor (127, 256)

    # Compute magnitude for SSIM/LPIPS (before normalization for consistency)
    ra_rendered_mag = torch.abs(ra_rendered_complex)
    ra_gt_mag = torch.abs(ra_gt_complex)

    # Extract real and imaginary parts for L2 loss
    ra_rendered_real = ra_rendered_complex.real
    ra_rendered_imag = ra_rendered_complex.imag
    ra_gt_real = ra_gt_complex.real
    ra_gt_imag = ra_gt_complex.imag

    # MIN-MAX normalization (matches inspect_ra_coir.py)
    # Two modes: separate (each normalized independently) or joint (using GT range)
    eps = 1e-12
    ra_rendered_min = torch.min(ra_rendered_mag)
    ra_rendered_max = torch.max(ra_rendered_mag)
    ra_rendered_range = torch.maximum(ra_rendered_max - ra_rendered_min, torch.tensor(eps))
    ra_rendered_mag_norm = (ra_rendered_mag - ra_rendered_min) / ra_rendered_range

    ra_gt_min = torch.min(ra_gt_mag)
    ra_gt_max = torch.max(ra_gt_mag)
    ra_gt_range = torch.maximum(ra_gt_max - ra_gt_min, torch.tensor(eps))
    ra_gt_mag_norm = (ra_gt_mag - ra_gt_min) / ra_gt_range

    # Scale real/imag to preserve phase
    ra_rendered_scale = ra_rendered_mag_norm / torch.maximum(ra_rendered_mag, torch.tensor(eps))
    ra_rendered_real = ra_rendered_real * ra_rendered_scale
    ra_rendered_imag = ra_rendered_imag * ra_rendered_scale

    ra_gt_scale = ra_gt_mag_norm / torch.maximum(ra_gt_mag, torch.tensor(eps))
    ra_gt_real = ra_gt_real * ra_gt_scale
    ra_gt_imag = ra_gt_imag * ra_gt_scale

    # Initialize total loss and loss dict
    total_loss = torch.tensor(0.0, requires_grad=True)
    loss_dict = {'ra_l2': 0.0, 'ra_ssim_loss': 0.0, 'ra_lpips_loss': 0.0, 'adc_phase_loss': 0.0,
                 'ra_use_log': ra_use_log, 'ra_loss_type': ra_loss_type, 'adc_phase_loss_type': adc_phase_loss_type}

    # Compute RA magnitude loss based on configuration
    if weight > 0:
        ra_mag_loss = torch.mean((ra_rendered_mag_norm - ra_gt_mag_norm) ** 2)

        total_loss = total_loss + weight * ra_mag_loss
        loss_dict['ra_l2'] = float(ra_mag_loss.item())  # Keep key name for compatibility

    # Step 4: Compute gradients (combined from all loss terms)
    total_loss.backward()

    grad_real_np = adc_real_torch.grad.numpy().flatten().astype(np.float32)
    grad_imag_np = adc_imag_torch.grad.numpy().flatten().astype(np.float32)

    # Step 5: Convert to DrJit tensors
    loss_drjit = mi.Float(float(total_loss.item()))
    grad_real_drjit = mi.Float(grad_real_np)
    grad_imag_drjit = mi.Float(grad_imag_np)

    return loss_drjit, grad_real_drjit, grad_imag_drjit, loss_dict
