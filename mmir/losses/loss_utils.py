"""Loss computation and image processing utilities for training."""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


# Min-Max Normalization Utilities

def minmax_normalize_numpy(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Min-max normalize numpy array to [0, 1] range."""
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + eps)


def minmax_normalize_torch(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Min-max normalize PyTorch tensor to [0, 1] range."""
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + eps)


def as_image2d_mag(t: torch.Tensor) -> torch.Tensor:
    """Convert real-imag ADC (...,2) to magnitude and reshape to (1,1,H,W). H is product of all dims
    except the last two; W is the second-to-last dim.
    """
    t2 = t.detach().float().contiguous()
    if t2.dim() < 2:
        mag = t2.abs()
        return mag.reshape(1, 1, 1, mag.numel())
    mag = t2
    if mag.dim() == 1:
        return mag.reshape(1, 1, 1, mag.numel())
    h = int(np.prod(mag.shape[:-1])) if mag.dim() >= 2 else 1
    w = int(mag.shape[-1])
    return mag.reshape(1, 1, h, w)


def compute_psnr(x: torch.Tensor, y: torch.Tensor, data_range: Optional[float] = None) -> float:
    """Compute Peak Signal-to-Noise Ratio between two tensors."""
    x_img = as_image2d_mag(x)
    y_img = as_image2d_mag(y)
    if data_range is None:
        dr = float((y_img.max() - y_img.min()).item())
        data_range = dr if dr > 0 else 1.0
    mse = F.mse_loss(x_img, y_img, reduction="mean").item()
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(float(data_range)) - 10.0 * math.log10(float(mse))


def gaussian_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    """Create Gaussian window for SSIM computation."""
    coords = torch.arange(window_size).float() - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = (g / g.sum()).unsqueeze(0)
    window_2d = (g.t() @ g).unsqueeze(0).unsqueeze(0)
    window = window_2d.repeat(channels, 1, 1, 1)
    return window


def compute_ssim(x: torch.Tensor, y: torch.Tensor, data_range: Optional[float] = None,
                 window_size: int = 11, sigma: float = 1.5) -> float:
    """Compute Structural Similarity Index between two tensors."""
    x_img = as_image2d_mag(x)
    y_img = as_image2d_mag(y)
    if data_range is None:
        dr = float((y_img.max() - y_img.min()).item())
        data_range = dr if dr > 0 else 1.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    channel_count = x_img.shape[1]
    window = gaussian_window(window_size, sigma, channel_count).to(x_img.device, x_img.dtype)

    padding = window_size // 2
    mu_x = F.conv2d(x_img, window, padding=padding, groups=channel_count)
    mu_y = F.conv2d(y_img, window, padding=padding, groups=channel_count)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x_img * x_img, window, padding=padding, groups=channel_count) - mu_x_sq
    sigma_y_sq = F.conv2d(y_img * y_img, window, padding=padding, groups=channel_count) - mu_y_sq
    sigma_xy = F.conv2d(x_img * y_img, window, padding=padding, groups=channel_count) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(ssim_map.mean().item())
