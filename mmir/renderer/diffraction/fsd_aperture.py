"""Virtual screen aperture construction for the fsdBSDF."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FsdEdge:
    """A single boundary edge on the virtual screen."""
    e: np.ndarray       # [2] 2D edge vector (u2 - u1) on screen
    v: np.ndarray       # [2] 2D edge midpoint on screen
    a: complex          # Complex beam amplitude at first vertex (includes √opacity)
    b: complex          # Complex beam amplitude at second vertex (includes √opacity)
    Phat: float         # Edge-diffracted power (for sampling weight)
    Phat_accum: float   # Cumulative power (for binary search in SIR)
    opacity: float = 1.0      # Fresnel opacity [0, 1] (for Phase B re-computation)
    face_idx: int = -1         # Global face index of owning triangle
    cos_theta: float = 1.0     # cos(θ_i) at this edge's triangle surface
    # Per-edge barycentric coordinates of the edge midpoint within the owning triangle.
    # Used for per-vertex material interpolation in Phase B (more accurate than centroid).
    bary_u: float = -1.0       # -1 = not set (fall back to per-face gather)
    bary_v: float = -1.0
    vert_idx_0: int = -1       # Global vertex indices of owning triangle
    vert_idx_1: int = -1
    vert_idx_2: int = -1
    # Jones polarization data (for Phase B complex correction)
    jones_real: float = 1.0    # Re(E_jones) from Phase A
    jones_imag: float = 0.0    # Im(E_jones) from Phase A
    tx_amp_s: float = 0.0      # TX polarization s-component (frozen geometry)
    tx_amp_p: float = 0.0      # TX polarization p-component (frozen geometry)
    rx_amp_s: float = 0.0      # RX polarization s-component (frozen geometry)
    rx_amp_p: float = 0.0      # RX polarization p-component (frozen geometry)


# Numerically integrated total powers (Eq. 32-33, Supplemental S2.36-S2.37)
_INTEGRAL_1 = 0.0045255085   # ∫(1-χ)|α₁|² for Phat
_INTEGRAL_2 = 0.114875434    # ∫(1-χ)|α₂|² for Phat
_INTEGRAL_1_FULL = 0.004973515  # ∫|α₁|² for full Pj (with central lobe)
_INTEGRAL_2_FULL = 14.2569085397  # ∫|α₂|² for full Pj (with central lobe)


def _jones_reflectance_numpy_batch(
    eps_real: np.ndarray,       # [K]
    eps_imag: np.ndarray,       # [K]
    cos_theta_i: np.ndarray,    # [K]
    wo: np.ndarray,             # [3] direction toward RX
    fn: np.ndarray,             # [K, 3] unit face normals
    tx_pol: np.ndarray,         # [3] TX polarization direction
    rx_pol: np.ndarray,         # [3] RX polarization direction
) -> tuple:
    """Batch-vectorized Jones reflectance. Returns tuple of [K] arrays: (jones_real, jones_imag,
    jones_power, tx_s, tx_p, rx_s, rx_p).
    """
    len(eps_real)

    # s = wo × fn: [K, 3]
    s_unnorm = np.cross(wo[np.newaxis, :], fn)  # [K, 3]
    s_len = np.linalg.norm(s_unnorm, axis=1, keepdims=True)  # [K, 1]

    # Fallback for grazing incidence
    fallback = np.cross(wo, np.array([1.0, 0.0, 0.0]))
    fl = np.linalg.norm(fallback)
    fallback = fallback / max(fl, 1e-10)

    need_fallback = (s_len.ravel() < 1e-6)  # [K]
    s = np.where(need_fallback[:, np.newaxis],
                 fallback[np.newaxis, :],
                 s_unnorm / np.maximum(s_len, 1e-10))

    # p = s × wo: [K, 3]
    p = np.cross(s, wo[np.newaxis, :])
    p_len = np.linalg.norm(p, axis=1, keepdims=True)
    p = p / np.maximum(p_len, 1e-10)

    # TX projections: [K]
    tx_s = np.einsum('j,ij->i', tx_pol, s)
    tx_p = np.einsum('j,ij->i', tx_pol, p)

    # Complex Fresnel r_s, r_p
    eps_mag = np.sqrt(eps_real**2 + eps_imag**2)
    n = np.sqrt((eps_mag + eps_real) / 2.0)
    kappa = np.sqrt(np.maximum(0.0, (eps_mag - eps_real) / 2.0))

    sin2 = 1.0 - cos_theta_i**2
    n2_real = n * n - kappa * kappa
    n2_imag = -2.0 * n * kappa

    xi_real = n2_real - sin2
    xi_imag = n2_imag
    xi_mag = np.sqrt(xi_real**2 + xi_imag**2)
    xi_arg = np.arctan2(xi_imag, xi_real)
    a = np.sqrt(xi_mag) * np.cos(xi_arg / 2.0)
    b = np.sqrt(xi_mag) * np.sin(xi_arg / 2.0)

    rs_num = (cos_theta_i - a) - 1j * b
    rs_den = (cos_theta_i + a) + 1j * b
    r_s = np.where(np.abs(rs_den) > 1e-10, rs_num / rs_den, 0.0 + 0.0j)

    # r_p = (ñ²cos_θ - (a+jb)) / (ñ²cos_θ + (a+jb))
    n2cos = (n2_real + 1j * n2_imag) * cos_theta_i
    rp_num = n2cos - (a + 1j * b)
    rp_den = n2cos + (a + 1j * b)
    r_p = np.where(np.abs(rp_den) > 1e-10, rp_num / rp_den, 0.0 + 0.0j)

    # Jones: E_s_out = r_s * tx_s, E_p_out = r_p * tx_p
    E_s_out = r_s * tx_s
    E_p_out = r_p * tx_p

    # RX projections (outgoing s/p basis)
    s_out = s
    p_out = np.cross(s_out, wo[np.newaxis, :])
    p_out_len = np.linalg.norm(p_out, axis=1, keepdims=True)
    p_out = p_out / np.maximum(p_out_len, 1e-10)

    rx_s = np.einsum('j,ij->i', rx_pol, s_out)
    rx_p = np.einsum('j,ij->i', rx_pol, p_out)

    E_jones = E_s_out * rx_s + E_p_out * rx_p

    jones_real = E_jones.real.astype(np.float64)
    jones_imag = E_jones.imag.astype(np.float64)
    jones_power = jones_real**2 + jones_imag**2

    return jones_real, jones_imag, jones_power, tx_s, tx_p, rx_s, rx_p
