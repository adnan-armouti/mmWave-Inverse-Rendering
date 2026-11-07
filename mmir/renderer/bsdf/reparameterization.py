"""Physics parameter reparameterization for differentiable optimization."""

import numpy as np
import drjit as dr
import mitsuba as mi


# NUMPY HELPERS

def _np_sigmoid(x):
    """Numerically stable sigmoid for numpy arrays."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))),
                    np.exp(np.clip(x, -50, 50)) / (1.0 + np.exp(np.clip(x, -50, 50))))


def _np_logit(p):
    """Inverse sigmoid for numpy arrays."""
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


# NUMPY REPARAMETERIZATION

def reparameterize_physics_params(raw_params: np.ndarray) -> np.ndarray:
    """Map unconstrained optimizer params → bounded physics params."""
    out = np.empty_like(raw_params)

    # ε' ∈ [1.5, 10]: sigmoid reparameterization
    out[:, 0] = 1.5 + 8.5 * _np_sigmoid(raw_params[:, 0])

    # ε'' ∈ [1e-3, ~9e6]: log-scale
    # Upper bound accommodates metals: ITU aluminum σ=3.8e7 → ε''≈8.8e6 at 77 GHz.
    # exp(16) ≈ 8.9e6 covers all ITU materials including perfect conductors.
    out[:, 1] = np.exp(np.clip(raw_params[:, 1], -7.0, 16.0))

    # σ_h ∈ [1e-7, 1e-3] m: log-scale
    out[:, 2] = np.exp(np.clip(raw_params[:, 2], -16.0, -7.0))

    # l_c ∈ [5e-4, 0.1] m: log-scale
    out[:, 3] = np.exp(np.clip(raw_params[:, 3], -7.6, -2.3))

    # τ ∈ [0.05, 0.95]: sigmoid
    out[:, 4] = 0.05 + 0.9 * _np_sigmoid(raw_params[:, 4])

    # d ∈ [1e-3, 0.3] m: log-scale
    out[:, 5] = np.exp(np.clip(raw_params[:, 5], -7.0, -1.2))

    return out


def inverse_reparameterize(physics_params: np.ndarray) -> np.ndarray:
    """Physics params → unconstrained optimizer space."""
    raw = np.empty_like(physics_params)
    raw[:, 0] = _np_logit((np.clip(physics_params[:, 0], 1.5, 10.0) - 1.5) / 8.5)
    raw[:, 1] = np.log(np.clip(physics_params[:, 1], 1e-3, 9e6))
    raw[:, 2] = np.log(np.clip(physics_params[:, 2], 1e-7, 1e-3))
    raw[:, 3] = np.log(np.clip(physics_params[:, 3], 5e-4, 0.1))
    raw[:, 4] = _np_logit((np.clip(physics_params[:, 4], 0.05, 0.95) - 0.05) / 0.9)
    raw[:, 5] = np.log(np.clip(physics_params[:, 5], 1e-3, 0.3))
    return raw


# DRJIT REPARAMETERIZATION (differentiable)

def reparameterize_physics_params_drjit(raw_params):
    """DrJit-differentiable reparameterization: unconstrained → bounded physics params."""
    from .mmwave_scalar import _sigmoid

    eps_real  = mi.Float(1.5) + mi.Float(8.5) * _sigmoid(raw_params[0])    # [1.5, 10]
    eps_imag  = dr.exp(dr.clamp(raw_params[1], -7.0, 16.0))                # [1e-3, ~9e6]
    sigma_h   = dr.exp(dr.clamp(raw_params[2], -16.0, -7.0))               # [1e-7, 1e-3]
    l_c       = dr.exp(dr.clamp(raw_params[3], -7.6, -2.3))                # [5e-4, 0.1]
    tau       = mi.Float(0.05) + mi.Float(0.9) * _sigmoid(raw_params[4])   # [0.05, 0.95]
    thickness = dr.exp(dr.clamp(raw_params[5], -7.0, -1.2))                # [1e-3, 0.3]
    return [eps_real, eps_imag, sigma_h, l_c, tau, thickness]


def create_drjit_raw_params(raw_params_np):
    """Create grad-enabled DrJit arrays from numpy raw params."""
    params = []
    for col in range(6):
        p = mi.Float(raw_params_np[:, col].astype(np.float32))
        dr.enable_grad(p)
        params.append(p)
    return params


# Per-parameter learning rate scales for physics-mode optimization.
# See Phase 5C gradient table for rationale.
LR_SCALES = np.array([0.3, 0.5, 0.3, 0.3, 0.5, 0.1], dtype=np.float32)
