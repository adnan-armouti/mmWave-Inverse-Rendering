"""Antenna element pattern functions for TX and RX arrays."""

import drjit as dr
import mitsuba as mi
from typing import Tuple, Optional


# Antenna Pattern Loading from File

import numpy as np


def evaluate_combined_gain(pattern_loader: 'AntennaPatternLoader',
                           direction: mi.Vector3f,
                           orientation: Optional['mi.Vector3f'] = None) -> mi.Float:
    """Evaluate antenna gain by combining E and H plane patterns (CORRECT approach)."""
    # Get both plane gains
    gain_E, gain_H = pattern_loader.evaluate_linear(direction, orientation)

    # Combine using product (standard for separable patterns)
    # This represents the full 3D radiation pattern
    gain_unscaled = gain_E * gain_H

    # Apply scaling factor C to anchor to absolute peak gain
    # C = G_max / P_max, computed at pattern load time
    # Use Python float directly to avoid blocking gradients
    gain_combined = pattern_loader.C_scale * gain_unscaled

    return gain_combined


class AntennaPatternLoader:
    """Load and evaluate antenna radiation patterns from file."""

    def __init__(self, pattern_file: str):
        """Load antenna pattern from .npy file."""
        # Load pattern data as numpy (for backup and info)
        self.pattern_data = np.load(pattern_file)

        if self.pattern_data.ndim != 2 or self.pattern_data.shape[1] != 2:
            raise ValueError(f"Expected pattern shape [N, 2], got {self.pattern_data.shape}")

        self.num_angles = self.pattern_data.shape[0]

        # Extract E and H plane patterns (in dB) - numpy arrays for info only
        E_plane_db_np = self.pattern_data[:, 0]
        H_plane_db_np = self.pattern_data[:, 1]

        # Convert to linear scale - numpy arrays for info only
        E_plane_linear_np = 10.0 ** (E_plane_db_np / 10.0)
        H_plane_linear_np = 10.0 ** (H_plane_db_np / 10.0)

        # CRITICAL: Store patterns as DrJit arrays for differentiability
        # These are the arrays that will be used in forward/backward pass
        self.E_plane_linear = mi.Float(E_plane_linear_np)
        self.H_plane_linear = mi.Float(H_plane_linear_np)

        # Angle grid (assume uniform spacing 0 to 360 degrees)
        self.angles_deg = np.linspace(0, 360, self.num_angles, dtype=np.float32)
        self.angles_rad = self.angles_deg * np.pi / 180.0

        # Step 1: Estimate G_max from the maximum dB value in either cut
        # This assumes the true 2D peak occurs near a principal plane peak
        G_max_dB = max(E_plane_db_np.max(), H_plane_db_np.max())
        G_max_lin = 10.0 ** (G_max_dB / 10.0)

        # Step 2: Compute P_max = max(G_E x G_H) over all angles
        # This is the peak of the separable product
        P_max = (E_plane_linear_np * H_plane_linear_np).max()

        # Step 3: Compute scaling factor C (as Python float - it's a constant)
        self.C_scale = float(G_max_lin / P_max)

        print(f"[AntennaPatternLoader] Loaded pattern from {pattern_file}")
        print(f"  Angles: {self.num_angles} samples from 0 to 360 degrees")
        print(f"  E-plane range: {E_plane_db_np.min():.2f} to {E_plane_db_np.max():.2f} dBi")
        print(f"  H-plane range: {H_plane_db_np.min():.2f} to {H_plane_db_np.max():.2f} dBi")
        print(f"  Estimated G_max: {G_max_dB:.2f} dBi ({G_max_lin:.4f} linear)")
        print(f"  Separable P_max: {10*np.log10(P_max):.2f} dBi ({P_max:.4f} linear)")
        print(f"  Scaling factor C: {self.C_scale:.6f} ({10*np.log10(self.C_scale):.2f} dB)")
        print("  Pattern data stored as DrJit arrays (differentiable)")

    def enable_gradients(self):
        """Enable gradient tracking for antenna pattern parameters."""
        dr.enable_grad(self.E_plane_linear)
        dr.enable_grad(self.H_plane_linear)
        print(f"  Gradients enabled for antenna pattern data ({self.num_angles} samples per plane)")

    def get_gradients(self):
        """Extract gradients for E-plane and H-plane pattern arrays."""
        e_grad = dr.grad(self.E_plane_linear)
        h_grad = dr.grad(self.H_plane_linear)
        if dr.width(e_grad) == 0 and dr.width(h_grad) == 0:
            return None
        e_np = np.array(e_grad, dtype=np.float32)
        h_np = np.array(h_grad, dtype=np.float32)
        return np.stack([e_np, h_np], axis=0)

    def apply_update(self, delta: np.ndarray):
        """Apply an additive update to the pattern parameters."""
        e_np = np.array(self.E_plane_linear, dtype=np.float32) + delta[0]
        h_np = np.array(self.H_plane_linear, dtype=np.float32) + delta[1]
        # Clamp to positive (pattern gains must be > 0)
        e_np = np.maximum(e_np, 1e-6)
        h_np = np.maximum(h_np, 1e-6)
        self.E_plane_linear = mi.Float(e_np)
        self.H_plane_linear = mi.Float(h_np)

    def evaluate_linear(self,
                       direction: mi.Vector3f,
                       orientation: Optional['mi.Vector3f'] = None) -> Tuple[mi.Float, mi.Float]:
        """Evaluate antenna pattern at given direction(s) in linear scale."""
        # If orientation provided, transform direction to antenna local frame
        y_local = dr.normalize(orientation)

        # Construct perpendicular X and Z axes
        # Use (1, 0, 0) as auxiliary vector, unless parallel to Y
        aux = mi.Vector3f(1.0, 0.0, 0.0)
        # If Y is nearly parallel to X axis, use Z axis instead
        parallel_threshold = 0.99
        is_parallel = dr.abs(dr.dot(y_local, aux)) > parallel_threshold
        aux = dr.select(is_parallel, mi.Vector3f(0.0, 0.0, 1.0), aux)

        # Z = normalize(Y x aux)  [up direction]
        z_local = dr.normalize(dr.cross(y_local, aux))
        # X = Z x Y  [right direction, completes right-handed frame]
        x_local = dr.cross(z_local, y_local)

        # Transform direction to local frame: dot with each basis vector
        direction_local = mi.Vector3f(
            dr.dot(direction, x_local),  # X component
            dr.dot(direction, y_local),  # Y component (along boresight)
            dr.dot(direction, z_local)   # Z component
        )

        # Normalize direction
        direction_local = dr.normalize(direction_local)

        # Convert to spherical coordinates with +Y as boresight
        x = direction_local.x
        y = direction_local.y
        z = direction_local.z

        # Phi_E: elevation angle in YZ plane (for E-plane)
        # Positive = upward (+Z), Negative = downward (-Z)
        # atan2(z, y) gives angle from +Y in YZ plane
        phi_E = dr.atan2(z, y)

        # Phi_H: azimuth angle in XY plane (for H-plane)
        # Positive = rightward (+X), Negative = leftward (-X)
        # atan2(x, y) gives angle from +Y in XY plane
        phi_H = dr.atan2(x, y)

        # E-plane: use elevation angle phi_E
        # phi_E in [-pi, pi] represents [-180deg, +180deg] in elevation
        # Add 180deg to map to [0deg, 360deg] for array indexing
        angle_E_deg = phi_E * 180.0 / dr.pi + 180.0

        # H-plane: use azimuth angle phi_H
        # phi_H in [-pi, pi] represents [-180deg, +180deg] in azimuth
        # Add 180deg to map to [0deg, 360deg] for array indexing
        angle_H_deg = phi_H * 180.0 / dr.pi + 180.0

        # Use differentiable interpolation
        gain_E = self._differentiable_interp(angle_E_deg, self.E_plane_linear)
        gain_H = self._differentiable_interp(angle_H_deg, self.H_plane_linear)

        return gain_E, gain_H

    def _differentiable_interp(self, angle_deg: mi.Float, pattern_data: mi.Float) -> mi.Float:
        """Differentiable interpolation of pattern data."""
        # pattern_data is already a DrJit array, use directly
        pattern_dr = pattern_data

        # Map angle to continuous index in [0, num_angles)
        # Pattern has 361 samples for 0-360 degrees (inclusive endpoints)
        idx_float = angle_deg * (self.num_angles - 1) / 360.0

        # Clamp to valid range [0, num_angles-1]
        # Pattern is periodic, so we can safely clamp
        max_idx = float(self.num_angles - 1)
        idx_float = dr.clamp(idx_float, 0.0, max_idx)

        # Get integer indices for linear interpolation
        idx_low = dr.floor(idx_float)
        idx_high = idx_low + 1.0

        # Compute interpolation weight
        weight = idx_float - idx_low

        # Wrap high index (circular boundary)
        idx_high = dr.select(idx_high >= float(self.num_angles),
                            idx_high - float(self.num_angles),
                            idx_high)

        # Gather values using integer indices
        idx_low_int = mi.UInt32(idx_low)
        idx_high_int = mi.UInt32(idx_high)

        val_low = dr.gather(mi.Float, pattern_dr, idx_low_int)
        val_high = dr.gather(mi.Float, pattern_dr, idx_high_int)

        # Linear interpolation
        gain = val_low * (1.0 - weight) + val_high * weight

        return gain


# RRTS-Compatible Antenna Pattern Functions

def load_pattern_rrts_format(npy_path: str) -> np.ndarray:
    """Load antenna pattern from NPY file in RRTS format."""
    data = np.load(npy_path)
    if data.shape != (361, 2):
        raise ValueError(f"Expected pattern shape [361, 2], got {data.shape}")

    # RRTS uses dB values directly - DO NOT convert to linear!
    E_dB = data[:, 0]  # [361]
    H_dB = data[:, 1]  # [361]

    # Build RRTS format: [361, 2, 5]
    # Only channel 0 (abs gain) is used by bilinear_gain_single
    pattern = np.zeros((361, 2, 5), dtype=np.float32)
    pattern[:, 0, 0] = E_dB  # E-plane gain in dB (RRTS uses raw dB values!)
    pattern[:, 1, 0] = H_dB  # H-plane gain in dB

    print(f"[load_pattern_rrts_format] Loaded {npy_path}")
    print(f"  Shape: {pattern.shape}")
    print(f"  E-plane dB range: [{E_dB.min():.2f}, {E_dB.max():.2f}] dB")
    print(f"  H-plane dB range: [{H_dB.min():.2f}, {H_dB.max():.2f}] dB")
    print("  NOTE: RRTS uses dB values directly as multipliers (not linear)")

    return pattern


def _pattern_scaling_constant_dB(pattern: np.ndarray) -> float:
    """Precompute the separable pattern scaling constant C (in dB)."""
    E_linear = np.power(10.0, pattern[:, 0, 0] / 10.0)
    H_linear = np.power(10.0, pattern[:, 1, 0] / 10.0)
    G_max_dB = max(pattern[:, 0, 0].max(), pattern[:, 1, 0].max())
    G_max_linear = np.power(10.0, G_max_dB / 10.0)
    P_max = (E_linear * H_linear).max()
    C = G_max_linear / P_max
    C_dB = 10.0 * np.log10(C + 1e-10)
    return float(C_dB)


def evaluate_gain_rrts_style_drjit_product(
    direction: 'mi.Vector3f',
    pattern: np.ndarray,
    orientation: 'mi.Vector3f',
) -> 'mi.Float':
    """Orientation-aware separable E×H antenna pattern evaluation (DrJit GPU version)."""
    import math
    nPhi = pattern.shape[0]  # 361

    # Upload E-plane and H-plane pattern data to GPU (361 floats each — tiny)
    pattern_E = mi.Float(pattern[:, 0, 0].astype(np.float32))  # [nPhi]
    pattern_H = mi.Float(pattern[:, 1, 0].astype(np.float32))  # [nPhi]

    # Precompute scaling constant (once, on CPU — 361 elements)
    C_dB = _pattern_scaling_constant_dB(pattern)

    # Build local coordinate frame:  Y = boresight, Z = up, X = right
    mi.Float(1e-10)

    # Normalize inputs
    dir_norm = dr.normalize(direction)
    y_local = dr.normalize(orientation)

    # Gram-Schmidt: choose auxiliary vector not parallel to Y
    parallel = dr.abs(y_local.x) > mi.Float(0.9)
    aux = mi.Vector3f(
        dr.select(parallel, mi.Float(0.0), mi.Float(1.0)),
        mi.Float(0.0),
        dr.select(parallel, mi.Float(1.0), mi.Float(0.0)),
    )
    z_local = dr.normalize(dr.cross(y_local, aux))
    x_local = dr.cross(z_local, y_local)

    # Project direction into local frame
    dir_x = dr.dot(dir_norm, x_local)
    dir_y = dr.dot(dir_norm, y_local)
    dir_z = dr.dot(dir_norm, z_local)

    # Compute E-plane and H-plane angles relative to boresight
    phi_E = dr.atan2(dir_z, dir_y)  # elevation in YZ plane  [-π, π]
    phi_H = dr.atan2(dir_x, dir_y)  # azimuth  in XY plane  [-π, π]

    # Map to degrees [0, 360]:  pattern index 0 = −180°, 180 = 0° (boresight)
    RAD2DEG = mi.Float(180.0 / math.pi)
    angle_E_deg = phi_E * RAD2DEG + mi.Float(180.0)
    angle_H_deg = phi_H * RAD2DEG + mi.Float(180.0)

    # E-plane interpolation
    scale = mi.Float((nPhi - 1) / 360.0)
    idx_E_f = dr.clamp(angle_E_deg * scale, mi.Float(0.0), mi.Float(nPhi - 1))
    idx_E_lo = mi.UInt32(dr.floor(idx_E_f))
    idx_E_hi = dr.minimum(idx_E_lo + mi.UInt32(1), mi.UInt32(nPhi - 1))
    w_E = idx_E_f - mi.Float(idx_E_lo)
    gain_E = dr.gather(mi.Float, pattern_E, idx_E_lo) * (mi.Float(1.0) - w_E) \
           + dr.gather(mi.Float, pattern_E, idx_E_hi) * w_E

    # H-plane interpolation
    idx_H_f = dr.clamp(angle_H_deg * scale, mi.Float(0.0), mi.Float(nPhi - 1))
    idx_H_lo = mi.UInt32(dr.floor(idx_H_f))
    idx_H_hi = dr.minimum(idx_H_lo + mi.UInt32(1), mi.UInt32(nPhi - 1))
    w_H = idx_H_f - mi.Float(idx_H_lo)
    gain_H = dr.gather(mi.Float, pattern_H, idx_H_lo) * (mi.Float(1.0) - w_H) \
           + dr.gather(mi.Float, pattern_H, idx_H_hi) * w_H

    # Combined gain: C_dB + gain_E_dB + gain_H_dB
    return mi.Float(C_dB) + gain_E + gain_H
