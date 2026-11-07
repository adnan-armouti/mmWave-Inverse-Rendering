"""Phase B differentiable re-synthesis from cached geometry."""

from typing import Optional
import drjit as dr
import mitsuba as mi
import numpy as np

from ..specular.paths import SpecularPaths


# Import diffraction modules (optional)
try:
    _HAS_DIFFRACTION = True
except ImportError:
    _HAS_DIFFRACTION = False

# Import pattern evaluation function
try:
    from mmir.sensor.element_patterns import (
        evaluate_combined_gain, AntennaPatternLoader, load_pattern_rrts_format,
        evaluate_gain_rrts_style_drjit_product,
    )
except ImportError:
    evaluate_combined_gain = None
    AntennaPatternLoader = None
    load_pattern_rrts_format = None
    evaluate_gain_rrts_style_numpy = None
    evaluate_gain_rrts_style_vectorized = None
    evaluate_gain_rrts_style_drjit = None
    evaluate_gain_rrts_style_drjit_product = None
    RRTSPatternLoader = None

# Physical Constants
C = 299792458.0


def _synthesize_specular_paths(
    self,
    specular_paths: SpecularPaths,
    adc_real_flat: 'mi.Float',
    adc_imag_flat: 'mi.Float',
    n_tx: int,
    n_rx: int,
    K: int,
    tx_boresights: Optional['mi.Vector3f'] = None,
    rx_boresights: Optional['mi.Vector3f'] = None,
    tx_pattern_rrts: Optional[np.ndarray] = None,
    rx_pattern_rrts: Optional[np.ndarray] = None,
    tx_pattern_loader: Optional['AntennaPatternLoader'] = None,
    rx_pattern_loader: Optional['AntennaPatternLoader'] = None,
    antenna_gain_linear: bool = False,
    verbose: bool = True,
):
    """Synthesize ADC contributions from deterministic specular paths."""
    from ..utils.math import gather_vector3f

    sp = specular_paths
    valid = sp.valid
    n_paths = sp.n_paths

    n_valid_spec = int(dr.sum(mi.UInt32(valid))[0]) if hasattr(dr.sum(mi.UInt32(valid)), '__getitem__') else int(dr.sum(mi.UInt32(valid)))

    if verbose:
        print(f"\n  [Specular Branch] Synthesizing {n_valid_spec:,} valid specular paths")

    if n_valid_spec == 0:
        return None  # Return None antenna gain

    # Step S1: Compute specular power
    # For deterministic specular paths, use integrated reflectance
    # R_specular = η × τ_eff × A (dimensionless), NOT the BSDF density.
    # This matches the reference renderer's Phase A which uses sp.R_specular.
    _eval_sr = getattr(self.bsdf, 'eval_specular_reflectance', None)
    prim_np = np.array(sp.prim_ids).astype(np.int64)
    prim_np = np.clip(prim_np, 0, self.triangle_materials.shape[0] - 1)
    mat = self.triangle_materials[prim_np]
    _sp_eps_r = mi.Float(mat[:, 0].astype(np.float32))
    _sp_eps_i = mi.Float(mat[:, 1].astype(np.float32))
    _sp_sh = mi.Float(mat[:, 2].astype(np.float32))
    _sp_lc = mi.Float(mat[:, 3].astype(np.float32))
    _sp_tau = mi.Float(mat[:, 4].astype(np.float32))
    _sp_thick = mi.Float(mat[:, 5].astype(np.float32))
    sp_reflectance = _eval_sr(
        cos_theta_i=sp.cos_theta_i,
        eps_real=_sp_eps_r, eps_imag=_sp_eps_i,
        sigma_h=_sp_sh, l_c=_sp_lc, tau=_sp_tau,
        thickness=_sp_thick,
    )
    specular_power = sp_reflectance * sp.cos_theta_i * sp.cos_theta_r * sp.A_tri

    # Step S2: Antenna pattern evaluation (if available)
    antenna_gain = mi.Float(1.0)
    pattern_mode = getattr(self.render_config, 'pattern_mode', 'fixed')

    # GPU-native path
    _use_gpu_spec = (
        pattern_mode == 'fixed'
        and evaluate_combined_gain is not None
        and (tx_pattern_loader is not None or rx_pattern_loader is not None)
    )

    if _use_gpu_spec:
        # Antenna direction: from antenna to specular point = -(dir_to_tx/rx)
        dir_tx_to_spec = -sp.dir_to_tx
        dir_rx_to_spec = -sp.dir_to_rx

        # Gather per-path boresights
        tx_boresights_exp = gather_vector3f(tx_boresights, sp.tx_idx) if tx_boresights is not None else None
        rx_boresights_exp = gather_vector3f(rx_boresights, sp.rx_idx) if rx_boresights is not None else None

        gain_tx_lin = evaluate_combined_gain(tx_pattern_loader, dir_tx_to_spec, tx_boresights_exp)

        gain_rx_lin = evaluate_combined_gain(rx_pattern_loader, dir_rx_to_spec, rx_boresights_exp)

        antenna_gain = gain_tx_lin * gain_rx_lin
        specular_power = specular_power * antenna_gain

    elif tx_pattern_rrts is not None or rx_pattern_rrts is not None:
        # DrJit GPU fallback: all-GPU evaluation (no CPU↔GPU transfers)
        dir_tx_to_spec = mi.Vector3f(-sp.dir_to_tx.x, -sp.dir_to_tx.y, -sp.dir_to_tx.z)
        dir_rx_to_spec = mi.Vector3f(-sp.dir_to_rx.x, -sp.dir_to_rx.y, -sp.dir_to_rx.z)

        if pattern_mode == 'legacy':
            if tx_pattern_rrts is not None and evaluate_gain_rrts_style_drjit is not None:
                gain_tx_dB = evaluate_gain_rrts_style_drjit(dir_tx_to_spec, tx_pattern_rrts)
            else:
                gain_tx_dB = dr.zeros(mi.Float, n_paths)
            if rx_pattern_rrts is not None and evaluate_gain_rrts_style_drjit is not None:
                gain_rx_dB = evaluate_gain_rrts_style_drjit(dir_rx_to_spec, rx_pattern_rrts)
            else:
                gain_rx_dB = dr.zeros(mi.Float, n_paths)
        else:
            if tx_pattern_rrts is not None and evaluate_gain_rrts_style_drjit_product is not None:
                tx_boresights_exp = gather_vector3f(tx_boresights, sp.tx_idx) \
                    if tx_boresights is not None else mi.Vector3f(0.0, 1.0, 0.0)
                gain_tx_dB = evaluate_gain_rrts_style_drjit_product(
                    dir_tx_to_spec, tx_pattern_rrts, tx_boresights_exp)
            else:
                gain_tx_dB = dr.zeros(mi.Float, n_paths)
            if rx_pattern_rrts is not None and evaluate_gain_rrts_style_drjit_product is not None:
                rx_boresights_exp = gather_vector3f(rx_boresights, sp.rx_idx) \
                    if rx_boresights is not None else mi.Vector3f(0.0, 1.0, 0.0)
                gain_rx_dB = evaluate_gain_rrts_style_drjit_product(
                    dir_rx_to_spec, rx_pattern_rrts, rx_boresights_exp)
            else:
                gain_rx_dB = dr.zeros(mi.Float, n_paths)

        if antenna_gain_linear or self.render_config.use_radar_equation:
            gain_tx_lin = dr.power(mi.Float(10.0), gain_tx_dB * mi.Float(0.1))
            gain_rx_lin = dr.power(mi.Float(10.0), gain_rx_dB * mi.Float(0.1))
        else:
            gain_tx_lin = gain_tx_dB
            gain_rx_lin = gain_rx_dB

        antenna_gain = gain_tx_lin * gain_rx_lin
        specular_power = specular_power * antenna_gain

    # Step S3: Compute amplitude
    # Path loss: 1/(d_tx² × d_rx²) — both distances appear explicitly
    # because specular branch is deterministic (no cosine hemisphere sampling)
    path_loss = mi.Float(1.0) / (sp.d_tx * sp.d_tx * sp.d_rx * sp.d_rx)

    radar_scale = mi.Float(self.radar_constant * self.rx_dBFS_scale * self.adc_scale)
    E_amplitude_spec = radar_scale * dr.sqrt(
        dr.maximum(specular_power * path_loss, mi.Float(0.0))
    )

    # Step S4: Phase computation and scatter-add
    R_total_spec = sp.d_tx + sp.d_rx
    tau_spec = R_total_spec / mi.Float(C)

    TWO_PI = 2.0 * np.pi
    phi_const_spec = mi.Float(TWO_PI * self.min_freq) * tau_spec
    phi_slope_spec = mi.Float(TWO_PI * self.slope) * tau_spec

    # Time grid
    t_grid_dr = self._t_grid_dr  # [K] (cached on GPU)

    # Expand to [n_paths × K]
    phi_const_3d = dr.tile(phi_const_spec, K)
    phi_slope_3d = dr.tile(phi_slope_spec, K)
    t_k = dr.repeat(t_grid_dr, n_paths)
    weight_3d = dr.tile(E_amplitude_spec, K)
    active_3d = dr.tile(valid, K)

    # Phase
    phi = phi_const_3d + phi_slope_3d * t_k
    cos_phi = dr.cos(phi)
    sin_phi = dr.sin(phi)
    if not self.render_config.enable_grad_phase:
        cos_phi = dr.detach(cos_phi)
        sin_phi = dr.detach(sin_phi)

    contrib_real = weight_3d * cos_phi
    contrib_imag = weight_3d * sin_phi

    # Flat indices
    tx_idx_3d = dr.tile(sp.tx_idx, K)
    rx_idx_3d = dr.tile(sp.rx_idx, K)
    base_idx = dr.arange(mi.UInt32, n_paths * K)
    k_idx = base_idx // mi.UInt32(n_paths)

    flat_idx = tx_idx_3d * mi.UInt32(n_rx * K) + rx_idx_3d * mi.UInt32(K) + k_idx

    # Accumulate into existing ADC arrays
    dr.scatter_add(adc_real_flat, contrib_real, flat_idx, active_3d)
    dr.scatter_add(adc_imag_flat, contrib_imag, flat_idx, active_3d)

    return antenna_gain
