"""End-to-end differentiable multibounce synthesis."""

from typing import Optional, Tuple
import drjit as dr
import mitsuba as mi
import numpy as np

from ..specular.paths import SpecularPaths

from mmir.renderer.utils.math import diff_ray_triangle_intersect, get_ad_triangle_vertices
from .mixture_sampling import _sample_cosine_hemisphere_drjit

# Import diffraction modules (optional)
try:
    from ..diffraction import eval_batch_drjit
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


# Multibounce End-to-End Differentiable Synthesis

def synthesize_end_to_end_multibounce(
    self,
    reservoir_hits,
    scene: 'mi.Scene',
    tx_positions: 'mi.Point3f',
    rx_positions: 'mi.Point3f',
    tx_boresights: Optional['mi.Vector3f'] = None,
    rx_boresights: Optional['mi.Vector3f'] = None,
    physics_params: list = None,
    normal_params: Optional[list] = None,
    pattern_loaders: Optional[dict] = None,
    tx_pattern_rrts: Optional[np.ndarray] = None,
    rx_pattern_rrts: Optional[np.ndarray] = None,
    tx_pattern_loader: Optional['AntennaPatternLoader'] = None,
    rx_pattern_loader: Optional['AntennaPatternLoader'] = None,
    antenna_gain_linear: bool = True,
    verbose: bool = False,
    specular_paths: Optional[object] = None,
    specular_chains: Optional[object] = None,
    include_diffuse: bool = True,
    include_specular: bool = True,
    include_diffraction: bool = True,
    include_energy_borrowing: bool = True,
    vertex_offset_params: Optional[list] = None,
    return_total_distance: bool = False,
    max_bounces: int = 2,
    nee_every_bounce: bool = True,
    rr_start_bounce: int = 3,
    rr_prob: float = 0.5,
    compute_boundary_viewpoints: bool = True,
) -> Tuple['mi.Float', 'mi.Float']:
    """Multibounce end-to-end differentiable ADC synthesis with inline differentiable
    re-intersection for full-chain gradient flow.
    """
    from ..utils.math import gather_point3f, gather_vector3f

    n_tx = dr.width(tx_positions)
    n_rx = reservoir_hits.n_rx
    n_valid = reservoir_hits.n_valid
    K = self.num_samples

    # Allocate ADC arrays
    adc_real_flat = dr.zeros(mi.Float, n_tx * n_rx * K)
    adc_imag_flat = dr.zeros(mi.Float, n_tx * n_rx * K)

    _dist_accum = [mi.Float(0.0)] if return_total_distance else None

    # Profiling support (same mechanism as single-bounce)
    getattr(self, '_profile', None)
    def _tic(label):
        pass
    def _toc(label):
        pass

    if physics_params is None:
        raise ValueError("physics_params required for end-to-end synthesis")

    # DIFFUSE MC CONTRIBUTION (Multibounce with inline diff tracing)
    if include_diffuse and n_valid > 0:
        _tic('mb_mimo_expansion')
        # Step 1: MIMO expansion [n_valid] -> [n_total]
        n_total = n_valid * n_tx * n_rx
        _n_mimo = n_tx * n_rx

        flat_arange = dr.arange(mi.UInt32, n_total)
        hit_idx = flat_arange // mi.UInt32(_n_mimo)
        tx_idx = (flat_arange // mi.UInt32(n_rx)) % mi.UInt32(n_tx)
        rx_idx = flat_arange % mi.UInt32(n_rx)

        tx_pos_expanded = gather_point3f(tx_positions, tx_idx)
        rx_pos_expanded = gather_point3f(rx_positions, rx_idx)

        # MC correction from reservoir sampling (non-diff)
        pdf_expanded = dr.gather(mi.Float, reservoir_hits.hit_pdf, hit_idx)
        rx_for_hit = dr.gather(mi.UInt32, reservoir_hits.rx_element_idx, hit_idx)
        n_attempted_expanded = mi.Float(dr.gather(
            mi.UInt32, reservoir_hits.n_attempted_per_rx, rx_for_hit))
        mc_first_bounce = mi.Float(1.0) / (
            dr.maximum(pdf_expanded, mi.Float(1e-8)) *
            dr.maximum(n_attempted_expanded, mi.Float(1.0)))
        _toc('mb_mimo_expansion')

        # Get mesh and vertex position buffer for diff re-intersection
        mesh = scene.shapes()[0]
        params = mi.traverse(scene)
        vp_key = None
        for k in params.keys():
            if k.endswith('vertex_positions'):
                vp_key = k
                break
        vp_buffer = params[vp_key or 'vertex_positions']

        # Step 2: First bounce — differentiable re-intersection
        _tic('mb_first_bounce_reintersect')
        prim_ids_1 = dr.gather(mi.UInt32, reservoir_hits.hit_prim_ids, hit_idx)
        current_active = dr.full(mi.Bool, True, n_total)

        # Get AD-attached vertices for first-bounce triangles
        v0_1, v1_1, v2_1, vi0_1, vi1_1, vi2_1 = get_ad_triangle_vertices(
            mesh, prim_ids_1, vp_buffer, current_active)

        # Ray from RX to first hit for diff re-intersection
        # Use detached reservoir hit_P for direction (BVH found this triangle)
        hit_P_detached = mi.Point3f(
            dr.gather(mi.Float, reservoir_hits.hit_P.x, hit_idx),
            dr.gather(mi.Float, reservoir_hits.hit_P.y, hit_idx),
            dr.gather(mi.Float, reservoir_hits.hit_P.z, hit_idx),
        )
        delta_rx_1 = hit_P_detached - rx_pos_expanded
        d_rx_1_detached = dr.maximum(dr.norm(delta_rx_1), mi.Float(1e-10))
        ray_d_1 = mi.Vector3f(
            delta_rx_1.x / d_rx_1_detached,
            delta_rx_1.y / d_rx_1_detached,
            delta_rx_1.z / d_rx_1_detached,
        )

        # Differentiable re-intersection: full AD through ray_o (rx_pos) and vertices
        _t1, _u1, _v1, hit_P_1 = diff_ray_triangle_intersect(
            rx_pos_expanded, ray_d_1, v0_1, v1_1, v2_1)

        # AD-attached normal at bounce 1
        nx_arr, ny_arr, nz_arr = normal_params
        bary_u_1 = _u1
        bary_v_1 = _v1
        bary_w_1 = mi.Float(1.0) - bary_u_1 - bary_v_1
        nx_interp = bary_w_1 * dr.gather(mi.Float, nx_arr, vi0_1) + \
                    bary_u_1 * dr.gather(mi.Float, nx_arr, vi1_1) + \
                    bary_v_1 * dr.gather(mi.Float, nx_arr, vi2_1)
        ny_interp = bary_w_1 * dr.gather(mi.Float, ny_arr, vi0_1) + \
                    bary_u_1 * dr.gather(mi.Float, ny_arr, vi1_1) + \
                    bary_v_1 * dr.gather(mi.Float, ny_arr, vi2_1)
        nz_interp = bary_w_1 * dr.gather(mi.Float, nz_arr, vi0_1) + \
                    bary_u_1 * dr.gather(mi.Float, nz_arr, vi1_1) + \
                    bary_v_1 * dr.gather(mi.Float, nz_arr, vi2_1)
        n_len = dr.maximum(dr.sqrt(dr.maximum(
            nx_interp*nx_interp + ny_interp*ny_interp + nz_interp*nz_interp,
            mi.Float(1e-20))), mi.Float(0.01))
        hit_N_1 = mi.Vector3f(nx_interp / n_len, ny_interp / n_len, nz_interp / n_len)

        # First segment distance (AD through hit_P_1 and rx_pos)
        d_rx_to_1 = dr.norm(hit_P_1 - rx_pos_expanded)

        # Direction from RX toward hit_1 (for BSDF wo at bounce 1)
        _d_safe_rx1 = dr.maximum(d_rx_to_1, mi.Float(1e-10))
        _delta_rx1 = hit_P_1 - rx_pos_expanded
        dir_rx_to_1 = mi.Vector3f(
            _delta_rx1.x / _d_safe_rx1,
            _delta_rx1.y / _d_safe_rx1,
            _delta_rx1.z / _d_safe_rx1,
        )

        _toc('mb_first_bounce_reintersect')

        if verbose:
            print(f"  [E2E-MB] MIMO expansion: {n_valid} x {n_tx} x {n_rx} = {n_total:,} paths")

        # Initialize path state for bounce loop
        current_P = hit_P_1              # AD-attached position
        current_N = hit_N_1              # AD-attached normal
        current_prim_ids = prim_ids_1    # Non-diff triangle IDs
        current_vi0 = vi0_1
        current_vi1 = vi1_1
        current_vi2 = vi2_1
        current_bary_u = _u1
        current_bary_v = _v1
        prev_dir = dir_rx_to_1           # Direction arriving at current bounce
        cumulative_weight = mi.Float(1.0)  # Product of BSDF values
        cumulative_dist = d_rx_to_1      # Total path distance (AD)
        cumulative_mc = mc_first_bounce  # MC correction accumulator
        cumulative_path_loss = mi.Float(1.0)  # Path loss for segments after first

        # PERF: Evaluate all AD-attached first-bounce state to materialize
        # the diff re-intersection graph.
        dr.eval(
            current_P.x, current_P.y, current_P.z,
            current_N.x, current_N.y, current_N.z,
            cumulative_dist, d_rx_to_1,
            _u1, _v1,
        )

        _eval_f_cos = getattr(self.bsdf, 'eval_f_cos_physics', self.bsdf.eval_f_cos)
        double_sided = getattr(self.render_config, 'double_sided', True)
        epsilon = 1e-4

        # Precompute antenna gains using first/last directions
        # (TX gain uses direction at last bounce → TX; RX gain uses direction from RX → hit_1)
        # We compute RX gain once since it's always from the same first-bounce direction
        dir_rx_to_hit1_for_gain = mi.Vector3f(
            prev_dir.x, prev_dir.y, prev_dir.z)  # Copy for later use

        # Base seed for direction sampling
        base_seed = getattr(self.render_config, 'seed', 42)

        # Per-bounce viewpoint centroids for boundary gradient computation.
        # At each bounce b, the viewpoint is the centroid of active hit
        # positions from the PREVIOUS bounce (or RX for bounce 1).
        # These are stored as ScalarPoint3f for use by the boundary
        # gradient computer after synthesis.
        self._multibounce_viewpoints = []
        # Bounce 1 viewpoint = RX centroid
        self._multibounce_viewpoints.append(mi.ScalarPoint3f(
            float(dr.mean(rx_pos_expanded.x)[0]),
            float(dr.mean(rx_pos_expanded.y)[0]),
            float(dr.mean(rx_pos_expanded.z)[0]),
        ))

        # Bounce loop: NEE at each bounce + continue tracing
        for b in range(1, max_bounces + 1):

            # NEE: Connect current bounce to TX
            if nee_every_bounce or b == max_bounces:
                _tic(f'mb_b{b}_nee_shadow')
                # Direction from current hit to TX (AD through both)
                delta_tx = tx_pos_expanded - current_P
                d_to_tx = dr.norm(delta_tx)
                _d_safe_tx = dr.maximum(d_to_tx, mi.Float(1e-10))
                dir_to_tx = mi.Vector3f(
                    delta_tx.x / _d_safe_tx,
                    delta_tx.y / _d_safe_tx,
                    delta_tx.z / _d_safe_tx,
                )

                # Shadow ray: check visibility to TX
                shadow_o = mi.Point3f(
                    current_P.x + epsilon * dir_to_tx.x,
                    current_P.y + epsilon * dir_to_tx.y,
                    current_P.z + epsilon * dir_to_tx.z,
                )
                shadow_ray = mi.Ray3f(shadow_o, dir_to_tx)
                shadow_ray.maxt = d_to_tx - 2 * epsilon
                occluded = scene.ray_test(shadow_ray)
                nee_active = current_active & (~occluded)
                _toc(f'mb_b{b}_nee_shadow')

                _tic(f'mb_b{b}_nee_bsdf')
                # BSDF at this bounce for NEE connection
                # wi = toward TX, wo = toward RX side (negate prev_dir)
                wi_nee = dir_to_tx
                wo_nee = mi.Vector3f(-prev_dir.x, -prev_dir.y, -prev_dir.z)
                n_nee = current_N

                # Double-sided normal flip
                if double_sided:
                    cos_out = dr.dot(wo_nee, n_nee)
                    need_flip = cos_out < 0
                    n_nee = dr.select(
                        need_flip,
                        mi.Vector3f(-n_nee.x, -n_nee.y, -n_nee.z),
                        n_nee)

                # Gather materials (AD through physics_params)
                class _MBGeom:
                    pass
                _geom = _MBGeom()
                _geom.hit_prim_ids = current_prim_ids
                _geom.active = nee_active
                _geom.hit_bary_u = current_bary_u
                _geom.hit_bary_v = current_bary_v
                _geom.hit_vertex_ids_0 = current_vi0
                _geom.hit_vertex_ids_1 = current_vi1
                _geom.hit_vertex_ids_2 = current_vi2
                per_hit = self.material_param.gather(physics_params, _geom)
                eps_real, eps_imag, sigma_h, l_c, tau_mat, thickness = per_hit

                # Evaluate BSDF (AD through materials + normals + directions)
                bsdf_nee = _eval_f_cos(
                    wo=wo_nee, wi=wi_nee, n=n_nee,
                    eps_real=eps_real, eps_imag=eps_imag,
                    sigma_h=sigma_h, l_c=l_c, tau=tau_mat,
                    thickness=thickness,
                )

                # NEE weight = cumulative_weight * bsdf at this bounce
                nee_weight = cumulative_weight * bsdf_nee
                _toc(f'mb_b{b}_nee_bsdf')

                # First-bounce diffraction (only at bounce 1)
                _tic(f'mb_b{b}_diffraction')
                if b == 1 and include_diffraction and _HAS_DIFFRACTION:
                    diff_config = self.diffraction_config
                    if (diff_config is not None and diff_config.enabled
                            and self.tri_hash is not None and self.fsd_bsdf is not None):
                        f_diff, _, _, beta_diff = self._compute_diffraction_e2e(
                            hit_P_expanded=current_P,
                            hit_N_expanded=current_N,
                            dir_hit_to_tx=wi_nee,
                            dir_hit_to_rx=wo_nee,
                            active=nee_active,
                            physics_params=physics_params,
                            n_valid=n_valid,
                            n_tx=n_tx, n_rx=n_rx,
                            vertex_ids_0=current_vi0,
                            vertex_ids_1=current_vi1,
                            vertex_ids_2=current_vi2,
                            bary_u_expanded=current_bary_u,
                            bary_v_expanded=current_bary_v,
                            scene=scene,
                            verbose=verbose,
                        )
                        if diff_config.energy_borrowing and include_energy_borrowing:
                            one_minus_beta = dr.maximum(mi.Float(1.0) - beta_diff, mi.Float(0.0))
                            nee_weight = nee_weight * one_minus_beta

                _toc(f'mb_b{b}_diffraction')

                # Antenna gain (using first and last directions)
                _tic(f'mb_b{b}_antenna_phase')
                antenna_gain = mi.Float(1.0)
                if pattern_loaders is not None and evaluate_combined_gain is not None:
                    tx_loader = pattern_loaders.get('tx')
                    rx_loader = pattern_loaders.get('rx')
                    if tx_loader is not None and tx_boresights is not None:
                        tx_bore_exp = gather_vector3f(tx_boresights, tx_idx)
                        gain_tx = evaluate_combined_gain(tx_loader, dir_to_tx, tx_bore_exp)
                        antenna_gain = antenna_gain * gain_tx
                    if rx_loader is not None and rx_boresights is not None:
                        rx_bore_exp = gather_vector3f(rx_boresights, rx_idx)
                        gain_rx = evaluate_combined_gain(
                            rx_loader, dir_rx_to_hit1_for_gain, rx_bore_exp)
                        antenna_gain = antenna_gain * gain_rx
                elif (tx_pattern_loader is not None or rx_pattern_loader is not None) \
                        and evaluate_combined_gain is not None:
                    # GPU-native non-differentiable path
                    if tx_pattern_loader is not None and tx_boresights is not None:
                        tx_bore_exp = gather_vector3f(tx_boresights, tx_idx)
                        gain_tx = evaluate_combined_gain(tx_pattern_loader, -dir_to_tx, tx_bore_exp)
                        antenna_gain = antenna_gain * gain_tx
                    if rx_pattern_loader is not None and rx_boresights is not None:
                        rx_bore_exp = gather_vector3f(rx_boresights, rx_idx)
                        gain_rx = evaluate_combined_gain(
                            rx_pattern_loader, dir_rx_to_hit1_for_gain, rx_bore_exp)
                        antenna_gain = antenna_gain * gain_rx
                elif tx_pattern_rrts is not None or rx_pattern_rrts is not None:
                    # DrJit GPU fallback: all-GPU evaluation (no CPU↔GPU transfers)
                    dir_tx_to_hit_dr = mi.Vector3f(-dir_to_tx.x, -dir_to_tx.y, -dir_to_tx.z)
                    gain_tx_dB = dr.zeros(mi.Float, n_total)
                    gain_rx_dB = dr.zeros(mi.Float, n_total)
                    if tx_pattern_rrts is not None and tx_boresights is not None and evaluate_gain_rrts_style_drjit_product is not None:
                        tx_bore_exp = gather_vector3f(tx_boresights, tx_idx)
                        gain_tx_dB = evaluate_gain_rrts_style_drjit_product(
                            dir_tx_to_hit_dr, tx_pattern_rrts, tx_bore_exp)
                    if rx_pattern_rrts is not None and rx_boresights is not None and evaluate_gain_rrts_style_drjit_product is not None:
                        rx_bore_exp = gather_vector3f(rx_boresights, rx_idx)
                        gain_rx_dB = evaluate_gain_rrts_style_drjit_product(
                            dir_rx_to_hit1_for_gain, rx_pattern_rrts, rx_bore_exp)
                    antenna_gain = dr.power(mi.Float(10.0), gain_tx_dB * mi.Float(0.1)) * \
                                   dr.power(mi.Float(10.0), gain_rx_dB * mi.Float(0.1))

                nee_weight = nee_weight * antenna_gain * cumulative_mc

                # Path loss: TX leg + intermediate legs
                d_safe_tx = dr.maximum(d_to_tx, mi.Float(1e-4))
                path_loss_nee = cumulative_path_loss / (d_safe_tx * d_safe_tx)
                radar_scale = mi.Float(self.radar_constant * self.rx_dBFS_scale * self.adc_scale)
                weight_nee = radar_scale * dr.sqrt(
                    dr.maximum(nee_weight * path_loss_nee, mi.Float(1e-20)))

                # Total distance for this path (AD through ALL positions)
                R_total_nee = cumulative_dist + d_to_tx

                # Phase computation + K-chunked scatter-add
                tau_delay = R_total_nee / mi.Float(C)
                TWO_PI = mi.Float(2.0 * np.pi)
                phi_const = TWO_PI * mi.Float(self.min_freq) * tau_delay
                phi_slope = TWO_PI * mi.Float(self.slope) * tau_delay

                # K-chunked phase computation + scatter-add
                # (matches specular path pattern — avoids massive n_total×K graph)
                CHUNK = getattr(self.render_config, 'e2e_phase_chunk_size', 64)

                for k_start in range(0, K, CHUNK):
                    k_end = min(k_start + CHUNK, K)
                    k_chunk = k_end - k_start

                    n_elems = n_total * k_chunk

                    path_idx_k = dr.arange(mi.UInt32, n_elems) // mi.UInt32(k_chunk)
                    k_local = dr.arange(mi.UInt32, n_elems) % mi.UInt32(k_chunk)
                    k_global = k_local + mi.UInt32(k_start)

                    pc = dr.gather(mi.Float, phi_const, path_idx_k)
                    ps = dr.gather(mi.Float, phi_slope, path_idx_k)
                    w = dr.gather(mi.Float, weight_nee, path_idx_k)
                    act = dr.gather(mi.Bool, nee_active, path_idx_k)

                    t_k = mi.Float(k_global) / mi.Float(self.sample_rate)
                    phi = pc + ps * t_k

                    _cos = dr.cos(phi)
                    _sin = dr.sin(phi)
                    if not self.render_config.enable_grad_phase:
                        _cos = dr.detach(_cos)
                        _sin = dr.detach(_sin)
                    contrib_real = w * _cos
                    contrib_imag = w * _sin

                    tx_for_elem = dr.gather(mi.UInt32, tx_idx, path_idx_k)
                    rx_for_elem = dr.gather(mi.UInt32, rx_idx, path_idx_k)
                    flat_adc_idx = (tx_for_elem * mi.UInt32(n_rx * K)
                                    + rx_for_elem * mi.UInt32(K)
                                    + k_global)

                    dr.scatter_add(adc_real_flat, contrib_real, flat_adc_idx, act)
                    dr.scatter_add(adc_imag_flat, contrib_imag, flat_adc_idx, act)

                # Materialize this bounce's ADC contributions to prevent graph
                # accumulation across bounces.
                dr.eval(adc_real_flat, adc_imag_flat)

                _toc(f'mb_b{b}_antenna_phase')

            # Continue tracing: sample direction for next bounce
            if b < max_bounces:
                _tic(f'mb_b{b}_continue')

                # Step A: Collapse MIMO → per-hit (n_valid)
                # Extract one representative per base hit (first MIMO pair)
                per_hit_idx = dr.arange(mi.UInt32, n_valid) * mi.UInt32(_n_mimo)

                ph_P = mi.Point3f(
                    dr.gather(mi.Float, current_P.x, per_hit_idx),
                    dr.gather(mi.Float, current_P.y, per_hit_idx),
                    dr.gather(mi.Float, current_P.z, per_hit_idx),
                )
                ph_N = mi.Vector3f(
                    dr.gather(mi.Float, current_N.x, per_hit_idx),
                    dr.gather(mi.Float, current_N.y, per_hit_idx),
                    dr.gather(mi.Float, current_N.z, per_hit_idx),
                )
                ph_prev_dir = mi.Vector3f(
                    dr.gather(mi.Float, prev_dir.x, per_hit_idx),
                    dr.gather(mi.Float, prev_dir.y, per_hit_idx),
                    dr.gather(mi.Float, prev_dir.z, per_hit_idx),
                )
                ph_active = dr.gather(mi.Bool, current_active, per_hit_idx)
                ph_prim_ids = dr.gather(mi.UInt32, current_prim_ids, per_hit_idx)
                ph_vi0 = dr.gather(mi.UInt32, current_vi0, per_hit_idx)
                ph_vi1 = dr.gather(mi.UInt32, current_vi1, per_hit_idx)
                ph_vi2 = dr.gather(mi.UInt32, current_vi2, per_hit_idx)
                ph_bary_u = dr.gather(mi.Float, current_bary_u, per_hit_idx)
                ph_bary_v = dr.gather(mi.Float, current_bary_v, per_hit_idx)

                # Step B: Per-hit continuation (n_valid)
                wo_cont_ph = mi.Vector3f(-ph_prev_dir.x, -ph_prev_dir.y, -ph_prev_dir.z)
                n_cont_ph = ph_N
                if double_sided:
                    cos_out_c = dr.dot(wo_cont_ph, n_cont_ph)
                    need_flip_c = cos_out_c < 0
                    n_cont_ph = dr.select(
                        need_flip_c,
                        mi.Vector3f(-n_cont_ph.x, -n_cont_ph.y, -n_cont_ph.z),
                        n_cont_ph)

                # Gather materials for continuation BSDF (n_valid)
                class _MBGeom2:
                    pass
                _geom2 = _MBGeom2()
                _geom2.hit_prim_ids = ph_prim_ids
                _geom2.active = ph_active
                _geom2.hit_bary_u = ph_bary_u
                _geom2.hit_bary_v = ph_bary_v
                _geom2.hit_vertex_ids_0 = ph_vi0
                _geom2.hit_vertex_ids_1 = ph_vi1
                _geom2.hit_vertex_ids_2 = ph_vi2
                per_hit_c = self.material_param.gather(physics_params, _geom2)
                eps_r_c, eps_i_c, sig_c, lc_c, tau_c, thick_c = per_hit_c

                # Direction sampling (n_valid): one cosine-weighted direction per base hit
                next_dir_ph, sample_pdf_ph, _ = _sample_cosine_hemisphere_drjit(
                    n_cont_ph, n_valid, base_seed + b * 1000)

                # Continuation BSDF (n_valid)
                bsdf_cont_ph = _eval_f_cos(
                    wo=wo_cont_ph, wi=next_dir_ph, n=n_cont_ph,
                    eps_real=eps_r_c, eps_imag=eps_i_c,
                    sigma_h=sig_c, l_c=lc_c, tau=tau_c,
                    thickness=thick_c,
                )

                # Ray tracing: one ray per base hit (n_valid)
                next_ray_o_ph = mi.Point3f(
                    ph_P.x + epsilon * next_dir_ph.x,
                    ph_P.y + epsilon * next_dir_ph.y,
                    ph_P.z + epsilon * next_dir_ph.z,
                )
                next_ray_ph = mi.Ray3f(next_ray_o_ph, next_dir_ph)
                si_next_ph = scene.ray_intersect(next_ray_ph, ph_active)
                next_hit_valid_ph = si_next_ph.is_valid() & ph_active

                # Diff re-intersection (n_valid)
                next_prim_ids_ph = si_next_ph.prim_index
                v0_n, v1_n, v2_n, vi0_n, vi1_n, vi2_n = get_ad_triangle_vertices(
                    mesh, next_prim_ids_ph, vp_buffer, next_hit_valid_ph)

                _t_n, _u_n, _v_n, hit_P_next_ph = diff_ray_triangle_intersect(
                    next_ray_o_ph, next_dir_ph, v0_n, v1_n, v2_n)

                # AD-attached normal at next bounce (n_valid)
                bw_n = mi.Float(1.0) - _u_n - _v_n
                nx_n = bw_n * dr.gather(mi.Float, nx_arr, vi0_n, next_hit_valid_ph) + \
                       _u_n * dr.gather(mi.Float, nx_arr, vi1_n, next_hit_valid_ph) + \
                       _v_n * dr.gather(mi.Float, nx_arr, vi2_n, next_hit_valid_ph)
                ny_n = bw_n * dr.gather(mi.Float, ny_arr, vi0_n, next_hit_valid_ph) + \
                       _u_n * dr.gather(mi.Float, ny_arr, vi1_n, next_hit_valid_ph) + \
                       _v_n * dr.gather(mi.Float, ny_arr, vi2_n, next_hit_valid_ph)
                nz_n = bw_n * dr.gather(mi.Float, nz_arr, vi0_n, next_hit_valid_ph) + \
                       _u_n * dr.gather(mi.Float, nz_arr, vi1_n, next_hit_valid_ph) + \
                       _v_n * dr.gather(mi.Float, nz_arr, vi2_n, next_hit_valid_ph)
                nl_n = dr.maximum(dr.sqrt(dr.maximum(
                    nx_n*nx_n + ny_n*ny_n + nz_n*nz_n, mi.Float(1e-20))), mi.Float(0.01))
                hit_N_next_ph = mi.Vector3f(nx_n / nl_n, ny_n / nl_n, nz_n / nl_n)

                # Segment distance (n_valid, AD-attached)
                d_segment_ph = dr.norm(hit_P_next_ph - ph_P)

                # Step C: MIMO re-expansion (n_valid → n_total)
                # All TX-RX pairs for the same base hit share the same
                # continuation geometry, preserving MIMO phase coherence.
                next_dir_exp = mi.Vector3f(
                    dr.gather(mi.Float, next_dir_ph.x, hit_idx),
                    dr.gather(mi.Float, next_dir_ph.y, hit_idx),
                    dr.gather(mi.Float, next_dir_ph.z, hit_idx),
                )
                hit_P_next_exp = mi.Point3f(
                    dr.gather(mi.Float, hit_P_next_ph.x, hit_idx),
                    dr.gather(mi.Float, hit_P_next_ph.y, hit_idx),
                    dr.gather(mi.Float, hit_P_next_ph.z, hit_idx),
                )
                hit_N_next_exp = mi.Vector3f(
                    dr.gather(mi.Float, hit_N_next_ph.x, hit_idx),
                    dr.gather(mi.Float, hit_N_next_ph.y, hit_idx),
                    dr.gather(mi.Float, hit_N_next_ph.z, hit_idx),
                )
                next_hit_valid_exp = dr.gather(mi.Bool, next_hit_valid_ph, hit_idx)
                bsdf_cont_exp = dr.gather(mi.Float, bsdf_cont_ph, hit_idx)
                sample_pdf_exp = dr.gather(mi.Float, sample_pdf_ph, hit_idx)
                d_segment_exp = dr.gather(mi.Float, d_segment_ph, hit_idx)
                next_prim_ids_exp = dr.gather(mi.UInt32, next_prim_ids_ph, hit_idx)
                vi0_exp = dr.gather(mi.UInt32, vi0_n, hit_idx)
                vi1_exp = dr.gather(mi.UInt32, vi1_n, hit_idx)
                vi2_exp = dr.gather(mi.UInt32, vi2_n, hit_idx)
                bary_u_exp = dr.gather(mi.Float, _u_n, hit_idx)
                bary_v_exp = dr.gather(mi.Float, _v_n, hit_idx)

                # Step D: Update cumulative state (n_total, using expanded values)
                cumulative_weight = dr.select(next_hit_valid_exp,
                    cumulative_weight * bsdf_cont_exp, cumulative_weight)
                cumulative_dist = dr.select(next_hit_valid_exp,
                    cumulative_dist + d_segment_exp, cumulative_dist)
                d_seg_safe = dr.maximum(d_segment_exp, mi.Float(1e-4))
                cumulative_path_loss = dr.select(next_hit_valid_exp,
                    cumulative_path_loss / (d_seg_safe * d_seg_safe),
                    cumulative_path_loss)
                cumulative_mc = dr.select(next_hit_valid_exp,
                    cumulative_mc / dr.maximum(sample_pdf_exp, mi.Float(1e-8)),
                    cumulative_mc)

                # Advance path state (n_total)
                prev_dir = dr.select(next_hit_valid_exp, next_dir_exp, prev_dir)
                current_P = dr.select(next_hit_valid_exp, hit_P_next_exp, current_P)
                current_N = dr.select(next_hit_valid_exp, hit_N_next_exp, current_N)
                current_prim_ids = dr.select(next_hit_valid_exp, next_prim_ids_exp, current_prim_ids)
                current_vi0 = dr.select(next_hit_valid_exp, vi0_exp, current_vi0)
                current_vi1 = dr.select(next_hit_valid_exp, vi1_exp, current_vi1)
                current_vi2 = dr.select(next_hit_valid_exp, vi2_exp, current_vi2)
                current_bary_u = dr.select(next_hit_valid_exp, bary_u_exp, current_bary_u)
                current_bary_v = dr.select(next_hit_valid_exp, bary_v_exp, current_bary_v)
                current_active = next_hit_valid_exp

                # PERF: Evaluate continuation state to materialize the diff
                # re-intersection AD graph before the next bounce.
                dr.eval(
                    current_P.x, current_P.y, current_P.z,
                    current_N.x, current_N.y, current_N.z,
                    cumulative_weight, cumulative_dist, cumulative_mc,
                    current_active,
                )

                _toc(f'mb_b{b}_continue')

        if verbose:
            print(f"  [E2E-MB] Multibounce diffuse synthesis complete (max_bounces={max_bounces})")

    # SPECULAR CONTRIBUTION (single-bounce SMS, same as synthesize_end_to_end)
    # Materialize diffuse AD graph before specular synthesis.
    dr.eval(adc_real_flat, adc_imag_flat)

    _tic('mb_specular_sms')
    if include_specular and specular_paths is not None:
        self._synthesize_specular_e2e(
            specular_paths=specular_paths,
            adc_real_flat=adc_real_flat,
            adc_imag_flat=adc_imag_flat,
            n_tx=n_tx, n_rx=n_rx, K=K,
            physics_params=physics_params,
            tx_positions=tx_positions,
            rx_positions=rx_positions,
            tx_boresights=tx_boresights,
            rx_boresights=rx_boresights,
            pattern_loaders=pattern_loaders,
            tx_pattern_rrts=tx_pattern_rrts,
            rx_pattern_rrts=rx_pattern_rrts,
            tx_pattern_loader=tx_pattern_loader,
            rx_pattern_loader=rx_pattern_loader,
            antenna_gain_linear=antenna_gain_linear,
            vertex_offset_params=vertex_offset_params,
            normal_params=normal_params,
            scene=scene,
            verbose=verbose,
            _dist_accum=_dist_accum,
        )

    # MULTIBOUNCE SPECULAR CHAINS (k-bounce SMS)

    _toc('mb_specular_sms')

    if return_total_distance:
        return adc_real_flat, adc_imag_flat, _dist_accum[0]
    return adc_real_flat, adc_imag_flat

def _synthesize_specular_e2e(
    self,
    specular_paths: 'SpecularPaths',
    adc_real_flat: 'mi.Float',
    adc_imag_flat: 'mi.Float',
    n_tx: int, n_rx: int, K: int,
    physics_params: list,
    tx_positions: 'mi.Point3f',
    rx_positions: 'mi.Point3f',
    tx_boresights: Optional['mi.Vector3f'] = None,
    rx_boresights: Optional['mi.Vector3f'] = None,
    pattern_loaders: Optional[dict] = None,
    tx_pattern_rrts: Optional[np.ndarray] = None,
    rx_pattern_rrts: Optional[np.ndarray] = None,
    tx_pattern_loader=None,
    rx_pattern_loader=None,
    antenna_gain_linear: bool = True,
    vertex_offset_params: Optional[list] = None,
    normal_params: Optional[list] = None,
    scene: Optional['mi.Scene'] = None,
    verbose: bool = False,
    _dist_accum: Optional[list] = None,
):
    """End-to-end specular path synthesis with IFT gradient flow."""
    from ..utils.math import gather_point3f, gather_vector3f

    sp = specular_paths
    n_spec = sp.n_paths
    sp_valid = sp.valid

    # Fine-grained profiling for specular synthesis
    getattr(self, '_profile', None)
    def _stic(label):
        pass
    def _stoc(label):
        pass

    _stic('count_valid')
    n_valid_spec = int(dr.sum(mi.UInt32(sp_valid))[0]) if hasattr(
        dr.sum(mi.UInt32(sp_valid)), '__getitem__') else int(dr.sum(mi.UInt32(sp_valid)))
    _stoc('count_valid')

    if verbose:
        print(f"\n  [E2E Specular] Synthesizing {n_valid_spec:,} valid specular paths")

    _stic('detach_and_ift')
    # Detach frozen specular geometry from stale AD graph.
    #
    # SceneContext loads vertex positions/normals with grad enabled.
    # The SMS solver (Phase A) computes specular paths using these
    # AD-attached arrays, so sp.hit_P, sp.hit_N, etc. inherit stale
    # AD connections. If not detached, dr.backward() tries to traverse
    # through the entire vertex position/normal AD graph → hangs.
    #
    # Only physics_params (and optionally vertex_offset_params /
    # normal_params) should carry AD connections into this function.
    sp.hit_P = mi.Point3f(dr.detach(sp.hit_P.x), dr.detach(sp.hit_P.y), dr.detach(sp.hit_P.z))
    sp.hit_N = mi.Vector3f(dr.detach(sp.hit_N.x), dr.detach(sp.hit_N.y), dr.detach(sp.hit_N.z))
    if sp.A_tri is not None:
        sp.A_tri = mi.Float(dr.detach(sp.A_tri))
    if sp.R_specular is not None:
        sp.R_specular = mi.Float(dr.detach(sp.R_specular))
    if sp.grad_info is not None:
        gi_raw = sp.grad_info
        if gi_raw.dp_du is not None:
            gi_raw.dp_du = mi.Vector3f(dr.detach(gi_raw.dp_du.x), dr.detach(gi_raw.dp_du.y), dr.detach(gi_raw.dp_du.z))
        if gi_raw.dp_dv is not None:
            gi_raw.dp_dv = mi.Vector3f(dr.detach(gi_raw.dp_dv.x), dr.detach(gi_raw.dp_dv.y), dr.detach(gi_raw.dp_dv.z))
        if gi_raw.tx_pos is not None:
            gi_raw.tx_pos = mi.Point3f(dr.detach(gi_raw.tx_pos.x), dr.detach(gi_raw.tx_pos.y), dr.detach(gi_raw.tx_pos.z))
        if gi_raw.rx_pos is not None:
            gi_raw.rx_pos = mi.Point3f(dr.detach(gi_raw.rx_pos.x), dr.detach(gi_raw.rx_pos.y), dr.detach(gi_raw.rx_pos.z))

    # Step S0: Gather live materials at specular hit triangles (AD)
    sp_prim_ids = sp.prim_ids

    class _SpecGeom:
        pass
    _geom = _SpecGeom()
    _geom.hit_prim_ids = sp_prim_ids
    _geom.active = sp_valid

    # Get vertex IDs and barycentrics for per-vertex mode
    mesh = scene.shapes()[0] if scene is not None and len(scene.shapes()) > 0 else None
    face_idx = mesh.face_indices(sp_prim_ids)
    _geom.hit_vertex_ids_0 = face_idx[0]
    _geom.hit_vertex_ids_1 = face_idx[1]
    _geom.hit_vertex_ids_2 = face_idx[2]
    _geom.hit_bary_u = sp.bary_u
    _geom.hit_bary_v = sp.bary_v

    sp_materials = self.material_param.gather(physics_params, _geom)

    sp_eps_real, sp_eps_imag, sp_sigma_h, sp_l_c, sp_tau, sp_thickness = sp_materials

    # Step S1: IFT vertex position attachment (vertex_offset_params)
    gi = sp.grad_info  # SpecularGradInfo from SMS Newton solver
    ift_active = (vertex_offset_params is not None
                  and gi is not None
                  and sp.bary_u is not None)

    nx_arr, ny_arr, nz_arr = normal_params
    mesh = scene.shapes()[0] if scene is not None and len(scene.shapes()) > 0 else None
    face_idx = mesh.face_indices(sp_prim_ids)
    svi0, svi1, svi2 = face_idx[0], face_idx[1], face_idx[2]
    su, sv = sp.bary_u, sp.bary_v
    sw = mi.Float(1.0) - su - sv

    sp_nx = sw * dr.gather(mi.Float, nx_arr, svi0, sp_valid) \
          + su * dr.gather(mi.Float, nx_arr, svi1, sp_valid) \
          + sv * dr.gather(mi.Float, nx_arr, svi2, sp_valid)
    sp_ny = sw * dr.gather(mi.Float, ny_arr, svi0, sp_valid) \
          + su * dr.gather(mi.Float, ny_arr, svi1, sp_valid) \
          + sv * dr.gather(mi.Float, ny_arr, svi2, sp_valid)
    sp_nz = sw * dr.gather(mi.Float, nz_arr, svi0, sp_valid) \
          + su * dr.gather(mi.Float, nz_arr, svi1, sp_valid) \
          + sv * dr.gather(mi.Float, nz_arr, svi2, sp_valid)

    n_len = dr.maximum(
        dr.sqrt(dr.maximum(sp_nx * sp_nx + sp_ny * sp_ny + sp_nz * sp_nz,
                            mi.Float(1e-20))),
        mi.Float(1e-10))
    sp_hit_N = mi.Vector3f(sp_nx / n_len, sp_ny / n_len, sp_nz / n_len)
    sp_hit_P = sp.hit_P

    _stoc('detach_and_ift')
    _stic('bsdf_antenna')
    # Step S2: Recompute distances/directions from live positions (AD)
    # TX/RX positions are already AD-attached from pose transform
    tx_pos_exp = gather_point3f(tx_positions, sp.tx_idx)
    rx_pos_exp = gather_point3f(rx_positions, sp.rx_idx)

    delta_tx = mi.Vector3f(
        tx_pos_exp.x - sp_hit_P.x,
        tx_pos_exp.y - sp_hit_P.y,
        tx_pos_exp.z - sp_hit_P.z)
    delta_rx = mi.Vector3f(
        rx_pos_exp.x - sp_hit_P.x,
        rx_pos_exp.y - sp_hit_P.y,
        rx_pos_exp.z - sp_hit_P.z)

    sp_d_tx = dr.norm(delta_tx)
    sp_d_rx = dr.norm(delta_rx)

    # Accumulate specular path distances for phase-only testing

    sp_dir_to_tx = mi.Vector3f(
        delta_tx.x / dr.maximum(sp_d_tx, mi.Float(1e-10)),
        delta_tx.y / dr.maximum(sp_d_tx, mi.Float(1e-10)),
        delta_tx.z / dr.maximum(sp_d_tx, mi.Float(1e-10)))
    sp_dir_to_rx = mi.Vector3f(
        delta_rx.x / dr.maximum(sp_d_rx, mi.Float(1e-10)),
        delta_rx.y / dr.maximum(sp_d_rx, mi.Float(1e-10)),
        delta_rx.z / dr.maximum(sp_d_rx, mi.Float(1e-10)))

    # Step S3: Double-sided normal flip
    double_sided = getattr(self.render_config, 'double_sided', True)
    if double_sided:
        cos_out = dr.dot(sp_dir_to_rx, sp_hit_N)
        need_flip = cos_out < 0
        sp_hit_N = dr.select(
            need_flip,
            mi.Vector3f(-sp_hit_N.x, -sp_hit_N.y, -sp_hit_N.z),
            sp_hit_N,
        )

    # Step S4: Specular BSDF evaluation (AD through materials)
    # Use eval_ka_only_physics (KA lobe only = R_jones × η × τ_eff × f_KA)
    # to match the reference renderer's Phase B differentiable path.
    # This gives richer angular gradient information than the flat
    # reflectance coefficient eval_specular_reflectance.
    # Note: Phase A (non-diff) uses R_specular = η × τ_eff × A, which is
    # different. The Phase B BSDF gives better gradient quality for
    # material optimization despite the forward value mismatch.
    sp_cos_theta_i = dr.maximum(dr.dot(sp_dir_to_tx, sp_hit_N), mi.Float(0.0))
    sp_cos_theta_r = dr.maximum(dr.dot(sp_dir_to_rx, sp_hit_N), mi.Float(0.0))

    _eval_ka = getattr(self.bsdf, 'eval_ka_only_physics', None)
    sp_brdf = _eval_ka(
        wo=sp_dir_to_rx,
        wi=sp_dir_to_tx,
        n=sp_hit_N,
        eps_real=sp_eps_real, eps_imag=sp_eps_imag,
        sigma_h=sp_sigma_h, l_c=sp_l_c, tau=sp_tau,
        thickness=sp_thickness,
    )

    specular_power = sp_brdf * sp_cos_theta_i * sp_cos_theta_r * sp.A_tri

    # Step S5: Antenna gain (skip RRTS when directions are AD-attached
    # to avoid np.array() triggering eager eval that confuses backward)
    if pattern_loaders is not None and evaluate_combined_gain is not None:
        # Differentiable antenna pattern evaluation
        tx_loader = pattern_loaders.get('tx')
        rx_loader = pattern_loaders.get('rx')

        sp_antenna_gain = mi.Float(1.0)
        if tx_loader is not None and tx_boresights is not None:
            tx_bore_exp = gather_vector3f(tx_boresights, sp.tx_idx)
            gain_tx = evaluate_combined_gain(tx_loader, sp_dir_to_tx, tx_bore_exp)
            sp_antenna_gain = sp_antenna_gain * gain_tx

        if rx_loader is not None and rx_boresights is not None:
            rx_bore_exp = gather_vector3f(rx_boresights, sp.rx_idx)
            dir_rx_to_hit = mi.Vector3f(
                -sp_dir_to_rx.x, -sp_dir_to_rx.y, -sp_dir_to_rx.z)
            gain_rx = evaluate_combined_gain(rx_loader, dir_rx_to_hit, rx_bore_exp)
            sp_antenna_gain = sp_antenna_gain * gain_rx

        specular_power = specular_power * sp_antenna_gain

    elif (tx_pattern_loader is not None or rx_pattern_loader is not None) \
            and evaluate_combined_gain is not None:
        # GPU-native non-differentiable antenna patterns (no GPU→CPU sync)
        sp_antenna_gain = mi.Float(1.0)
        if tx_pattern_loader is not None and tx_boresights is not None:
            tx_bore_exp = gather_vector3f(tx_boresights, sp.tx_idx)
            gain_tx = evaluate_combined_gain(tx_pattern_loader, -sp_dir_to_tx, tx_bore_exp)
            sp_antenna_gain = sp_antenna_gain * gain_tx
        if rx_pattern_loader is not None and rx_boresights is not None:
            rx_bore_exp = gather_vector3f(rx_boresights, sp.rx_idx)
            gain_rx = evaluate_combined_gain(rx_pattern_loader, -sp_dir_to_rx, rx_bore_exp)
            sp_antenna_gain = sp_antenna_gain * gain_rx
        specular_power = specular_power * sp_antenna_gain

    elif tx_pattern_rrts is not None or rx_pattern_rrts is not None:
        # DrJit GPU fallback: all-GPU evaluation (no CPU↔GPU transfers)
        dir_tx_to_spec = mi.Vector3f(-sp_dir_to_tx.x, -sp_dir_to_tx.y, -sp_dir_to_tx.z)
        dir_rx_to_spec = mi.Vector3f(-sp_dir_to_rx.x, -sp_dir_to_rx.y, -sp_dir_to_rx.z)

        pattern_mode = getattr(self.render_config, 'pattern_mode', 'fixed')

        if pattern_mode == 'legacy':
            if tx_pattern_rrts is not None and evaluate_gain_rrts_style_drjit is not None:
                gain_tx_dB = evaluate_gain_rrts_style_drjit(dir_tx_to_spec, tx_pattern_rrts)
            else:
                gain_tx_dB = dr.zeros(mi.Float, n_spec)
            if rx_pattern_rrts is not None and evaluate_gain_rrts_style_drjit is not None:
                gain_rx_dB = evaluate_gain_rrts_style_drjit(dir_rx_to_spec, rx_pattern_rrts)
            else:
                gain_rx_dB = dr.zeros(mi.Float, n_spec)
        else:
            if tx_pattern_rrts is not None and evaluate_gain_rrts_style_drjit_product is not None:
                tx_boresights_exp = gather_vector3f(tx_boresights, sp.tx_idx) \
                    if tx_boresights is not None else mi.Vector3f(0.0, 1.0, 0.0)
                gain_tx_dB = evaluate_gain_rrts_style_drjit_product(
                    dir_tx_to_spec, tx_pattern_rrts, tx_boresights_exp)
            else:
                gain_tx_dB = dr.zeros(mi.Float, n_spec)
            if rx_pattern_rrts is not None and evaluate_gain_rrts_style_drjit_product is not None:
                rx_boresights_exp = gather_vector3f(rx_boresights, sp.rx_idx) \
                    if rx_boresights is not None else mi.Vector3f(0.0, 1.0, 0.0)
                gain_rx_dB = evaluate_gain_rrts_style_drjit_product(
                    dir_rx_to_spec, rx_pattern_rrts, rx_boresights_exp)
            else:
                gain_rx_dB = dr.zeros(mi.Float, n_spec)

        if verbose:
            with dr.suspend_grad():
                _v = np.array(sp_valid)
                _g_tx = np.array(gain_tx_dB)
                _g_rx = np.array(gain_rx_dB)
                print(f"  [E2E-SPEC-DEBUG] RRTS gain_tx_dB[valid]: mean={_g_tx[_v].mean():.4f}, "
                      f"range=[{_g_tx[_v].min():.4f}, {_g_tx[_v].max():.4f}]")
                print(f"  [E2E-SPEC-DEBUG] RRTS gain_rx_dB[valid]: mean={_g_rx[_v].mean():.4f}, "
                      f"range=[{_g_rx[_v].min():.4f}, {_g_rx[_v].max():.4f}]")
                print(f"  [E2E-SPEC-DEBUG] antenna_gain_linear={antenna_gain_linear}, "
                      f"use_radar_equation={self.render_config.use_radar_equation}")

        if antenna_gain_linear or self.render_config.use_radar_equation:
            sp_antenna_gain = dr.power(mi.Float(10.0), gain_tx_dB * mi.Float(0.1)) * \
                              dr.power(mi.Float(10.0), gain_rx_dB * mi.Float(0.1))
        else:
            sp_antenna_gain = gain_tx_dB * gain_rx_dB

        specular_power = specular_power * sp_antenna_gain

    # Step S6: Path loss and amplitude (AD through distances)
    # Two-way path loss for specular: 1/(d_tx² × d_rx²)
    path_loss = mi.Float(1.0) / (sp_d_tx * sp_d_tx * sp_d_rx * sp_d_rx)

    radar_scale = mi.Float(self.radar_constant * self.rx_dBFS_scale * self.adc_scale)
    E_amp = radar_scale * dr.sqrt(dr.maximum(specular_power * path_loss, mi.Float(1e-20)))

    _stoc('bsdf_antenna')
    _stic('phase_scatter')
    # Step S7: Phase computation + scatter-add (AD through distances)
    # K-chunked loop to bound peak GPU memory.
    R_total_spec = sp_d_tx + sp_d_rx
    tau_spec = R_total_spec / mi.Float(C)

    TWO_PI = 2.0 * np.pi
    phi_const = mi.Float(TWO_PI * self.min_freq) * tau_spec
    phi_slope = mi.Float(TWO_PI * self.slope) * tau_spec

    # K-chunked gather loop (same pattern as diffuse path).
    # dr.tile()/dr.repeat() create AD graph structures that cause
    # dr.backward() to hang; dr.gather() in chunks works reliably.
    CHUNK = getattr(self.render_config, 'e2e_phase_chunk_size', 64)

    for k_start in range(0, K, CHUNK):
        k_end = min(k_start + CHUNK, K)
        k_chunk = k_end - k_start

        n_elems = n_spec * k_chunk

        # Path index: which specular path does each element belong to
        path_idx = dr.arange(mi.UInt32, n_elems) // mi.UInt32(k_chunk)
        k_local = dr.arange(mi.UInt32, n_elems) % mi.UInt32(k_chunk)
        k_global = k_local + mi.UInt32(k_start)

        # Gather per-path quantities (AD-attached)
        pc = dr.gather(mi.Float, phi_const, path_idx)
        ps = dr.gather(mi.Float, phi_slope, path_idx)
        ea = dr.gather(mi.Float, E_amp, path_idx)
        act = dr.gather(mi.Bool, sp_valid, path_idx)

        # Time samples for this K chunk
        t_k = mi.Float(k_global) / mi.Float(self.sample_rate)

        # Phase + weighted contributions
        phi = pc + ps * t_k
        _cos = dr.cos(phi)
        _sin = dr.sin(phi)
        if not self.render_config.enable_grad_phase:
            _cos = dr.detach(_cos)
            _sin = dr.detach(_sin)
        contrib_real = ea * _cos
        contrib_imag = ea * _sin

        # Flat ADC index: tx_idx * (n_rx * K) + rx_idx * K + k_global
        tx_for_elem = dr.gather(mi.UInt32, sp.tx_idx, path_idx)
        rx_for_elem = dr.gather(mi.UInt32, sp.rx_idx, path_idx)
        flat_adc_idx = (tx_for_elem * mi.UInt32(n_rx * K)
                        + rx_for_elem * mi.UInt32(K)
                        + k_global)

        dr.scatter_add(adc_real_flat, contrib_real, flat_adc_idx, act)
        dr.scatter_add(adc_imag_flat, contrib_imag, flat_adc_idx, act)

    _stoc('phase_scatter')


def _compute_diffraction_e2e(
    self,
    hit_P_expanded: 'mi.Point3f',       # [n_total] hit positions
    hit_N_expanded: 'mi.Vector3f',       # [n_total] hit normals (AD if normal_params)
    dir_hit_to_tx: 'mi.Vector3f',        # [n_total] TX directions (AD if pose)
    dir_hit_to_rx: 'mi.Vector3f',        # [n_total] RX directions (wo)
    active: 'mi.Bool',                   # [n_total] active mask
    physics_params: list,                 # 6 AD-attached material arrays
    n_valid: int,                         # number of unique hits
    n_tx: int, n_rx: int,                # MIMO layout
    vertex_ids_0: 'mi.UInt32' = None,    # [n_total] vertex index 0
    vertex_ids_1: 'mi.UInt32' = None,    # [n_total] vertex index 1
    vertex_ids_2: 'mi.UInt32' = None,    # [n_total] vertex index 2
    bary_u_expanded: 'mi.Float' = None,  # [n_total] barycentric u
    bary_v_expanded: 'mi.Float' = None,  # [n_total] barycentric v
    scene: Optional['mi.Scene'] = None,  # scene for vertex position access
    verbose: bool = False,
) -> Tuple['mi.Float', 'mi.Float', 'mi.Float', 'mi.Float']:
    """Compute inline FSD diffraction for E2E pipeline."""
    n_total = n_valid * n_tx * n_rx
    zero = dr.zeros(mi.Float, n_total)

    diff_config = self.diffraction_config
    if not (_HAS_DIFFRACTION and diff_config is not None and diff_config.enabled
            and self.tri_hash is not None and self.fsd_bsdf is not None):
        return zero, zero, zero, zero

    k = self.fsd_bsdf.k
    beam_sigma = self.fsd_bsdf.beam_sigma

    # D1: Detach geometry to numpy for aperture construction
    hit_P_np = np.stack([np.array(dr.detach(hit_P_expanded.x)),
                         np.array(dr.detach(hit_P_expanded.y)),
                         np.array(dr.detach(hit_P_expanded.z))], axis=1)
    wo_np = np.stack([np.array(dr.detach(dir_hit_to_rx.x)),
                      np.array(dr.detach(dir_hit_to_rx.y)),
                      np.array(dr.detach(dir_hit_to_rx.z))], axis=1)
    wi_np = np.stack([np.array(dr.detach(dir_hit_to_tx.x)),
                      np.array(dr.detach(dir_hit_to_tx.y)),
                      np.array(dr.detach(dir_hit_to_tx.z))], axis=1)
    active_np = np.array(active)

    # D2: Construct apertures (numpy, non-diff topology)
    # Use reference positions (TX=0, RX=0) for each hit
    ref_flat_indices = np.arange(n_valid) * n_tx * n_rx
    hit_pos_ref = hit_P_np[ref_flat_indices]
    wo_ref = wo_np[ref_flat_indices]

    # Pre-filter: only process hits with at least one visible path
    vis_2d = active_np.reshape(n_valid, n_tx * n_rx)
    visible_per_hit = np.any(vis_2d, axis=1)
    vis_hit_indices = np.where(visible_per_hit)[0]

    if len(vis_hit_indices) == 0:
        return zero, zero, zero, zero

    # Per-triangle material arrays for numpy-based material opacity
    tri_eps_r_np, tri_eps_i_np = self.diffraction_tri_eps
    use_mat_opacity = (diff_config.material_opacity
                       and tri_eps_r_np is not None
                       and tri_eps_i_np is not None)

    # Jones polarization
    jones_mode = diff_config.jones_polarization and use_mat_opacity
    tx_pol_np = rx_pol_np = None
    _tpx, _tpy, _tpz = self.bsdf.tx_polarization.x, self.bsdf.tx_polarization.y, self.bsdf.tx_polarization.z
    _rpx, _rpy, _rpz = self.bsdf.rx_polarization.x, self.bsdf.rx_polarization.y, self.bsdf.rx_polarization.z
    dr.eval(_tpx, _tpy, _tpz, _rpx, _rpy, _rpz)
    tx_pol_np = np.array([float(_tpx[0]), float(_tpy[0]), float(_tpz[0])])
    rx_pol_np = np.array([float(_rpx[0]), float(_rpy[0]), float(_rpz[0])])

    # GPU aperture construction (CPU path is deprecated)
    from ..diffraction.fsd_aperture_gpu import construct_apertures_batch_gpu
    result = construct_apertures_batch_gpu(
        hit_pos_ref[vis_hit_indices],
        wo_ref[vis_hit_indices],
        self._gpu_spatial_hash, k, beam_sigma,
        diff_config.max_tessellation_depth,
        diff_config.max_edges_per_hit,
        diff_config.fill_min, diff_config.fill_max,
        tri_eps_r_np if use_mat_opacity else None,
        tri_eps_i_np if use_mat_opacity else None,
        tx_pol_np if jones_mode else None,
        rx_pol_np if jones_mode else None,
        jones_mode, self.fsd_bsdf.beta_max, verbose,
        edge_angle_threshold_deg=diff_config.edge_angle_threshold_deg,
        tri_hash=self.tri_hash,
    )

    if result is None:
        return zero, zero, zero, zero

    flat, active_local_idx = result
    active_hit_orig = vis_hit_indices[active_local_idx]
    n_diffraction_hits = flat.n_apertures

    if n_diffraction_hits == 0:
        return zero, zero, zero, zero

    # D3: Build evaluation arrays (path → aperture mapping)
    bases = active_hit_orig * n_tx * n_rx
    pair_offsets = np.arange(n_tx * n_rx, dtype=np.int32)
    all_pairs = bases[:, None] + pair_offsets[None, :]
    vis_expanded = active_np[all_pairs]
    flat.ap_to_path_indices = [
        all_pairs[a, vis_expanded[a]].astype(np.int32)
        for a in range(flat.n_apertures)
    ]

    path_counts = np.array([len(p) for p in flat.ap_to_path_indices], dtype=np.int32)
    total_evals = int(np.sum(path_counts))

    if total_evals == 0:
        return zero, zero, zero, zero

    eval_path_idx = np.concatenate(flat.ap_to_path_indices)
    eval_ap_idx = np.repeat(np.arange(flat.n_apertures, dtype=np.int32), path_counts)

    # D4: Compute AD-attached inputs
    # (a) wi as DrJit tuple — AD-attached for pose gradients
    wi_x_live = dr.gather(mi.Float, dir_hit_to_tx.x, mi.UInt32(eval_path_idx))
    wi_y_live = dr.gather(mi.Float, dir_hit_to_tx.y, mi.UInt32(eval_path_idx))
    wi_z_live = dr.gather(mi.Float, dir_hit_to_tx.z, mi.UInt32(eval_path_idx))

    # (b) cos_theta_live — AD-attached for normal gradients
    cos_theta_live = None
    if vertex_ids_0 is not None:
        # Per-edge: get owning face normals from AD-attached hit normals
        # The edge's cos_theta = |dot(face_normal, wo)| — recompute from AD normals
        # Gather the aperture's wo direction (frozen screen frame)
        E_total = flat.n_edges
        edge_ap_map = np.zeros(E_total, dtype=np.int32)
        for a in range(flat.n_apertures):
            s = flat.ap_edge_start[a]
            edge_ap_map[s:s + flat.ap_edge_count[a]] = a

        # Get a representative path index for each edge's aperture (for normal lookup)
        edge_hit_orig = active_hit_orig[edge_ap_map]  # [E_total] → hit index
        edge_ref_path = edge_hit_orig * n_tx * n_rx  # ref path (TX=0, RX=0)

        # Gather AD normals at edge's hit location
        edge_ref_path_dr = mi.UInt32(edge_ref_path)
        nx_e = dr.gather(mi.Float, hit_N_expanded.x, edge_ref_path_dr)
        ny_e = dr.gather(mi.Float, hit_N_expanded.y, edge_ref_path_dr)
        nz_e = dr.gather(mi.Float, hit_N_expanded.z, edge_ref_path_dr)

        # wo per edge (frozen screen frame z-axis)
        wo_x_e = mi.Float(flat.ap_wo_dir[edge_ap_map, 0])
        wo_y_e = mi.Float(flat.ap_wo_dir[edge_ap_map, 1])
        wo_z_e = mi.Float(flat.ap_wo_dir[edge_ap_map, 2])

        cos_theta_live = dr.abs(nx_e * wo_x_e + ny_e * wo_y_e + nz_e * wo_z_e)

    # (c) Live edge geometry — AD-attached for vertex position gradients
    live_edge_ex = live_edge_ey = live_edge_vx = live_edge_vy = None
    if (vertex_ids_0 is not None and flat.edge_ba_u is not None
            and flat.ap_hit_pos is not None and len(flat.ap_hit_pos) > 0):
        # Gather live vertex positions for each edge's triangle
        g_vi0 = mi.UInt32(flat.edge_vert_idx_0.astype(np.int32))
        g_vi1 = mi.UInt32(flat.edge_vert_idx_1.astype(np.int32))
        g_vi2 = mi.UInt32(flat.edge_vert_idx_2.astype(np.int32))

        # Get per-vertex positions from the scene (AD-attached if vertex_offset_params)
        # In E2E, vertex positions in the mesh are AD-attached through scene vertices.
        mesh = None
        if scene is not None:
            try:
                shapes = scene.shapes()
                if len(shapes) > 0:
                    mesh = shapes[0]
            except Exception:
                pass

        if mesh is not None:
            # Get AD-attached vertex positions from the mesh
            vp = mesh.vertex_positions_buffer()
            dr.width(vp) // 3
            all_vx = dr.gather(mi.Float, vp, g_vi0 * 3 + 0)
            all_vy = dr.gather(mi.Float, vp, g_vi0 * 3 + 1)
            all_vz = dr.gather(mi.Float, vp, g_vi0 * 3 + 2)
            all_v1x = dr.gather(mi.Float, vp, g_vi1 * 3 + 0)
            all_v1y = dr.gather(mi.Float, vp, g_vi1 * 3 + 1)
            all_v1z = dr.gather(mi.Float, vp, g_vi1 * 3 + 2)
            all_v2x = dr.gather(mi.Float, vp, g_vi2 * 3 + 0)
            all_v2y = dr.gather(mi.Float, vp, g_vi2 * 3 + 1)
            all_v2z = dr.gather(mi.Float, vp, g_vi2 * 3 + 2)

            # Endpoint A: bary (ba_u, ba_v) in triangle (vi0, vi1, vi2)
            ba_u = mi.Float(flat.edge_ba_u.astype(np.float32))
            ba_v = mi.Float(flat.edge_ba_v.astype(np.float32))
            ba_w = mi.Float(1.0) - ba_u - ba_v
            pa_x = ba_w * all_vx + ba_u * all_v1x + ba_v * all_v2x
            pa_y = ba_w * all_vy + ba_u * all_v1y + ba_v * all_v2y
            pa_z = ba_w * all_vz + ba_u * all_v1z + ba_v * all_v2z

            # Endpoint B: bary (bb_u, bb_v)
            bb_u = mi.Float(flat.edge_bb_u.astype(np.float32))
            bb_v = mi.Float(flat.edge_bb_v.astype(np.float32))
            bb_w = mi.Float(1.0) - bb_u - bb_v
            pb_x = bb_w * all_vx + bb_u * all_v1x + bb_v * all_v2x
            pb_y = bb_w * all_vy + bb_u * all_v1y + bb_v * all_v2y
            pb_z = bb_w * all_vz + bb_u * all_v1z + bb_v * all_v2z

            # Per-aperture hit position and screen frame (frozen)
            hit_x = mi.Float(flat.ap_hit_pos[edge_ap_map, 0].astype(np.float32))
            hit_y = mi.Float(flat.ap_hit_pos[edge_ap_map, 1].astype(np.float32))
            hit_z = mi.Float(flat.ap_hit_pos[edge_ap_map, 2].astype(np.float32))
            tang_x = mi.Float(flat.ap_tangent[edge_ap_map, 0])
            tang_y = mi.Float(flat.ap_tangent[edge_ap_map, 1])
            tang_z = mi.Float(flat.ap_tangent[edge_ap_map, 2])
            btang_x = mi.Float(flat.ap_bitangent[edge_ap_map, 0])
            btang_y = mi.Float(flat.ap_bitangent[edge_ap_map, 1])
            btang_z = mi.Float(flat.ap_bitangent[edge_ap_map, 2])

            # Project endpoint A to screen space
            da_x = pa_x - hit_x; da_y = pa_y - hit_y; da_z = pa_z - hit_z
            sa_x = da_x * tang_x + da_y * tang_y + da_z * tang_z
            sa_y = da_x * btang_x + da_y * btang_y + da_z * btang_z

            # Project endpoint B to screen space
            db_x = pb_x - hit_x; db_y = pb_y - hit_y; db_z = pb_z - hit_z
            sb_x = db_x * tang_x + db_y * tang_y + db_z * tang_z
            sb_y = db_x * btang_x + db_y * btang_y + db_z * btang_z

            # Edge vector and midpoint in screen space (AD-attached)
            live_edge_ex = sb_x - sa_x
            live_edge_ey = sb_y - sa_y
            live_edge_vx = (sa_x + sb_x) * mi.Float(0.5)
            live_edge_vy = (sa_y + sb_y) * mi.Float(0.5)

    # D5: Extract AD-attached material params for eval_batch_drjit
    eps_real_vertex = physics_params[0] if physics_params is not None else None
    eps_imag_vertex = physics_params[1] if physics_params is not None else None

    # D6: Call eval_batch_drjit with AD inputs
    f_diff_batch, psi_r_batch, psi_i_batch = eval_batch_drjit(
        flat,
        wi_world=(wi_x_live, wi_y_live, wi_z_live),  # AD tuple
        k=k,
        aperture_indices=eval_ap_idx,
        eps_real_vertex=eps_real_vertex,
        eps_imag_vertex=eps_imag_vertex,
        jones_mode=jones_mode,
        cos_theta_live=cos_theta_live,
        live_edge_ex=live_edge_ex,
        live_edge_ey=live_edge_ey,
        live_edge_vx=live_edge_vx,
        live_edge_vy=live_edge_vy,
    )

    # D7: Scatter results to [n_total] arrays
    f_diff = dr.zeros(mi.Float, n_total)
    psi_real = dr.zeros(mi.Float, n_total)
    psi_imag = dr.zeros(mi.Float, n_total)

    eval_path_idx_dr = mi.UInt32(eval_path_idx)
    dr.scatter(f_diff, f_diff_batch, eval_path_idx_dr)
    dr.scatter(psi_real, psi_r_batch, eval_path_idx_dr)
    dr.scatter(psi_imag, psi_i_batch, eval_path_idx_dr)

    # Beta from aperture construction (frozen, non-diff)
    beta_np = np.zeros(n_total, dtype=np.float64)
    if diff_config.energy_borrowing:
        beta_per_eval = flat.ap_beta[eval_ap_idx]
        beta_np[eval_path_idx] = beta_per_eval
    beta = mi.Float(beta_np)

    return f_diff, psi_real, psi_imag, beta
