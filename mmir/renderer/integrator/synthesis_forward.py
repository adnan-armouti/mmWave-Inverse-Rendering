"""Forward render synthesis pipeline (Phase A)."""

from typing import Optional
import drjit as dr
import mitsuba as mi
import numpy as np

from ..sampler import ReservoirHits
from ..specular.paths import SpecularPaths

from .cached_geometry import ADCResult

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


# This was a loop-based implementation that is 135-146x slower than
# synthesize_shared_hits_vectorized(). Use synthesize_shared_hits_vectorized() instead.


def synthesize_shared_hits_vectorized(
    self,
    tx_positions: 'mi.Point3f',
    rx_positions: 'mi.Point3f',
    tx_boresights: Optional['mi.Vector3f'] = None,
    rx_boresights: Optional['mi.Vector3f'] = None,
    reservoir_hits: ReservoirHits = None,
    scene: 'mi.Scene' = None,
    tx_pattern_rrts: Optional[np.ndarray] = None,
    rx_pattern_rrts: Optional[np.ndarray] = None,
    tx_pattern_loader: Optional['AntennaPatternLoader'] = None,
    rx_pattern_loader: Optional['AntennaPatternLoader'] = None,
    antenna_gain_linear: bool = False,
    verbose: bool = True,
    return_components: bool = False,
    use_non_ka_bsdf: bool = False,
    specular_paths: Optional[SpecularPaths] = None,
    patch_has_specular: Optional['mi.Bool'] = None,
    tri_to_patch: Optional['mi.Int32'] = None,
    return_cached_geometry: bool = False,
) -> ADCResult:
    """FULLY VECTORIZED ADC synthesis with radar equation support - NO PYTHON LOOPS."""
    from ..utils.math import (
        expand_point3f_for_mimo, expand_vector3f_for_mimo, generate_mimo_indices, gather_point3f,
        gather_vector3f,
    )

    n_tx = dr.width(tx_positions)
    n_rx = reservoir_hits.n_rx
    K = self.num_samples

    if verbose:
        print("\n[SBRIntegratorRef] Synthesizing ADC signals (FULLY VECTORIZED - NO LOOPS)")
        print(f"  TX elements: {n_tx}, RX elements: {n_rx}")
        print(f"  ADC samples: {K}")
        print(f"  f0 = {self.min_freq/1e9:.3f} GHz")
        print(f"  S = {self.slope/1e12:.3f} THz/s")
        print(f"  λ = {self.wavelength*1000:.2f} mm")
        gain_scale_str = "linear" if antenna_gain_linear else "logarithmic dB"
        print(f"  Antenna gain scale: {gain_scale_str}")
        if self.render_config.use_radar_equation:
            print("  Radar equation: ENABLED")
            print(f"    Radar constant = {self.radar_constant:.6e}")
            print(f"    Total scale factor = {self.radar_constant * self.rx_dBFS_scale * self.adc_scale:.6e}")
        else:
            print("  Radar equation: DISABLED (legacy BRDF-only mode)")

    # Step 1: Extract and flatten all valid hits
    # Get all hit data
    hit_P_all = reservoir_hits.hit_P
    hit_N_all = reservoir_hits.hit_N
    hit_rho_all = reservoir_hits.hit_rho  # [n_total_slots, 3] numpy array

    # Count valid hits on GPU (avoids full boolean transfer)
    n_valid = int(dr.sum(mi.UInt32(reservoir_hits.valid))[0])

    if verbose:
        print(f"  Total valid hits: {n_valid}")

    # Get valid hit indices (compressed on GPU, then to numpy for remaining indexing)
    valid_indices_dr = dr.compress(reservoir_hits.valid)
    valid_indices_np = np.array(valid_indices_dr)

    # Gather valid hits into DrJit arrays
    valid_P = gather_point3f(hit_P_all, valid_indices_dr)  # [n_valid]
    valid_N = gather_vector3f(hit_N_all, valid_indices_dr)  # [n_valid]
    valid_rho = hit_rho_all[valid_indices_np]  # [n_valid, 3] numpy

    # Extract primitive IDs for differentiable re-synthesis (dr.gather on materials)

    # Extract barycentric coordinates + vertex indices for differentiable
    # normal/vertex recomputation in Phase B (needed regardless of material mode)
    valid_vertex_ids_0 = valid_vertex_ids_1 = valid_vertex_ids_2 = None

    # Extract PDF and n_attempted for MC probability chain (Option B normalization)
    # hit_pdf: [n_total_slots] -> [n_valid] (PDF for each valid hit)
    # n_attempted_per_rx: [n_rx] -> need to map to [n_valid] based on which RX owns each hit
    valid_pdf = None
    valid_n_attempted = None
    if reservoir_hits.hit_pdf is not None and reservoir_hits.n_attempted_per_rx is not None:
        valid_pdf = reservoir_hits.hit_pdf[valid_indices_np]  # [n_valid] numpy

        # Map each valid hit to its owning RX to get n_attempted
        # The flat index i maps to rx_idx = i // n_hits_per_rx
        valid_hit_slot_indices = valid_indices_np  # Original slot indices for valid hits
        valid_rx_indices = valid_hit_slot_indices // reservoir_hits.n_hits_per_rx
        valid_n_attempted = reservoir_hits.n_attempted_per_rx[valid_rx_indices]  # [n_valid]

        if verbose:
            print(f"  MC probability chain: PDF range [{valid_pdf.min():.4f}, {valid_pdf.max():.4f}]")
            print(f"  MC probability chain: n_attempted range [{valid_n_attempted.min()}, {valid_n_attempted.max()}]")

    # Step 2: MIMO Expansion - [n_valid] -> [n_valid × n_tx × n_rx]
    n_total = n_valid * n_tx * n_rx

    if verbose:
        print(f"  MIMO expansion: {n_valid} hits × {n_tx} TX × {n_rx} RX = {n_total:,} paths")

    # Expand hit positions and normals
    hit_P_expanded = expand_point3f_for_mimo(valid_P, n_tx, n_rx)  # [n_total]
    hit_N_expanded = expand_vector3f_for_mimo(valid_N, n_tx, n_rx)  # [n_total]

    # Expand material properties
    # Detect physics mode from column count
    physics_mode = valid_rho.shape[1] == 6

    # Lazy-upload materials to GPU for dr.gather-based expansion
    if self.triangle_materials is not None:
        if (self._gpu_material_cols is None
                or id(self.triangle_materials) != self._gpu_material_src_id):
            self._gpu_material_cols = [
                mi.Float(self.triangle_materials[:, c].astype(np.float32))
                for c in range(self.triangle_materials.shape[1])
            ]
            self._gpu_material_src_id = id(self.triangle_materials)

    # Try GPU-resident material expansion via dr.gather (avoids numpy->DrJit transfer)
    _n_mimo = n_tx * n_rx
    _use_gpu_mats = (
        self._gpu_material_cols is not None
        and reservoir_hits.hit_ID is not None
        and len(self._gpu_material_cols) >= (6 if physics_mode else 3)
    )

    if _use_gpu_mats:
        # Get per-hit prim_ids (reuse if already computed for cached geometry)
        _vpi = dr.gather(mi.Int32, reservoir_hits.hit_ID, valid_indices_dr)
        _vpi = mi.UInt32(dr.maximum(_vpi, mi.Int32(0)))
        _n_tris_gpu = dr.width(self._gpu_material_cols[0])
        _vpi = dr.minimum(_vpi, mi.UInt32(_n_tris_gpu - 1))

        rho_cols_expanded = [
            dr.repeat(dr.gather(mi.Float, self._gpu_material_cols[c], _vpi), _n_mimo)
            for c in range(6)
        ]
        rho_eps_real = rho_cols_expanded[0]
        rho_eps_imag = rho_cols_expanded[1]
        rho_sigma_h = rho_cols_expanded[2]
        rho_l_c = rho_cols_expanded[3]
        rho_tau = rho_cols_expanded[4]
        rho_thickness = rho_cols_expanded[5]
        rho_x_expanded = None
        rho_y_expanded = None
        rho_z_expanded = None
    else:
        # Fallback: numpy -> DrJit path
        rho_x_expanded = mi.Float(np.repeat(valid_rho[:, 0], n_tx * n_rx))
        rho_y_expanded = mi.Float(np.repeat(valid_rho[:, 1], n_tx * n_rx))
        rho_z_expanded = mi.Float(np.repeat(valid_rho[:, 2], n_tx * n_rx))

    # Expand PDF and n_attempted for MC probability chain (Option B normalization)
    pdf_expanded = None
    n_attempted_expanded = None
    if valid_pdf is not None and valid_n_attempted is not None:
        # Each valid hit is replicated n_tx × n_rx times in MIMO expansion
        pdf_expanded = mi.Float(np.repeat(valid_pdf, n_tx * n_rx))  # [n_total]
        n_attempted_expanded = mi.Float(np.repeat(valid_n_attempted, n_tx * n_rx))  # [n_total]

    # Generate TX/RX indices for each expanded path
    tx_idx, rx_idx = generate_mimo_indices(n_valid, n_tx, n_rx)

    # Gather TX and RX positions for each path
    tx_pos_expanded = gather_point3f(tx_positions, tx_idx)  # [n_total]
    rx_pos_expanded = gather_point3f(rx_positions, rx_idx)  # [n_total]

    # Gather TX and RX boresights for pattern evaluation (if provided)
    tx_boresights_expanded = None
    if tx_boresights is not None:
        tx_boresights_expanded = gather_vector3f(tx_boresights, tx_idx)  # [n_total]

    rx_boresights_expanded = None
    if rx_boresights is not None:
        rx_boresights_expanded = gather_vector3f(rx_boresights, rx_idx)  # [n_total]

    # Step 3: Vectorized Geometry Computation
    # Distance from RX to hit
    delta_rx = hit_P_expanded - rx_pos_expanded
    d_rx_to_hit = dr.norm(delta_rx)  # [n_total]

    # Distance from hit to TX
    delta_tx = tx_pos_expanded - hit_P_expanded
    d_hit_to_tx = dr.norm(delta_tx)  # [n_total]

    # Direction from hit to RX (wo - outgoing)
    dir_hit_to_rx = mi.Vector3f(
        -delta_rx.x / dr.maximum(d_rx_to_hit, 1e-10),
        -delta_rx.y / dr.maximum(d_rx_to_hit, 1e-10),
        -delta_rx.z / dr.maximum(d_rx_to_hit, 1e-10)
    )

    # Direction from hit to TX (wi - incident/light direction)
    dir_hit_to_tx = mi.Vector3f(
        delta_tx.x / dr.maximum(d_hit_to_tx, 1e-10),
        delta_tx.y / dr.maximum(d_hit_to_tx, 1e-10),
        delta_tx.z / dr.maximum(d_hit_to_tx, 1e-10)
    )

    # cos theta values (initial, before potential normal flip)
    cos_theta_out_raw = dr.dot(dir_hit_to_rx, hit_N_expanded)  # [n_total]
    cos_theta_in_raw = dr.dot(dir_hit_to_tx, hit_N_expanded)   # [n_total]

    # Double-sided rendering: flip normals for backfacing surfaces
    # When enabled, surfaces with normals pointing away from the RX are
    # treated as if they face the RX. This is essential for:
    # - LiDAR meshes with inconsistent normal orientation
    # - Thin walls/guardrails that should scatter from both sides
    # - Improving sidewall visibility in staircase scenes
    getattr(self.render_config, 'double_sided', True)

    need_flip = cos_theta_out_raw < 0
    hit_N_expanded = dr.select(
        need_flip,
        mi.Vector3f(-hit_N_expanded.x, -hit_N_expanded.y, -hit_N_expanded.z),
        hit_N_expanded
    )
    # Recompute cos_theta with potentially flipped normals
    cos_theta_out = dr.abs(cos_theta_out_raw)
    cos_theta_in = dr.select(need_flip, -cos_theta_in_raw, cos_theta_in_raw)

    if verbose:
        n_flipped = int(dr.sum(mi.Float(need_flip))[0]) if hasattr(dr.sum(mi.Float(need_flip)), '__getitem__') else int(dr.sum(mi.Float(need_flip)))
        print(f"  Double-sided: flipped {n_flipped:,} normals ({100*n_flipped/n_total:.1f}%)")
        # Diagnostic: check cos_theta distribution after flip
        cos_out_np = np.array(cos_theta_out)
        cos_in_np = np.array(cos_theta_in)
        print(f"  After flip: cos_theta_out range [{cos_out_np.min():.4f}, {cos_out_np.max():.4f}]")
        print(f"  After flip: cos_theta_in range [{cos_in_np.min():.4f}, {cos_in_np.max():.4f}]")
        n_still_backfacing_in = np.sum(cos_in_np < 0)
        n_grazing_out = np.sum((cos_out_np >= 0) & (cos_out_np < 0.1))
        n_grazing_in = np.sum((cos_in_np >= 0) & (cos_in_np < 0.1))
        print(f"  After flip: {n_still_backfacing_in:,} paths still have cos_theta_in < 0 ({100*n_still_backfacing_in/n_total:.1f}%)")
        print(f"  After flip: {n_grazing_out:,} paths have grazing cos_theta_out (0-0.1)")
        print(f"  After flip: {n_grazing_in:,} paths have grazing cos_theta_in (0-0.1)")

    # Step 4: Vectorized Visibility Check (SINGLE CALL)
    epsilon = 1e-4
    shadow_origins = mi.Point3f(
        hit_P_expanded.x + epsilon * dir_hit_to_tx.x,
        hit_P_expanded.y + epsilon * dir_hit_to_tx.y,
        hit_P_expanded.z + epsilon * dir_hit_to_tx.z
    )

    shadow_rays = mi.Ray3f(shadow_origins, dir_hit_to_tx)
    shadow_rays.maxt = d_hit_to_tx - 2 * epsilon

    # Single visibility check for ALL paths
    occluded = scene.ray_test(shadow_rays)
    visible = ~occluded  # [n_total]

    # Step 5: Vectorized BRDF Computation
    # Handles both scalar BSDF (power) and complex Jones BSDF (Fresnel phase)
    # Per-patch adaptive BSDF: when patch_has_specular is provided, select
    # eval_non_ka_cos vs eval_f_cos per-hit based on patch specular coverage
    # Physics mode: dispatches to *_physics() methods with 6-param signature
    # Check if BSDF uses Fresnel phase (Jones with use_fresnel_phase=True)
    use_fresnel_phase = getattr(self.bsdf, 'use_fresnel_phase', False)

    # Build BSDF call kwargs based on mode
    if physics_mode:
        bsdf_kwargs = dict(
            wo=dir_hit_to_rx, wi=dir_hit_to_tx, n=hit_N_expanded,
            eps_real=rho_eps_real, eps_imag=rho_eps_imag,
            sigma_h=rho_sigma_h, l_c=rho_l_c, tau=rho_tau,
            thickness=rho_thickness,
        )
        _eval_f_cos = getattr(self.bsdf, 'eval_f_cos_physics', self.bsdf.eval_f_cos)
        _eval_non_ka_cos = getattr(self.bsdf, 'eval_non_ka_cos_physics', self.bsdf.eval_non_ka_cos)
        _eval_f_cos_complex = getattr(self.bsdf, 'eval_f_cos_complex_physics',
                                       getattr(self.bsdf, 'eval_f_cos_complex', None))
        _eval_f_cos_components = getattr(self.bsdf, 'eval_f_cos_components_physics',
                                          getattr(self.bsdf, 'eval_f_cos_components', None))
    else:
        bsdf_kwargs = dict(
            wo=dir_hit_to_rx, wi=dir_hit_to_tx, n=hit_N_expanded,
            albedo=rho_x_expanded, roughness=rho_y_expanded,
            metallic=rho_z_expanded,
        )
        _eval_f_cos = self.bsdf.eval_f_cos
        _eval_non_ka_cos = getattr(self.bsdf, 'eval_non_ka_cos', None)
        _eval_f_cos_complex = getattr(self.bsdf, 'eval_f_cos_complex', None)
        getattr(self.bsdf, 'eval_f_cos_components', None)

    if use_non_ka_bsdf and _eval_non_ka_cos is not None:
        brdf_weight = _eval_non_ka_cos(**bsdf_kwargs)
        E_coh_real = None
        E_coh_imag = None
        f_inc_cos = None
        use_fresnel_phase = False

        if verbose:
            print("  Using eval_non_ka_cos (KA excluded, handled by image method)")
    else:
        brdf_weight = _eval_f_cos(**bsdf_kwargs)
        E_coh_real = None
        E_coh_imag = None
        f_inc_cos = None

    if verbose and physics_mode:
        print("  Physics mode: 6-param direct BSDF dispatch")

    # Step 5b: Evaluate BSDF components (if requested)

    # Step 5c: Free-Space Diffraction BSDF (fsdBSDF)
    # Additive combination: f_total = (1-β)×f_refl + f_diff
    # Only evaluated at hit points near diffracting edges.
    #
    # Optimization: aperture depends on (hit_pos, wo) where wo points
    # toward RX. Same (hit, RX) → same aperture, reused across all TX.
    # MIMO layout: flat_idx = h*(n_tx*n_rx) + t*n_rx + r
    # Initialize diffraction cache variables (used later in CachedGeometry)

    diff_config = self.diffraction_config
    if (_HAS_DIFFRACTION and diff_config is not None and diff_config.enabled
            and self.tri_hash is not None and self.fsd_bsdf is not None):

        # Convert DrJit arrays to numpy for per-hit aperture construction
        hit_P_np = np.stack([np.array(hit_P_expanded.x),
                             np.array(hit_P_expanded.y),
                             np.array(hit_P_expanded.z)], axis=1)  # [n_total, 3]
        wo_np = np.stack([np.array(dir_hit_to_rx.x),
                          np.array(dir_hit_to_rx.y),
                          np.array(dir_hit_to_rx.z)], axis=1)      # [n_total, 3]
        wi_np = np.stack([np.array(dir_hit_to_tx.x),
                          np.array(dir_hit_to_tx.y),
                          np.array(dir_hit_to_tx.z)], axis=1)      # [n_total, 3]

        k = self.fsd_bsdf.k
        beam_sigma = self.fsd_bsdf.beam_sigma

        n_diffraction_hits = 0
        visible_np = np.array(visible)

        # Per-triangle material arrays for material-dependent diffraction
        tri_eps_r_np, tri_eps_i_np = self.diffraction_tri_eps
        use_mat_opacity = (diff_config.material_opacity
                           and tri_eps_r_np is not None
                           and tri_eps_i_np is not None)

        # Jones polarization: extract TX/RX E-field directions from BSDF
        jones_mode = diff_config.jones_polarization and use_mat_opacity
        tx_pol_np = None
        rx_pol_np = None
        if jones_mode:
            tx_pol_np = np.array([
                float(self.bsdf.tx_polarization.x[0]),
                float(self.bsdf.tx_polarization.y[0]),
                float(self.bsdf.tx_polarization.z[0]),
            ])
            rx_pol_np = np.array([
                float(self.bsdf.rx_polarization.x[0]),
                float(self.bsdf.rx_polarization.y[0]),
                float(self.bsdf.rx_polarization.z[0]),
            ])

        # Batch aperture construction (replaces serial loop)
        # Reference positions for all valid hits (TX=0, RX=0)
        ref_flat_indices = np.arange(n_valid) * n_tx * n_rx
        hit_pos_ref = hit_P_np[ref_flat_indices]   # [n_valid, 3]
        wo_ref = wo_np[ref_flat_indices]            # [n_valid, 3]

        # Pre-filter: only process hits with at least one visible path
        vis_2d = visible_np.reshape(n_valid, n_tx * n_rx)
        visible_per_hit = np.any(vis_2d, axis=1)    # [n_valid]
        vis_hit_indices = np.where(visible_per_hit)[0]

        flat = None

        if len(vis_hit_indices) > 0:
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

            if result is not None:
                flat, active_local_idx = result

                # Map batch-local indices back to original hit indices
                active_hit_orig = vis_hit_indices[active_local_idx]  # [A]
                n_diffraction_hits = flat.n_apertures
                n_valid - n_diffraction_hits

                # Build ap_to_path_indices: for each aperture, find visible paths
                bases = active_hit_orig * n_tx * n_rx  # [A]
                pair_offsets = np.arange(n_tx * n_rx, dtype=np.int32)
                all_pairs = bases[:, None] + pair_offsets[None, :]  # [A, n_tx*n_rx]
                vis_expanded = visible_np[all_pairs]  # [A, n_tx*n_rx]
                flat.ap_to_path_indices = [
                    all_pairs[a, vis_expanded[a]].astype(np.int32)
                    for a in range(flat.n_apertures)
                ]

        # Phase B: Batch-evaluate all (aperture, TX) pairs at once
        # Use GPU (DrJit) path: results stay on GPU, no CPU→GPU round-trip
        f_diff_dr = dr.zeros(mi.Float, n_total)
        beta_np = np.zeros(n_total, dtype=np.float64)
        psi_diff_real_dr = dr.zeros(mi.Float, n_total)
        psi_diff_imag_dr = dr.zeros(mi.Float, n_total)

        if flat is not None and flat.n_apertures > 0:

            # Build evaluation arrays using vectorized concat + repeat
            path_counts = np.array([len(p) for p in flat.ap_to_path_indices], dtype=np.int32)
            total_evals = int(np.sum(path_counts))

            if total_evals > 0:
                eval_path_idx = np.concatenate(flat.ap_to_path_indices)  # [N_eval]
                eval_ap_idx = np.repeat(np.arange(flat.n_apertures, dtype=np.int32), path_counts)
                eval_wi = wi_np[eval_path_idx]  # [N_eval, 3]

                # GPU eval: returns DrJit mi.Float arrays
                f_diff_batch, psi_r_batch, psi_i_batch = eval_batch_drjit(
                    flat, eval_wi, k, aperture_indices=eval_ap_idx
                )

                # Scatter results directly on GPU (no CPU→GPU transfer)
                eval_path_idx_dr = mi.UInt32(eval_path_idx)
                dr.scatter(f_diff_dr, f_diff_batch, eval_path_idx_dr)

                if diff_config.complex_output:
                    dr.scatter(psi_diff_real_dr, psi_r_batch, eval_path_idx_dr)
                    dr.scatter(psi_diff_imag_dr, psi_i_batch, eval_path_idx_dr)

                if diff_config.energy_borrowing:
                    beta_per_eval = flat.ap_beta[eval_ap_idx]
                    beta_np[eval_path_idx] = beta_per_eval

        # Energy-conserving edge diffraction for backscatter radar:
        #
        # The FSD BSDF (Steinberg et al. 2024) is a TRANSMISSIVE lobe —
        # the reference implementation (free_space_diffraction.cpp) explicitly
        # restricts evaluation to the opposite hemisphere from wi:
        #   eval():  cosθ(wo) × cosθ(wi) < 0   (line 112)
        #   FSDpossible(): dot(wi, -wo) > 0.35  (line 270)
        # Both guards prevent FSD from producing backscatter.
        #
        # For our single-bounce backscatter radar, edges affect reflection
        # through shadow/blocking: β fraction of the beam is intercepted by
        # edges, reducing reflected energy by (1-β).  The intercepted energy
        # goes to forward diffraction (transmission) — a loss in our renderer,
        # analogous to how we model transmission through materials as a loss.
        #
        # Energy budget per path:
        #   Non-intercepted: (1-β) × R  →  reflected back (captured)
        #   Edge-intercepted: β          →  forward-diffracted (lost)
        #   Absorbed:         (1-β)(1-R) →  absorbed (lost)
        if diff_config.energy_borrowing and np.any(beta_np > 0):
            one_minus_beta = mi.Float(1.0 - beta_np)
            brdf_weight = brdf_weight * one_minus_beta

    # Step 6: Vectorized Antenna Pattern Evaluation (with orientation support)
    # Track combined scale factor for component weights (Steps 6+7)
    antenna_gain_combined = None
    mc_correction = None

    # GPU-native path: use AntennaPatternLoader (DrJit, no GPU→CPU sync)
    # Note: pattern_mode check removed to match E2E path (synthesis_e2e.py)
    # which always uses evaluate_combined_gain when loaders are available
    _use_gpu_pattern = (
        getattr(self.render_config, 'antenna_gain_model', 'loader') == 'loader'
        and evaluate_combined_gain is not None
        and (tx_pattern_loader is not None or rx_pattern_loader is not None)
    )

    if _use_gpu_pattern:
        # Direction from antenna to hit (negate hit-to-antenna vectors)
        dir_rx_to_hit = -dir_hit_to_rx
        dir_tx_to_hit = -dir_hit_to_tx

        # RX gain (GPU-native: C × gain_E × gain_H, returns linear)
        gain_rx_linear = evaluate_combined_gain(
            rx_pattern_loader, dir_rx_to_hit, rx_boresights_expanded)

        # TX gain (GPU-native)
        gain_tx_linear = evaluate_combined_gain(
            tx_pattern_loader, dir_tx_to_hit, tx_boresights_expanded)

        # evaluate_combined_gain returns linear scale directly
        antenna_gain_combined = gain_rx_linear * gain_tx_linear

        brdf_weight = brdf_weight * antenna_gain_combined

    elif tx_pattern_rrts is not None or rx_pattern_rrts is not None:
        getattr(self.render_config, 'pattern_mode', 'fixed')
        # DrJit GPU fallback: all-GPU evaluation (no CPU↔GPU transfers)
        dir_rx_to_hit = mi.Vector3f(-dir_hit_to_rx.x, -dir_hit_to_rx.y, -dir_hit_to_rx.z)
        dir_tx_to_hit = mi.Vector3f(-dir_hit_to_tx.x, -dir_hit_to_tx.y, -dir_hit_to_tx.z)

        gain_rx_dB = evaluate_gain_rrts_style_drjit_product(
            dir_rx_to_hit, rx_pattern_rrts,
            rx_boresights_expanded if rx_boresights_expanded is not None
            else mi.Vector3f(0.0, 1.0, 0.0),
        )

        gain_tx_dB = evaluate_gain_rrts_style_drjit_product(
            dir_tx_to_hit, tx_pattern_rrts,
            tx_boresights_expanded if tx_boresights_expanded is not None
            else mi.Vector3f(0.0, 1.0, 0.0),
        )

        gain_rx_linear = dr.power(mi.Float(10.0), gain_rx_dB * mi.Float(0.1))
        gain_tx_linear = dr.power(mi.Float(10.0), gain_tx_dB * mi.Float(0.1))

        antenna_gain_combined = gain_rx_linear * gain_tx_linear
        brdf_weight = brdf_weight * antenna_gain_combined

        if verbose:
            print("  Antenna pattern: DrJit GPU fallback")

    # Step 7: Apply Monte Carlo correction (Option B: 1/(pdf × n_attempted))
    # According to rx_centric_radar_mc_estimators.tex Eq. 4:
    # P_r = (P_t λ²)/(4π)² × (1/N_r) × Σⱼ [G_t G_r c_t f V] / [d_t² × ρ_rx(ωⱼ)]
    #
    # Option B normalization: normalize by n_attempted (not n_hits)
    # This accounts for the conditional distribution p(ω|hit) = p(ω) × 1[hit(ω)] / P_hit
    # where P_hit = n_hits / n_attempted
    #
    # The correction factor is: 1 / (ρ_rx(ω) × n_attempted_per_rx)
    EPS_PDF = 1e-8
    mc_correction = mi.Float(1.0) / (dr.maximum(pdf_expanded, EPS_PDF) * n_attempted_expanded)
    brdf_weight = brdf_weight * mc_correction

    if verbose:
        mc_corr_np = np.array(mc_correction)
        valid_mc = mc_corr_np[np.array(visible) & (np.array(cos_theta_out) > 1e-6) & (np.array(cos_theta_in) > 1e-6)]
        if len(valid_mc) > 0:
            print(f"  MC correction range: [{valid_mc.min():.4e}, {valid_mc.max():.4e}]")

    # DEBUG: Match e2e debug prints for comparison
    if verbose:
        dr.eval(brdf_weight, antenna_gain_combined if antenna_gain_combined is not None else mi.Float(1.0))
        _bw = np.array(brdf_weight)
        _act_mask = np.array(visible) & (np.array(cos_theta_out) > 1e-6) & (np.array(cos_theta_in) > 1e-6)
        if _act_mask.sum() > 0:
            print("  [FWD-DEBUG] After BSDF+antenna+MC:")
            print(f"    brdf_weight[active]: mean={_bw[_act_mask].mean():.6e}, max={_bw[_act_mask].max():.6e}, min={_bw[_act_mask].min():.6e}")
            if antenna_gain_combined is not None:
                _ag = np.array(antenna_gain_combined)
                print(f"    antenna_gain[active]: mean={_ag[_act_mask].mean():.6e}, max={_ag[_act_mask].max():.6e}")

    # Step 8: Active mask (visibility + geometry validity)
    active = visible & (cos_theta_out > 1e-6) & (cos_theta_in > 1e-6)

    # Count visible paths
    active_count = dr.sum(mi.Float(active))
    total_paths = int(active_count[0]) if hasattr(active_count, '__getitem__') else int(active_count)

    if verbose:
        print(f"  Visible paths: {total_paths:,}")

    # Step 9: Compute weight based on radar equation or legacy mode
    R_total = d_rx_to_hit + d_hit_to_tx  # [n_total]

    d_t = d_hit_to_tx  # Distance from hit to TX only [n_total]

    # Path loss: 1 / d_t² (NOT 1/R⁴)
    path_loss = mi.Float(1.0) / (d_t * d_t)
    dr.sqrt(path_loss)

    # Radar scaling factor
    radar_scale = mi.Float(self.radar_constant * self.rx_dBFS_scale * self.adc_scale)

    E_amplitude = radar_scale * dr.sqrt(brdf_weight * path_loss)
    weight = E_amplitude

    # Step 10: Vectorized Phase Computation + Scatter-Add Accumulation
    tau = R_total / C  # [n_total]

    # Phase constants (propagation phase)
    TWO_PI = 2.0 * np.pi
    phi_const = TWO_PI * self.min_freq * tau   # [n_total]
    phi_slope = TWO_PI * self.slope * tau      # [n_total]

    # Time grid as DrJit array
    t_grid_dr = self._t_grid_dr  # [K] (cached on GPU)

    # Accumulate over the K time samples in chunks so that n_total * chunk_K stays
    # under Dr.Jit's 2^32 element limit (the 100x100 dense array needs this).
    MAX_ELEMENTS = 2_000_000_000
    chunk_K = max(1, min(K, MAX_ELEMENTS // max(n_total, 1)))

    adc_real_flat = dr.zeros(mi.Float, n_tx * n_rx * K)
    adc_imag_flat = dr.zeros(mi.Float, n_tx * n_rx * K)

    for k_start in range(0, K, chunk_K):
        cK = min(k_start + chunk_K, K) - k_start

        # Expand to [n_total × cK] (dr.tile layout: k = i // n_total)
        phi_const_3d = dr.tile(phi_const, cK)
        phi_slope_3d = dr.tile(phi_slope, cK)
        t_chunk = dr.repeat(dr.gather(mi.Float, t_grid_dr,
                                      dr.arange(mi.UInt32, cK) + mi.UInt32(k_start)), n_total)
        active_3d = dr.tile(active, cK)

        # Propagation phase and complex phasor
        phi = phi_const_3d + phi_slope_3d * t_chunk
        cos_phi = dr.cos(phi)
        sin_phi = dr.sin(phi)

        weight_3d = dr.tile(weight, cK)
        contrib_real = weight_3d * cos_phi
        contrib_imag = weight_3d * sin_phi

        # Flat indices into [n_tx, n_rx, K] and atomic accumulation
        tx_idx_3d = dr.tile(tx_idx, cK)
        rx_idx_3d = dr.tile(rx_idx, cK)
        k_idx = dr.arange(mi.UInt32, n_total * cK) // mi.UInt32(n_total) + mi.UInt32(k_start)
        flat_idx = tx_idx_3d * mi.UInt32(n_rx * K) + rx_idx_3d * mi.UInt32(K) + k_idx
        dr.scatter_add(adc_real_flat, contrib_real, flat_idx, active_3d)
        dr.scatter_add(adc_imag_flat, contrib_imag, flat_idx, active_3d)
        if chunk_K < K:
            dr.eval(adc_real_flat, adc_imag_flat)

    # Step 12b: Specular path synthesis (image method branch)
    if specular_paths is not None and specular_paths.n_paths > 0:
        sp_antenna_gain = self._synthesize_specular_paths(
            specular_paths=specular_paths,
            adc_real_flat=adc_real_flat,
            adc_imag_flat=adc_imag_flat,
            n_tx=n_tx, n_rx=n_rx, K=K,
            tx_boresights=tx_boresights,
            rx_boresights=rx_boresights,
            tx_pattern_rrts=tx_pattern_rrts,
            rx_pattern_rrts=rx_pattern_rrts,
            tx_pattern_loader=tx_pattern_loader,
            rx_pattern_loader=rx_pattern_loader,
            antenna_gain_linear=antenna_gain_linear,
            verbose=verbose,
        )

    # Reshape to [n_tx, n_rx, K]
    adc_real = np.array(adc_real_flat).reshape(n_tx, n_rx, K)
    adc_imag = np.array(adc_imag_flat).reshape(n_tx, n_rx, K)

    # Step 13: Component-level scatter-add (if requested)

    if verbose:
        print("  ADC synthesis complete")

    adc_result = ADCResult(
        adc_real=adc_real.astype(np.float32),
        adc_imag=adc_imag.astype(np.float32),
        n_tx=n_tx,
        n_rx=n_rx,
        n_samples=K,
        total_paths=total_paths
    )

    if return_cached_geometry:
        return adc_result, None
    return adc_result
