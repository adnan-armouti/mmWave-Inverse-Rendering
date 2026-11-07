"""Render configuration for the FMCW radar renderer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .integrator import ADCResult
    from .sampler import ReservoirHits


@dataclass
class RenderConfigRef:
    """Configuration for RRTS-style rendering."""
    # Reservoir sampling parameters
    n_hits_per_rx: int = 500
    n_rays_per_res: int = 16
    max_distance: float = 1e10

    # General options
    seed: int = 42
    verbose: bool = True

    # Use RRTS-style shared hits (each hit processed for ALL TX-RX pairs)
    # This is required for proper azimuth estimation in MIMO radar
    use_shared_hits: bool = True

    # Antenna gain scale: False = logarithmic dB, True = linear scale (recommended)
    # When False: dB values from antenna pattern are used directly as multipliers
    # When True (default): dB values are converted to linear scale (10^(dB/10)) before multiplication
    # NOTE: Linear scale is correct for physics-based radar equation
    antenna_gain_linear: bool = True

    # Antenna pattern evaluation mode:
    # 'fixed' (default): Orientation-aware E×H separable pattern (correct physics)
    # 'legacy': Original bilinear interpolation (has front/back discrimination bug)
    pattern_mode: str = 'fixed'

    # Antenna gain evaluation in forward rendering:
    # 'loader' (training): the GPU pattern loader, which also carries learned patterns.
    # 'rrts': per-element pattern tables evaluated with pattern_mode, which the paper's
    # radar-transfer and 3D-occupancy evaluations use.
    antenna_gain_model: str = 'loader'

    # Enable/disable radar equation scaling
    # If False, use legacy BRDF-only weighting (no physical power scaling)
    # If True, apply full radar equation: Pr = (Pt * Gt * Gr * λ² * σ * L) / ((4π)³ * R⁴)
    use_radar_equation: bool = True

    # Transmit power in dBm (typical AWR2243: 12-13 dBm per TX)
    Pt_dBm: float = 13.0

    # Programmable receiver gain in dB (AWR2243 range: 0-48 dB, typical: 30 dB)
    rx_gain_dB: float = 30.0

    # Round-trip antenna/PCB loss in dB (TI recommends ~4 dB: 1.5-2 dB each way)
    antenna_loss_dB: float = 4.0

    # LO phase noise loss in dB (for SNR calculation, ~10 dB typical)
    phase_noise_loss_dB: float = 10.0

    # Noise figure in dB (AWR2243 typical: 13-14 dB)
    NF_dB: float = 13.0

    # dBm to dBFS conversion offset (TI RadarStudio convention)
    # Full-scale complex sinusoid corresponds to 13 dBm at ADC input
    dBm_to_dBFS_offset: float = 13.0

    # ADC quantization parameters
    # These convert from physical E-field (sqrt(Watts)) to ADC counts
    # ADC_scale = sqrt(Z_0) * ADC_full_scale, where Z_0 = system impedance
    adc_full_scale: float = 32768.0  # 16-bit ADC full scale (signed)
    system_impedance: float = 50.0   # System impedance in ohms (typical RF)

    # DEPRECATED: Vectorized synthesis is now always used (~135-210x faster)
    # The loop-based synthesize() and synthesize_shared_hits() were removed.
    # This field is kept for backward compatibility but is ignored.
    use_vectorized_synthesis: bool = True

    # Monte Carlo Normalization (RX-centric solid angle sampling)
    # The RX-centric MC estimator (see rx_centric_radar_mc_estimators.tex Eq. 4):
    # P_r = (P_t λ²)/(4π)² × (1/N_attempted) × Σ [G_t G_r c_t f V] / [d_t² × ρ_rx(ω_r)]
    #
    # Key insight: The MC normalization is now handled EXPLICITLY in the integrator:
    # - 1/ρ_rx(ω_r) = π/cos(θ) is applied per-hit using stored hit_pdf
    # - 1/N_attempted is applied using stored n_attempted_per_rx
    #
    # This mc_normalization factor is now PURELY for physics calibration (if needed).
    # Setting to 1.0 means all MC normalization is explicit and traceable.
    #
    # Previously this was 1/(8π²) which mixed MC normalization with physics,
    # making it impossible to verify the probability chain independently.
    mc_normalization: float = 1.0

    # Pattern Importance Sampling (Phase 1 variance reduction)
    # Enable antenna pattern-weighted importance sampling for variance reduction.
    # When True, sampling distribution is ρ_rx(ω) ∝ cos(θ) × G_r(ω) instead of
    # just cos(θ)/π. This concentrates samples in high-gain regions.
    use_pattern_importance_sampling: bool = False

    # Resolution for pattern importance sampling CDF grid

    # BSDF Model Selection
    # Select which BSDF model to use for surface scattering:
    # - 'mmwave_jones': Polarization-aware Jones BSDF with Fresnel phase (default)
    # - 'mmwave_scalar': Physics-first KA+SPM hybrid for 77 GHz (CSV-BSDF paper)
    # - 'rrts': Original RRTS-style GGX specular + Lambertian diffuse (legacy)
    bsdf_model: str = 'mmwave_jones'

    # Polarization for mmwave BSDF models ('vertical', 'horizontal', 'unpolarized')
    mmwave_polarization: str = 'vertical'

    # Enable ITU slab Fresnel model (Phase 2) for mmwave_scalar BSDF.
    # When True, uses ITU-R P.2040-4 single-layer slab with thickness.
    # When False, uses legacy single-interface Fresnel.
    enable_slab_fresnel: bool = True

    # Default slab thickness (m) for the ITU slab Fresnel model.
    # Used when per-vertex thickness is not available.
    default_thickness: float = 0.1

    # Enable coherent backscatter enhancement (CBS) for mmwave BSDF models.
    # Factor-of-2 enhancement at exact retroreflection (monostatic radar).
    enable_cbs: bool = True

    # Number of material columns per triangle.
    # 3 = legacy (albedo, roughness, metallic)
    # 6 = direct physics (eps_real, eps_imag, sigma_h, l_c, tau, thickness)
    material_columns: int = 3

    # Hemisphere Sampling Mode
    # 'cosine': Cosine-weighted hemisphere (default, biased toward boresight)
    # 'uniform': Uniform hemisphere (better coverage of off-axis surfaces)
    #
    # Use 'uniform' to improve visibility of surfaces perpendicular to boresight
    # (e.g., sidewalls in staircase scenes). This increases variance for on-axis
    # surfaces but provides better coverage of the full hemisphere.
    hemisphere_sampling: str = 'cosine'

    # Image Method Specular Path Refinement
    # Enable image method for deterministic specular path computation.
    # When True:
    # - MC branch uses eval_non_ka (everything except KA lobe)
    # - Image method branch computes exact specular points for unique triangles
    # - Specular weight: R = η × τ × A (deterministic, not MC-sampled)
    # When False: original pipeline with eval_f_cos (full BSDF)

    # Counter size for specular path deduplication hash table
    specular_dedup_counter_size: int = 10000

    # Number of independent hash functions for deduplication
    num_hash_functions: int = 2

    # Patch clustering: group coplanar adjacent triangles for image method
    # When True (and image method enabled), the image method tests specular points
    # against all triangles in a planar patch instead of just the hit triangle.
    # This dramatically increases valid specular paths for fine-grained meshes.
    use_patch_clustering: bool = True
    patch_angle_threshold_deg: float = 10.0  # Max normal deviation for coplanarity (LiDAR meshes need ~10deg)
    patch_distance_threshold: float = 0.05   # Max plane offset difference in meters
    patch_max_tris: int = 2000               # Cap per-patch triangle count (memory guard)
    use_spatial_adjacency: bool = False       # Use spatial proximity adjacency (bridges topological gaps)
    spatial_radius: float = 0.05             # Vertex proximity radius for spatial adjacency (meters)

    # Specular Manifold Sampling (SMS) - Zeltner et al. SIGGRAPH 2020
    # SMS replaces the image method for finding specular paths on fine meshes.
    # Uses Newton iteration on the half-vector constraint to walk from seed
    # positions to exact specular reflection points, crossing triangle boundaries.
    # When enabled, SMS is used instead of image method for Step 2b.
    enable_sms: bool = False

    # Maximum Newton iterations per seed point
    sms_max_iterations: int = 10

    # Convergence threshold for ||C|| (half-vector constraint residual)
    sms_solver_threshold: float = 1e-5

    # Use smooth (per-vertex) normals for half-vector constraint.
    # False = geometric face normals (simpler Jacobian, more robust).
    # True = interpolated vertex normals (smoother manifold, needs dn/du).
    sms_use_smooth_normals: bool = False

    # Normal Mode: per-vertex (smooth) vs per-triangle (flat)
    # When True, uses si.sh_frame.n (Mitsuba's interpolated vertex normal) which
    # provides smooth shading across triangle boundaries. This is the physically
    # correct choice for curved surfaces reconstructed from LiDAR point clouds.
    #
    # When False, uses si.n (geometric face normal) which gives flat shading
    # with discontinuous normals at triangle edges.
    #
    # Default: True (per-vertex, smooth normals)
    use_vertex_normals: bool = True

    # Double-Sided Rendering (for single-sided mesh geometry)
    # Enable double-sided rendering to handle surfaces with normals pointing away.
    # When True, backfacing surfaces (cos_theta < 0) have their normals flipped
    # so they contribute to the rendered image.
    #
    # This is essential for:
    # - LiDAR-derived meshes where normal orientation may be inconsistent
    # - Thin walls/guardrails that should scatter from both sides
    # - Improving sidewall visibility in staircase/corridor scenes
    #
    # Default: True (recommended for most radar scenes)
    double_sided: bool = True

    # Free-Space Diffraction BSDF (Steinberg et al. SIGGRAPH 2024)
    # When set to a DiffractionConfig instance, enables edge diffraction.
    # Diffraction is evaluated additively with energy borrowing:
    #   f_total = (1-β)×f_refl + f_diff
    # Set to None to disable (default).
    diffraction_config: object = None  # Optional[DiffractionConfig]

    # End-to-End Differentiable Rendering
    # Enable single-pass end-to-end AD (vs two-phase cached mode).
    # When True, ray tracing occurs every iteration and hit positions carry
    # AD gradients through vertex positions and pose.
    use_end_to_end_ad: bool = True

    # Enable projective boundary gradient sampling (Zhang et al. 2023).
    # Corrects missing visibility discontinuity term for geometry/pose grads.
    enable_boundary_gradients: bool = True

    # Number of silhouette boundary samples for primary visibility term.

    # K-chunk size for phase computation (bounds peak GPU memory).
    # Smaller = less memory but more kernel launches.
    e2e_phase_chunk_size: int = 64

    # SMS specular path cache refresh interval for dynamic geometry.
    # When geometry changes (LEARN_VTX or LEARN_POSE), SMS paths are cached
    # and reused for this many iterations before being recomputed.
    # IFT handles small geometry perturbations between refreshes.
    # 0 = recompute every iteration (no caching with dynamic geometry).
    sms_cache_interval: int = 0

    # Rotate random seed every N iterations (for MC variance exploration).
    e2e_ray_seed_rotation_interval: int = 10

    # Block gradient flow through phase (cos/sin of propagation delay).
    # When False (default), detaches phase trig before amplitude multiplication,
    # preserving amplitude gradients while blocking noisy phase->delay->pose/vtx
    # gradient path (at 77 GHz, 1mm = ~1.6 rad phase change).
    enable_grad_phase: bool = False

    # Multibounce Configuration
    # Maximum number of bounces for multibounce end-to-end rendering.
    # 1 = single bounce (original behavior), 2+ = multibounce with diff
    # re-intersection for full-chain gradient flow.
    max_bounces: int = 1

    # NEE (Next-Event Estimation) at every bounce.
    # When True, connects to TX at every bounce (gives 1-bounce, 2-bounce, ...
    # contributions in one trace). When False, only the last bounce connects.
    nee_every_bounce: bool = True

    # Russian roulette start bounce (disable early termination before this).
    # Paths shorter than this are never terminated by RR.
    rr_start_bounce: int = 3

    # Russian roulette continuation probability (0-1).
    # At bounce >= rr_start_bounce, terminate with probability (1 - rr_prob).
    # Survivors have weight multiplied by 1/rr_prob to maintain unbiasedness.
    rr_prob: float = 0.5


@dataclass
class RenderResultRef:
    """Result of RRTS-style rendering."""
    adc_result: ADCResult
    reservoir_hits: ReservoirHits
    config: RenderConfigRef
