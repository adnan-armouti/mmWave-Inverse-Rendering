"""Configuration for free-space diffraction BSDF."""

from dataclasses import dataclass


@dataclass
class DiffractionConfig:
    """Configuration parameters for the fsdBSDF diffraction model."""

    enabled: bool = True

    # Beam profile: Gaussian with spatial standard deviation = sigma_wavelengths * lambda
    # The search radius is 3 * sigma (capturing >99.7% of the Gaussian beam).
    # Paper default: sigma = 25 lambda. At 77 GHz (lambda ~ 3.9 mm): sigma ~ 97 mm.
    beam_sigma_wavelengths: float = 25.0

    # Maximum edges to retain per hit point (after tessellation).
    # The reference code allocates 100 * n_triangles; 256 is a safe cap.
    max_edges_per_hit: int = 256

    # Minimum dihedral angle (degrees) for pre-filtering candidate triangles.
    # This is only used for the optional sharp-edge pre-filter BVH, NOT for
    # the actual boundary edge detection (which is view-dependent).
    # Raised from 15 to 20 to reduce false positives on noisy LiDAR meshes.
    edge_angle_threshold_deg: float = 20.0

    # Filter E: Minimum edge length in wavelengths.
    # Edges shorter than this are suppressed — too short for meaningful diffraction.
    # At 77 GHz (λ ≈ 3.9mm), default 2λ ≈ 7.8mm.  Set to 0 to disable.
    min_edge_length_wavelengths: float = 2.0

    # Filter R: Region-based edge classification (replaces old Filter D).
    # Segment mesh into approximately planar regions via flood-fill, then
    # suppress intra-region edges (noise on flat surfaces).  Inter-region
    # edges (boundaries between distinct surfaces) are preserved.
    # Set region_angle_threshold_deg to 0 to disable.
    region_angle_threshold_deg: float = 20.0

    # Intra-region safety cap: edges with dihedral ABOVE this are kept even
    # if both faces belong to the same region (real features the region
    # growing couldn't separate).  0 = no cap (suppress all intra-region).
    region_dihedral_cap_deg: float = 45.0

    # Minimum region size (faces).  Regions smaller than this are merged
    # into their largest neighbor.  Inter-region edges where BOTH regions
    # are below this size are also suppressed (noise cluster boundaries).
    min_region_faces: int = 10

    # Dihedral-dependent inter-region filter: for inter-region edges with
    # dihedral below region_dihedral_cap_deg, suppress if the SMALLER
    # adjacent region has fewer than this many faces.  Catches noise patches
    # (10-50 faces) adjacent to large structural surfaces.  0 = disabled.
    inter_region_min_faces: int = 50

    # Filter C: Mesh boundary erosion depth (face-hops from boundary).
    # Faces within N hops of a mesh boundary edge are suppressed for
    # diffraction — outer rim of Poisson reconstruction is pathological.
    # Reduced from 3 to 2: hop-3 edges are statistically indistinguishable
    # from hop-4, and the extra layer catches ~19 real features (>80° dihedral).
    # Set to 0 to disable.
    boundary_erosion_hops: int = 2

    # Filter F: Minimum connected chain length in wavelengths.
    # Diffracting edges not part of a connected chain with total geometric
    # length >= this are suppressed.  Real features (building ridges, curbs)
    # form long chains; isolated noise is short.
    # At 77 GHz (λ ≈ 3.9mm), default 50λ ≈ 195mm.  Set to 0 to disable.
    min_chain_length_wavelengths: float = 50.0

    # Optional collinearity threshold for chain connectivity (degrees).
    # Two edges sharing a vertex are only "connected" if the angle between
    # their directions is < this.  0 = disabled (any shared vertex connects).
    chain_collinearity_threshold_deg: float = 0.0

    # Energy borrowing: scale reflection by (1-beta) and add diffraction.
    # When False, diffraction is purely additive (not energy-conserving).
    energy_borrowing: bool = True

    # Clamp beta to [0, beta_max] to prevent extreme redistribution.
    beta_max: float = 0.5

    # Return complex amplitude (True) or scalar intensity (False).
    # Must be True for MIMO coherence.
    complex_output: bool = True

    # Triangle spatial hash cell size in meters.
    # Should be ~ search_radius for optimal bucket occupancy.
    # If 0, auto-computed as 3 * beam_sigma.
    hash_cell_size: float = 0.0

    # Maximum tessellation depth (recursive subdivision of long projected edges).
    # Reference code uses 5.
    max_tessellation_depth: int = 5

    # Early-exit thresholds: skip diffraction if projected triangle fill fraction
    # is above fill_max or below fill_min.
    fill_min: float = 1e-6
    fill_max: float = 1.0 - 1e-6

    # Material-dependent diffraction: modulate edge amplitudes by Fresnel opacity.
    # When True, diffracting edges of partially transparent materials (glass, drywall)
    # produce weaker diffraction than opaque materials (metal, concrete).
    # Opacity = polarization-averaged Fresnel power reflectance (Luebbers 1984).
    # Requires per-triangle material parameters (eps_real, eps_imag) to be available.
    material_opacity: bool = True

    # Jones polarization: use full complex Fresnel with TX/RX polarization
    # instead of polarization-averaged scalar opacity.
    # When True, edge amplitudes carry Fresnel phase and polarization-dependent
    # magnitude. Requires BSDFmmWaveJones (or explicit TX/RX polarization vectors).
    # When False, falls back to scalar opacity (|R_s|² + |R_p|²)/2.
    jones_polarization: bool = True

    # GPU-accelerated aperture topology construction (always enabled).
    # The CPU NumPy path (construct_apertures_batch) is deprecated and
    # commented out.  This flag is kept for API compatibility but ignored.
    use_gpu_apertures: bool = True
