"""SBR Integrator package for FMCW ADC signal synthesis."""

# Standalone dataclasses and utilities (no class dependencies)
from .cached_geometry import ADCResult
from mmir.renderer.utils.math import diff_ray_triangle_intersect, get_ad_triangle_vertices

# Core class definition (includes __init__, _precompute_radar_constants, etc.)
from .core import SBRIntegratorRef

# Import method implementations
from .synthesis_forward import synthesize_shared_hits_vectorized
from .synthesis_e2e import (
    synthesize_end_to_end_multibounce, _synthesize_specular_e2e, _compute_diffraction_e2e,
)
from .synthesis_differentiable import _synthesize_specular_paths

# Attach forward synthesis methods
SBRIntegratorRef.synthesize_shared_hits_vectorized = synthesize_shared_hits_vectorized

# Attach end-to-end methods
SBRIntegratorRef.synthesize_end_to_end_multibounce = synthesize_end_to_end_multibounce
SBRIntegratorRef._synthesize_specular_e2e = _synthesize_specular_e2e
SBRIntegratorRef._compute_diffraction_e2e = _compute_diffraction_e2e

# Attach differentiable synthesis methods
SBRIntegratorRef._synthesize_specular_paths = _synthesize_specular_paths

__all__ = [
    'SBRIntegratorRef',
    'ADCResult',
    'diff_ray_triangle_intersect',
    'get_ad_triangle_vertices',
]
