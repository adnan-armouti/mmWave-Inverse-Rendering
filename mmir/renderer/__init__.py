"""Differentiable FMCW Radar Renderer."""

# Scene context and logging (inlined from mmir/renderer)
from .scene_context import SceneContext, RenderLogger

# Core renderer and integrator
from .config import RenderConfigRef, RenderResultRef
from .renderer import FMCWRendererRef
from .integrator import SBRIntegratorRef, ADCResult
from .sampler import ReservoirSampler, ReservoirHits, ReservoirHitsDrJit

# BSDF classes
from mmir.renderer.bsdf.base import BSDFBase
from mmir.renderer.bsdf.mmwave_jones import BSDFmmWaveJones
from .bsdf.reparameterization import (
    reparameterize_physics_params,
    inverse_reparameterize,
    reparameterize_physics_params_drjit,
    create_drjit_raw_params,
    LR_SCALES,
)

# Specular path finding
from mmir.renderer.specular.hashing import SpecularDeduplicator, PlaneHasher, hash_fnv1a_uint32
from mmir.renderer.specular.paths import SpecularPaths
from mmir.renderer.specular.sms import (
    SpecularManifoldSampler,
    SpecularGradInfo,
    MultibounceSpecularChain,
    MultibounceSpecularGradInfo,
)

# Materials
from mmir.renderer.materials.itu_materials import (
    get_itu_properties, get_material_properties, ITU_MATERIAL_PROPERTIES, DEFAULT_ROUGHNESS,
    DEFAULT_THICKNESS, WALL_PRESETS,
)
from mmir.renderer.materials.parameterization import (
    MaterialParameterization, PerVertexParameterization, vertex_to_triangle_materials,
)

# Utilities
from mmir.renderer.utils.math import (
    perp_stark, to_global, sample_cosine_hemisphere_concentric, safe_normalize, gather_point3f,
    gather_vector3f,
)

# Free-space diffraction BSDF (Steinberg et al. SIGGRAPH 2024)
try:
    from .diffraction import DiffractionConfig, TriangleSpatialHash, FsdBSDF
except ImportError:
    pass

# Backward-compatible import path for material_parameterization
from .materials import parameterization as material_parameterization

__all__ = [
    'FMCWRendererRef',
    'RenderConfigRef',
    'RenderResultRef',
    'SBRIntegratorRef',
    'ADCResult',
    'ReservoirSampler',
    'ReservoirHits',
    'ReservoirHitsDrJit',
    'BSDFBase',
    'BSDFmmWaveJones',
    'reparameterize_physics_params',
    'inverse_reparameterize',
    'reparameterize_physics_params_drjit',
    'create_drjit_raw_params',
    'LR_SCALES',
    'SpecularPaths',
    'SpecularManifoldSampler',
    'SpecularGradInfo',
    'MultibounceSpecularChain',
    'MultibounceSpecularGradInfo',
    'SpecularDeduplicator',
    'PlaneHasher',
    'hash_fnv1a_uint32',
    'MaterialParameterization',
    'PerVertexParameterization',
    'vertex_to_triangle_materials',
    'get_itu_properties',
    'get_material_properties',
    'ITU_MATERIAL_PROPERTIES',
    'DEFAULT_ROUGHNESS',
    'DEFAULT_THICKNESS',
    'WALL_PRESETS',
    'perp_stark',
    'to_global',
    'sample_cosine_hemisphere_concentric',
    'safe_normalize',
    'gather_point3f',
    'gather_vector3f',
    'DiffractionConfig',
    'TriangleSpatialHash',
    'FsdBSDF',
    'SceneContext',
    'RenderLogger',
    'material_parameterization',
]
