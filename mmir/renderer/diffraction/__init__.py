"""Free-Space Diffraction BSDF for mmWave Radar."""

from .fsd_config import DiffractionConfig
from .triangle_search import TriangleSpatialHash
from .fsd_aperture import FsdEdge
from .fsd_bsdf import (FsdBSDF, FlatEdgeData, eval_batch_drjit, _jones_reflectance_drjit)
# CPU aperture construction is deprecated; GPU path is always used.
from .fsd_aperture_gpu import construct_apertures_batch_gpu, GPUSpatialHashData, build_gpu_spatial_hash

__all__ = [
    'DiffractionConfig',
    'TriangleSpatialHash',
    'FsdEdge',
    'FsdBSDF',
    'FlatEdgeData',
    'eval_batch_drjit',
    '_jones_reflectance_drjit',
    'construct_apertures_batch_gpu',
    'GPUSpatialHashData',
    'build_gpu_spatial_hash',
]
