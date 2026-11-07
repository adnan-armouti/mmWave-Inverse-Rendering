"""Abstract base class for BSDF implementations."""

from abc import ABC, abstractmethod
from typing import Union
import numpy as np


class BSDFBase(ABC):
    """Abstract base class for BSDF implementations."""

    # Metal BRDF calibration for 77 GHz radar
    F0_DIELECTRIC = 0.04  # Fresnel at normal incidence for dielectrics
    F0_METAL = 0.95       # Fresnel at normal incidence for metals at 77 GHz

    @abstractmethod
    def eval_f_cos(
        self,
        wo: Union[np.ndarray, 'mi.Vector3f'],
        wi: Union[np.ndarray, 'mi.Vector3f'],
        n: Union[np.ndarray, 'mi.Vector3f'],
        albedo: Union[np.ndarray, 'mi.Float'],
        roughness: Union[np.ndarray, 'mi.Float'],
        metallic: Union[np.ndarray, 'mi.Float'],
    ) -> Union[np.ndarray, 'mi.Float']:
        """Evaluate BSDF × |cos(theta_i)|."""


__all__ = ['BSDFBase']
