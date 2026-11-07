"""Material parameterization strategies for differentiable rendering."""

from abc import ABC
from typing import Optional

import numpy as np
import drjit as dr
import mitsuba as mi


class MaterialParameterization(ABC):
    """Abstract base class for material parameterization strategies."""


class PerVertexParameterization(MaterialParameterization):
    """Per-vertex material parameterization with differentiable barycentric interpolation."""

    def __init__(self, n_vertices: int, faces: Optional[np.ndarray] = None):
        self._n_vertices = n_vertices
        # Mesh faces array (n_triangles, 3) int32 — needed for vertex->triangle
        # conversion in the diffraction pipeline (edges reference face indices).
        self._faces = faces.astype(np.int32) if faces is not None else None

    def gather(self, physics_params, cached_geom):
        active = cached_geom.active
        # Barycentric coordinates
        u = cached_geom.hit_bary_u   # weight for vertex 1
        v = cached_geom.hit_bary_v   # weight for vertex 2
        w = mi.Float(1.0) - u - v    # weight for vertex 0

        # Vertex indices for each hit's triangle
        vi0 = cached_geom.hit_vertex_ids_0  # UInt32 [n_total]
        vi1 = cached_geom.hit_vertex_ids_1  # UInt32 [n_total]
        vi2 = cached_geom.hit_vertex_ids_2  # UInt32 [n_total]

        result = []
        for i in range(len(physics_params)):
            p0 = dr.gather(mi.Float, physics_params[i], vi0, active)
            p1 = dr.gather(mi.Float, physics_params[i], vi1, active)
            p2 = dr.gather(mi.Float, physics_params[i], vi2, active)
            # Differentiable barycentric interpolation
            interp = w * p0 + u * p1 + v * p2
            result.append(interp)
        return result


def vertex_to_triangle_materials(vertex_params: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Convert per-vertex materials to per-triangle by centroid averaging."""
    v0 = vertex_params[faces[:, 0]]  # (n_triangles, n_cols)
    v1 = vertex_params[faces[:, 1]]
    v2 = vertex_params[faces[:, 2]]
    return (v0 + v1 + v2) / 3.0
