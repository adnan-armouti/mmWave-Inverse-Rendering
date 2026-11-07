"""Per-vertex positions and normals of the scene mesh."""

import drjit as dr
import mitsuba as mi
import numpy as np


class VertexAttribute:
    """A per-vertex 3-vector attribute of a PLY mesh, stored as a Mitsuba array."""

    MESH_ATTRIBUTE = None  # trimesh attribute name
    ARRAY_TYPE = None      # mi.Point3f or mi.Vector3f
    LABEL = None           # for log messages

    def __init__(self, num_vertices: int, values=None):
        self.num_vertices = num_vertices
        self.values = values

    @classmethod
    def from_ply(cls, ply_file: str, enable_grad: bool = False) -> 'VertexAttribute':
        import trimesh
        print(f"[{cls.__name__}] Loading from {ply_file}")
        mesh = trimesh.load(ply_file, force='mesh', process=False)
        a = np.array(getattr(mesh, cls.MESH_ATTRIBUTE), dtype=np.float32)
        print(f"  Loaded {a.shape[0]} {cls.LABEL}")
        return cls(a.shape[0], cls.ARRAY_TYPE(mi.Float(a[:, 0]), mi.Float(a[:, 1]), mi.Float(a[:, 2])))


class VertexPositions(VertexAttribute):
    MESH_ATTRIBUTE, ARRAY_TYPE, LABEL = 'vertices', mi.Point3f, 'vertices'

    @property
    def positions(self):
        return self.values

    @positions.setter
    def positions(self, value):
        self.values = value


class VertexNormals(VertexAttribute):
    MESH_ATTRIBUTE, ARRAY_TYPE, LABEL = 'vertex_normals', mi.Vector3f, 'vertex normals'

    @property
    def normals(self):
        return self.values

    @normals.setter
    def normals(self, value):
        self.values = value

    @classmethod
    def from_ply(cls, ply_file: str, enable_grad: bool = False) -> 'VertexNormals':
        instance = super().from_ply(ply_file, enable_grad)
        instance.normals = instance.normals / dr.maximum(dr.norm(instance.normals), 1e-6)
        return instance
