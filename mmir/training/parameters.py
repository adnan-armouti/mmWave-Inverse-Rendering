"""Optimized scene parameters, the Adam optimizer, and checkpoint saving and loading."""

import os
import json
import numpy as np
from typing import Dict, Optional, Any


import drjit as dr

import mitsuba as mi

from mmir.renderer import (
    FMCWRendererRef, reparameterize_physics_params, inverse_reparameterize, LR_SCALES,
    create_drjit_raw_params, get_material_properties, vertex_to_triangle_materials,
    PerVertexParameterization,
)

try:
    import trimesh
except ImportError:
    trimesh = None

from mmir.training.config import TrainingConfig


class ParameterManager:
    """Manages all differentiable parameters for training."""

    def __init__(self, renderer: FMCWRendererRef, config: TrainingConfig,
                 mesh_path: str):
        self.renderer = renderer
        self.config = config
        self.mesh_path = mesh_path

        # Load mesh geometry via trimesh (HFMO pattern)
        if trimesh is None:
            raise ImportError("trimesh is required for ParameterManager")
        mesh = trimesh.load(mesh_path)
        self.mesh_vertices = np.array(mesh.vertices, dtype=np.float32)
        self.mesh_faces = np.array(mesh.faces, dtype=np.int32)
        self.n_vertices = len(self.mesh_vertices)
        self.n_triangles = len(self.mesh_faces)

        # Determine optimization size and set material parameterization
        self.n_opt_params = self.n_vertices
        self.material_param = PerVertexParameterization(
            self.n_vertices, faces=self.mesh_faces)
        # Propagate to renderer (sets on both renderer and integrator)
        renderer.material_param = self.material_param

        # Material State (NumPy, 6-param physics)
        self.raw_params = np.zeros((self.n_opt_params, 6), dtype=np.float32)
        self._init_default_physics_params()

        # Pose State (NumPy, 6 scalars)
        self.pose_params_np = np.zeros(6, dtype=np.float32)

        # Normal State (NumPy)
        self.normal_params_np = None
        if config.LEARN_NRM:
            self.normal_params_np = np.array(mesh.vertex_normals, dtype=np.float32)

        # Geometry State (NumPy offsets)
        self.vertex_offset_np = None
        self.laplacian_neighbors = None

        # Antenna Pattern Loaders
        self.tx_pattern_loader = getattr(renderer, 'tx_pattern_loader', None)
        self.rx_pattern_loader = getattr(renderer, 'rx_pattern_loader', None)

        # Wavelength
        self.wavelength = 3e8 / 77e9  # ~3.9mm

        # Radar params
        self.NT = renderer.scene_ctx.tx_array.num_elements
        self.NR = renderer.scene_ctx.rx_array.num_elements
        self.K = renderer.integrator.num_samples

    def _init_default_physics_params(self):
        """Initialize with neutral physics defaults."""
        defaults = np.array([4.0, 0.1, 3e-5, 0.01, 0.5, 0.1], dtype=np.float32)
        physics_params = np.tile(defaults, (self.n_opt_params, 1))
        self.raw_params = inverse_reparameterize(physics_params)

    def initialize_from_itu(self, material_name: str):
        """Initialize physics params from ITU material database."""
        props = get_material_properties(material_name, 77e9)
        physics_params = reparameterize_physics_params(self.raw_params)

        physics_params[:, 0] = props['eps_real']
        physics_params[:, 1] = props['eps_imag']
        if 'sigma_h' in props:
            physics_params[:, 2] = props['sigma_h']
        if 'l_c' in props:
            physics_params[:, 3] = props['l_c']
        if 'thickness' in props:
            physics_params[:, 5] = props['thickness']

        self.raw_params = inverse_reparameterize(physics_params)

        phys = reparameterize_physics_params(self.raw_params)
        print(f"ITU init '{material_name}':")
        print(f"  eps_real={phys[:, 0].mean():.3f}, eps_imag={phys[:, 1].mean():.4f}")
        print(f"  sigma_h={phys[:, 2].mean():.2e}, l_c={phys[:, 3].mean():.4f}")
        print(f"  tau={phys[:, 4].mean():.3f}, thickness={phys[:, 5].mean():.4f}")

    def set_renderer_materials(self):
        """Push raw_params -> physics -> renderer triangle materials."""
        physics_params = reparameterize_physics_params(self.raw_params)
        tri_materials = vertex_to_triangle_materials(physics_params, self.mesh_faces)
        self.renderer.triangle_materials = tri_materials

    def create_grad_enabled_params(self):
        """Create fresh DrJit params with gradients enabled for AD."""
        raw_drjit = create_drjit_raw_params(self.raw_params)

        pose_dr = None

        normal_dr = None
        if self.config.LEARN_NRM and self.normal_params_np is not None \
                and 'normals' not in self.config.frozen_params:
            normal_dr = [mi.Float(self.normal_params_np[:, i].astype(np.float32))
                         for i in range(3)]
            for n in normal_dr:
                dr.enable_grad(n)

        vertex_offset_dr = None

        pattern_loaders_dr = None
        if self.config.LEARN_PAT and 'patterns' not in self.config.frozen_params:
            pattern_loaders_dr = {}
            for key, loader in [('tx', self.tx_pattern_loader), ('rx', self.rx_pattern_loader)]:
                if loader is not None:
                    loader.enable_gradients()
                    pattern_loaders_dr[key] = loader

        return raw_drjit, pose_dr, normal_dr, vertex_offset_dr, pattern_loaders_dr

    def post_step_constraints(self):
        """Apply post-step constraints."""
        cfg = self.config

        # Material raw_params: optional physical bounds (see TrainingConfig)
        for col, lower, upper in ((1, -7.0, cfg.eps_imag_raw_upper),
                                  (2, -16.0, cfg.sigma_h_raw_upper),
                                  (5, -7.0, cfg.thickness_raw_upper)):
            if upper is not None:
                self.raw_params[:, col] = np.clip(self.raw_params[:, col], lower, upper)

        # Pose rotation: clamp to ±max_rad
        self.pose_params_np[:3] = np.clip(
            self.pose_params_np[:3],
            -cfg.pose_rotation_max_rad, cfg.pose_rotation_max_rad
        )
        # Pose translation: clamp norm to max_m
        t_norm = np.linalg.norm(self.pose_params_np[3:])

        # Normal re-normalization
        if self.normal_params_np is not None:
            norms = np.linalg.norm(self.normal_params_np, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            self.normal_params_np /= norms

    def get_physics_param_means(self) -> Dict[str, float]:
        """Get mean values of all 6 physics params (for history tracking)."""
        phys = reparameterize_physics_params(self.raw_params)
        names = ['eps_real', 'eps_imag', 'sigma_h', 'l_c', 'tau', 'thickness']
        return {f'{n}_mean': float(phys[:, i].mean()) for i, n in enumerate(names)}


class SimpleAdam:
    """Adam optimizer for physics-parameterized training."""

    def __init__(self, param_manager: ParameterManager, config: TrainingConfig):
        self.pm = param_manager
        self.config = config
        self.beta1 = config.adam_beta1
        self.beta2 = config.adam_beta2
        self.eps = config.adam_eps
        self.t = 0

        # Per-group Adam states (m, v as NumPy arrays)
        self.states = {}
        self._init_states()

    def _init_states(self):
        """Initialize Adam momentum buffers for all parameter groups."""
        n = self.pm.n_opt_params
        # Materials: 6 separate columns
        self.states['materials'] = {
            'm': [np.zeros(n, dtype=np.float32) for _ in range(6)],
            'v': [np.zeros(n, dtype=np.float32) for _ in range(6)],
        }
        # Pose: 6 scalars
        self.states['pose_rotation'] = {
            'm': np.zeros(3, dtype=np.float32),
            'v': np.zeros(3, dtype=np.float32),
        }
        self.states['pose_translation'] = {
            'm': np.zeros(3, dtype=np.float32),
            'v': np.zeros(3, dtype=np.float32),
        }
        # Normals: n_vertices x 3
        nv = self.pm.n_vertices
        self.states['normals'] = {
            'm': [np.zeros(nv, dtype=np.float32) for _ in range(3)],
            'v': [np.zeros(nv, dtype=np.float32) for _ in range(3)],
        }
        # Vertex positions: n_vertices x 3
        self.states['vertex_positions'] = {
            'm': [np.zeros(nv, dtype=np.float32) for _ in range(3)],
            'v': [np.zeros(nv, dtype=np.float32) for _ in range(3)],
        }
        # Patterns: placeholder (sized when first gradient arrives)
        self.states['patterns'] = {}

    def step(self, gradients: Dict[str, Any], lr_scale: float = 1.0):
        """Perform one Adam step with extracted gradients."""
        self.t += 1
        cfg = self.config

        for group_name, grad in gradients.items():
            if group_name in cfg.frozen_params:
                continue

            # Get learning rate
            lr_map = {
                'materials': cfg.lr_materials,
                'pose_rotation': cfg.lr_pose_rotation,
                'pose_translation': cfg.lr_pose_translation,
                'normals': cfg.lr_normals,
                'vertex_positions': cfg.lr_vertex_positions,
                'patterns': cfg.lr_patterns,
            }
            base_lr = lr_map.get(group_name, 0.01)
            lr = base_lr * lr_scale

            state = self.states.get(group_name)
            if state is None:
                continue

            if group_name == 'materials':
                # grad shape: (n_opt, 6) - update per column with LR_SCALES
                for i in range(6):
                    g = grad[:, i] * LR_SCALES[i]
                    state['m'][i] = self.beta1 * state['m'][i] + (1 - self.beta1) * g
                    state['v'][i] = self.beta2 * state['v'][i] + (1 - self.beta2) * g ** 2
                    m_hat = state['m'][i] / (1 - self.beta1 ** self.t)
                    v_hat = state['v'][i] / (1 - self.beta2 ** self.t)
                    self.pm.raw_params[:, i] -= lr * m_hat / (np.sqrt(v_hat) + self.eps)

            elif group_name == 'normals' and self.pm.normal_params_np is not None:
                # Exponential-map update on unit sphere
                for i in range(3):
                    g = grad[:, i] if grad.ndim == 2 else grad[i]
                    state['m'][i] = self.beta1 * state['m'][i] + (1 - self.beta1) * g
                    state['v'][i] = self.beta2 * state['v'][i] + (1 - self.beta2) * g ** 2
                    m_hat = state['m'][i] / (1 - self.beta1 ** self.t)
                    v_hat = state['v'][i] / (1 - self.beta2 ** self.t)
                    step_i = lr * m_hat / (np.sqrt(v_hat) + self.eps)
                    self.pm.normal_params_np[:, i] -= step_i

            elif group_name == 'patterns' and isinstance(grad, dict):
                # Pattern gradients: dict of {key: np.ndarray}
                for pat_key, pat_grad in grad.items():
                    if pat_key not in state:
                        state[pat_key] = {
                            'm': np.zeros_like(pat_grad),
                            'v': np.zeros_like(pat_grad),
                        }
                    ps = state[pat_key]
                    ps['m'] = self.beta1 * ps['m'] + (1 - self.beta1) * pat_grad
                    ps['v'] = self.beta2 * ps['v'] + (1 - self.beta2) * pat_grad ** 2
                    m_hat = ps['m'] / (1 - self.beta1 ** self.t)
                    v_hat = ps['v'] / (1 - self.beta2 ** self.t)
                    step_p = lr * m_hat / (np.sqrt(v_hat) + self.eps)
                    loader = self.pm.tx_pattern_loader if pat_key == 'tx' else self.pm.rx_pattern_loader
                    if loader is not None:
                        loader.apply_update(-step_p)


def _extract_pattern_data(pm) -> Optional[Dict]:
    """Extract current pattern loader state as numpy arrays for saving."""
    data = {}
    for key, loader in [('tx', getattr(pm, 'tx_pattern_loader', None)),
                         ('rx', getattr(pm, 'rx_pattern_loader', None))]:
        if loader is not None and hasattr(loader, 'E_plane_linear'):
            data[f'{key}_E_plane'] = np.array(loader.E_plane_linear, dtype=np.float32)
            data[f'{key}_H_plane'] = np.array(loader.H_plane_linear, dtype=np.float32)
    return data if data else None


def save_final_results(pm: ParameterManager, best_raw_params: np.ndarray,
                        history: Dict, ra_gt: np.ndarray, output_dir: str,
                        best_normal_params=None, best_pose_params=None,
                        best_vertex_offset=None, best_pattern_data=None):
    """Save best materials and all learned parameters as separate files."""
    # Best materials (always saved)
    phys = reparameterize_physics_params(best_raw_params)
    np.savez(os.path.join(output_dir, 'best_materials.npz'),
             raw_params=best_raw_params,
             eps_real=phys[:, 0], eps_imag=phys[:, 1],
             sigma_h=phys[:, 2], l_c=phys[:, 3],
             tau=phys[:, 4], thickness=phys[:, 5])

    # Best normals (if normals were learned)
    if best_normal_params is not None:
        np.savez(os.path.join(output_dir, 'best_normals.npz'),
                 normal_params=best_normal_params)
        print(f"  Saved best_normals.npz ({best_normal_params.shape})")

    # Best patterns (if patterns were learned)
    if best_pattern_data is not None:
        np.savez(os.path.join(output_dir, 'best_patterns.npz'), **best_pattern_data)
        print(f"  Saved best_patterns.npz ({list(best_pattern_data.keys())})")

    # Best pose (if pose was learned)
    if best_pose_params is not None:
        np.savez(os.path.join(output_dir, 'best_pose.npz'),
                 pose_params=best_pose_params)
        print(f"  Saved best_pose.npz ({best_pose_params.shape})")

    # History
    serializable_history = {}
    for k, v in history.items():
        serializable_history[k] = [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(serializable_history, f, indent=2, default=str)
