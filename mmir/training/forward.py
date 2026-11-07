"""Renderer construction, error-map initialization, and the forward and backward passes."""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple

import torch

import drjit as dr

import mitsuba as mi

from mmir.renderer import (
    FMCWRendererRef, RenderConfigRef, reparameterize_physics_params, DiffractionConfig,
)

from mmir.data.ra_utils import adc_to_ra_image
from mmir.losses.drjit_ra_loss import compute_ra_loss_with_gradients
from mmir.losses.loss_utils import minmax_normalize_numpy

from mmir.training.config import LossConfig, TrainingConfig
from mmir.training.parameters import ParameterManager


def create_render_config(config: TrainingConfig) -> RenderConfigRef:
    """Create the renderer configuration from the training config."""
    return RenderConfigRef(
        n_hits_per_rx=config.n_hits_per_rx,
        n_rays_per_res=config.n_rays_per_res,
        max_distance=1e10,
        seed=42,
        verbose=False,
        use_radar_equation=True,
        use_pattern_importance_sampling=config.use_pattern_importance_sampling,
        bsdf_model=config.bsdf_model,
        mmwave_polarization=config.mmwave_polarization,
        hemisphere_sampling=config.hemisphere_sampling,
        double_sided=config.double_sided,
        enable_sms=config.enable_sms,
        sms_max_iterations=config.sms_max_iterations,
        sms_solver_threshold=config.sms_solver_threshold,
        sms_use_smooth_normals=config.sms_use_smooth_normals,
        use_patch_clustering=config.use_patch_clustering,
        patch_angle_threshold_deg=config.patch_angle_threshold_deg,
        patch_distance_threshold=config.patch_distance_threshold,
        patch_max_tris=config.patch_max_tris,
        use_spatial_adjacency=config.use_spatial_adjacency,
        spatial_radius=config.spatial_radius,
        material_columns=config.material_columns,
        use_vertex_normals=(config.param_mode == 'per_vertex'),
        diffraction_config=DiffractionConfig(use_gpu_apertures=True) if config.enable_diffraction else None,
        # End-to-end differentiable & multibounce
        use_end_to_end_ad=config.use_end_to_end_ad,
        max_bounces=config.max_bounces,
        nee_every_bounce=config.nee_every_bounce,
        rr_start_bounce=config.rr_start_bounce,
        rr_prob=config.rr_prob,
        e2e_ray_seed_rotation_interval=config.e2e_ray_seed_rotation_interval,
        e2e_phase_chunk_size=config.e2e_phase_chunk_size,
        enable_grad_phase=config.enable_grad_phase,
        enable_boundary_gradients=config.enable_boundary_gradients,
        sms_cache_interval=config.sms_cache_interval,
    )


def create_renderer(config: TrainingConfig) -> FMCWRendererRef:
    """Initialize the mmIR renderer from config."""
    render_config = create_render_config(config)
    renderer = FMCWRendererRef.from_files(
        mesh_file=config.scene_file,
        config_file=config.config_file,
        tx_pattern_file=config.tx_pattern_file,
        rx_pattern_file=config.rx_pattern_file,
        material_type="metal",
        render_config=render_config,
        verbose=False,
    )
    return renderer


class BinTriangleMapper:
    """Maps RA bins to contributing mesh triangles for per-triangle error-map initialization."""

    def __init__(self, mesh_vertices: np.ndarray, mesh_faces: np.ndarray,
                 radar_origin: np.ndarray, radar_boresight: np.ndarray,
                 num_az: int, num_r: int, range_res: float,
                 near_field_m: float = 1.5):
        self.mesh_vertices = mesh_vertices
        self.mesh_faces = mesh_faces
        self.radar_origin = radar_origin.astype(np.float32)
        self.radar_boresight = radar_boresight.astype(np.float32)
        self.radar_boresight /= np.linalg.norm(self.radar_boresight)
        self.near_field_m = near_field_m
        self.num_az = num_az
        self.num_r = num_r
        self.range_res = range_res

        self._create_bin_edges()
        self.bin_to_triangles: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}

    def _create_bin_edges(self):
        """Create bin edges matching the radar RA grid."""
        try:
            from mmir.evaluation.utils.single_view_proc import make_angle_grids_np, _centers_to_edges
            fft_size_az = self.num_az + 1
            az_centers, _ = make_angle_grids_np(fft_size_az, 2)
            self.az_edges = _centers_to_edges(
                az_centers, low_clip=-np.pi/2, high_clip=np.pi/2).astype(np.float32)
        except ImportError:
            # Fallback: uniform azimuth bins
            self.az_edges = np.linspace(-np.pi/2, np.pi/2, self.num_az + 1).astype(np.float32)
        self.r_edges = np.arange(self.num_r + 1, dtype=np.float32) * self.range_res

    def compute_mapping(self):
        """Compute triangle-to-RA-bin mapping using radar-frame coordinates."""
        print("[BinTriangleMapper] Computing triangle-to-RA-bin mapping...")
        t_start = time.time()

        # Triangle centroids and areas
        v0 = self.mesh_vertices[self.mesh_faces[:, 0]]
        v1 = self.mesh_vertices[self.mesh_faces[:, 1]]
        v2 = self.mesh_vertices[self.mesh_faces[:, 2]]
        centroids = (v0 + v1 + v2) / 3.0
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

        # Transform to radar frame
        pts_local = centroids - self.radar_origin
        R = self._compute_rotation_matrix()
        pts_radar = (R @ pts_local.T).T

        # Spherical coordinates
        x, y, z = pts_radar[:, 0], pts_radar[:, 1], pts_radar[:, 2]
        r = np.linalg.norm(pts_radar, axis=1)
        valid_mask = r >= self.near_field_m
        az = np.arctan2(-x, y)

        # Bin indices
        az_idx = np.searchsorted(self.az_edges, az, side='right') - 1
        r_idx = np.searchsorted(self.r_edges, r, side='right') - 1
        valid_bins = valid_mask & (az_idx >= 0) & (az_idx < self.num_az) & \
                     (r_idx >= 0) & (r_idx < self.num_r)

        # Build mapping
        self.bin_to_triangles = {}
        valid_indices = np.where(valid_bins)[0]
        for idx in valid_indices:
            key = (int(az_idx[idx]), int(r_idx[idx]))
            if key not in self.bin_to_triangles:
                self.bin_to_triangles[key] = []
            self.bin_to_triangles[key].append((int(idx), float(areas[idx])))

        # Normalize weights
        for key in self.bin_to_triangles:
            entries = self.bin_to_triangles[key]
            total = sum(w for _, w in entries)
            if total > 0:
                self.bin_to_triangles[key] = [(t, w / total) for t, w in entries]

        print(f"  Mapped {len(valid_indices)}/{len(self.mesh_faces)} triangles "
              f"to {len(self.bin_to_triangles)} RA bins ({time.time()-t_start:.2f}s)")
        return self.bin_to_triangles


def run_error_map_initialization(pm: ParameterManager,
                                  ra_gt: np.ndarray, range_res: float):
    """Per-triangle error-map initialization using BinTriangleMapper."""
    print("\n--- Error-map initialization ---")
    pm.set_renderer_materials()
    result = pm.renderer.render(seed=42)
    adc_np = pm.renderer.get_adc_numpy()

    # Convert to RA
    adc_torch = torch.from_numpy(adc_np).float()
    ra_rendered = adc_to_ra_image(adc_torch).detach().cpu().numpy()

    # Compute normalized error map (rendered - GT)
    error_map = minmax_normalize_numpy(ra_rendered) - minmax_normalize_numpy(ra_gt)

    # Try per-triangle mapping via BinTriangleMapper
    try:
        radar_origin = np.array(pm.renderer.scene_ctx.tx_array.positions[0], dtype=np.float32)
        radar_boresight = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        num_az, num_r = ra_gt.shape

        mapper = BinTriangleMapper(
            mesh_vertices=pm.mesh_vertices,
            mesh_faces=pm.mesh_faces,
            radar_origin=radar_origin,
            radar_boresight=radar_boresight,
            num_az=num_az, num_r=num_r,
            range_res=range_res,
        )
        mapper.compute_mapping()

        # Accumulate per-triangle error
        n_tri = len(pm.mesh_faces)
        tri_error = np.zeros(n_tri, dtype=np.float32)
        tri_count = np.zeros(n_tri, dtype=np.float32)

        for (az_bin, r_bin), tri_weights in mapper.bin_to_triangles.items():
            if az_bin >= num_az or r_bin >= num_r:
                continue
            err = error_map[az_bin, r_bin]
            for tri_id, weight in tri_weights:
                tri_error[tri_id] += err * weight
                tri_count[tri_id] += weight

        mask = tri_count > 0
        tri_error[mask] /= tri_count[mask]

        # Map to per-optimization-unit (per_vertex or per_triangle)
        if pm.config.param_mode == 'per_vertex':
            # Average triangle errors to vertices
            opt_error = np.zeros(pm.n_opt_params, dtype=np.float32)
            opt_count = np.zeros(pm.n_opt_params, dtype=np.float32)
            for ti in range(n_tri):
                for vi in pm.mesh_faces[ti]:
                    opt_error[vi] += tri_error[ti]
                    opt_count[vi] += 1
            msk = opt_count > 0
            opt_error[msk] /= opt_count[msk]
        else:
            opt_error = tri_error[:pm.n_opt_params]

        # Per-triangle adjustments
        over_mask = opt_error > 0.05
        under_mask = opt_error < -0.05

        n_over = int(np.sum(over_mask))
        n_under = int(np.sum(under_mask))
        n_ok = pm.n_opt_params - n_over - n_under

        if np.any(over_mask):
            pm.raw_params[over_mask, 2] += 0.5    # sigma_h: rougher
            pm.raw_params[over_mask, 0] -= 0.5    # eps_real: lower

        if np.any(under_mask):
            pm.raw_params[under_mask, 0] += 1.0   # eps_real: higher
            pm.raw_params[under_mask, 2] -= 1.0   # sigma_h: smoother
            pm.raw_params[under_mask, 4] += 0.5   # tau: more KA

        print(f"  Per-triangle error-init: {n_over} over, {n_under} under, {n_ok} OK")

    except Exception as e:
        print(f"  Warning: BinTriangleMapper failed ({e}), falling back to uniform adjustment")
        # Fallback: uniform adjustment
        over_frac = (error_map > 0.05).mean()
        under_frac = (error_map < -0.05).mean()
        if under_frac > over_frac:
            pm.raw_params[:, 0] += 0.5
            pm.raw_params[:, 2] -= 0.5
            pm.raw_params[:, 4] += 0.3

    phys = reparameterize_physics_params(pm.raw_params)
    print(f"  Post-adjustment: eps'={phys[:, 0].mean():.2f}, tau={phys[:, 4].mean():.3f}")
    print("--- Error-map initialization done ---\n")


def run_end_to_end_forward(pm: 'ParameterManager', iteration: int):
    """End-to-end differentiable forward pass (single AD graph)."""
    cfg = pm.config

    # Create fresh grad-enabled parameters
    raw_drjit, pose_dr, normal_dr, vertex_offset_dr, pattern_loaders_dr = \
        pm.create_grad_enabled_params()

    # Set renderer materials for non-diff auxiliary operations (SMS solver setup, etc.)
    pm.set_renderer_materials()

    # Rotate seed for MC variance reduction
    seed_interval = max(cfg.e2e_ray_seed_rotation_interval, 1)
    seed = 42 + (iteration // seed_interval) * 7

    # Enable forward timing debug for first few iterations
    pm.renderer._debug_forward_timing = True

    # End-to-end render (single-bounce or multibounce depending on config.max_bounces)
    adc_real, adc_imag = pm.renderer.render_end_to_end(
        raw_drjit,
        pose_params=pose_dr,
        normal_params=normal_dr,
        vertex_offset_params=vertex_offset_dr,
        pattern_loaders=pattern_loaders_dr,
        seed=seed,
    )

    return adc_real, adc_imag, raw_drjit, pose_dr, normal_dr, vertex_offset_dr, pattern_loaders_dr


def extract_adc_for_metrics(adc_real: mi.Float, adc_imag: mi.Float,
                             NT: int, NR: int, K: int) -> np.ndarray:
    """Convert flat DrJit ADC to complex numpy (NR, NT, K) for metrics."""
    with dr.suspend_grad():
        real_np = np.array(adc_real).reshape(NT, NR, K)
        imag_np = np.array(adc_imag).reshape(NT, NR, K)
    return (real_np + 1j * imag_np).transpose(1, 0, 2)  # (NR, NT, K)


def extract_ra_from_adc(adc_real: mi.Float, adc_imag: mi.Float,
                         NT: int, NR: int, K: int) -> np.ndarray:
    """Extract RA magnitude from DrJit ADC (suspend grad)."""
    with dr.suspend_grad():
        real_np = np.array(adc_real).reshape(NT, NR, K)
        imag_np = np.array(adc_imag).reshape(NT, NR, K)
    adc_ri = np.stack([real_np, imag_np], axis=-1)
    adc_torch = torch.from_numpy(adc_ri).float()
    ra_polar = adc_to_ra_image(adc_torch).detach().cpu().numpy()
    return ra_polar


def clip_gradients(gradients: Dict[str, np.ndarray],
                   config: TrainingConfig) -> Dict[str, np.ndarray]:
    """Per-group RMS gradient clipping."""
    clip_map = {
        'materials': config.clip_materials,
        'pose_rotation': config.clip_pose_rotation,
        'pose_translation': config.clip_pose_translation,
        'normals': config.clip_normals,
        'vertex_positions': config.clip_vertex_positions,
        'patterns': config.clip_patterns,
    }
    for name, grad in gradients.items():
        max_norm = clip_map.get(name, 1.0)
        if max_norm <= 0:
            continue
        if isinstance(grad, np.ndarray):
            np.sqrt(np.mean(grad ** 2))
    return gradients


def run_backward_and_extract_gradients(
    adc_real: mi.Float,
    adc_imag: mi.Float,
    gt_adc_torch: torch.Tensor,
    loss_config: LossConfig,
    raw_drjit, pose_dr, normal_dr, vertex_offset_dr, pattern_loaders_dr,
    config: TrainingConfig,
    NT: int, NR: int, K: int,
    gt_all_chirps: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Run backward pass and extract sanitized gradients."""
    loss_dict = {}

    # Compute RA loss via PyTorch bridge
    ra_grad_real = None
    ra_grad_imag = None
    multi_chirp_grad_real = None
    multi_chirp_grad_imag = None

    if loss_config.ra_mag_weight > 0:
        # Single-chirp RA loss via PyTorch bridge
        _, ra_grad_real_dr, ra_grad_imag_dr, ra_loss_dict = compute_ra_loss_with_gradients(
            adc_real, adc_imag, gt_adc_torch, NT, NR, K,
            normalize=True,
            weight=loss_config.ra_mag_weight,
            ssim_weight=loss_config.ra_mag_ssim_weight,
            lpips_weight=loss_config.ra_mag_perceptual_weight,
            ra_use_log=loss_config.ra_mag_use_log,
            ra_loss_type=loss_config.ra_mag_loss_type,
            log_epsilon=loss_config.log_epsilon,
            log_scale=loss_config.log_scale,
            adc_phase_weight=loss_config.adc_phase_weight,
            adc_phase_loss_type=loss_config.adc_phase_loss_type,
            phase_magnitude_weighting=loss_config.phase_magnitude_weighting,
            phase_magnitude_power=loss_config.phase_magnitude_power,
            phase_reference_channel=loss_config.phase_reference_channel,
            phase_use_gt_magnitude=loss_config.phase_use_gt_magnitude,
            phase_magnitude_clip_max=loss_config.phase_magnitude_clip_max,
            use_joint_normalization=loss_config.use_joint_normalization,
        )
        ra_grad_real = ra_grad_real_dr
        ra_grad_imag = ra_grad_imag_dr
        if ra_loss_dict:
            loss_dict.update(ra_loss_dict)
            loss_dict['total'] = ra_loss_dict.get('ra_l2', 0.0)

    # Compute ADC magnitude loss (DrJit, single-chirp only)
    adc_loss = None

    # Determine backward mode
    use_adc = adc_loss is not None
    effective_ra_grad_real = multi_chirp_grad_real if multi_chirp_grad_real is not None else ra_grad_real
    effective_ra_grad_imag = multi_chirp_grad_imag if multi_chirp_grad_imag is not None else ra_grad_imag
    use_ra = effective_ra_grad_real is not None

    # Backward pass
    if use_ra and not use_adc:
        # RA-ONLY MODE: Inject gradients and traverse
        dr.set_grad(adc_real, effective_ra_grad_real)
        dr.set_grad(adc_imag, effective_ra_grad_imag)
        dr.enqueue(dr.ADMode.Backward, adc_real)
        dr.enqueue(dr.ADMode.Backward, adc_imag)
        dr.traverse(dr.ADMode.Backward)

    elif use_ra and use_adc:
        # COMBINED ADC+RA MODE: ADC backward first, then accumulate RA grads
        dr.backward(adc_loss, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)

        # Accumulate RA gradients with ADC gradients on ADC outputs
        existing_grad_real = dr.grad(adc_real)
        existing_grad_imag = dr.grad(adc_imag)
        accumulated_real = (existing_grad_real + effective_ra_grad_real) if existing_grad_real is not None else effective_ra_grad_real
        accumulated_imag = (existing_grad_imag + effective_ra_grad_imag) if existing_grad_imag is not None else effective_ra_grad_imag

        dr.set_grad(adc_real, accumulated_real)
        dr.set_grad(adc_imag, accumulated_imag)
        dr.enqueue(dr.ADMode.Backward, adc_real)
        dr.enqueue(dr.ADMode.Backward, adc_imag)
        dr.traverse(dr.ADMode.Backward)

    elif use_adc:
        # ADC-ONLY MODE: Standard backward
        dr.backward(adc_loss)

    # Extract & sanitize gradients
    gradients = {}

    # Materials (6 physics params)
    if 'materials' not in config.frozen_params:
        mat_grads = np.zeros((config.n_opt_params_hint, 6), dtype=np.float32) \
            if hasattr(config, 'n_opt_params_hint') else None
        mat_grads_list = []
        for i in range(6):
            g = np.array(dr.grad(raw_drjit[i]))
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            mat_grads_list.append(g)
        gradients['materials'] = np.stack(mat_grads_list, axis=-1)

    # Normals (n_vertices x 3)
    if normal_dr is not None and 'normals' not in config.frozen_params:
        normal_grads_list = []
        for i in range(3):
            g = np.array(dr.grad(normal_dr[i]))
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            normal_grads_list.append(g)
        gradients['normals'] = np.stack(normal_grads_list, axis=-1)

    # Patterns
    if pattern_loaders_dr is not None and 'patterns' not in config.frozen_params:
        pat_grads = {}
        for key, loader in pattern_loaders_dr.items():
            g = loader.get_gradients()
            if g is not None:
                g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                pat_grads[key] = g
        if pat_grads:
            gradients['patterns'] = pat_grads

    return gradients, loss_dict
