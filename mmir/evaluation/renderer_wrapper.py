"""Thin wrapper around FMCWRendererRef — single import point for renderer switching."""

import gc
import os
import time
from typing import Optional

import numpy as np

# SINGLE IMPORT POINT — change here to switch renderers
import mitsuba as mi
if mi.variant() is None:
    mi.set_variant("cuda_ad_rgb")
import drjit as dr

from mmir.renderer import (
    FMCWRendererRef,
    RenderConfigRef,
)
from mmir.evaluation.base_evaluator import load_learned_normals, load_learned_patterns


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_TX_PATTERN = os.path.join(_PROJECT_ROOT, "assets", "antenna_pattern", "MMWCAS", "tx1_76.npy")
_DEFAULT_RX_PATTERN = os.path.join(_PROJECT_ROOT, "assets", "antenna_pattern", "MMWCAS", "rx1_76.npy")


class RendererWrapper:
    """Thin wrapper for forward rendering with trained materials."""

    def __init__(
        self,
        mesh_file: str,
        config_file: str,
        render_config: Optional[RenderConfigRef] = None,
        tx_pattern_file: Optional[str] = None,
        rx_pattern_file: Optional[str] = None,
        verbose: bool = True,
    ):
        self.mesh_file = mesh_file
        self.config_file = config_file
        self.verbose = verbose
        self._render_config = render_config or self._default_config()

        # Use MMWCAS antenna patterns by default (matching training)
        if tx_pattern_file is None:
            tx_pattern_file = _DEFAULT_TX_PATTERN
        if rx_pattern_file is None:
            rx_pattern_file = _DEFAULT_RX_PATTERN

        self.renderer = FMCWRendererRef.from_files(
            mesh_file=mesh_file,
            config_file=config_file,
            tx_pattern_file=tx_pattern_file,
            rx_pattern_file=rx_pattern_file,
            material_type="metal",
            render_config=self._render_config,
            verbose=verbose,
        )

    @classmethod
    def from_training_dir(
        cls,
        mesh_file: str,
        config_file: str,
        training_dir: str,
        tx_pattern_file: Optional[str] = None,
        rx_pattern_file: Optional[str] = None,
        verbose: bool = True,
    ) -> "RendererWrapper":
        """Create wrapper with render config matching the training run."""
        import json
        from mmir.evaluation.base_evaluator import load_train_config

        train_cfg = load_train_config(training_dir)

        # Also load config.json which may have additional fields (e.g. pattern paths)
        config_json_path = os.path.join(training_dir, "config.json")
        with open(config_json_path) as f:
            full_cfg = json.load(f)
        # Merge: config.json provides defaults, train_config.json overrides
        merged = {**full_cfg, **train_cfg}

        # Antenna patterns: explicit overrides take precedence over training config
        if tx_pattern_file is None:
            tx_pattern_file = merged.get("tx_pattern_file", _DEFAULT_TX_PATTERN)
        if rx_pattern_file is None:
            rx_pattern_file = merged.get("rx_pattern_file", _DEFAULT_RX_PATTERN)

        # Build RenderConfigRef matching training
        render_config = _config_from_train_dict(merged)

        if verbose:
            print(f"  RendererWrapper: loaded config from {training_dir}")
            print(f"    TX pattern: {os.path.basename(tx_pattern_file)}")
            print(f"    RX pattern: {os.path.basename(rx_pattern_file)}")

        return cls(
            mesh_file=mesh_file,
            config_file=config_file,
            render_config=render_config,
            tx_pattern_file=tx_pattern_file,
            rx_pattern_file=rx_pattern_file,
            verbose=verbose,
        )

    def load_materials(self, training_dir: str):
        """Load best_materials.npz and set per-triangle physics params on renderer."""
        from mmir.evaluation.base_evaluator import load_our_physics_params
        from mmir.renderer.materials.parameterization import (
            vertex_to_triangle_materials,
        )

        physics_params, meta = load_our_physics_params(training_dir)

        # Convert per-vertex → per-triangle (matching training pipeline)
        import trimesh
        mesh = trimesh.load(self.mesh_file, force='mesh', process=False)
        faces = np.array(mesh.faces, dtype=np.int32)
        n_verts = len(mesh.vertices)
        n_mat = physics_params.shape[0]

        # Handle off-by-one: some scenes have materials 1 row short of vertex count.
        # Pad with a copy of the last row so face indices don't go out of bounds.
        if n_mat < n_verts:
            pad_rows = n_verts - n_mat
            padding = np.tile(physics_params[-1:], (pad_rows, 1))
            physics_params = np.concatenate([physics_params, padding], axis=0)
            if self.verbose:
                print(f"  Padded materials: {n_mat} → {n_verts} (+{pad_rows} rows)")

        tri_params = vertex_to_triangle_materials(physics_params, faces)
        self.renderer.triangle_materials = tri_params.astype(np.float32)

        if self.verbose:
            print(f"  Loaded materials from {training_dir} "
                  f"({n_mat} vertices → {tri_params.shape[0]} triangles, 6 params)")
        return meta

    def load_learned_normals(self, training_dir: str) -> bool:
        """Load best_normals.npz and apply to Mitsuba mesh vertex normals."""
        normals_np = load_learned_normals(training_dir)
        if normals_np is None:
            return False

        # Modify Mitsuba mesh vertex normals in-place
        params = mi.traverse(self.renderer.scene_ctx.scene)
        flat = normals_np.astype(np.float32).ravel()  # (n_vertices*3,) interleaved
        params['mesh.vertex_normals'] = mi.Float(flat)
        params.update()

        if self.verbose:
            print(f"  Applied learned normals ({normals_np.shape[0]} vertices) from {training_dir}")
        return True

    def load_learned_patterns(self, training_dir: str) -> bool:
        """Load best_patterns.npz and apply to antenna pattern loaders."""
        pattern_data = load_learned_patterns(training_dir)
        if pattern_data is None:
            return False

        applied = False
        for key, loader_attr in [('tx', 'tx_pattern_loader'), ('rx', 'rx_pattern_loader')]:
            loader = getattr(self.renderer, loader_attr, None)
            if loader is None:
                continue
            e_key = f'{key}_E_plane'
            h_key = f'{key}_H_plane'
            if e_key in pattern_data and h_key in pattern_data:
                loader.E_plane_linear = mi.Float(pattern_data[e_key])
                loader.H_plane_linear = mi.Float(pattern_data[h_key])
                applied = True
                if self.verbose:
                    print(f"  Applied learned {key.upper()} pattern ({len(pattern_data[e_key])} angles)")

        return applied

    def load_all_learned_params(self, training_dir: str) -> dict:
        """Load materials, normals, and patterns from training output."""
        meta = self.load_materials(training_dir)
        self.load_learned_normals(training_dir)
        self.load_learned_patterns(training_dir)
        return meta

    def render_forward(self, seed: int = 42) -> np.ndarray:
        """Forward render → returns ADC array (n_tx, n_rx, n_adc, 2) float32."""
        t0 = time.time()
        result = self.renderer.render(seed=seed)
        elapsed = time.time() - t0
        if self.verbose:
            shape = result.adc_result.get_ri_array().shape
            print(f"  Render complete in {elapsed:.1f}s → ADC shape {shape}")
        return result.adc_result.get_ri_array()

    def render_batched(
        self,
        batch_size: int = 5000,
        seed: int = 42,
        n_avg: int = 1,
    ) -> np.ndarray:
        """Batched forward render for large arrays (e.g. dense 100×100)."""
        total_hits = self._render_config.n_hits_per_rx
        n_batches = max(1, (total_hits + batch_size - 1) // batch_size)
        hits_per_batch = min(batch_size, total_hits)

        if self.verbose:
            print(f"  Batched render: {total_hits} hits/rx, {n_batches} batches of {hits_per_batch}, {n_avg} avg samples")

        # Temporarily set hits per rx to batch size
        original_hits = self._render_config.n_hits_per_rx

        adc_accum = None
        total_renders = n_batches * n_avg

        for avg_idx in range(n_avg):
            for batch_idx in range(n_batches):
                # Update BOTH config and sampler's internal hit count
                self._render_config.n_hits_per_rx = hits_per_batch
                self.renderer.config = self._render_config
                self.renderer.sampler.n_hits_per_rx = hits_per_batch

                t_batch = time.time()
                batch_seed = seed + avg_idx * n_batches + batch_idx
                result = self.renderer.render(seed=batch_seed)
                adc_batch = result.adc_result.get_ri_array()  # (n_tx, n_rx, n_adc, 2)

                if adc_accum is None:
                    adc_accum = adc_batch.astype(np.float64)
                else:
                    adc_accum += adc_batch.astype(np.float64)

                # Memory cleanup between batches
                dr.sync_thread()
                del result, adc_batch

                render_num = avg_idx * n_batches + batch_idx + 1
                elapsed = time.time() - t_batch
                if self.verbose:
                    print(f"    Batch {render_num}/{total_renders} done in {elapsed:.1f}s")

        # Restore original config
        self._render_config.n_hits_per_rx = original_hits
        self.renderer.config = self._render_config
        self.renderer.sampler.n_hits_per_rx = original_hits

        # Average across all renders
        adc_accum /= total_renders
        return adc_accum.astype(np.float32)

    def update_antenna_config(self, config_dict: dict):
        """Update TX/RX antenna positions and boresights from a config dict."""

        scene_ctx = self.renderer.scene_ctx

        # Parse TX positions and orientations from config dict
        tx_elements = config_dict.get("tx_array", [])
        if tx_elements:
            tx_pos = []
            tx_bor = []
            for elem in tx_elements:
                pos = np.array(elem["pos_mm"], dtype=np.float32) / 1000.0
                tx_pos.append(pos)
                tx_bor.append(np.array(elem.get("boresight", [0, 1, 0]), dtype=np.float32))
            tx_pos = np.array(tx_pos)  # (N_TX, 3)
            tx_bor = np.array(tx_bor)  # (N_TX, 3)
            scene_ctx.tx_array.positions = mi.Point3f(
                mi.Float(tx_pos[:, 0]), mi.Float(tx_pos[:, 1]), mi.Float(tx_pos[:, 2])
            )
            scene_ctx.tx_array.orientations = mi.Vector3f(
                mi.Float(tx_bor[:, 0]), mi.Float(tx_bor[:, 1]), mi.Float(tx_bor[:, 2])
            )

        # Parse RX positions and orientations from config dict
        rx_elements = config_dict.get("rx_array", [])
        if rx_elements:
            rx_pos = []
            rx_bor = []
            for elem in rx_elements:
                pos = np.array(elem["pos_mm"], dtype=np.float32) / 1000.0
                rx_pos.append(pos)
                rx_bor.append(np.array(elem.get("boresight", [0, 1, 0]), dtype=np.float32))
            rx_pos = np.array(rx_pos)  # (N_RX, 3)
            rx_bor = np.array(rx_bor)  # (N_RX, 3)
            scene_ctx.rx_array.positions = mi.Point3f(
                mi.Float(rx_pos[:, 0]), mi.Float(rx_pos[:, 1]), mi.Float(rx_pos[:, 2])
            )
            scene_ctx.rx_array.orientations = mi.Vector3f(
                mi.Float(rx_bor[:, 0]), mi.Float(rx_bor[:, 1]), mi.Float(rx_bor[:, 2])
            )

    def cleanup(self):
        """Release GPU memory."""
        del self.renderer
        self.renderer = None
        dr.sync_thread()
        gc.collect()
        if hasattr(dr, "flush_malloc_cache"):
            dr.flush_malloc_cache()


def _config_from_train_dict(cfg: dict) -> RenderConfigRef:
    """Build a RenderConfigRef from a training config.json dict."""
    # Diffraction: disable in postprocess forward rendering to avoid
    # per-vertex vs per-face material indexing mismatch in fsd_aperture_batch.
    diff_cfg = None

    return RenderConfigRef(
        antenna_gain_model="rrts",
        n_hits_per_rx=cfg.get("n_hits_per_rx", 1500),
        n_rays_per_res=cfg.get("n_rays_per_res", 16),
        use_shared_hits=True,
        bsdf_model=cfg.get("bsdf_model", "mmwave_jones"),
        mmwave_polarization=cfg.get("mmwave_polarization", "vertical"),
        hemisphere_sampling=cfg.get("hemisphere_sampling", "cosine"),
        double_sided=cfg.get("double_sided", True),
        enable_sms=False,
        sms_max_iterations=cfg.get("sms_max_iterations", 20),
        sms_solver_threshold=cfg.get("sms_solver_threshold", 1e-5),
        sms_use_smooth_normals=cfg.get("sms_use_smooth_normals", False),
        use_vertex_normals=cfg.get("use_vertex_normals", False),
        material_columns=cfg.get("material_columns", 6),
        diffraction_config=diff_cfg,
        use_patch_clustering=cfg.get("use_patch_clustering", False),
        patch_angle_threshold_deg=cfg.get("patch_angle_threshold_deg", 10.0),
        patch_distance_threshold=cfg.get("patch_distance_threshold", 0.05),
        patch_max_tris=cfg.get("patch_max_tris", 2000),
        use_spatial_adjacency=cfg.get("use_spatial_adjacency", False),
        spatial_radius=cfg.get("spatial_radius", 0.05),
    )
