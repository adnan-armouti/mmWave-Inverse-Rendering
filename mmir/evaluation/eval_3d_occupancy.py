"""Evaluation #3: 3D RA Occupancy — Single Dense Frame."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .base_evaluator import BaseEvaluator, aggregate_metrics
from .scene_registry import SceneInfo


class OccupancyEvaluator(BaseEvaluator):
    """Evaluate 3D occupancy from dense virtual array rendering."""

    def __init__(
        self,
        scene: SceneInfo,
        output_root: str,
        batch_size: int = 5000,
        radar_percentile: float = 99.9,
        near_field_m: float = 1.5,
        coloradar_dataset_dir: str = None,
        coloradar_calib_dir: str = None,
        viz_radar_percentile: float = 99.9,
        verbose: bool = True,
    ):
        super().__init__(scene, output_root, verbose)
        self.batch_size = batch_size
        self.radar_percentile = radar_percentile
        self.near_field_m = near_field_m
        self.coloradar_dataset_dir = coloradar_dataset_dir
        self.coloradar_calib_dir = coloradar_calib_dir
        self.viz_radar_percentile = viz_radar_percentile

    @property
    def eval_name(self) -> str:
        return "3d_occupancy"

    def _ensure_dense_config(self) -> bool:
        """Write dense_frame_<F>.json to the output folder: a 100×100 uniform virtual array with
        half-wavelength spacing on the cascade radar's board plane (center + boresight).
        """
        source = self.scene.cascaded_config  # aligned cascade radar pose
        if not source or not os.path.isfile(source):
            self.log("No cascaded config available — cannot generate dense config")
            return False

        output_path = os.path.join(self.output_dir, f"dense_frame_{self.scene.frame}.json")
        self.log("Generating dense 100×100 config from cascaded config...")
        self.log(f"  Cascaded: {source}")
        self.log(f"  Output:   {output_path}")

        try:
            from mmir.evaluation.utils.dense_array import (
                generate_dense_config,
            )
            generate_dense_config(
                cascaded_config_path=source,
                output_path=output_path,
                num_antennas=100,
                verbose=self.verbose,
            )
        except Exception as e:
            self.log(f"  Failed to generate dense config: {e}")
            return False

        # Update scene info so downstream code can find it
        self.scene.dense_config = output_path
        return True

    def render(self) -> Optional[np.ndarray]:
        """Phase 1: Dense array forward render (100×100)."""
        if not self._ensure_dense_config():
            self.log("No dense config available for this scene")
            return None

        if not self.scene.our_training_dir:
            self.log("No training output directory available")
            return None

        # Skip if already rendered
        adc_path = os.path.join(self.output_dir, "adc_dense.npy")
        if os.path.isfile(adc_path):
            self.log(f"Dense ADC already exists at {adc_path}, skipping render")
            return np.load(adc_path)

        self.log("Rendering dense array (100×100)...")
        self.log(f"  Config: {self.scene.dense_config}")
        self.log(f"  Mesh: {self.scene.mesh_file}")

        from .renderer_wrapper import RendererWrapper, RenderConfigRef

        # Dense array config optimized for 100×100 virtual arrays:
        # - SMS/image method disabled: O(n_unique × n_tx × iters) is prohibitive
        # - use_shared_hits=True: required for correct MIMO azimuth estimation
        config = RenderConfigRef(
            antenna_gain_model="rrts",
            n_hits_per_rx=200,
            use_shared_hits=True,
            bsdf_model="mmwave_jones",
            mmwave_polarization="vertical",
            hemisphere_sampling="cosine",
            double_sided=True,
            enable_sms=False,
            use_vertex_normals=True,
            material_columns=6,
        )

        wrapper = RendererWrapper(
            mesh_file=self.scene.mesh_file,
            config_file=self.scene.dense_config,
            render_config=config,
            verbose=self.verbose,
        )
        wrapper.load_all_learned_params(self.scene.our_training_dir)

        # Batched render with K-chunking in synthesizer to handle DrJit 2^32 limit
        adc = wrapper.render_batched(batch_size=100, seed=42)
        self.log(f"  Rendered ADC shape: {adc.shape}")

        # Save
        adc_path = os.path.join(self.output_dir, "adc_dense.npy")
        np.save(adc_path, adc)
        self.log(f"  Saved → {adc_path}")

        wrapper.cleanup()
        return adc

    def run(self) -> dict:
        """Phase 2: 3D evaluation."""
        metrics = {"scene": self.scene.name}

        adc_path = os.path.join(self.output_dir, "adc_dense.npy")

        # Use single middle LiDAR frame (index=4) for metrics — fairer comparison
        # since the dense radar render is also a single-viewpoint capture.
        # Falls back to aggregated pcl.npy if single frames aren't available.
        lidar_path_for_metrics = self._get_single_lidar_frame(index=4)

        self.log(f"Running 3D occupancy evaluation (LiDAR: {os.path.basename(lidar_path_for_metrics)})...")

        # Compute and save RAE cube separately (run_single_view doesn't return it)
        from mmir.evaluation.utils.single_view_viz import (
            compute_rae_cube,
            collapse_to_ra,
            load_runtime_config,
            run_single_view,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        params = load_runtime_config(self.scene.dense_config, Path(self.output_dir))
        rae_cube = compute_rae_cube(Path(adc_path), params, device)
        np.save(os.path.join(self.output_dir, "rae_cube.npy"), rae_cube)
        self.log(f"  Saved RAE cube {rae_cube.shape} → rae_cube.npy")

        # Also save collapsed RA map for visualization
        ra_map, az_deg, r_m = collapse_to_ra(rae_cube, params)
        np.save(os.path.join(self.output_dir, "ra_map.npy"), ra_map)

        result = run_single_view(
            adc_file=adc_path,
            config_path=self.scene.dense_config,
            lidar_pcl_path=lidar_path_for_metrics,
            lidar_intensity_pcl_path=None,
            mesh_path=self.scene.mesh_file,
            output_directory=Path(self.output_dir),
            device=device,
            near_field_m=self.near_field_m,
            radar_threshold_percentile=self.radar_percentile,
            lidar_voxelize=True,
            # Always apply config-derived transforms
            radar_shift_to_config=True,
            radar_orient_to_config=True,
            show_radar=False,
            show_scene=False,
        )

        # Extract point clouds from run_single_view result.
        # Keys: 'radar_pcd' (Open3D PointCloud), 'lidar_pcd_voxelized' (Open3D PointCloud)
        radar_pts = self._extract_points(result, "radar_pcd")
        lidar_pts = self._extract_points(result, "lidar_pcd_voxelized")

        if radar_pts is not None and lidar_pts is not None:
            # Clip LiDAR to mesh bounding box — only evaluate within the
            # scene geometry that the radar actually renders against.
            if self.scene.mesh_file and os.path.isfile(self.scene.mesh_file):
                n_before = len(lidar_pts)
                lidar_pts = self._clip_to_mesh_bbox(lidar_pts, self.scene.mesh_file)
                self.log(f"  Clipped LiDAR to mesh bbox: {n_before} → {len(lidar_pts)} points")

            from .utils.metrics import compute_3d_occupancy_metrics

            occ = compute_3d_occupancy_metrics(
                radar_pts, lidar_pts, distance_threshold=0.5
            )
            metrics.update(occ)
            p = occ.get('distance_precision')
            r = occ.get('distance_recall')
            acc = occ.get('distance_accuracy')
            rmse = occ.get('pc_rmse')
            chamfer = occ.get('chamfer_distance')
            self.log(f"  Precision={p:.3f}, Recall={r:.3f}, Accuracy={acc:.3f}" if all(
                v is not None for v in (p, r, acc)) else f"  Metrics: {occ}")
            if rmse is not None:
                self.log(f"  RMSE={rmse:.3f}m, Chamfer={chamfer:.3f}m")

            # Save point clouds
            np.save(os.path.join(self.output_dir, "radar_pcl.npy"), radar_pts)
            np.save(os.path.join(self.output_dir, "lidar_pcl.npy"), lidar_pts)

            # Save radar intensity from Open3D pcd colors (for visualization filtering)
            radar_intensity = self._extract_intensity(result, "radar_pcd")
            if radar_intensity is not None:
                np.save(os.path.join(self.output_dir, "radar_intensity.npy"), radar_intensity)

        # ColoRadar CASCADE baseline metrics
        coloradar_metrics = self._compute_coloradar_metrics(lidar_path_for_metrics)
        if coloradar_metrics:
            metrics.update(coloradar_metrics)

        self.save_metrics(metrics)
        return metrics

    def _compute_coloradar_metrics(self, lidar_path: str) -> Optional[dict]:
        """Compute 3D occupancy metrics for ColoRadar CASCADE heatmap point cloud."""
        if not self.scene.dense_config:
            return None

        try:
            from .utils.coloradar_loader import load_coloradar_for_scene

            radar_xyz, radar_intensities, sensor_position = load_coloradar_for_scene(
                self.scene.name,
                dataset_dir=self.coloradar_dataset_dir,
                calib_dir=self.coloradar_calib_dir,
                intensity_threshold=0.2,
                min_range=10,
            )
        except Exception as e:
            self.log(f"  ColoRadar metrics: failed to load — {e}")
            return None

        if radar_xyz is None or len(radar_xyz) == 0:
            self.log("  ColoRadar metrics: no points loaded")
            return None

        # Apply board-frame alignment
        try:
            from mmir.evaluation.utils.single_view_viz import extract_board_frame_from_config
            import json as _json

            v2_az, boresight, v1_el = extract_board_frame_from_config(self.scene.dense_config)
            R_align = np.column_stack([v2_az, boresight, v1_el])

            with open(self.scene.dense_config, 'r') as f:
                cfg = _json.load(f)
            all_pos = []
            for tx in cfg.get('tx_array', []):
                all_pos.append(tx['pos_mm'])
            for rx in cfg.get('rx_array', []):
                all_pos.append(rx['pos_mm'])
            cfg_center = np.mean(np.array(all_pos), axis=0) / 1000.0

            radar_xyz = (radar_xyz @ R_align.T).astype(np.float32)
            translation_offset = cfg_center - sensor_position
            radar_xyz = radar_xyz + translation_offset.reshape(1, 3).astype(np.float32)
        except Exception as e:
            self.log(f"  ColoRadar metrics: alignment failed — {e}")
            return None

        # Load LiDAR for comparison
        if not lidar_path or not os.path.isfile(lidar_path):
            self.log("  ColoRadar metrics: no LiDAR available")
            return None

        lidar_data = np.load(lidar_path)
        lidar_pts = lidar_data[:, :3] if lidar_data.ndim == 2 and lidar_data.shape[1] >= 3 else lidar_data

        # Clip both to mesh bounding box
        if self.scene.mesh_file and os.path.isfile(self.scene.mesh_file):
            radar_xyz = self._clip_to_mesh_bbox(radar_xyz, self.scene.mesh_file)
            lidar_pts = self._clip_to_mesh_bbox(lidar_pts, self.scene.mesh_file)

        if len(radar_xyz) == 0 or len(lidar_pts) == 0:
            self.log("  ColoRadar metrics: no points after clipping")
            return None

        self.log(f"  ColoRadar CASCADE: {len(radar_xyz)} radar pts vs {len(lidar_pts)} LiDAR pts")

        from .utils.metrics import compute_3d_occupancy_metrics

        occ = compute_3d_occupancy_metrics(radar_xyz, lidar_pts, distance_threshold=0.5)

        # Prefix all keys with "coloradar_"
        result = {f"coloradar_{k}": v for k, v in occ.items()}

        p = occ.get('distance_precision')
        r = occ.get('distance_recall')
        acc = occ.get('distance_accuracy')
        rmse = occ.get('pc_rmse')
        chamfer = occ.get('chamfer_distance')
        self.log(f"  ColoRadar: Precision={p:.3f}, Recall={r:.3f}, Accuracy={acc:.3f}" if all(
            v is not None for v in (p, r, acc)) else f"  ColoRadar metrics: {occ}")
        if rmse is not None:
            self.log(f"  ColoRadar: RMSE={rmse:.3f}m, Chamfer={chamfer:.3f}m")

        # Save ColoRadar point cloud
        np.save(os.path.join(self.output_dir, "coloradar_pcl.npy"), radar_xyz)

        return result

    def _get_single_lidar_frame(self, index: int = 4) -> Optional[str]:
        """Get the nth LiDAR frame file (0-indexed) from data/{scene}/lidar/."""
        lidar_dir = os.path.join(self.scene.data_dir, "lidar")
        if not os.path.isdir(lidar_dir):
            return None
        import glob
        frames = sorted(glob.glob(os.path.join(lidar_dir, "lidar_frame_*.npy")))
        if index < len(frames):
            self.log(f"  Using single LiDAR frame: {os.path.basename(frames[index])}")
            return frames[index]

    def _extract_points(self, result: dict, key: str) -> Optional[np.ndarray]:
        """Extract point array from run_single_view result."""
        if result is None or key not in result:
            return None

        obj = result[key]
        # Could be an Open3D point cloud or numpy array
        if isinstance(obj, np.ndarray):
            return obj
        if hasattr(obj, "points"):
            return np.asarray(obj.points)

    def _extract_intensity(self, result: dict, key: str) -> Optional[np.ndarray]:
        """Extract intensity proxy from Open3D pcd colors (luminance from colormap)."""
        if result is None or key not in result:
            return None
        obj = result[key]
        if hasattr(obj, "colors"):
            colors = np.asarray(obj.colors)
            if len(colors) > 0:
                # Convert RGB to luminance as intensity proxy
                return 0.2989 * colors[:, 0] + 0.5870 * colors[:, 1] + 0.1140 * colors[:, 2]


def run_3d_occupancy_all(
    scenes: List[SceneInfo],
    output_root: str,
    render: bool = True,
    batch_size: int = 5000,
    coloradar_dataset_dir: str = None,
    coloradar_calib_dir: str = None,
    viz_radar_percentile: float = 99.9,
    skip_latex: bool = False,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Run 3D occupancy evaluation for all scenes."""
    per_scene = {}

    for scene in scenes:

        evaluator = OccupancyEvaluator(
            scene, output_root, batch_size=batch_size,
            coloradar_dataset_dir=coloradar_dataset_dir,
            coloradar_calib_dir=coloradar_calib_dir,
            viz_radar_percentile=viz_radar_percentile,
            verbose=verbose,
        )
        if render:
            evaluator.render()
        metrics = evaluator.run()
        per_scene[scene.name] = metrics

    # Aggregate (keys from compute_distance_based_occupancy_metrics + chamfer)
    # Ours (dense rendered) + ColoRadar CASCADE baseline
    ours_keys = ["distance_precision", "distance_recall", "distance_accuracy",
                 "pc_rmse", "chamfer_distance", "relative_chamfer_distance"]
    coloradar_keys = [f"coloradar_{k}" for k in ours_keys]
    metric_keys = ours_keys + coloradar_keys
    flat_metrics = {}
    for scene_name, m in per_scene.items():
        flat_metrics[scene_name] = {k: m.get(k) for k in metric_keys}

    agg = aggregate_metrics(flat_metrics, metric_keys)

    agg_dir = os.path.join(output_root, "3d_occupancy")
    os.makedirs(agg_dir, exist_ok=True)

    with open(os.path.join(agg_dir, "aggregate_metrics.json"), "w") as f:
        json.dump({"per_scene": flat_metrics, "aggregate": agg}, f, indent=2)

    if not skip_latex:
        from .utils.visualization import generate_3d_occupancy_table
        generate_3d_occupancy_table(
            flat_metrics, agg,
            os.path.join(agg_dir, "table3_3d_occupancy.tex"),
        )

    if verbose:
        print("\n=== 3D Occupancy Aggregate ===")
        for key, stats in agg.items():
            print(f"  {key}: {stats['mean']:.3f} ± {stats['std']:.3f}")

    return per_scene
