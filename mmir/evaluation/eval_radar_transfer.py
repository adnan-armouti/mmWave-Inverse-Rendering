"""Evaluation #2: Radar Transfer (cascaded → single-chip)."""

import os
from typing import Dict, List, Optional

import numpy as np

from .base_evaluator import BaseEvaluator, aggregate_metrics
from .scene_registry import SceneInfo

# Single-chip antenna pattern (TI IWR1443) — NOT the cascaded MMWCAS patterns
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SC_ANTENNA_PATTERN = os.path.join(_PROJECT_ROOT, "assets", "antenna_pattern", "IWR1443", "pattern_76.npy")


class RadarTransferEvaluator(BaseEvaluator):
    """Evaluate radar transfer from cascaded to single-chip."""

    def __init__(
        self,
        scene: SceneInfo,
        output_root: str,
        single_chip_frame: Optional[int] = None,
        n_hits_per_rx: Optional[int] = None,
        verbose: bool = True,
    ):
        super().__init__(scene, output_root, verbose)
        # Auto-detect closest single-chip frame if not specified
        if single_chip_frame is None:
            single_chip_frame = scene.get_closest_single_chip_frame()
        self.single_chip_frame = single_chip_frame
        self._n_hits_override = n_hits_per_rx

    @property
    def eval_name(self) -> str:
        return "radar_transfer"

    def _ensure_aligned_sc_config(self) -> Optional[str]:
        """Ensure an aligned single-chip config exists for this scene."""
        if hasattr(self, "_cached_aligned_sc_config"):
            return self._cached_aligned_sc_config

        if self.single_chip_frame is None:
            self._cached_aligned_sc_config = None
            return None

        sc_frame = self.single_chip_frame

        # Primary: aligned config in output dir (training-specific)
        aligned_config = os.path.join(
            self.output_dir, f"single_chip_frame_{sc_frame}_aligned.json"
        )

        if os.path.isfile(aligned_config):
            self.log(f"  Aligned SC config exists (output dir): {aligned_config}")
            self._cached_aligned_sc_config = aligned_config
            return aligned_config

        # Fallback: check data dir for pre-existing aligned config
        configs_dir = os.path.join(self.scene.data_dir, "configs")
        data_dir_aligned = os.path.join(
            configs_dir, f"single_chip_frame_{sc_frame}_aligned.json"
        )

        # Check if we have learned normals/patterns that change rendering.
        # If so, we must re-align even if data_dir has a config, because
        # the learned normals affect the rendered RA used for alignment.
        has_learned_normals = os.path.isfile(
            os.path.join(self.scene.our_training_dir, "best_normals.npz")
        ) if self.scene.our_training_dir else False
        has_learned_patterns = os.path.isfile(
            os.path.join(self.scene.our_training_dir, "best_patterns.npz")
        ) if self.scene.our_training_dir else False

        if os.path.isfile(data_dir_aligned) and not has_learned_normals and not has_learned_patterns:
            self.log(f"  Aligned SC config exists (data dir): {data_dir_aligned}")
            self._cached_aligned_sc_config = data_dir_aligned
            return data_dir_aligned

        # Run alignment pipeline, saving to output dir
        self.log(f"  Running alignment pipeline (learned normals={has_learned_normals}, "
                 f"learned patterns={has_learned_patterns})...")
        from .sc_alignment import run_sc_alignment

        result = run_sc_alignment(
            scene=self.scene,
            output_config_path=aligned_config,
            verbose=self.verbose,
        )

        self._cached_aligned_sc_config = aligned_config
        return aligned_config

    def render(self) -> Optional[np.ndarray]:
        """Phase 1: Forward render with single-chip config + trained materials."""
        if self.single_chip_frame is None:
            self.log("No single-chip frame available, skipping render")
            return None

        # Ensure aligned SC config (runs Stage 1 + Stage 2 if needed)
        sc_config = self._ensure_aligned_sc_config()
        if sc_config is None:
            self.log(f"No single-chip config found for frame {self.single_chip_frame}")
            return None

        # Skip main render if already done
        adc_path = os.path.join(self.output_dir, "adc_rendered_single_chip.npy")
        adc = None

        if adc is None and not self.scene.our_training_dir:
            self.log("No training output directory available")
            return None

        if adc is None:
            self.log(f"Rendering single-chip frame {self.single_chip_frame}...")
            self.log(f"  Config: {sc_config}")
            self.log(f"  Mesh: {self.scene.mesh_file}")

            from .renderer_wrapper import RendererWrapper

            # Render with trained materials
            # Use from_training_dir() to reproduce exact renderer settings
            # (BSDF model, etc.) — matching Eval #1 behavior.
            # Override antenna patterns with single-chip pattern (not cascaded MMWCAS).
            wrapper = RendererWrapper.from_training_dir(
                mesh_file=self.scene.mesh_file,
                config_file=sc_config,
                training_dir=self.scene.our_training_dir,
                tx_pattern_file=_SC_ANTENNA_PATTERN,
                rx_pattern_file=_SC_ANTENNA_PATTERN,
                verbose=self.verbose,
            )
            meta = wrapper.load_all_learned_params(self.scene.our_training_dir)
            self.log(f"  All learned params loaded (training corr={meta.get('correlation', 'N/A')})")

            adc = wrapper.render_forward(seed=42)
            self.log(f"  Rendered ADC shape: {adc.shape}")

            np.save(adc_path, adc)
            self.log(f"  Saved trained → {adc_path}")
            wrapper.cleanup()

        # Render with default (untrained) materials
        default_path = os.path.join(self.output_dir, "adc_default_single_chip.npy")
        if not os.path.isfile(default_path) and self.scene.our_training_dir:
            from .renderer_wrapper import RendererWrapper as _RW
            self.log("  Rendering default (untrained) baseline...")
            wrapper_default = _RW.from_training_dir(
                mesh_file=self.scene.mesh_file,
                config_file=sc_config,
                training_dir=self.scene.our_training_dir,
                tx_pattern_file=_SC_ANTENNA_PATTERN,
                rx_pattern_file=_SC_ANTENNA_PATTERN,
                verbose=False,
            )
            # No load_materials() call — uses renderer's default ITU materials
            adc_default = wrapper_default.render_forward(seed=42)
            np.save(default_path, adc_default)
            self.log(f"  Saved default → {default_path}")
            wrapper_default.cleanup()

        return adc

    def run(self) -> dict:
        """Phase 2: Evaluate transfer quality."""
        metrics = {
            "scene": self.scene.name,
            "single_chip_frame": self.single_chip_frame,
        }

        if self.single_chip_frame is None:
            self.log("No single-chip frame available")
            self.save_metrics(metrics)
            return metrics

        # Check if rendered ADC exists
        adc_path = os.path.join(self.output_dir, "adc_rendered_single_chip.npy")
        self.log(f"Loading pre-rendered ADC from {adc_path}")

        # Load GT single-chip ADC
        gt_path = self.scene.get_single_chip_gt_adc(self.single_chip_frame)
        if gt_path is None or not os.path.isfile(gt_path):
            self.log(f"No GT single-chip ADC for frame {self.single_chip_frame}")
            self.save_metrics(metrics)
            return metrics

        self.log("Computing transfer metrics...")

        # Beamform ADC → RA image via single-chip virtual array (3TX×4RX → 8-el)
        # This matches the RA images shown in generate_figures().
        from mmir.data.ra_utils import adc_to_ra_image_single_chip
        from .utils.metrics import compute_ra_metrics

        adc_rendered = np.load(adc_path)  # (3, 4, 128, 2) float32

        # GT single-chip is (128, 4, 3, 128) complex128 — convert:
        #   arr[0] → (4, 3, 128), transpose → (3, 4, 128), split complex → (3, 4, 128, 2)
        gt_raw = np.load(gt_path)
        gt_complex = gt_raw[0].transpose(1, 0, 2)  # (3, 4, 128) complex
        adc_gt = np.stack([gt_complex.real, gt_complex.imag], axis=-1).astype(np.float32)

        # Beamform to RA polar images (63, 128) — Hann-windowed, FFT-shifted
        ra_rendered = adc_to_ra_image_single_chip(adc_rendered)
        ra_gt = adc_to_ra_image_single_chip(adc_gt)

        # Save RA polar arrays for visualization
        np.save(os.path.join(self.output_dir, "ra_rendered_polar.npy"), ra_rendered)
        np.save(os.path.join(self.output_dir, "ra_gt_polar.npy"), ra_gt)

        self.log(f"  RA shapes: rendered={ra_rendered.shape}, GT={ra_gt.shape}")

        # Skip first N range bins to remove TX-RX coupling leakage
        # (hardware artifact at near-zero range that the renderer doesn't model)
        SKIP_RANGE_BINS = 5
        ra_rendered_c = ra_rendered[:, SKIP_RANGE_BINS:]
        ra_gt_c = ra_gt[:, SKIP_RANGE_BINS:]

        # Compute metrics on beamformed RA (linear scale, after leakage removal)
        ra_metrics = compute_ra_metrics(ra_rendered_c, ra_gt_c)
        metrics["transfer_corr"] = ra_metrics["correlation"]
        metrics["transfer_psnr"] = ra_metrics["psnr"]
        metrics["transfer_ssim"] = ra_metrics["ssim"]
        metrics["transfer_mse"] = ra_metrics["mse"]

        # Also compute dB-scale correlation (more standard for radar)
        from scipy.stats import pearsonr as _pearsonr
        def _to_db(x, floor=-40.0):
            x_n = x / max(x.max(), 1e-30)
            return np.maximum(20.0 * np.log10(np.maximum(x_n, 1e-10)), floor)

        ra_rend_db = _to_db(ra_rendered_c)
        ra_gt_db = _to_db(ra_gt_c)
        corr_db, _ = _pearsonr(ra_rend_db.flatten(), ra_gt_db.flatten())
        metrics["transfer_corr_dB"] = float(corr_db)

        # Cartesian RA metrics (matching Eval #1 methodology)
        from mmir.data.ra_utils import ra_polar_to_cartesian
        from .utils.metrics import compute_cart_ra_metrics

        range_res = self._get_range_resolution()
        ra_rend_cart = ra_polar_to_cartesian(ra_rendered_c, range_res)
        ra_gt_cart = ra_polar_to_cartesian(ra_gt_c, range_res)
        cart_metrics = compute_cart_ra_metrics(ra_gt_cart, ra_rend_cart)
        metrics["transfer_cart_corr"] = cart_metrics["cart_corr"]
        metrics["transfer_cart_psnr"] = cart_metrics["psnr"]
        metrics["transfer_cart_ssim"] = cart_metrics["ssim"]
        metrics["transfer_cart_mse"] = cart_metrics["mse"]
        metrics["transfer_cart_rmse"] = cart_metrics["rmse"]

        # ADC-level metrics (matching training script methodology)
        from mmir.training.metrics import compute_adc_metrics as _compute_adc_metrics

        rendered_complex = adc_rendered[..., 0] + 1j * adc_rendered[..., 1]  # (3, 4, 128)
        adc_metrics = _compute_adc_metrics(rendered_complex, gt_complex)
        metrics["transfer_adc_log_mag_mse"] = adc_metrics["adc_log_mag_mse"]
        metrics["transfer_adc_log_mag_rmse"] = adc_metrics["adc_log_mag_rmse"]
        metrics["transfer_adc_minmax_mag_mse"] = adc_metrics["adc_minmax_mag_mse"]
        metrics["transfer_adc_minmax_mag_rmse"] = adc_metrics["adc_minmax_mag_rmse"]
        metrics["transfer_adc_phase_mse"] = adc_metrics["adc_phase_mse"]
        metrics["transfer_adc_phase_rmse"] = adc_metrics["adc_phase_rmse"]

        # Default (untrained) baseline
        default_adc_path = os.path.join(self.output_dir, "adc_default_single_chip.npy")
        if os.path.isfile(default_adc_path):
            ra_default = adc_to_ra_image_single_chip(np.load(default_adc_path))
            np.save(os.path.join(self.output_dir, "ra_default_polar.npy"), ra_default)
            ra_default_c = ra_default[:, SKIP_RANGE_BINS:]
            default_ra_metrics = compute_ra_metrics(ra_default_c, ra_gt_c)
            metrics["default_corr"] = default_ra_metrics["correlation"]
            metrics["default_psnr"] = default_ra_metrics["psnr"]
            metrics["default_ssim"] = default_ra_metrics["ssim"]
            # dB-scale default
            ra_def_db = _to_db(ra_default_c)
            corr_def_db, _ = _pearsonr(ra_def_db.flatten(), ra_gt_db.flatten())
            metrics["default_corr_dB"] = float(corr_def_db)
            # Cartesian RA metrics for default
            ra_def_cart = ra_polar_to_cartesian(ra_default_c, range_res)
            default_cart_metrics = compute_cart_ra_metrics(ra_gt_cart, ra_def_cart)
            metrics["default_cart_corr"] = default_cart_metrics["cart_corr"]
            metrics["default_cart_psnr"] = default_cart_metrics["psnr"]
            metrics["default_cart_ssim"] = default_cart_metrics["ssim"]
            metrics["default_cart_mse"] = default_cart_metrics["mse"]
            metrics["default_cart_rmse"] = default_cart_metrics["rmse"]
            # ADC metrics for default
            default_adc = np.load(default_adc_path)
            default_complex = default_adc[..., 0] + 1j * default_adc[..., 1]
            default_adc_metrics = _compute_adc_metrics(default_complex, gt_complex)
            metrics["default_adc_log_mag_mse"] = default_adc_metrics["adc_log_mag_mse"]
            metrics["default_adc_log_mag_rmse"] = default_adc_metrics["adc_log_mag_rmse"]
            metrics["default_adc_minmax_mag_mse"] = default_adc_metrics["adc_minmax_mag_mse"]
            metrics["default_adc_minmax_mag_rmse"] = default_adc_metrics["adc_minmax_mag_rmse"]
            metrics["default_adc_phase_mse"] = default_adc_metrics["adc_phase_mse"]
            metrics["default_adc_phase_rmse"] = default_adc_metrics["adc_phase_rmse"]

        # Load cascaded training correlation for degradation ratio
        if self.scene.our_training_dir:
            from .base_evaluator import load_our_training_history
            try:
                history = load_our_training_history(self.scene.our_training_dir)
                cart_corr = history.get("cart_corr", [])
                valid = [c for c in cart_corr if c is not None]
                if valid:
                    metrics["train_corr"] = max(valid)
                    if metrics["train_corr"] > 0:
                        metrics["degradation_pct"] = (
                            1.0 - metrics["transfer_corr"] / metrics["train_corr"]
                        ) * 100.0
            except Exception:
                pass

        self.log(f"  Transfer corr={metrics.get('transfer_corr', 'N/A'):.3f} "
                 f"(dB={metrics.get('transfer_corr_dB', 'N/A'):.3f}), "
                 f"Default corr={metrics.get('default_corr', 'N/A'):.3f} "
                 f"(dB={metrics.get('default_corr_dB', 'N/A'):.3f}), "
                 f"Train corr={metrics.get('train_corr', 'N/A')}")
        self.log(f"  Cartesian RA: transfer cart_corr={metrics.get('transfer_cart_corr', 'N/A'):.3f}, "
                 f"PSNR={metrics.get('transfer_cart_psnr', 'N/A'):.2f}, "
                 f"SSIM={metrics.get('transfer_cart_ssim', 'N/A'):.3f}, "
                 f"MSE={metrics.get('transfer_cart_mse', 'N/A'):.6f}, "
                 f"RMSE={metrics.get('transfer_cart_rmse', 'N/A'):.4f}")
        if "default_cart_corr" in metrics:
            self.log(f"  Cartesian RA: default  cart_corr={metrics['default_cart_corr']:.3f}, "
                     f"PSNR={metrics['default_cart_psnr']:.2f}, "
                     f"SSIM={metrics['default_cart_ssim']:.3f}, "
                     f"MSE={metrics['default_cart_mse']:.6f}, "
                     f"RMSE={metrics['default_cart_rmse']:.4f}")
        self.log(f"  ADC (log mag): transfer MSE={metrics.get('transfer_adc_log_mag_mse', 'N/A'):.6f}, "
                 f"RMSE={metrics.get('transfer_adc_log_mag_rmse', 'N/A'):.4f}")
        self.log(f"  ADC (minmax): transfer MSE={metrics.get('transfer_adc_minmax_mag_mse', 'N/A'):.6f}, "
                 f"RMSE={metrics.get('transfer_adc_minmax_mag_rmse', 'N/A'):.4f}")
        self.log(f"  ADC (phase):  transfer MSE={metrics.get('transfer_adc_phase_mse', 'N/A'):.6f}, "
                 f"RMSE={metrics.get('transfer_adc_phase_rmse', 'N/A'):.4f}")
        if "default_adc_log_mag_mse" in metrics:
            self.log(f"  ADC (log mag): default  MSE={metrics['default_adc_log_mag_mse']:.6f}, "
                     f"RMSE={metrics['default_adc_log_mag_rmse']:.4f}")
            self.log(f"  ADC (minmax): default  MSE={metrics['default_adc_minmax_mag_mse']:.6f}, "
                     f"RMSE={metrics['default_adc_minmax_mag_rmse']:.4f}")
            self.log(f"  ADC (phase):  default  MSE={metrics['default_adc_phase_mse']:.6f}, "
                     f"RMSE={metrics['default_adc_phase_rmse']:.4f}")

        self.save_metrics(metrics)
        return metrics

    def _get_range_resolution(self) -> float:
        """Compute range resolution from single-chip config."""
        sc_config = self._ensure_aligned_sc_config()
        if sc_config is None:
            return 0.0375  # default for 77 GHz, 4 GHz BW
        try:
            from mmir.data.ra_utils import compute_range_res_from_cfg
            return compute_range_res_from_cfg(sc_config)
        except Exception:
            return 0.0375


def run_radar_transfer_all(
    scenes: List[SceneInfo],
    output_root: str,
    render: bool = True,
    skip_latex: bool = False,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Run radar transfer evaluation for all scenes."""
    per_scene = {}

    for scene in scenes:
        evaluator = RadarTransferEvaluator(scene, output_root, verbose=verbose)
        if render:
            evaluator.render()
        metrics = evaluator.run()
        per_scene[scene.name] = metrics

    # Aggregate
    flat_metrics = {}
    for scene_name, m in per_scene.items():
        flat_metrics[scene_name] = {
            "train_corr": m.get("train_corr"),
            "transfer_corr": m.get("transfer_corr"),
            "transfer_psnr": m.get("transfer_psnr"),
            "transfer_ssim": m.get("transfer_ssim"),
            "transfer_cart_corr": m.get("transfer_cart_corr"),
            "transfer_cart_psnr": m.get("transfer_cart_psnr"),
            "transfer_cart_ssim": m.get("transfer_cart_ssim"),
            "transfer_cart_mse": m.get("transfer_cart_mse"),
            "transfer_cart_rmse": m.get("transfer_cart_rmse"),
            "transfer_adc_log_mag_mse": m.get("transfer_adc_log_mag_mse"),
            "transfer_adc_minmax_mag_mse": m.get("transfer_adc_minmax_mag_mse"),
            "transfer_adc_phase_mse": m.get("transfer_adc_phase_mse"),
            "default_corr": m.get("default_corr"),
            "default_psnr": m.get("default_psnr"),
            "default_ssim": m.get("default_ssim"),
            "default_cart_corr": m.get("default_cart_corr"),
            "default_cart_psnr": m.get("default_cart_psnr"),
            "default_cart_ssim": m.get("default_cart_ssim"),
            "default_cart_mse": m.get("default_cart_mse"),
            "default_cart_rmse": m.get("default_cart_rmse"),
            "default_adc_log_mag_mse": m.get("default_adc_log_mag_mse"),
            "default_adc_minmax_mag_mse": m.get("default_adc_minmax_mag_mse"),
            "default_adc_phase_mse": m.get("default_adc_phase_mse"),
            "degradation_pct": m.get("degradation_pct"),
        }

    agg = aggregate_metrics(flat_metrics, [
        "train_corr", "transfer_corr", "transfer_psnr", "transfer_ssim",
        "transfer_cart_corr", "transfer_cart_psnr", "transfer_cart_ssim",
        "transfer_cart_mse", "transfer_cart_rmse",
        "transfer_adc_log_mag_mse", "transfer_adc_minmax_mag_mse",
        "transfer_adc_phase_mse",
        "default_corr", "default_psnr", "default_ssim",
        "default_cart_corr", "default_cart_psnr", "default_cart_ssim",
        "default_cart_mse", "default_cart_rmse",
        "default_adc_log_mag_mse", "default_adc_minmax_mag_mse",
        "default_adc_phase_mse",
        "degradation_pct",
    ])

    agg_dir = os.path.join(output_root, "radar_transfer")
    os.makedirs(agg_dir, exist_ok=True)

    import json as _json
    with open(os.path.join(agg_dir, "aggregate_metrics.json"), "w") as f:
        _json.dump({"per_scene": flat_metrics, "aggregate": agg}, f, indent=2)

    if not skip_latex:
        from .utils.visualization import generate_radar_transfer_table
        generate_radar_transfer_table(
            flat_metrics, agg,
            os.path.join(agg_dir, "table2_radar_transfer.tex"),
        )

    if verbose:
        print("\n=== Radar Transfer Aggregate ===")
        for key, stats in agg.items():
            print(f"  {key}: {stats['mean']:.3f} ± {stats['std']:.3f}")

    return per_scene
