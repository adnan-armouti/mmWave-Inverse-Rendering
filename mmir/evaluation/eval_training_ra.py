"""Evaluation #1 v2: Training RA — reads metrics from training output when available."""

import json
import os
from typing import Dict, List, Optional

import numpy as np

from .base_evaluator import BaseEvaluator, aggregate_metrics
from mmir.evaluation.base_evaluator import load_our_training_history
from .scene_registry import SceneInfo

from .utils.metrics import compute_cart_ra_metrics as compute_ra_metrics
from mmir.training.metrics import _compute_va_phase_coherence


# VA Phase Coherence (standalone function)

# Evaluator v2

class TrainingRAEvaluator(BaseEvaluator):
    """Evaluate training RA results, preferring pre-computed metrics from training."""

    @property
    def eval_name(self) -> str:
        return "training_ra"

    # Main entry point

    def run(self) -> dict:
        """Compute RA Cartesian images and metrics for both methods."""
        ours_dir = os.path.join(self.output_dir, "ours")
        os.makedirs(ours_dir, exist_ok=True)

        # GT RA Cartesian
        ra_gt_cart, range_res = self._get_gt_ra_cartesian()
        if ra_gt_cart is None:
            self.log("No GT RA available — skipping")
            return {"scene": self.scene.name, "ours": {}}

        # Ours
        ours_metrics = self._eval_ours(ra_gt_cart, range_res, ours_dir)

        metrics = {
            "scene": self.scene.name,
            "ours": ours_metrics,
        }
        return metrics

    # ADC metrics helper

    _ADC_METRIC_KEYS = (
        "adc_log_mag_mse", "adc_log_mag_rmse",
        "adc_minmax_mag_mse", "adc_minmax_mag_rmse",
        "adc_phase_mse", "adc_phase_rmse",
        "va_phase_coherence",
    )

    def _read_adc_metrics(
        self, training_dir: str, best_metrics: Optional[dict] = None,
    ) -> dict:
        """Read ADC metrics from training output."""
        adc = {}

        # 1) Try best_metrics.json
        if best_metrics:
            pass

        # 2) Fall back to training_history.json at best_iteration
        if not training_dir:
            return adc

        history_path = os.path.join(training_dir, "training_history.json")
        if not os.path.isfile(history_path):
            return adc

        try:
            with open(history_path) as f:
                raw = json.load(f)

            # Determine best iteration index
            h = raw
            best_iter = None
            # Best iteration from best_metrics.json, else the max cart_corr
            bm_path = os.path.join(training_dir, "best_metrics.json")
            if os.path.isfile(bm_path):
                with open(bm_path) as f2:
                    best_iter = json.load(f2).get("best_iteration")

            iterations = h.get("iteration", [])
            if best_iter is None or best_iter not in iterations:
                return adc
            idx = iterations.index(best_iter)

            # Read ADC metrics at best iteration
            # Training history stores the phase metrics as 'adc_relphase_*'
            key_map = {
                "adc_log_mag_mse": "adc_log_mag_mse",
                "adc_log_mag_rmse": "adc_log_mag_rmse",
                "adc_minmax_mag_mse": "adc_minmax_mag_mse",
                "adc_minmax_mag_rmse": "adc_minmax_mag_rmse",
                "adc_phase_mse": "adc_relphase_mse",
                "adc_phase_rmse": "adc_relphase_rmse",
                "va_phase_coherence": "va_phase_coherence",
            }
            for out_key, hist_key in key_map.items():
                vals = h.get(hist_key, [])
                if idx < len(vals):
                    adc[out_key] = vals[idx]

        except Exception as e:
            self.log(f"  Could not read ADC metrics from history: {e}")

        return adc

    # Ours evaluation

    def _eval_ours(self, ra_gt_cart: np.ndarray, range_res: float, ours_dir: str) -> dict:
        """Evaluate our method: try best_metrics.json first, else fall back to re-render."""
        self.log("Evaluating ours...")
        ours_metrics = {}

        # Try reading pre-computed metrics from training
        if self.scene.our_training_dir:
            best_metrics_path = os.path.join(self.scene.our_training_dir, "best_metrics.json")
            if os.path.isfile(best_metrics_path):
                self.log(f"  Reading pre-computed metrics from {best_metrics_path}")
                with open(best_metrics_path) as f:
                    best = json.load(f)

                ours_metrics = {
                    "cart_corr": best.get("cart_corr"),
                    "psnr": best.get("cart_psnr"),
                    "ssim": best.get("cart_ssim"),
                    "mse": best.get("cart_mse"),
                    "rmse": best.get("cart_rmse"),
                    "best_iteration": best.get("best_iteration"),
                    "polar_corr": best.get("polar_corr"),
                }

                # ADC metrics (from best_metrics.json or training_history.json)
                adc = self._read_adc_metrics(
                    self.scene.our_training_dir, best_metrics=best)
                ours_metrics.update(adc)

                self.log("  " + "  ".join(
                    f"{name}={ours_metrics[key]:{fmt}}" if isinstance(ours_metrics.get(key), (int, float))
                    else f"{name}=N/A"
                    for name, key, fmt in (("corr", "cart_corr", ".4f"), ("MSE", "mse", ".6f"),
                                           ("RMSE", "rmse", ".4f"), ("PSNR", "psnr", ".2f"),
                                           ("SSIM", "ssim", ".4f"))))
                if adc:
                    self.log(f"  ADC: log_mag_mse={adc.get('adc_log_mag_mse', 'N/A'):.4f} "
                             f"minmax_mag_mse={adc.get('adc_minmax_mag_mse', 'N/A'):.6f} "
                             f"phase_mse={adc.get('adc_phase_mse', 'N/A'):.4f} "
                             f"va_phase_coh={adc.get('va_phase_coherence', 'N/A')}")

                # Compute va_phase_coherence if missing (old training runs)
                if ours_metrics.get("va_phase_coherence") is None:
                    va_coh = self._compute_va_phase_coherence_fallback(ours_dir)
                    if va_coh is not None:
                        ours_metrics["va_phase_coherence"] = va_coh
                        self.log(f"  Computed va_phase_coherence={va_coh:.4f} (re-rendered)")

                # Copy ra_rendered_cart.npy from training dir
                os.path.join(self.scene.our_training_dir, "ra_rendered_cart.npy")

                # Copy rendered .png files from training dir
                for png_name in ("rendered_ra_dB.png", "rendered_ra_linear.png"):
                    os.path.join(self.scene.our_training_dir, png_name)

                # Save per-method metrics
                ours_full = {"scene": self.scene.name, "method": "ours", **ours_metrics}
                self._save_json(os.path.join(ours_dir, "metrics.json"), ours_full)
                return ours_metrics

        # Fallback: re-render like v1

    # Data loading helpers (same as v1)

    def _get_gt_ra_cartesian(self) -> tuple:
        """Load GT cascaded ADC → RA Cartesian. Returns (ra_cart, range_res) or (None, None)."""
        # Try loading from training dir first
        if self.scene.our_training_dir:
            os.path.join(self.scene.our_training_dir, "ra_gt_cart.npy")

        if not self.scene.gt_cascaded_adc or not os.path.isfile(self.scene.gt_cascaded_adc):
            return None, None

        from mmir.data.ra_utils import adc_to_ra_image_numpy, ra_polar_to_cartesian
        from mmir.data.ra_utils import compute_range_res_from_cfg

        range_res = compute_range_res_from_cfg(self.scene.cascaded_config)
        gt_raw = np.load(self.scene.gt_cascaded_adc)
        gt_complex = gt_raw[0].transpose(1, 0, 2)
        gt_adc = np.stack([gt_complex.real, gt_complex.imag], axis=-1).astype(np.float32)
        ra_polar = adc_to_ra_image_numpy(gt_adc)
        ra_cart = ra_polar_to_cartesian(ra_polar, range_res)
        return ra_cart, range_res

    # VA Phase Coherence fallback (for old training runs without this metric)

    def _compute_va_phase_coherence_fallback(self, method_dir: str) -> Optional[float]:
        """Re-render ADC and compute VA phase coherence when not in training history."""
        if not self.scene.our_training_dir:
            return None

        # Check for cached ADC first
        cached_path = os.path.join(method_dir, "adc_rendered_cascaded.npy")
        rendered_adc = None
        try:
            from .renderer_wrapper import RendererWrapper
            self.log("  Re-rendering ADC for va_phase_coherence...")
            wrapper = RendererWrapper.from_training_dir(
                mesh_file=self.scene.mesh_file,
                config_file=self.scene.cascaded_config,
                training_dir=self.scene.our_training_dir,
                verbose=False,
            )
            wrapper.load_materials(self.scene.our_training_dir)
            rendered_adc = wrapper.render_forward(seed=42)
            np.save(cached_path, rendered_adc)
            self.log(f"  Rendered ADC → {cached_path}")
            wrapper.cleanup()
        except Exception as e:
            self.log(f"  Could not render ADC for va_phase_coherence: {e}")
            return None

        if rendered_adc is None:
            return None

        # Load GT ADC
        if not self.scene.gt_cascaded_adc or not os.path.isfile(self.scene.gt_cascaded_adc):
            return None

        gt_raw = np.load(self.scene.gt_cascaded_adc)
        gt_complex = gt_raw[0].transpose(1, 0, 2)  # [NT, NR, K]

        # Convert rendered ADC [NT, NR, K, 2] to complex
        rendered_complex = rendered_adc[..., 0] + 1j * rendered_adc[..., 1]

        return _compute_va_phase_coherence(rendered_complex, gt_complex)

    # Utility

    def _save_json(self, path: str, data: dict):
        """Save dict as JSON with numpy support."""
        from .base_evaluator import _json_default
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_json_default)
        self.log(f"  Saved {path}")


# Batch runner

def run_training_ra_all(
    scenes: List[SceneInfo],
    output_root: str,
    skip_latex: bool = False,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Run Training RA evaluation for all scenes and generate aggregate outputs."""
    per_scene = {}

    for scene in scenes:
        evaluator = TrainingRAEvaluator(scene, output_root, verbose=verbose)
        metrics = evaluator.run()
        per_scene[scene.name] = metrics

    # Build flat metrics for aggregation
    flat_metrics = {}
    for scene_name, m in per_scene.items():
        ours = m.get("ours", {})
        flat_metrics[scene_name] = {
            "ours_corr": ours.get("cart_corr"),
            "ours_psnr": ours.get("psnr"),
            "ours_ssim": ours.get("ssim"),
            "ours_mse": ours.get("mse"),
            "ours_rmse": ours.get("rmse"),
            "ours_adc_log_mag_mse": ours.get("adc_log_mag_mse"),
            "ours_adc_minmax_mag_mse": ours.get("adc_minmax_mag_mse"),
            "ours_adc_phase_mse": ours.get("adc_phase_mse"),
            "ours_va_phase_coherence": ours.get("va_phase_coherence"),
        }

    agg_keys = [
        "ours_corr", "ours_psnr", "ours_ssim", "ours_mse", "ours_rmse",
        "ours_adc_log_mag_mse", "ours_adc_minmax_mag_mse", "ours_adc_phase_mse",
        "ours_va_phase_coherence",
    ]
    agg = aggregate_metrics(flat_metrics, agg_keys)

    # Save aggregate
    agg_dir = os.path.join(output_root, "training_ra")
    os.makedirs(agg_dir, exist_ok=True)

    with open(os.path.join(agg_dir, "aggregate_metrics.json"), "w") as f:
        json.dump({"per_scene": flat_metrics, "aggregate": agg}, f, indent=2)

    # LaTeX table
    if not skip_latex:
        from mmir.evaluation.utils.visualization import generate_training_ra_table
        generate_training_ra_table(
            flat_metrics, agg,
            os.path.join(agg_dir, "table1_training_ra.tex"),
        )

    if verbose:
        print("\n=== Training RA Aggregate ===")
        for key, stats in agg.items():
            print(f"  {key}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    return per_scene
