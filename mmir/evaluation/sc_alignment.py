"""Single-chip radar alignment pipeline."""

import os
from typing import Optional


def run_sc_alignment(
    scene,
    output_config_path: str,
    n_hits_per_rx: int = 2000,
    coarse_range_m: float = 3.0,
    coarse_azimuth_deg: float = 15.0,
    coarse_rotation_deg: float = 8.0,
    coarse_steps: int = 13,
    fine_range_m: float = 0.5,
    fine_azimuth_deg: float = 2.5,
    fine_rotation_deg: float = 1.5,
    fine_steps: int = 9,
    verbose: bool = True,
) -> Optional[str]:
    """Run full two-stage SC alignment for a scene."""
    sc_frame = scene.get_closest_single_chip_frame()

    configs_dir = os.path.join(scene.data_dir, "configs")

    # Pre-requisites
    casc_orig = os.path.join(configs_dir, f"cascaded_frame_{scene.frame}.json")
    casc_aligned = scene.cascaded_config
    sc_orig = os.path.join(configs_dir, f"single_chip_frame_{sc_frame}.json")


    gt_adc_path = scene.get_single_chip_gt_adc(sc_frame)

    if verbose:
        print(f"  SC alignment for {scene.name} (SC frame {sc_frame})")

    from mmir.evaluation.cascaded_sc_alignment import (
        transfer_alignment_to_sc,
        multi_resolution_grid_search,
    )

    # Stage 1: Procrustes transfer
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        stage1_path = os.path.join(tmpdir, f"single_chip_frame_{sc_frame}_stage1.json")

        if verbose:
            print(f"  Stage 1: Transferring cascaded alignment → SC frame {sc_frame}...")

        stage1_path = transfer_alignment_to_sc(
            casc_orig_config=casc_orig,
            casc_aligned_config=casc_aligned,
            sc_orig_config=sc_orig,
            output_path=stage1_path,
        )

        if verbose:
            print(f"  Stage 1 done → {stage1_path}")

        # Stage 2: Multi-resolution grid search
        if verbose:
            print("  Stage 2: Multi-resolution grid search (metric=cart_corr)...")

        result = multi_resolution_grid_search(
            base_config=stage1_path,
            gt_adc_path=gt_adc_path,
            mesh_file=scene.mesh_file,
            training_dir=scene.our_training_dir,
            output_config_path=output_config_path,
            coarse_range_m=coarse_range_m,
            coarse_azimuth_deg=coarse_azimuth_deg,
            coarse_rotation_deg=coarse_rotation_deg,
            coarse_steps=coarse_steps,
            fine_range_m=fine_range_m,
            fine_azimuth_deg=fine_azimuth_deg,
            fine_rotation_deg=fine_rotation_deg,
            fine_steps=fine_steps,
            n_hits_per_rx=n_hits_per_rx,
            metric="cart_corr",
            verbose=verbose,
        )

    if verbose:
        print(f"  Alignment done: best cart_corr={result['best_corr_dB']:.4f}")
        print(f"  Saved: {output_config_path}")

    return output_config_path
