#!/usr/bin/env python3
"""Cascaded Radar Alignment Orchestrator ======================================"""

import json
import os
from typing import Dict, List, Optional, Tuple

# Scene definitions

SCENES: List[Tuple[str, int]] = [
    ("seq_0_frame_135", 135),
    ("seq_0_frame_390", 390),
    ("seq_1_frame_185", 185),
    ("seq_1_frame_438", 438),
    ("seq_2_frame_105", 105),
    ("seq_2_frame_160", 160),
    ("seq_2_frame_300", 300),
]

# Validated alignment results
# Correlation scores measured at 1500 renderer hits.
# ``winner`` is max(cc_lidar_4dof, cc_renderer_2dof).
# ``config_suffix`` is the file suffix of the winning aligned config.

ALIGNMENT_RESULTS: Dict[str, dict] = {
    "seq_0_frame_135": {
        "frame": 135,
        "winner": "lidar_4dof",
        "cc_lidar_4dof": 0.476,
        "cc_renderer_2dof": 0.466,
        "config_suffix": "_aligned_gpu",
    },
    "seq_0_frame_390": {
        "frame": 390,
        "winner": "lidar_4dof",
        "cc_lidar_4dof": 0.363,
        "cc_renderer_2dof": 0.172,
        "config_suffix": "_aligned_gpu",
    },
    "seq_1_frame_185": {
        "frame": 185,
        "winner": "lidar_4dof",
        "cc_lidar_4dof": 0.463,
        "cc_renderer_2dof": 0.283,
        "config_suffix": "_aligned_gpu",
    },
    "seq_1_frame_438": {
        "frame": 438,
        "winner": "renderer_2dof",
        "cc_lidar_4dof": 0.584,
        "cc_renderer_2dof": 0.613,
        "config_suffix": "_aligned_2dof",
    },
    "seq_2_frame_105": {
        "frame": 105,
        "winner": "renderer_2dof",
        "cc_lidar_4dof": 0.113,
        "cc_renderer_2dof": 0.244,
        "config_suffix": "_aligned_2dof",
    },
    "seq_2_frame_160": {
        "frame": 160,
        "winner": "renderer_2dof",
        "cc_lidar_4dof": 0.393,
        "cc_renderer_2dof": 0.577,
        "config_suffix": "_aligned_2dof",
    },
    "seq_2_frame_300": {
        "frame": 300,
        "winner": "lidar_4dof",
        "cc_lidar_4dof": 0.432,
        "cc_renderer_2dof": 0.389,
        "config_suffix": "_aligned_gpu",
    },
}


# The aligned configs used for the paper ship with the code. Regenerating them with
# run_all() gives configs within about 1 mm, which changes training results slightly.
PAPER_ALIGNMENT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "assets", "alignment"))


def get_aligned_config_path(
    scene_name: str,
    frame: int,
    data_root: str = "data",
    alignment: str = "paper",
    suffix: Optional[str] = None,
) -> str:
    """Return the path to an aligned cascade config for a scene."""
    if alignment not in ("paper", "data"):
        raise ValueError(f"alignment must be 'paper' or 'data', got {alignment!r}")
    if suffix is None:
        suffix = ALIGNMENT_RESULTS.get(scene_name, {}).get("config_suffix", "_aligned_gpu")
    name = f"cascaded_frame_{frame}{suffix}.json"
    if alignment == "paper":
        return os.path.join(PAPER_ALIGNMENT_DIR, scene_name, name)
    return os.path.join(data_root, scene_name, "configs", name)


def select_best_alignment(
    cc_lidar_4dof: float,
    cc_renderer_2dof: float,
) -> str:
    """Given two correlation scores, return the winning method name."""
    if cc_lidar_4dof >= cc_renderer_2dof:
        return "lidar_4dof"
    return "renderer_2dof"


# Summary / verification

def print_summary(data_root: str = "data") -> None:
    """Print a table of alignment results and verify config files exist."""
    print("=" * 78)
    print("Cascaded Radar Alignment — Per-Scene Results (1500-hit validation)")
    print("=" * 78)
    hdr = (
        f"{'Scene':<22} {'Winner':<16} {'CC_lidar':<10} "
        f"{'CC_renderer':<12} {'Config suffix'}"
    )
    print(hdr)
    print("-" * 78)

    all_ok = True
    winner_ccs = []
    for scene_name, frame in SCENES:
        info = ALIGNMENT_RESULTS[scene_name]
        cc_best = max(info["cc_lidar_4dof"], info["cc_renderer_2dof"])
        winner_ccs.append(cc_best)

        path = get_aligned_config_path(scene_name, frame, data_root, alignment="data")
        exists = os.path.isfile(path)
        status = "✓" if exists else "MISSING"
        if not exists:
            all_ok = False

        print(
            f"{scene_name:<22} {info['winner']:<16} "
            f"{info['cc_lidar_4dof']:<10.3f} "
            f"{info['cc_renderer_2dof']:<12.3f} "
            f"{info['config_suffix']}  [{status}]"
        )

    print("-" * 78)
    print(f"Mean winner CC: {sum(winner_ccs) / len(winner_ccs):.4f}")
    print(f"Config files: {'all present' if all_ok else 'SOME MISSING'}")
    print("=" * 78)


# LiDAR-based 4-DOF alignment

def run_lidar_4dof(config_path: str, lidar_path: str, gt_adc_path: str,
                   output_path: str, coarse_steps: int = 5) -> dict:
    """Run the LiDAR-based 4-DOF alignment and write the aligned config."""
    from .lidar_alignment import (
        create_aligned_config, load_point_cloud, load_radar_config, optimize_alignment_gpu,
        radar_gt_to_ra_map,
    )

    params, base_origin, base_boresight = load_radar_config(config_path)
    max_range = params["num_adc"] * params["range_resolution"]
    results = optimize_alignment_gpu(
        load_point_cloud(lidar_path), radar_gt_to_ra_map(gt_adc_path), params,
        base_boresight, base_origin,
        (-max_range * 0.1, max_range * 0.1), (-15.0, 15.0), (-10.0, 10.0), (-10.0, 10.0),
        metric="correlation", coarse_steps=coarse_steps, near_field_m=1.5,
        batch_size=100, verbose=False,
    )
    opt = results["optimal_params"]
    create_aligned_config(
        config_path=config_path,
        delta_range_m=opt["delta_range_m"],
        delta_azimuth_deg=opt["delta_azimuth_deg"],
        rotation_elev_deg=opt["rotation_elev_deg"],
        rotation_azim_deg=opt["rotation_azim_deg"],
        output_path=output_path,
    )
    return results


# Full pipeline (run both methods + select)

def run_all(
    data_root: str = "data",
    output_root: str = "output/alignment",
    tx_pattern: Optional[str] = None,
    rx_pattern: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Run both alignment methods for all scenes and select winners."""
    # Lazy imports — these pull in mitsuba / drjit / open3d
    import mitsuba as mi
    if mi.variant() is None:
        mi.set_variant("cuda_ad_rgb")

    import drjit as dr
    import time

    from mmir.renderer import FMCWRendererRef, RenderConfigRef
    from .cascaded_renderer import (
        apply_2dof_to_config,
        update_renderer_antennas,
        render_and_evaluate,
        grid_search_2dof,
        refine_2dof,
        save_aligned_config,
        load_gt_ra_cart,
    )
    from mmir.data.ra_utils import compute_range_res_from_cfg

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    _tx = tx_pattern or os.path.join(
        repo_root, "assets", "antenna_pattern", "MMWCAS", "tx1_76.npy"
    )
    _rx = rx_pattern or os.path.join(
        repo_root, "assets", "antenna_pattern", "MMWCAS", "rx1_76.npy"
    )

    os.makedirs(output_root, exist_ok=True)
    all_results = {}

    if verbose:
        print("=" * 78)
        print("Cascaded Radar Alignment — Full Pipeline")
        print("=" * 78)

    for scene_name, frame in SCENES:
        if verbose:
            print(f"\n{'='*70}")
            print(f"Scene: {scene_name}")
            print(f"{'='*70}")

        data_dir = os.path.join(data_root, scene_name)
        mesh_path = os.path.join(data_dir, "scene", "mesh.ply")
        gt_adc_path = os.path.join(
            data_dir, "radar", f"cascaded_frame_{frame}.npy"
        )
        base_config_path = os.path.join(
            data_dir, "configs", f"cascaded_frame_{frame}.json"
        )

        # Check prerequisites
        missing = [
            p for p in [mesh_path, gt_adc_path, base_config_path]
            if not os.path.isfile(p)
        ]
        if missing:
            if verbose:
                for p in missing:
                    print(f"  SKIP: not found: {p}")
            continue

        with open(base_config_path) as f:
            base_config = json.load(f)
        range_res = compute_range_res_from_cfg(base_config_path)
        ra_gt_cart = load_gt_ra_cart(gt_adc_path, range_res)

        # Method 1: LiDAR-based 4-DOF (run if no aligned config exists)
        aligned_gpu_path = os.path.join(
            data_dir, "configs",
            f"cascaded_frame_{frame}_aligned_gpu.json",
        )

        cc_lidar = None
        if not os.path.isfile(aligned_gpu_path):
            if verbose:
                print("  LiDAR 4DOF: running alignment...")
            run_lidar_4dof(base_config_path,
                           os.path.join(data_dir, "scene", "pcl.npy"),
                           gt_adc_path, aligned_gpu_path)
        if verbose:
            print(f"  LiDAR 4DOF: using existing {os.path.basename(aligned_gpu_path)}")
        # Quick renderer validation at 1500 hits
        render_cfg = RenderConfigRef(
            n_hits_per_rx=1500,
            use_shared_hits=True,
            bsdf_model="mmwave_jones",
            mmwave_polarization="vertical",
            hemisphere_sampling="cosine",
            double_sided=True,
            enable_sms=False,
            use_vertex_normals=False,
            material_columns=6,
        )
        renderer = FMCWRendererRef.from_files(
            mesh_file=mesh_path,
            config_file=aligned_gpu_path,
            tx_pattern_file=_tx,
            rx_pattern_file=_rx,
            material_type="metal",
            render_config=render_cfg,
            verbose=False,
        )
        cc_lidar = render_and_evaluate(
            renderer, range_res, ra_gt_cart, seed=42
        )
        del renderer
        dr.sync_thread()
        if verbose:
            print(f"  LiDAR 4DOF CC (1500 hits): {cc_lidar:.4f}")

        # Method 2: Renderer-based 2-DOF
        aligned_2dof_path = os.path.join(
            data_dir, "configs",
            f"cascaded_frame_{frame}_aligned_2dof.json",
        )

        cc_renderer = None
        if verbose:
            print("  Renderer 2DOF: running alignment...")
        t0 = time.time()

        render_cfg_fast = RenderConfigRef(
            n_hits_per_rx=800,
            use_shared_hits=True,
            bsdf_model="mmwave_jones",
            mmwave_polarization="vertical",
            hemisphere_sampling="cosine",
            double_sided=True,
            enable_sms=False,
            use_vertex_normals=False,
            material_columns=6,
        )
        renderer = FMCWRendererRef.from_files(
            mesh_file=mesh_path,
            config_file=base_config_path,
            tx_pattern_file=_tx,
            rx_pattern_file=_rx,
            material_type="metal",
            render_config=render_cfg_fast,
            verbose=False,
        )

        grid_params, grid_cc, _ = grid_search_2dof(
            renderer, base_config, range_res, ra_gt_cart, seed=42
        )
        refined_params, refined_cc, _ = refine_2dof(
            renderer, base_config, grid_params, range_res, ra_gt_cart,
            max_evals=30, seed=42,
        )

        # Validate at 1500 hits
        renderer.config = RenderConfigRef(
            n_hits_per_rx=1500, use_shared_hits=True,
            bsdf_model="mmwave_jones", mmwave_polarization="vertical",
            hemisphere_sampling="cosine", double_sided=True,
            enable_sms=False,
            use_vertex_normals=False, material_columns=6,
        )
        renderer.sampler.n_hits_per_rx = 1500
        tx_mm, rx_mm, bs = apply_2dof_to_config(
            base_config, refined_params[0], refined_params[1]
        )
        update_renderer_antennas(renderer, tx_mm, rx_mm, bs)
        cc_renderer = render_and_evaluate(
            renderer, range_res, ra_gt_cart, seed=42
        )

        save_aligned_config(
            base_config,
            refined_params[0], refined_params[1],
            aligned_2dof_path,
        )
        del renderer
        dr.sync_thread()
        if verbose:
            print(
                f"  Renderer 2DOF CC (1500 hits): {cc_renderer:.4f} "
                f"({time.time()-t0:.1f}s)"
            )

        # Select winner
        if cc_lidar is not None and cc_renderer is not None:
            winner = select_best_alignment(cc_lidar, cc_renderer)
        elif cc_lidar is not None:
            winner = "lidar_4dof"
        elif cc_renderer is not None:
            winner = "renderer_2dof"
        else:
            if verbose:
                print(f"  WARNING: no alignment available for {scene_name}")
            continue

        suffix = (
            "_aligned_gpu" if winner == "lidar_4dof" else "_aligned_2dof"
        )
        winner_cc = (
            cc_lidar if winner == "lidar_4dof" else cc_renderer
        )

        scene_result = {
            "frame": frame,
            "winner": winner,
            "cc_lidar_4dof": float(cc_lidar) if cc_lidar is not None else None,
            "cc_renderer_2dof": float(cc_renderer) if cc_renderer is not None else None,
            "config_suffix": suffix,
            "aligned_config": os.path.join(
                data_dir, "configs",
                f"cascaded_frame_{frame}{suffix}.json",
            ),
        }
        all_results[scene_name] = scene_result

        if verbose:
            print(
                f"\n  >> Winner: {winner} (CC={winner_cc:.4f})"
                f"  config: cascaded_frame_{frame}{suffix}.json"
            )

        # Save per-scene log
        scene_out = os.path.join(output_root, scene_name)
        os.makedirs(scene_out, exist_ok=True)
        with open(os.path.join(scene_out, "alignment_result.json"), "w") as f:
            json.dump(scene_result, f, indent=2)

    # Save global results
    with open(os.path.join(output_root, "alignment_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    if verbose:
        print()
        print_summary(data_root)

    return all_results


# CLI entry point

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cascaded radar alignment orchestrator",
    )
    parser.add_argument(
        "--data-root", default="data",
        help="Root of preprocessed data tree (default: data)",
    )
    parser.add_argument(
        "--output-root", default="output/alignment",
        help="Output directory for logs (default: output/alignment)",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Just print the alignment summary table (no alignment run)",
    )
    args = parser.parse_args()

    run_all(
        data_root=args.data_root,
        output_root=args.output_root,
    )
