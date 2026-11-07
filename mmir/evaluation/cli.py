"""Evaluate trained scenes.

Usage:
    python -m mmir.evaluation training-ra [--scenes seq_0_frame_135,...]
    python -m mmir.evaluation radar-transfer [--scenes seq_0_frame_135,...]
    python -m mmir.evaluation occupancy --coloradar-dataset-dir /path/to/kitti/2_28_2021_outdoors_run

training-ra     metrics of the trained cascade view (paper Table 1)
radar-transfer  frozen scene rendered for the single-chip radar (paper Table 2)
occupancy       3D occupancy from a dense 100x100 virtual array vs. LiDAR (paper Table 3);
                the ColoRadar baseline needs the raw dataset
"""

import argparse
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="mmIR evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("training-ra", "training-view RA metrics"),
                            ("radar-transfer", "cascade-to-single-chip transfer"),
                            ("occupancy", "3D occupancy from the dense virtual array")):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--scenes", type=str, default=None,
                         help="Comma-separated scene names (default: all paper scenes)")
        sub.add_argument("--data-root", type=str, default=None,
                         help="Folder holding <scene>/ (default: data)")
        sub.add_argument("--training-root", type=str, default=None,
                         help="Folder holding the trained <scene>/ runs (default: output/training)")
        sub.add_argument("--output-root", type=str, default=None,
                         help="Output directory (default: output/evaluation)")
        sub.add_argument("--alignment", choices=("paper", "data"), default="paper",
                         help="aligned cascade configs: 'paper' (assets/alignment/, default) or "
                              "'data' (regenerated into data/<scene>/configs/)")
        sub.add_argument("--skip-latex", action="store_true", help="Skip LaTeX table generation")
        sub.add_argument("--quiet", action="store_true")
        if name == "occupancy":
            sub.add_argument("--coloradar-dataset-dir", type=str, required=True,
                             help="Raw ColoRadar kitti prefix; the sequence number is appended "
                                  "(e.g. /path/to/kitti/2_28_2021_outdoors_run)")
            sub.add_argument("--coloradar-calib-dir", type=str, default=None,
                             help="ColoRadar calibration (default: mmir/preprocessing/calib)")
    return parser.parse_args()


def resolve_scenes(args):
    from . import scene_registry
    scene_registry.configure(args.data_root, args.training_root, args.alignment)
    if not args.scenes:
        return scene_registry.get_benchmark_scenes()
    scenes = []
    for name in (s.strip() for s in args.scenes.split(",")):
        scene = scene_registry.get_scene(name)
        if scene is None:
            print(f"Warning: scene '{name}' not found, skipping")
        else:
            scenes.append(scene)
    return scenes


def main():
    args = parse_args()
    verbose = not args.quiet
    scenes = resolve_scenes(args)
    from .scene_registry import PROJECT_ROOT
    output_root = args.output_root or os.path.join(PROJECT_ROOT, "output", "evaluation")
    os.makedirs(output_root, exist_ok=True)
    print(f"mmIR evaluation: {args.command} on {[s.name for s in scenes]} -> {output_root}")

    import mitsuba as mi
    if mi.variant() is None:
        mi.set_variant("cuda_ad_rgb")

    t_start = time.time()
    if args.command == "training-ra":
        from .eval_training_ra import run_training_ra_all
        run_training_ra_all(scenes, output_root, skip_latex=args.skip_latex, verbose=verbose)
    elif args.command == "radar-transfer":
        from .eval_radar_transfer import run_radar_transfer_all
        run_radar_transfer_all(scenes, output_root, render=True,
                               skip_latex=args.skip_latex, verbose=verbose)
    else:
        from .eval_3d_occupancy import run_3d_occupancy_all
        calib = args.coloradar_calib_dir or os.path.join(PROJECT_ROOT, "mmir", "preprocessing", "calib")
        run_3d_occupancy_all(scenes, output_root, render=True,
                             coloradar_dataset_dir=args.coloradar_dataset_dir,
                             coloradar_calib_dir=calib,
                             skip_latex=args.skip_latex, verbose=verbose)
    print(f"\nTotal time: {time.time() - t_start:.1f}s")
