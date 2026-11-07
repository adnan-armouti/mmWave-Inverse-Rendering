"""Train mmIR on one scene with the paper's settings."""

import argparse

from mmir.training.config import DEFAULT_DATA_ROOT, scene_config
from mmir.training.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Train mmIR on one scene.")
    parser.add_argument('--scene', required=True, help="scene name, e.g. seq_0_frame_135")
    parser.add_argument('--data_root', default=DEFAULT_DATA_ROOT, help="folder holding <scene>/")
    parser.add_argument('--alignment', choices=('paper', 'data'), default='paper',
                        help="aligned cascade config: 'paper' (assets/alignment/, default) or "
                             "'data' (regenerated into data/<scene>/configs/)")
    parser.add_argument('--config', default=None,
                        help="optional JSON file overriding training settings (field names of TrainingConfig)")
    parser.add_argument('--output_dir', default=None, help="default: output/training/<scene>")
    parser.add_argument('--iterations', type=int, default=None, help="default: 500")
    args = parser.parse_args()

    config = scene_config(args.scene, args.data_root, args.config, args.alignment)
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.iterations:
        config.num_iterations = args.iterations
    train(config)


if __name__ == '__main__':
    main()
