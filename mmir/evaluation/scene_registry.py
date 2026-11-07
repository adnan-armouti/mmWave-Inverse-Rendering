"""Scene registry: the paper scenes, their files, and exclusions."""

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Project root — all paths relative to this
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

EXCLUDED_SCENES = {"seq_1_frame_277", "seq_0_frame_451"}

# Trained scenes are read from <OUR_TRAINING_ROOT>/<scene>/
OUR_TRAINING_ROOT = os.path.join(OUTPUT_DIR, "training")

# Aligned cascade configs: "paper" (assets/alignment/) or "data" (data/<scene>/configs/)
ALIGNMENT = "paper"


@dataclass
class SceneInfo:
    """All paths and metadata for one scene."""

    seq: int
    frame: int
    name: str  # e.g. "seq_1_frame_185"
    data_dir: str  # e.g. "<project>/data/seq_1_frame_185"

    # Configs
    cascaded_config: str  # aligned cascaded config JSON
    single_chip_configs: List[str] = field(default_factory=list)
    dense_config: Optional[str] = None

    # Scene geometry
    mesh_file: str = ""

    # Ground truth
    gt_cascaded_adc: str = ""
    gt_single_chip_adcs: Dict[int, str] = field(default_factory=dict)  # {frame_num: path}
    lidar_pcl: str = ""

    # Training outputs
    our_training_dir: str = ""

    def get_closest_single_chip_frame(self, target_frame: Optional[int] = None) -> Optional[int]:
        """Return the single-chip frame number closest to target (default: cascaded_frame * 2)."""
        if not self.gt_single_chip_adcs:
            return None
        if target_frame is None:
            target_frame = self.frame * 2
        return min(self.gt_single_chip_adcs.keys(), key=lambda f: abs(f - target_frame))

    def get_single_chip_gt_adc(self, sc_frame: int) -> Optional[str]:
        """Return GT single-chip ADC path for a given frame number."""
        return self.gt_single_chip_adcs.get(sc_frame)


def _discover_scene(scene_dir: str) -> Optional[SceneInfo]:
    """Discover a scene from its data directory structure."""
    name = os.path.basename(scene_dir)
    match = re.match(r"seq_(\d+)_frame_(\d+)", name)
    if not match:
        return None

    seq, frame = int(match.group(1)), int(match.group(2))
    configs_dir = os.path.join(scene_dir, "configs")
    radar_dir = os.path.join(scene_dir, "radar")
    scene_subdir = os.path.join(scene_dir, "scene")

    our_training_dir = os.path.join(OUR_TRAINING_ROOT, name)
    # Same radar pose as training: the scene's winning alignment (LiDAR 4-DOF or renderer 2-DOF)
    from mmir.preprocessing.alignment.cascaded_alignment import get_aligned_config_path
    cascaded_config = get_aligned_config_path(name, frame, DATA_DIR, ALIGNMENT)

    mesh_file = os.path.join(scene_subdir, "mesh.ply")

    # Single-chip configs
    sc_configs = sorted(glob.glob(os.path.join(configs_dir, "single_chip_frame_*.json")))

    # GT cascaded ADC
    gt_cascaded = os.path.join(radar_dir, f"cascaded_frame_{frame}.npy")

    # GT single-chip ADCs: extract frame numbers from filenames
    gt_sc_adcs = {}
    for f_path in sorted(glob.glob(os.path.join(radar_dir, "single_chip_frame_*.npy"))):
        m = re.search(r"single_chip_frame_(\d+)\.npy", f_path)
        if m:
            gt_sc_adcs[int(m.group(1))] = f_path

    # LiDAR point cloud
    lidar_pcl = os.path.join(scene_subdir, "pcl.npy")

    return SceneInfo(
        seq=seq,
        frame=frame,
        name=name,
        data_dir=scene_dir,
        cascaded_config=cascaded_config,
        single_chip_configs=sc_configs,
        mesh_file=mesh_file,
        gt_cascaded_adc=gt_cascaded,
        gt_single_chip_adcs=gt_sc_adcs,
        lidar_pcl=lidar_pcl,
        our_training_dir=our_training_dir if os.path.isdir(our_training_dir) else "",
    )


def get_scene(name: str) -> Optional[SceneInfo]:
    """Return a single scene by name, or None if not found."""
    scene_dir = os.path.join(DATA_DIR, name)
    if os.path.isdir(scene_dir):
        return _discover_scene(scene_dir)
    return None


def configure(data_root: str = None, training_root: str = None, alignment: str = None) -> None:
    """Point the registry at a data folder, a folder of training runs and an alignment source."""
    global DATA_DIR, OUR_TRAINING_ROOT, ALIGNMENT
    if alignment:
        ALIGNMENT = alignment
    if data_root:
        DATA_DIR = os.path.abspath(data_root)
    if training_root:
        OUR_TRAINING_ROOT = os.path.abspath(training_root)
