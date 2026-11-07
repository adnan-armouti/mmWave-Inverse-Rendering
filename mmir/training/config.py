"""Training configuration: loss and training settings, with the paper's values as defaults."""

import os
import re
import json
from typing import List, Optional
from dataclasses import dataclass, field

from mmir.preprocessing.alignment.cascaded_alignment import get_aligned_config_path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "data")
DEFAULT_OUTPUT_ROOT = os.path.join(REPO_ROOT, "output", "training")


@dataclass
class LossConfig:
    """Loss settings. The paper trains on the min-max normalized RA magnitude (no log compression)."""
    # Loss weights (0 = disabled)
    adc_mag_weight: float = 0.0              # ADC magnitude loss
    adc_phase_weight: float = 0.0            # ADC phase loss
    ra_mag_weight: float = 1.0               # RA magnitude loss (the training loss)
    ra_mag_ssim_weight: float = 0.0          # SSIM on RA
    ra_mag_perceptual_weight: float = 0.0    # LPIPS on RA

    # ADC magnitude loss settings
    adc_mag_use_log: bool = False
    adc_mag_loss_type: str = "l2"

    # ADC phase loss settings
    adc_phase_loss_type: str = "none"

    # RA magnitude loss settings
    ra_mag_use_log: bool = False             # log-compress before the loss
    ra_mag_loss_type: str = "l2"

    # Log-compression parameters
    log_epsilon: float = 1e-6
    log_scale: float = 1.0

    # Phase loss parameters
    phase_magnitude_weighting: bool = True
    phase_magnitude_power: float = 0.5
    phase_reference_channel: int = 0
    phase_use_gt_magnitude: bool = True
    phase_magnitude_clip_max: float = 0.0

    # Normalization mode
    use_joint_normalization: bool = False

    # Multi-chirp loss
    use_multi_chirp_loss: bool = False
    multi_chirp_phase_weight: float = 0.0
    multi_chirp_top_percentile: float = 0.9


@dataclass
class TrainingConfig:
    """Training settings. Defaults are the settings behind the paper's results."""

    # Files & Paths
    config_file: str = ""
    scene_file: str = ""
    gt_adc_file: str = ""
    output_dir: str = ""
    tx_pattern_file: str = ""
    rx_pattern_file: str = ""

    # Renderer
    n_hits_per_rx: int = 1500
    n_rays_per_res: int = 16
    bsdf_model: str = "mmwave_jones"
    mmwave_polarization: str = "vertical"
    hemisphere_sampling: str = "cosine"
    double_sided: bool = True
    use_pattern_importance_sampling: bool = False

    # Specular Path Finding
    enable_sms: bool = True
    sms_max_iterations: int = 20
    sms_solver_threshold: float = 1e-5
    sms_use_smooth_normals: bool = False
    sms_cache_interval: int = 0  # 0=recompute every iter, N=cache for N iters
    enable_diffraction: bool = True
    use_patch_clustering: bool = False
    patch_angle_threshold_deg: float = 10.0
    patch_distance_threshold: float = 0.05
    patch_max_tris: int = 2000
    use_spatial_adjacency: bool = False
    spatial_radius: float = 0.05

    # End-to-End Differentiable & Multibounce
    use_end_to_end_ad: bool = True
    max_bounces: int = 3
    nee_every_bounce: bool = True
    rr_start_bounce: int = 3
    rr_prob: float = 0.5
    e2e_ray_seed_rotation_interval: int = 9999
    e2e_phase_chunk_size: int = 64
    enable_grad_phase: bool = False
    enable_boundary_gradients: bool = True

    # Physics Material Model
    physics_mode: bool = True
    param_mode: str = "per_vertex"
    initial_material: str = "concrete"
    enable_error_map_init: bool = True
    material_columns: int = 6

    # Material bounds
    # Optional raw-space clamps applied after each Adam step (physics = exp(clamp(raw))).
    # None leaves the column unclamped, as in the paper's outdoor results. The paper's
    # indoor results used eps'' <= exp(4.6), sigma_h <= exp(-2.3) and thickness <= exp(-0.7).
    eps_imag_raw_upper: Optional[float] = None
    sigma_h_raw_upper: Optional[float] = None
    thickness_raw_upper: Optional[float] = None

    # Optimization
    num_iterations: int = 500
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    # Per-Group Learning Rates
    lr_materials: float = 0.5
    lr_pose_rotation: float = 5e-3
    lr_pose_translation: float = 5e-4
    lr_normals: float = 0.01
    lr_vertex_positions: float = 5e-4
    lr_patterns: float = 0.05

    # Gradient Clipping
    clip_materials: float = 1.0
    clip_pose_rotation: float = 0.5
    clip_pose_translation: float = 0.5
    clip_normals: float = 0.5
    clip_vertex_positions: float = 0.1
    clip_patterns: float = 1.0

    # Parameter Learning Flags
    LEARN_MAT: bool = True
    LEARN_POSE: bool = False
    LEARN_NRM: bool = True
    LEARN_VTX: bool = False
    LEARN_PAT: bool = True

    # Staged Training
    enable_staged_training: bool = False
    stage1_iterations: int = 50

    # LR Scheduling
    enable_lr_schedule: bool = True
    lr_warmup_iters: int = 5
    lr_warmup_factor: float = 0.3

    # Post-Step Constraints
    pose_rotation_max_rad: float = 0.0349
    pose_translation_max_m: float = 1.95e-3
    geom_laplacian_weight: float = 0.3

    # Pose Initialization
    init_pitch_deg: float = 0.0
    init_roll_deg: float = 0.0
    init_yaw_deg: float = 0.0
    init_translation_x: float = 0.0
    init_translation_y: float = 0.0
    init_translation_z: float = 0.0

    # Loss Configuration
    loss_config: LossConfig = field(default_factory=LossConfig)

    # Visualization
    visualization_interval: int = 99999
    checkpoint_interval: int = 99999
    log_interval: int = 10
    viz_scale: str = "dB"
    viz_db_floor: float = -40.0

    # Convergence
    target_correlation: float = 0.90
    target_mse: float = 1e-4
    enable_early_stopping: bool = False

    # Antenna Patterns
    pattern_mode: str = "shared"

    # Checkpoint Loading
    init_checkpoint: str = ""

    # Computed
    frozen_params: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Measured MMWCAS antenna patterns shipped with the repository
        if not self.tx_pattern_file:
            self.tx_pattern_file = os.path.join(REPO_ROOT, "assets/antenna_pattern/MMWCAS/tx1_76.npy")
            self.rx_pattern_file = os.path.join(REPO_ROOT, "assets/antenna_pattern/MMWCAS/rx1_76.npy")

        # Physics mode always uses 6 columns
        if self.physics_mode:
            self.material_columns = 6

        # Compute frozen_params from LEARN_* flags
        if not self.frozen_params:
            self.frozen_params = []
            if not self.LEARN_POSE:
                self.frozen_params.extend(['pose_rotation', 'pose_translation'])
            if not self.LEARN_VTX:
                self.frozen_params.append('vertex_positions')


def load_config_from_json(json_path: str) -> TrainingConfig:
    """Load config from JSON file."""
    with open(json_path, 'r') as f:
        config_dict = json.load(f)

    config = TrainingConfig()

    for key, value in config_dict.items():
        if hasattr(config, key):
            if key == 'loss_config' and isinstance(value, dict):
                loss_cfg = LossConfig()
                for lk, lv in value.items():
                    if hasattr(loss_cfg, lk):
                        setattr(loss_cfg, lk, lv)
                config.loss_config = loss_cfg
            else:
                setattr(config, key, value)

    # Reset frozen_params so __post_init__ recomputes from LEARN_* flags
    config.frozen_params = []
    config.__post_init__()
    return config


def scene_config(scene: str, data_root: str = DEFAULT_DATA_ROOT,
                 config_json: str = None, alignment: str = "paper") -> TrainingConfig:
    """Paper settings for ``scene`` (``seq_<s>_frame_<F>``), with the scene's files under
    ``data_root``. Values in ``config_json`` override the defaults. ``alignment`` picks the
    aligned cascade config: ``"paper"`` (shipped in assets/alignment/) or ``"data"`` (regenerated
    into data/<scene>/configs/).
    """
    match = re.fullmatch(r"seq_\d+_frame_(\d+)", scene)
    if match is None:
        raise SystemExit(f"Scene name {scene!r} must look like seq_<s>_frame_<F>.")
    frame = match.group(1)
    config = load_config_from_json(config_json) if config_json else TrainingConfig()
    scene_dir = os.path.join(data_root, scene)
    # Each scene trains on the configuration that won its alignment (LiDAR 4-DOF or renderer 2-DOF)
    config.config_file = get_aligned_config_path(scene, int(frame), data_root, alignment)
    config.scene_file = os.path.join(scene_dir, "scene", "mesh.ply")
    config.gt_adc_file = os.path.join(scene_dir, "radar", f"cascaded_frame_{frame}.npy")
    config.output_dir = config.output_dir or os.path.join(DEFAULT_OUTPUT_ROOT, scene)
    missing = [p for p in (config.config_file, config.scene_file, config.gt_adc_file) if not os.path.isfile(p)]
    if missing:
        raise SystemExit("Scene data not found: " + ", ".join(missing)
                         + ". Set --data_root, or see data/README.md.")
    return config
