"""Scene context and logging utilities for the FMCW renderer."""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import json

import drjit as dr
import mitsuba as mi

if TYPE_CHECKING:
    from ..sensor.config import FMCWConfig, TxArray, RxArray
    from ..sensor.element_patterns import AntennaPatternLoader
    from .scene_params.vertex_attributes import VertexPositions, VertexNormals


@dataclass
class SceneContext:
    """Immutable scene context holding all data required for rendering."""
    # Mitsuba scene
    scene: 'mi.Scene'

    # FMCW configuration
    config: 'FMCWConfig'

    # Antenna arrays
    tx_array: 'TxArray'
    rx_array: 'RxArray'

    # Differentiable geometry (optional)
    vertex_positions: Optional['VertexPositions'] = None
    vertex_normals: Optional['VertexNormals'] = None

    # Antenna patterns (optional)
    tx_pattern: Optional['AntennaPatternLoader'] = None
    rx_pattern: Optional['AntennaPatternLoader'] = None

    # Cached values
    epsilon: float = 1e-4

    @property
    def n_tx(self) -> int:
        """Number of TX elements."""
        return self.tx_array.num_elements

    @property
    def n_rx(self) -> int:
        """Number of RX elements."""
        return self.rx_array.num_elements

    @property
    def wavelength(self) -> float:
        """Wavelength at center frequency."""
        return self.config.c / self.config.center_freq

    @classmethod
    def from_files(cls,
                   config_file: str,
                   scene_file: str,
                   pattern_file: Optional[str] = None,
                   tx_pattern_file: Optional[str] = None,
                   rx_pattern_file: Optional[str] = None,
                   edge_file: Optional[str] = None,
                   material_type: str = "metal",
                   enable_gradients: bool = True,
                   verbose: bool = True) -> 'SceneContext':
        """Load SceneContext from files."""
        from ..sensor.config import FMCWConfig, TxArray, RxArray
        from ..sensor.element_patterns import AntennaPatternLoader
        from .scene_params.vertex_attributes import VertexPositions, VertexNormals

        if verbose:
            print("=" * 80)
            print("[SceneContext] Loading from files")
            print("=" * 80)

        # Step 1: Load FMCW configuration
        if verbose:
            print(f"\n[Step 1/6] Loading FMCW configuration from {config_file}")
        config = FMCWConfig.from_json(config_file)
        if verbose:
            print(f"  Center frequency: {config.center_freq/1e9:.2f} GHz")
            print(f"  Bandwidth: {config.bandwidth/1e9:.2f} GHz")
            print(f"  ADC samples: {config.num_adc_samples}")

        # Step 2: Load antenna patterns
        if verbose:
            print("\n[Step 2/6] Loading antenna patterns")

        if tx_pattern_file is None or rx_pattern_file is None:
            raise ValueError(
                "If using separate patterns, must specify both tx_pattern_file and rx_pattern_file"
            )
        tx_pattern_path = tx_pattern_file
        rx_pattern_path = rx_pattern_file
        if verbose:
            print(f"  TX pattern: {tx_pattern_file}")
            print(f"  RX pattern: {rx_pattern_file}")

        tx_pattern = AntennaPatternLoader(tx_pattern_path)
        rx_pattern = AntennaPatternLoader(rx_pattern_path)

        # Step 3: Create TX/RX arrays
        if verbose:
            print("\n[Step 3/6] Creating TX/RX arrays")

        with open(config_file, 'r') as f:
            config_data = json.load(f)

        tx_array = TxArray.from_config(config_data)
        rx_array = RxArray.from_config(config_data)

        # Store pattern data in arrays
        tx_array.pattern_data = tx_pattern.pattern_data
        rx_array.pattern_data = rx_pattern.pattern_data
        tx_array.pattern_loader = tx_pattern
        rx_array.pattern_loader = rx_pattern

        if verbose:
            print(f"  TX array: {tx_array.num_elements} elements")
            print(f"  RX array: {rx_array.num_elements} elements")

        # Step 4: Load scene geometry
        if verbose:
            print(f"\n[Step 4/6] Loading scene from {scene_file}")

        vertex_positions = VertexPositions.from_ply(scene_file, enable_grad=enable_gradients)

        # Step 5: Load vertex normals
        if verbose:
            print("\n[Step 5/6] Loading vertex normals")
        vertex_normals = VertexNormals.from_ply(scene_file, enable_grad=enable_gradients)


        # Build Mitsuba scene from PLY file
        scene = cls._build_mitsuba_scene(scene_file)

        # Compute scene epsilon
        bbox = scene.bbox()
        diagonal = float(dr.norm(bbox.max - bbox.min))
        epsilon = diagonal * 1e-5
        epsilon = max(1e-6, min(epsilon, 1e-2))

        if verbose:
            print("\n[SceneContext] Initialization complete")
            print(f"  Scene bounding box diagonal: {diagonal:.2f} m")
            print(f"  Shadow ray epsilon: {epsilon:.2e} m")

        return cls(
            scene=scene,
            config=config,
            tx_array=tx_array,
            rx_array=rx_array,
            vertex_positions=vertex_positions,
            vertex_normals=vertex_normals,
            tx_pattern=tx_pattern,
            rx_pattern=rx_pattern,
            epsilon=epsilon
        )

    @staticmethod
    def _build_mitsuba_scene(scene_file: str) -> 'mi.Scene':
        """Build Mitsuba scene from PLY file."""
        scene_dict = {
            'type': 'scene',
            'mesh': {
                'type': 'ply',
                'filename': scene_file,
                'bsdf': {
                    'type': 'conductor',
                    'material': 'Al'
                }
            }
        }
        scene = mi.load_dict(scene_dict)
        return scene


class RenderLogger:
    """Hierarchical logging with toggleable verbosity."""

    def __init__(self, verbosity: int = 0):
        self.verbosity = verbosity
        self._indent = 0
