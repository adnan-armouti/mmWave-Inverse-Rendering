"""Configuration classes for FMCW ADC accumulation."""

import json
from dataclasses import dataclass
from typing import Optional
import numpy as np
import drjit as dr
import mitsuba as mi


@dataclass
class FMCWConfig:
    """FMCW radar and SBR simulation configuration"""

    # FMCW parameters
    center_freq: float          # f0: Center frequency (Hz)
    bandwidth: float            # Chirp bandwidth (Hz)
    chirp_slope: float          # S: Chirp slope (Hz/s)
    chirp_duration: float       # Chirp duration (seconds)

    # ADC parameters
    num_adc_samples: int        # K: Number of ADC samples per chirp
    adc_sample_rate: float      # Samples per second

    # RX array
    num_rx: int                 # Number of RX elements

    # Optional parameters with defaults
    adc_start_time: float = 0.0 # ADC start time (seconds) - defines minimum range

    # SBR parameters
    max_depth: int = 10         # Maximum bounce depth
    rr_depth: int = 3           # Russian roulette starts at this depth
    rr_threshold: float = 0.1   # Threshold for RR probability computation
    samples_per_tx: int = 1000000  # Rays launched per TX
    min_gain: float = 1e-10     # Minimum E-field magnitude to continue

    # Sampling allocation (for mixed cosine + edge sampling)
    edge_allocation_fraction: float = 0.2   # Fraction of samples for edge diffraction (0.1-0.4)
    min_edge_samples: int = 50              # Minimum edge samples regardless of allocation
    min_cosine_samples: int = 50            # Minimum surface samples regardless of allocation

    # Russian Roulette parameters
    rr_max_prob: float = 0.95   # Maximum RR survival probability (caps throughput)

    # Numerical stability
    shadow_ray_epsilon: float = 1e-6  # Absolute offset (meters) to prevent self-intersection

    # Material interaction
    reset_tube_on_diffuse: bool = True  # Reset tube weight for diffuse (Sionna-RT style)

    # Physical constants
    c: float = 299792458.0      # Speed of light (m/s)

    # TX Phase Coherence (toggleable)
    # Controls whether TX elements trace the same paths or different paths:
    # - True (default): All TXs trace SAME paths (shared ray directions) -> full phase coherence
    # - False: Each TX traces DIFFERENT paths (independent rays) -> RX coherence only
    # Note: RX phase coherence is ALWAYS enabled (all paths evaluated by all RXs)
    use_coherent_tx: bool = True

    # Path loss
    enable_path_loss: bool = True  # Apply free-space path loss (1/R decay) to E-field for physically-based rendering
    path_loss_exponent: float = 2.0  # Exponent for path loss: 1=E-field decay (1/d), 2=power decay (1/d^2)

    @property
    def wavelength(self) -> float:
        """Compute wavelength from center frequency: lambda = c / f0"""
        return self.c / self.center_freq

    @classmethod
    def from_json(cls, config_path: str) -> 'FMCWConfig':
        """Load configuration from JSON file (cascaded_frame_*.json style)"""
        with open(config_path, 'r') as f:
            data = json.load(f)

        # Handle different JSON field names
        # carrier frequency
        center_freq = (data.get('carrierFrequency') or
                      data.get('center_freq') or
                      data.get('start_freq'))
        if center_freq is None:
            raise ValueError("Config must specify 'carrierFrequency', 'center_freq', or 'start_freq'")

        # chirp slope
        chirp_slope = (data.get('freqSlope') or
                      data.get('chirp_slope') or
                      data.get('freq_slope'))
        if chirp_slope is None:
            raise ValueError("Config must specify 'freqSlope' or 'chirp_slope'")

        # ADC samples
        num_adc_samples = (data.get('numAdcSamples') or
                          data.get('num_adc_samples'))
        if num_adc_samples is None:
            raise ValueError("Config must specify 'numAdcSamples' or 'num_adc_samples'")

        # Sample rate
        sample_rate = (data.get('sampleRate') or
                      data.get('adc_sample_rate') or
                      data.get('sample_rate'))

        # Compute derived parameters
        # Chirp duration from rampEndTime or compute from other params
        if 'rampEndTime' in data:
            chirp_duration = data['rampEndTime']
        elif 'chirp_duration' in data:
            chirp_duration = data['chirp_duration']
        else:
            # Compute from sample rate and num samples
            if sample_rate is None:
                raise ValueError("Cannot determine chirp duration")
            chirp_duration = num_adc_samples / sample_rate

        # Bandwidth from slope and duration
        bandwidth = chirp_slope * chirp_duration

        # Number of RX elements
        if 'rx_array' in data:
            num_rx = len(data['rx_array'])
        elif 'num_rx' in data:
            num_rx = data['num_rx']
        else:
            raise ValueError("Config must specify 'rx_array' or 'num_rx'")

        # ADC start time (defines minimum range)
        adc_start_time = (data.get('adcStartTime') or
                          data.get('adc_start_time') or
                          0.0)

        return cls(
            center_freq=center_freq,
            bandwidth=bandwidth,
            chirp_slope=chirp_slope,
            chirp_duration=chirp_duration,
            num_adc_samples=num_adc_samples,
            adc_sample_rate=sample_rate,
            adc_start_time=adc_start_time,
            num_rx=num_rx,
            max_depth=data.get('max_depth', 10),
            rr_depth=data.get('rr_depth', 3),
            samples_per_tx=data.get('samples_per_tx', 1000000),
            min_gain=data.get('min_gain', 1e-10),
        )


@dataclass
class AntennaArray:
    """Positions (m) and boresight directions of the TX or RX elements of a radar config."""

    positions: 'mi.Point3f'      # [N, 3]
    orientations: 'mi.Vector3f'  # [N, 3] boresight unit vectors
    pattern_data: Optional[np.ndarray] = None
    pattern_loader: Optional[object] = None

    KEY = None  # 'tx_array' or 'rx_array'

    @property
    def num_elements(self) -> int:
        return dr.width(self.positions.x)

    @classmethod
    def from_config(cls, config_data: dict) -> 'AntennaArray':
        """Load the elements listed under ``cls.KEY`` of a radar config dictionary."""
        if cls.KEY not in config_data:
            raise ValueError(f"Config must specify '{cls.KEY}'")
        positions, orientations = [], []
        for elem in config_data[cls.KEY]:
            if 'pos_mm' in elem:
                positions.append(np.array(elem['pos_mm'], dtype=np.float32) / 1000.0)
            elif 'pos' in elem:
                positions.append(np.array(elem['pos'], dtype=np.float32))
            else:
                raise ValueError(f"{cls.KEY} element must have 'pos_mm' or 'pos'")
            boresight = elem.get('boresight', elem.get('orientation', [0, 1, 0]))
            orientations.append(np.array(boresight, dtype=np.float32))
        # Mitsuba expects (3, N)
        return cls(positions=mi.Point3f(np.array(positions, dtype=np.float32).T),
                   orientations=mi.Vector3f(np.array(orientations, dtype=np.float32).T))


class TxArray(AntennaArray):
    KEY = 'tx_array'


class RxArray(AntennaArray):
    KEY = 'rx_array'

