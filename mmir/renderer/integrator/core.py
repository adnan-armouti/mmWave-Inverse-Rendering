"""SBR Integrator core: class definition, initialization, and radar constants."""

from typing import Optional, TYPE_CHECKING
import mitsuba as mi
import numpy as np

from ..bsdf.mmwave_jones import BSDFmmWaveJones
from ..materials.parameterization import MaterialParameterization

# Import diffraction modules (optional)
try:
    _HAS_DIFFRACTION = True
except ImportError:
    _HAS_DIFFRACTION = False

if TYPE_CHECKING:
    from mmir.sensor.config import FMCWConfig
    from ..config import RenderConfigRef

# Physical Constants
C = 299792458.0  # Speed of light (m/s)
K_BOLTZMANN = 1.38e-23  # Boltzmann constant (J/K)
T_SYSTEM = 290.0  # System temperature (K)


class SBRIntegratorRef:
    """RRTS-exact FMCW ADC synthesis integrator."""

    def __init__(
        self,
        config: 'FMCWConfig',
        render_config: 'RenderConfigRef' = None,
    ):
        """Initialize integrator."""
        self.config = config

        # Import RenderConfigRef here to avoid circular import
        from ..config import RenderConfigRef
        self.render_config = render_config if render_config is not None else RenderConfigRef()

        # Extract FMCW parameters
        self.center_freq = config.center_freq
        self.bandwidth = config.bandwidth
        self.chirp_duration = config.chirp_duration
        self.num_samples = config.num_adc_samples
        self.sample_rate = config.adc_sample_rate

        # Derived parameters
        # CRITICAL: carrierFrequency IS the start frequency (f0) in FMCW
        # RRTS uses MIN_FREQ = carrierFrequency directly (77 GHz)
        # The chirp sweeps from f0 to f0 + bandwidth
        self.min_freq = self.center_freq  # f0 = carrierFrequency (NOT center - BW/2!)
        self.slope = config.chirp_slope   # S = freqSlope from config directly

        # Wavelength computation
        self.wavelength = C / self.center_freq  # λ in meters
        self.lambda_squared = self.wavelength ** 2

        # CRITICAL: Time grid uses k/sample_rate, NOT linspace!
        # This is a key difference from mmIR
        self._build_time_grid()

        # Precompute radar equation constants if enabled
        if self.render_config.use_radar_equation:
            self._precompute_radar_constants()

        # Initialize BSDF evaluator based on config
        # All BRDF evaluation goes through this class
        bsdf_model = getattr(self.render_config, 'bsdf_model', 'rrts')

        # Read new material model config options
        getattr(self.render_config, 'enable_slab_fresnel', True)
        getattr(self.render_config, 'default_thickness', 0.1)
        enable_cbs = getattr(self.render_config, 'enable_cbs', True)

        if bsdf_model == 'mmwave_jones':
            # Polarization-enabled Jones BSDF with complex Fresnel (magnitude only)
            mmwave_pol = getattr(self.render_config, 'mmwave_polarization', 'vertical')
            if mmwave_pol == 'vertical':
                tx_pol = mi.Vector3f(0, 0, 1)
                rx_pol = mi.Vector3f(0, 0, 1)
            elif mmwave_pol == 'horizontal':
                tx_pol = mi.Vector3f(1, 0, 0)
                rx_pol = mi.Vector3f(1, 0, 0)
            else:
                tx_pol = mi.Vector3f(0, 0, 1)
                rx_pol = mi.Vector3f(0, 0, 1)
            self.bsdf = BSDFmmWaveJones(
                tx_polarization=tx_pol,
                rx_polarization=rx_pol,
                enable_incoherent=True,
                use_fresnel_phase=False,
                enable_cbs=enable_cbs,
            )
        elif bsdf_model == 'mmwave_jones_phase':
            # Polarization-enabled Jones BSDF WITH Fresnel phase in coherent term
            mmwave_pol = getattr(self.render_config, 'mmwave_polarization', 'vertical')
            if mmwave_pol == 'vertical':
                tx_pol = mi.Vector3f(0, 0, 1)
                rx_pol = mi.Vector3f(0, 0, 1)
            elif mmwave_pol == 'horizontal':
                tx_pol = mi.Vector3f(1, 0, 0)
                rx_pol = mi.Vector3f(1, 0, 0)
            else:
                tx_pol = mi.Vector3f(0, 0, 1)
                rx_pol = mi.Vector3f(0, 0, 1)
            self.bsdf = BSDFmmWaveJones(
                tx_polarization=tx_pol,
                rx_polarization=rx_pol,
                enable_incoherent=True,
                use_fresnel_phase=True,
                enable_cbs=enable_cbs,
            )
        else:
            raise ValueError(f"Unknown bsdf_model: '{bsdf_model}'. "
                           "Supported: 'mmwave_scalar', 'mmwave_jones', 'mmwave_jones_phase'")

        # Material parameterization strategy (default: per-triangle)
        self.material_param: Optional[MaterialParameterization] = None

        # Free-space diffraction BSDF (initialized lazily when triangle hash is set)
        self.fsd_bsdf = None
        self.tri_hash = None
        # FSD CDF tables for importance sampling (set by renderer when direction_sampling='bsdf')
        self.diffraction_config = getattr(self.render_config, 'diffraction_config', None)

        # Per-triangle material arrays for material-dependent diffraction (Phase A).
        # Set by renderer from triangle_materials when diffraction_config.material_opacity is True.
        # Shape: (eps_real_np [n_faces], eps_imag_np [n_faces]) or (None, None).
        self.diffraction_tri_eps: tuple = (None, None)

        # GPU-resident spatial hash data (set by renderer when use_gpu_apertures=True)
        self._gpu_spatial_hash = None

        # Per-triangle material array for Phase A specular BSDF evaluation.
        # Set by renderer. Shape: [n_triangles, 6] for physics mode, None otherwise.
        self.triangle_materials: Optional[np.ndarray] = None

        # GPU-resident material columns (lazy-cached from triangle_materials)
        self._gpu_material_cols: Optional[list] = None
        self._gpu_material_src_id: Optional[int] = None

    def _precompute_radar_constants(self):
        """Precompute constants for radar equation."""
        rc = self.render_config

        # Convert TX power from dBm to Watts: P_t = 10^(Pt_dBm/10) / 1000
        self.Pt_watts = 10 ** (rc.Pt_dBm / 10) / 1000.0

        # Antenna loss (linear): L = 10^(-loss_dB/10) (loss is subtracted, so negative)
        self.antenna_loss_linear = 10 ** (-rc.antenna_loss_dB / 10)

        # (4π)² normalization factor for RX-centric Monte Carlo estimator
        # See rx_centric_radar_mc_estimators.tex Eq. 4:
        # P_r = (P_t λ²)/(4π)² × integral term
        # NOT (4π)³ - the RX solid-angle sampling absorbs one factor of 4π
        self.four_pi_squared = (4 * np.pi) ** 2

        # RX gain and dBFS scaling factor (amplitude domain uses /20, not /10)
        self.rx_dBFS_scale = 10 ** ((rc.rx_gain_dB - rc.dBm_to_dBFS_offset) / 20)

        # ADC quantization scale: converts E-field from sqrt(Watts) to ADC counts
        # E-field in sqrt(W) needs to be converted to voltage: V = sqrt(P * Z_0) = E * sqrt(Z_0)
        # Then voltage to ADC counts: ADC = V * (ADC_full_scale / V_full_scale)
        # For TI radars, V_full_scale ≈ 1V corresponds to 0 dBFS = 13 dBm
        self.adc_scale = np.sqrt(rc.system_impedance) * rc.adc_full_scale

        # Monte Carlo normalization factor for solid angle sampling
        # This accounts for the hemisphere integral and sampling PDF compensation
        # NOTE: This factor is calibrated for n_hits_per_rx = 500 (the default).
        # For coherent phasor summation with deterministic phases, signal scales
        # approximately with sqrt(N), so using different n_hits_per_rx will affect
        # the absolute magnitude.
        self.mc_normalization = rc.mc_normalization

        # Combined constant: sqrt(Pt * L_ant * λ² * mc_norm) / sqrt((4π)²)
        # For RX-centric MC estimator: P_r ∝ (Pt λ²)/(4π)² × [G_t G_r c_t f V / d_t²]
        # The mc_normalization factor provides calibration for the sampling geometry
        self.radar_constant = np.sqrt(
            self.Pt_watts * self.antenna_loss_linear * self.lambda_squared *
            self.mc_normalization / self.four_pi_squared
        )

    def _build_time_grid(self):
        """Build time grid for phase computation."""
        K = self.num_samples

        # RRTS time grid: t[k] = k / sample_rate
        self.t_grid = np.arange(K) / self.sample_rate
        self._t_grid_dr = mi.Float(self.t_grid)  # Cache on GPU once
