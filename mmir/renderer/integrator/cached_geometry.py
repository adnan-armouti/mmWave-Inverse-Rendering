"""Cached geometry dataclasses for two-phase differentiable rendering."""

from dataclasses import dataclass
import numpy as np


@dataclass
class ADCResult:
    """Result of ADC synthesis."""
    adc_real: np.ndarray
    adc_imag: np.ndarray
    n_tx: int
    n_rx: int
    n_samples: int
    total_paths: int

    def get_ri_array(self) -> np.ndarray:
        """Get real/imag stacked as [n_tx, n_rx, n_samples, 2]."""
        return np.stack([self.adc_real, self.adc_imag], axis=-1)
