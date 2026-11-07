"""ITU-R P.2040-4 material property database for mmWave/sub-THz propagation."""

import numpy as np

# CONSTANTS

EPSILON_0 = 8.854187817e-12   # Vacuum permittivity (F/m)
C0 = 299792458.0              # Speed of light (m/s)

# ITU-R P.2040-4 TABLE 3: MATERIAL PROPERTIES
#
# Model: eps_r = a * f_GHz^b,  sigma = c * f_GHz^d  (S/m)
#
# Each material maps to one or more frequency bands:
#
# Materials with multiple bands have distinct parameterizations per band.
# When querying, the band containing the requested frequency is selected.

ITU_MATERIAL_PROPERTIES = {

    # P.2040-3 materials (also in Sionna)

    'concrete': {
        (1.0, 100.0):   (5.24,   0.0,     0.0462,  0.7822),
        (110.0, 330.0):  (5.17,   0.0,     0.0145,  1.09),      # P.2040-4
    },
    'brick': {
        (1.0, 40.0):     (3.91,   0.0,     0.0238,  0.16),
        (110.0, 330.0):  (4.15,   0.0,     0.0006,  1.5712),    # P.2040-4
    },
    'plasterboard': {
        (1.0, 100.0):    (2.73,   0.0,     0.0085,  0.9395),
        (110.0, 330.0):  (2.56,   0.0,     0.0001,  1.7799),    # P.2040-4
    },
    'wood': {
        (0.001, 100.0):  (1.99,   0.0,     0.0047,  1.0718),
    },
    'glass': {
        (0.1, 100.0):    (6.31,   0.0,     0.0036,  1.3394),
        (220.0, 450.0):  (5.79,   0.0,     0.0004,  1.658),     # P.2040-4
    },
    'ceiling_board': {
        (1.0, 100.0):    (1.48,   0.0,     0.0011,  1.0750),
        (220.0, 450.0):  (1.52,   0.0,     0.0029,  1.029),     # P.2040-4
    },
    'chipboard': {
        (1.0, 100.0):    (2.58,   0.0,     0.0217,  0.7800),
        (100.0, 200.0):  (2.16,   0.0,     0.0023,  1.359),     # P.2040-4
    },
    'plywood': {
        (1.0, 40.0):     (2.71,   0.0,     0.33,    0.0),
    },
    'marble': {
        (1.0, 60.0):     (7.074,  0.0,     0.0055,  0.9262),
    },
    'floorboard': {
        (50.0, 100.0):   (3.66,   0.0,     0.0044,  1.3515),
    },
    'metal': {
        (1.0, 100.0):    (1.0,    0.0,     1e7,     0.0),
    },
    'very_dry_ground': {
        (1.0, 10.0):     (3.0,    0.0,     0.00015, 2.52),
    },
    'medium_dry_ground': {
        (1.0, 10.0):     (15.0,  -0.1,     0.035,   1.63),
    },
    'wet_ground': {
        (1.0, 10.0):     (30.0,  -0.4,     0.15,    1.30),
    },

    # P.2040-4 new materials

    'glass_alt': {
        # Alternative glass measurement, 100-400 GHz
        (100.0, 400.0):  (6.5767, 0.0,     0.0012,  1.4697),
    },
    'plasterboard_alt': {
        # Alternative plasterboard measurement, 100-400 GHz
        (100.0, 400.0):  (2.65,   0.0,     0.0002,  1.598),
    },
    'clear_acrylic': {
        (110.0, 330.0):  (2.58,   0.0,     0.0001,  1.6524),
    },
    'asphalt_concrete': {
        (1.0, 40.0):     (4.83,   0.0,     0.0108,  1.3969),
    },
    'vinyl_tile': {
        (1.0, 40.0):     (3.62,   0.0,     0.0051,  0.8422),
    },
    'carpet_tile': {
        (1.0, 40.0):     (2.08,   0.0,     0.0009,  0.82),
    },

    # Non-ITU practical materials for automotive mmWave

    'aluminum': {
        # Modeled as metal with very high conductivity
        (1.0, 100.0):    (1.0,    0.0,     3.538e7, 0.0),
    },
    'steel': {
        (1.0, 100.0):    (1.0,    0.0,     1.450e6, 0.0),
    },
    'painted_metal': {
        # Thin paint layer over metal; effective permittivity of paint
        (1.0, 100.0):    (3.0,    0.0,     0.01,    0.5),
    },
    'vegetation': {
        # Approximate; highly variable with moisture content
        (1.0, 100.0):    (1.5,    0.0,     0.05,    0.7),
    },
    'plastic_abs': {
        # ABS plastic (bumpers, housings)
        (1.0, 100.0):    (2.5,    0.0,     0.003,   0.8),
    },
}


# DEFAULT ROUGHNESS STATISTICS PER MATERIAL
#
# sigma_h: RMS surface height (meters)
# l_c: Correlation length (meters)
#
# Sources: Ulaby, Moore & Fung "Microwave Remote Sensing" Vol II;
#          Ruck et al. "Radar Cross Section Handbook";
#          Pinel & Bourlier "Electromagnetic Wave Scattering from Random
#          Rough Surfaces"

DEFAULT_ROUGHNESS = {
    'concrete':          (1.0e-3,     1.5e-2),     # Cast concrete, moderate roughness
    'brick':             (2.0e-3,     1.0e-2),     # Standard clay brick face
    'glass':             (2.0e-5,     3.0e-2),     # Window glass (very smooth)
    'glass_alt':         (2.0e-5,     3.0e-2),
    'wood':              (5.0e-4,     1.0e-2),     # Sawn lumber surface
    'plasterboard':      (1.5e-4,     8.0e-3),     # Painted drywall
    'plasterboard_alt':  (1.5e-4,     8.0e-3),
    'metal':             (5.0e-6,     5.0e-3),     # Sheet metal
    'aluminum':          (5.0e-6,     5.0e-3),     # Polished/rolled aluminum
    'steel':             (2.0e-5,     5.0e-3),     # Rolled steel
    'painted_metal':     (1.0e-5,     3.0e-3),     # Painted sheet metal
    'plywood':           (3.0e-4,     8.0e-3),     # Sanded plywood
    'chipboard':         (5.0e-4,     5.0e-3),     # Particle board
    'ceiling_board':     (2.0e-4,     1.0e-2),     # Acoustic tile
    'marble':            (5.0e-5,     2.0e-2),     # Polished marble
    'floorboard':        (4.0e-4,     1.2e-2),     # Hardwood flooring
    'asphalt_concrete':  (5.0e-3,     5.0e-2),     # Road surface
    'vinyl_tile':        (1.0e-4,     1.0e-2),     # Vinyl flooring
    'carpet_tile':       (3.0e-3,     2.0e-2),     # Carpet (fibrous)
    'clear_acrylic':     (1.0e-5,     5.0e-3),     # Polished acrylic
    'very_dry_ground':   (1.0e-2,     1.0e-1),     # Sandy soil
    'medium_dry_ground': (8.0e-3,     8.0e-2),     # Typical earth
    'wet_ground':        (5.0e-3,     5.0e-2),     # Muddy/wet soil
    'vegetation':        (5.0e-2,     1.0e-1),     # Canopy (very rough/diffuse)
    'plastic_abs':       (5.0e-6,     2.0e-3),     # Injection-molded ABS
}


# DEFAULT SLAB THICKNESSES
#
# Used by the ITU single-layer slab Fresnel model when no explicit thickness
# is given for a material.

DEFAULT_THICKNESS = {
    'concrete':          0.15,        # Typical wall/slab
    'brick':             0.10,        # Single wythe
    'glass':             0.006,       # Window pane
    'glass_alt':         0.006,
    'wood':              0.02,        # Typical panel
    'plasterboard':      0.012,       # Standard drywall sheet
    'plasterboard_alt':  0.012,
    'metal':             0.003,       # Sheet metal
    'aluminum':          0.003,
    'steel':             0.003,
    'painted_metal':     0.003,
    'plywood':           0.012,       # Standard sheet
    'chipboard':         0.018,       # Particle board panel
    'ceiling_board':     0.015,       # Acoustic tile
    'marble':            0.02,        # Marble slab/tile
    'floorboard':        0.02,        # Hardwood flooring
    'asphalt_concrete':  0.05,        # Road surface layer
    'vinyl_tile':        0.003,       # Vinyl flooring tile
    'carpet_tile':       0.01,        # Carpet + backing
    'clear_acrylic':     0.005,       # Acrylic sheet
    'very_dry_ground':   0.5,         # Ground (effectively infinite)
    'medium_dry_ground': 0.5,
    'wet_ground':        0.5,
    'vegetation':        0.1,         # Approximate canopy depth
    'plastic_abs':       0.003,       # Bumper/housing thickness
}

# Default thickness for unknown materials
DEFAULT_THICKNESS_FALLBACK = 0.1  # 10 cm


# MULTI-LAYER WALL PRESETS
#
# Each preset is a list of (material_name, thickness_m) tuples, ordered from
# the exterior (incident) side to the interior (transmitted) side.

WALL_PRESETS = {
    'exterior_brick_wall': [
        ('brick', 0.10),
        ('air', 0.09),
        ('plasterboard', 0.012),
    ],
    'interior_wall': [
        ('plasterboard', 0.012),
        ('air', 0.09),
        ('plasterboard', 0.012),
    ],
    'double_glazing': [
        ('glass', 0.006),
        ('air', 0.012),
        ('glass', 0.006),
    ],
    'insulated_concrete_wall': [
        ('plasterboard', 0.012),
        ('air', 0.09),
        ('concrete', 0.15),
    ],
    'concrete_wall': [
        ('concrete', 0.15),
    ],
}

# Air properties for multi-layer walls (eps_r=1, sigma=0 at all frequencies)
AIR_PROPERTIES = {
    (0.001, 1000.0): (1.0, 0.0, 0.0, 0.0),
}


# QUERY FUNCTIONS

def get_itu_properties(material_name: str, freq_hz: float) -> tuple:
    """Look up ITU material properties at a given frequency."""
    if material_name == 'air':
        return 1.0, 0.0

    if material_name not in ITU_MATERIAL_PROPERTIES:
        available = sorted(ITU_MATERIAL_PROPERTIES.keys())
        raise ValueError(
            f"Unknown material '{material_name}'. "
            f"Available: {available}"
        )

    props = ITU_MATERIAL_PROPERTIES[material_name]
    freq_ghz = freq_hz / 1e9

    # Find the band containing this frequency
    for (f_min, f_max), (a, b, c, d) in props.items():
        if f_min <= freq_ghz <= f_max:
            eps_r = a * (freq_ghz ** b)
            sigma = c * (freq_ghz ** d)
            return eps_r, sigma

    # Frequency outside all defined bands — use nearest band with warning
    bands = list(props.keys())
    band_centers = [(f_min + f_max) / 2 for f_min, f_max in bands]
    nearest_idx = int(np.argmin([abs(freq_ghz - c) for c in band_centers]))
    nearest_band = bands[nearest_idx]
    a, b, c, d = props[nearest_band]
    eps_r = a * (freq_ghz ** b)
    sigma = c * (freq_ghz ** d)

    import warnings
    warnings.warn(
        f"Frequency {freq_ghz:.1f} GHz is outside defined bands for "
        f"'{material_name}' ({bands}). Using nearest band {nearest_band}.",
        stacklevel=2,
    )
    return eps_r, sigma


def complex_relative_permittivity(eps_r: float, sigma: float, freq_hz: float) -> complex:
    """Compute complex relative permittivity eta = eps_r - j * sigma / (omega * eps_0)."""
    omega = 2.0 * np.pi * freq_hz
    eps_i = sigma / (omega * EPSILON_0)
    return complex(eps_r, -eps_i)


def get_material_properties(material_name: str, freq_hz: float) -> dict:
    """Get complete material properties at a given frequency."""
    eps_r, sigma = get_itu_properties(material_name, freq_hz)
    eta = complex_relative_permittivity(eps_r, sigma, freq_hz)

    omega = 2.0 * np.pi * freq_hz
    eps_imag = sigma / (omega * EPSILON_0)

    roughness = DEFAULT_ROUGHNESS.get(material_name, (1e-4, 1e-2))
    thickness = DEFAULT_THICKNESS.get(material_name, DEFAULT_THICKNESS_FALLBACK)

    return {
        'eps_r': eps_r,
        'sigma': sigma,
        'eps_real': eps_r,
        'eps_imag': eps_imag,
        'sigma_h': roughness[0],
        'l_c': roughness[1],
        'thickness': thickness,
        'eta_complex': eta,
    }
