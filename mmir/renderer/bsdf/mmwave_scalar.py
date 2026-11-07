"""Physics-first hybrid BSDF for 77 GHz mmWave radar (CSV-BSDF equivalent)."""

import numpy as np
import drjit as dr
import mitsuba as mi


# CONSTANTS

WAVELENGTH_77GHZ = 3.9e-3  # λ at 77 GHz (m)
K_77GHZ = 2.0 * np.pi / WAVELENGTH_77GHZ  # Wavenumber ≈ 1611 m⁻¹


# HELPER FUNCTIONS

def _sigmoid(x: 'mi.Float') -> 'mi.Float':
    """Sigmoid function: σ(x) = 1 / (1 + exp(-x))"""
    # For numerical stability:
    # If x >= 0: sigmoid(x) = 1 / (1 + exp(-x))
    # If x < 0:  sigmoid(x) = exp(x) / (1 + exp(x))
    pos_mask = x >= mi.Float(0.0)

    # Clamp to avoid overflow
    x_clamped = dr.clamp(x, mi.Float(-50.0), mi.Float(50.0))

    # Standard sigmoid
    exp_neg_x = dr.exp(-x_clamped)
    sigmoid_pos = mi.Float(1.0) / (mi.Float(1.0) + exp_neg_x)

    # Alternative form for negative x
    exp_x = dr.exp(x_clamped)
    sigmoid_neg = exp_x / (mi.Float(1.0) + exp_x)

    return dr.select(pos_mask, sigmoid_pos, sigmoid_neg)


# PHASE 1: CORE UTILITIES

def permittivity_to_ior(
    eps_real: 'mi.Float',
    eps_imag: 'mi.Float'
) -> tuple:
    """Convert complex permittivity to complex index of refraction."""
    eps_mag = dr.sqrt(dr.maximum(eps_real * eps_real + eps_imag * eps_imag, mi.Float(1e-20)))
    n = dr.sqrt(dr.maximum((eps_mag + eps_real) / 2.0, mi.Float(1e-20)))
    kappa = dr.sqrt(dr.maximum((eps_mag - eps_real) / 2.0, mi.Float(1e-20)))
    return n, kappa


def fresnel_complex_power(
    n: 'mi.Float',
    kappa: 'mi.Float',
    cos_theta_i: 'mi.Float',
) -> tuple:
    """Compute Fresnel power reflectance |r_s|² and |r_p|²."""
    sin2_theta_i = 1.0 - cos_theta_i * cos_theta_i

    # ñ² = (n - jκ)² = n² - κ² - 2jnκ
    n2_sq_real = n * n - kappa * kappa
    n2_sq_imag = -2.0 * n * kappa

    # ξ = ñ² - sin²(θ_i)
    xi_real = n2_sq_real - sin2_theta_i
    xi_imag = n2_sq_imag

    # sqrt(ξ) = a + jb (epsilon in sqrt/atan2 to prevent NaN backward at degenerate inputs)
    xi_mag_sq = xi_real * xi_real + xi_imag * xi_imag
    xi_mag = dr.sqrt(dr.maximum(xi_mag_sq, mi.Float(1e-20)))
    xi_arg = dr.atan2(xi_imag, dr.maximum(xi_real, mi.Float(1e-10)))  # safe atan2
    a = dr.sqrt(dr.maximum(xi_mag, mi.Float(1e-20))) * dr.cos(xi_arg / 2.0)
    b = dr.sqrt(dr.maximum(xi_mag, mi.Float(1e-20))) * dr.sin(xi_arg / 2.0)

    # R_s = |(cos_θ - (a+jb)) / (cos_θ + (a+jb))|²
    rs_num_sq = (cos_theta_i - a) * (cos_theta_i - a) + b * b
    rs_den_sq = dr.maximum((cos_theta_i + a) * (cos_theta_i + a) + b * b, mi.Float(1e-10))
    R_s = rs_num_sq / rs_den_sq

    # R_p = |(ñ² cos_θ - (a+jb)) / (ñ² cos_θ + (a+jb))|²
    n2cos_real = n2_sq_real * cos_theta_i
    n2cos_imag = n2_sq_imag * cos_theta_i

    rp_num_sq = (n2cos_real - a) * (n2cos_real - a) + (n2cos_imag - b) * (n2cos_imag - b)
    rp_den_sq = dr.maximum((n2cos_real + a) * (n2cos_real + a) + (n2cos_imag + b) * (n2cos_imag + b), mi.Float(1e-10))
    R_p = rp_num_sq / rp_den_sq

    return R_s, R_p


# PHASE 2: ITU SINGLE-LAYER SLAB FRESNEL (ITU-R P.2040-4, Eqs 43-44)

def _cpx_sqrt(x_real: 'mi.Float', x_imag: 'mi.Float') -> tuple:
    """Complex square root using the algebraic formula. Matches Sionna's cpx_sqrt implementation."""
    r = dr.sqrt(dr.maximum(x_real * x_real + x_imag * x_imag, mi.Float(1e-20)))
    y_real = dr.sqrt(dr.maximum(mi.Float(0.5) * (r + x_real), mi.Float(1e-20)))
    y_imag = dr.sign(x_imag) * dr.sqrt(dr.maximum(mi.Float(0.5) * (r - x_real), mi.Float(1e-20)))
    return y_real, y_imag


def _cpx_mul(a_r, a_i, b_r, b_i):
    """Complex multiply: (a_r + j*a_i) * (b_r + j*b_i)."""
    return a_r * b_r - a_i * b_i, a_r * b_i + a_i * b_r


def _cpx_div(a_r, a_i, b_r, b_i):
    """Complex divide: (a_r + j*a_i) / (b_r + j*b_i)."""
    denom = dr.maximum(b_r * b_r + b_i * b_i, mi.Float(1e-30))
    return (a_r * b_r + a_i * b_i) / denom, (a_i * b_r - a_r * b_i) / denom


def _cpx_sq(a_r, a_i):
    """Complex square: (a_r + j*a_i)^2."""
    return a_r * a_r - a_i * a_i, mi.Float(2.0) * a_r * a_i


def _cpx_exp(a_r, a_i):
    """Complex exp: exp(a_r + j*a_i) = exp(a_r) * (cos(a_i) + j*sin(a_i))."""
    e = dr.exp(a_r)
    return e * dr.cos(a_i), e * dr.sin(a_i)


def _cpx_abs_sq(a_r, a_i):
    """Complex magnitude squared: |a_r + j*a_i|^2."""
    return a_r * a_r + a_i * a_i


def itu_slab_fresnel(
    eps_real: 'mi.Float',
    eps_imag: 'mi.Float',
    cos_theta_i: 'mi.Float',
    thickness: 'mi.Float',
    wavelength: float = WAVELENGTH_77GHZ,
) -> tuple:
    """ITU-R P.2040-4 single-layer slab Fresnel model (Eqs 43a, 43b, 44)."""
    sin2_theta = mi.Float(1.0) - cos_theta_i * cos_theta_i

    # Complex permittivity: eta = eps_real - j*eps_imag
    eta_r = eps_real
    eta_i = -eps_imag  # Convention: eta = eps_r - j*eps_i

    arg_r = eta_r - sin2_theta
    arg_i = eta_i
    a_r, a_i = _cpx_sqrt(arg_r, arg_i)

    # Single-interface Fresnel coefficients (Eqs 37a, 37b)
    rte_num_r = cos_theta_i - a_r
    rte_num_i = -a_i
    rte_den_r = cos_theta_i + a_r
    rte_den_i = a_i
    rte_r, rte_i = _cpx_div(rte_num_r, rte_num_i, rte_den_r, rte_den_i)

    # eta * cos_theta
    etac_r = eta_r * cos_theta_i
    etac_i = eta_i * cos_theta_i
    rtm_num_r = etac_r - a_r
    rtm_num_i = etac_i - a_i
    rtm_den_r = etac_r + a_r
    rtm_den_i = etac_i + a_i
    rtm_r, rtm_i = _cpx_div(rtm_num_r, rtm_num_i, rtm_den_r, rtm_den_i)

    # Phase delay through slab: q = 2*pi*d/lambda * a  (Eq 44)
    phase_scale = mi.Float(2.0 * np.pi) * thickness / mi.Float(wavelength)
    q_r = phase_scale * a_r
    q_i = phase_scale * a_i

    # exp(-j*q) = exp(-j*(q_r + j*q_i)) = exp(q_i) * exp(-j*q_r)
    # = exp(q_i) * (cos(q_r) - j*sin(q_r))
    ejq_r, ejq_i = _cpx_exp(q_i, -q_r)         # exp(-j*q)
    ej2q_r, ej2q_i = _cpx_exp(mi.Float(2.0) * q_i, mi.Float(-2.0) * q_r)  # exp(-j*2q)

    # r_p^2
    rte_sq_r, rte_sq_i = _cpx_sq(rte_r, rte_i)
    rtm_sq_r, rtm_sq_i = _cpx_sq(rtm_r, rtm_i)

    # Denominator: 1 - r_p^2 * exp(-j*2q)
    prod_te_r, prod_te_i = _cpx_mul(rte_sq_r, rte_sq_i, ej2q_r, ej2q_i)
    denom_te_r = mi.Float(1.0) - prod_te_r
    denom_te_i = -prod_te_i

    prod_tm_r, prod_tm_i = _cpx_mul(rtm_sq_r, rtm_sq_i, ej2q_r, ej2q_i)
    denom_tm_r = mi.Float(1.0) - prod_tm_r
    denom_tm_i = -prod_tm_i

    # Slab reflection (Eq 43a): R = r_p * (1 - exp(-j*2q)) / denom
    one_minus_ej2q_r = mi.Float(1.0) - ej2q_r
    one_minus_ej2q_i = -ej2q_i

    rte_num2_r, rte_num2_i = _cpx_mul(rte_r, rte_i, one_minus_ej2q_r, one_minus_ej2q_i)
    R_TE_r, R_TE_i = _cpx_div(rte_num2_r, rte_num2_i, denom_te_r, denom_te_i)

    rtm_num2_r, rtm_num2_i = _cpx_mul(rtm_r, rtm_i, one_minus_ej2q_r, one_minus_ej2q_i)
    R_TM_r, R_TM_i = _cpx_div(rtm_num2_r, rtm_num2_i, denom_tm_r, denom_tm_i)

    # Slab transmission (Eq 43b): T = (1 - r_p^2) * exp(-j*q) / denom
    one_minus_rte_sq_r = mi.Float(1.0) - rte_sq_r
    one_minus_rte_sq_i = -rte_sq_i
    tte_num_r, tte_num_i = _cpx_mul(one_minus_rte_sq_r, one_minus_rte_sq_i, ejq_r, ejq_i)
    T_TE_r, T_TE_i = _cpx_div(tte_num_r, tte_num_i, denom_te_r, denom_te_i)

    one_minus_rtm_sq_r = mi.Float(1.0) - rtm_sq_r
    one_minus_rtm_sq_i = -rtm_sq_i
    ttm_num_r, ttm_num_i = _cpx_mul(one_minus_rtm_sq_r, one_minus_rtm_sq_i, ejq_r, ejq_i)
    T_TM_r, T_TM_i = _cpx_div(ttm_num_r, ttm_num_i, denom_tm_r, denom_tm_i)

    return (R_TE_r, R_TE_i, R_TM_r, R_TM_i,
            T_TE_r, T_TE_i, T_TM_r, T_TM_i)


def itu_slab_fresnel_rough(
    eps_real: 'mi.Float',
    eps_imag: 'mi.Float',
    cos_theta_i: 'mi.Float',
    thickness: 'mi.Float',
    sigma_h: 'mi.Float',
    wavelength: float = WAVELENGTH_77GHZ,
) -> tuple:
    """Slab Fresnel with roughness attenuation on the transmitted field."""
    (R_TE_r, R_TE_i, R_TM_r, R_TM_i,
     T_TE_r, T_TE_i, T_TM_r, T_TM_i) = itu_slab_fresnel(
        eps_real, eps_imag, cos_theta_i, thickness, wavelength
    )

    k = mi.Float(2.0 * np.pi / wavelength)

    # Transmission Rayleigh factor
    sin2_theta_i = mi.Float(1.0) - cos_theta_i * cos_theta_i
    sin2_theta_t = sin2_theta_i / dr.maximum(eps_real, mi.Float(1.001))
    cos_theta_t = dr.sqrt(dr.maximum(mi.Float(1.0) - sin2_theta_t, mi.Float(1e-20)))
    sqrt_eps = dr.sqrt(dr.maximum(eps_real, mi.Float(1.001)))

    delta_kz = k * sigma_h * (cos_theta_i - sqrt_eps * cos_theta_t)
    g_t = delta_kz * delta_kz
    rayleigh_t = dr.exp(-g_t)

    # Attenuate coherent transmission
    T_TE_r = T_TE_r * rayleigh_t
    T_TE_i = T_TE_i * rayleigh_t
    T_TM_r = T_TM_r * rayleigh_t
    T_TM_i = T_TM_i * rayleigh_t

    return (R_TE_r, R_TE_i, R_TM_r, R_TM_i,
            T_TE_r, T_TE_i, T_TM_r, T_TM_i)


def compute_slab_energy_gate(
    eps_real: 'mi.Float',
    eps_imag: 'mi.Float',
    cos_theta_i: 'mi.Float',
    thickness: 'mi.Float',
    frac_s: 'mi.Float',
    frac_p: 'mi.Float',
    wavelength: float = WAVELENGTH_77GHZ,
    sigma_h: 'mi.Float' = None,
) -> 'mi.Float':
    """Compute energy gate A(omega_i) using the ITU slab Fresnel model."""
    use_slab = thickness > mi.Float(1e-6)

    # Single-interface Fresnel (fallback)
    n_ior, kappa = permittivity_to_ior(eps_real, eps_imag)
    R_s_single, R_p_single = fresnel_complex_power(n_ior, kappa, cos_theta_i)
    A_single = frac_s * R_s_single + frac_p * R_p_single

    # Slab Fresnel (with optional rough transmission)
    (R_TE_r, R_TE_i, R_TM_r, R_TM_i,
     T_TE_r, T_TE_i, T_TM_r, T_TM_i) = itu_slab_fresnel_rough(
        eps_real, eps_imag, cos_theta_i, thickness,
        sigma_h, wavelength  # Note: sigma_h before wavelength per function signature
    )
    R_s_slab = _cpx_abs_sq(R_TE_r, R_TE_i)
    R_p_slab = _cpx_abs_sq(R_TM_r, R_TM_i)
    A_slab = frac_s * R_s_slab + frac_p * R_p_slab

    A = dr.select(use_slab, A_slab, A_single)
    return dr.clamp(A, mi.Float(0.0), mi.Float(1.0))


def enforce_spm_validity(
    sigma_h: 'mi.Float',
    l_c: 'mi.Float',
    wavelength: float = WAVELENGTH_77GHZ,
) -> tuple:
    """Clamp parameters to SPM validity regime."""
    k = 2.0 * np.pi / wavelength

    # Constraint 1: kh < 0.1 (≪ 1)
    h_max_1 = mi.Float(0.1 / k)  # ≈ 62 μm at 77 GHz

    # Set minimum l_c to avoid division issues
    l_c_min = mi.Float(wavelength * 0.5)  # Minimum correlation length = λ/2
    l_c_clamped = dr.maximum(l_c, l_c_min)

    # Constraint 3: √2 h/l < 0.3 → h < 0.21 × l
    h_max_3 = 0.21 * l_c_clamped
    h_max = dr.minimum(h_max_1, h_max_3)

    # Minimum h for numerical stability
    h_min = mi.Float(1e-7)  # 0.1 μm

    sigma_h_clamped = dr.clamp(sigma_h, h_min, h_max)

    # Verify constraint 2: k³h²l < 0.1
    k3 = mi.Float(k * k * k)
    h_max_2 = dr.sqrt(mi.Float(0.1) / (k3 * l_c_clamped))
    sigma_h_clamped = dr.minimum(sigma_h_clamped, h_max_2)

    return sigma_h_clamped, l_c_clamped


def compute_validity_aware_blend(
    tau_base: 'mi.Float',
    cos_theta_i: 'mi.Float',
    sigma_h: 'mi.Float',
    l_c: 'mi.Float',
    wavelength: float = WAVELENGTH_77GHZ,
) -> 'mi.Float':
    """Compute blend coefficient τ with validity weighting."""
    k = mi.Float(2.0 * np.pi / wavelength)

    # Angle-based blend
    # θ = 20° → cos(20°) ≈ 0.94
    THETA_THRESHOLD_COS = mi.Float(0.94)
    tau_angle = _sigmoid((cos_theta_i - THETA_THRESHOLD_COS) * mi.Float(20.0))

    # KA validity
    kl = k * l_c
    v_KA = _sigmoid((kl - mi.Float(6.0)) * mi.Float(2.0))

    # SPM validity (kh < 0.3 is "good")
    kh = k * sigma_h
    v_SPM = _sigmoid((mi.Float(0.3) - kh) * mi.Float(10.0))

    # Weighted blend
    # At small angles: prefer KA if valid
    # At large angles: prefer SPM if valid
    w_KA = tau_angle * tau_base * v_KA
    w_SPM = (mi.Float(1.0) - tau_angle) * (mi.Float(1.0) - tau_base) * v_SPM

    # Normalize
    total = dr.maximum(w_KA + w_SPM, mi.Float(1e-6))
    tau_eff = w_KA / total

    return dr.clamp(tau_eff, mi.Float(0.01), mi.Float(0.99))


def map_renderer_params_to_physical(
    albedo: 'mi.Float',
    roughness: 'mi.Float',
    metallic: 'mi.Float',
    wavelength: float = WAVELENGTH_77GHZ,
) -> tuple:
    """Map CG-style parameters to physical mmWave parameters."""
    k = 2.0 * np.pi / wavelength

    # Electromagnetic properties
    eps_real = dr.select(
        metallic > mi.Float(0.5),
        mi.Float(1.0),
        mi.Float(2.0) + albedo * mi.Float(4.0)
    )

    eps_imag = dr.select(
        metallic > mi.Float(0.5),
        mi.Float(1e6),
        mi.Float(0.05) + albedo * mi.Float(0.3)
    )

    # Roughness statistics
    # Map GGX alpha to physical roughness
    # Use SPM validity bounds: h < 0.1/k ≈ 62 μm at 77 GHz
    h_max = mi.Float(0.1 / k)
    sigma_h = roughness * h_max
    sigma_h = dr.clamp(sigma_h, mi.Float(1e-7), h_max)

    # Correlation length: smooth surfaces have larger l_c
    # Ensure √2 h/l < 0.3 → l > √2 h / 0.3 ≈ 4.7 h
    l_c_min = mi.Float(5.0) * sigma_h  # Slightly above minimum for stability
    l_c = mi.Float(wavelength) * (mi.Float(1.0) + mi.Float(20.0) * (mi.Float(1.0) - roughness))
    l_c = dr.maximum(l_c, l_c_min)

    # Blending coefficient
    # Metals: more specular (higher τ)
    # Dielectrics: more diffuse (lower τ)
    # Rougher surfaces: more diffuse
    tau = dr.select(
        metallic > mi.Float(0.5),
        mi.Float(0.9) - roughness * mi.Float(0.4),
        mi.Float(0.6) - roughness * mi.Float(0.5)
    )
    tau = dr.clamp(tau, mi.Float(0.1), mi.Float(0.95))

    return eps_real, eps_imag, sigma_h, l_c, tau


# PHASE 2: LOBE IMPLEMENTATIONS

def _eval_lambda_ggx(alpha_sq: 'mi.Float', cos_theta: 'mi.Float') -> 'mi.Float':
    """GGX Lambda function for Smith masking."""
    valid = cos_theta > mi.Float(0)
    cos_theta_sq = cos_theta * cos_theta
    tan_theta_sq = dr.maximum(mi.Float(1.0) - cos_theta_sq, mi.Float(0.0)) / dr.maximum(cos_theta_sq, mi.Float(1e-10))
    result = mi.Float(0.5) * (mi.Float(-1.0) + dr.sqrt(mi.Float(1.0) + alpha_sq * tan_theta_sq))
    return dr.select(valid, result, mi.Float(0.0))


def eval_lobe_KA(
    wo: 'mi.Vector3f',
    wi: 'mi.Vector3f',
    n: 'mi.Vector3f',
    sigma_h: 'mi.Float',
    l_c: 'mi.Float',
    wavelength: float = WAVELENGTH_77GHZ,
) -> 'mi.Float':
    """Evaluate KA (Kirchhoff Approximation) specular lobe."""
    cos_theta_i = dr.dot(wi, n)
    cos_theta_o = dr.dot(wo, n)

    # Map physical roughness to GGX alpha
    #
    # The plan's formula α = 4π × σ_h / λ produces very small values (0.02-0.1)
    # which create extremely peaked specular (D ∝ 1/α²). This doesn't match
    # typical rendering behavior where α ∈ [0.1, 0.9].
    #
    # We use a scaled mapping: α = sqrt(4π × σ_h / λ)
    # This preserves the physical relationship while producing GGX-compatible values:
    #   σ_h = 6 μm  → α = sqrt(0.02) ≈ 0.14 (smooth)
    #   σ_h = 24 μm → α = sqrt(0.08) ≈ 0.28 (moderate)
    #   σ_h = 35 μm → α = sqrt(0.11) ≈ 0.33 (rough)
    #
    alpha_raw = mi.Float(4.0 * np.pi) * sigma_h / mi.Float(wavelength)
    alpha = dr.sqrt(alpha_raw)
    alpha = dr.clamp(alpha, mi.Float(0.05), mi.Float(0.95))

    # Half-vector (safe normalize to avoid NaN backward at grazing angles
    # where wo + wi ≈ 0, since dr.normalize backward → 1/norm → inf)
    h_unnorm = wo + wi
    h_len_sq = dr.dot(h_unnorm, h_unnorm)
    h_len = dr.sqrt(dr.maximum(h_len_sq, mi.Float(1e-20)))
    h = h_unnorm / h_len
    h_dot_n = dr.dot(h, n)

    # GGX NDF: D(h) = α² / (π × ((n·h)² × (α² - 1) + 1)²)
    a2 = alpha * alpha
    d_denom = (h_dot_n * h_dot_n) * (a2 - mi.Float(1.0)) + mi.Float(1.0)
    D = a2 / (d_denom * d_denom * mi.Float(np.pi))

    # Smith GGX masking-shadowing: G = 1 / (1 + Λ(wi) + Λ(wo))
    lambda_i = _eval_lambda_ggx(a2, cos_theta_i)
    lambda_o = _eval_lambda_ggx(a2, cos_theta_o)
    G = mi.Float(1.0) / (mi.Float(1.0) + lambda_i + lambda_o)

    # Standard Cook-Torrance:
    # f_KA = D × G / (4 × cos_i × cos_o)
    # NO Fresnel here - that's in the energy gate
    denom = mi.Float(4.0) * dr.maximum(cos_theta_i * cos_theta_o, mi.Float(1e-6))
    f_KA = D * G / denom

    valid = (cos_theta_i > mi.Float(1e-6)) & (cos_theta_o > mi.Float(1e-6))
    return dr.select(valid, f_KA, mi.Float(0.0))


def eval_lobe_SPM(
    wo: 'mi.Vector3f',
    wi: 'mi.Vector3f',
    n: 'mi.Vector3f',
    sigma_h: 'mi.Float',
    l_c: 'mi.Float',
    wavelength: float = WAVELENGTH_77GHZ,
    eps_real: 'mi.Float' = None,
    eps_imag: 'mi.Float' = None,
) -> 'mi.Float':
    """Evaluate SPM (Small Perturbation Method) diffuse lobe."""
    cos_theta_i = dr.dot(wi, n)
    cos_theta_o = dr.dot(wo, n)

    # Specular reflection direction: r = 2(n·d)n - d
    wi_reflected = mi.Float(2.0) * dr.dot(wi, n) * n - wi
    cos_dev = dr.dot(wo, wi_reflected)

    # SPM lobe width: kappa determines concentration
    #
    # The SPM lobe should be broader than KA (specular) but still directive.
    # Using sqrt(l_c / λ) gives reasonable concentration values:
    # - l_c = 5λ  → kappa ≈ 2.2 (broader lobe)
    # - l_c = 20λ → kappa ≈ 4.5 (moderate concentration)
    #
    # This keeps kappa in a range where vMF gives meaningful non-zero values
    # while still being narrower than Lambertian.
    l_c_wavelengths = l_c / mi.Float(wavelength)
    # Couple SPM lobe width to surface slope (sigma_h / l_c).
    # Factor of 3 (vs 5 for directive lobe) because SPM is already more peaked.
    roughness_slope = sigma_h / dr.maximum(l_c, mi.Float(1e-8))
    kappa_vmf = dr.sqrt(l_c_wavelengths) / (mi.Float(1.0) + mi.Float(3.0) * roughness_slope)
    kappa_vmf = dr.clamp(kappa_vmf, mi.Float(0.5), mi.Float(10.0))

    # von Mises-Fisher distribution: f ∝ exp(κ × (cos_dev - 1))
    # The -1 normalizes the peak to exp(0) = 1 at specular
    f_vmf = dr.exp(kappa_vmf * (cos_dev - mi.Float(1.0)))

    # Normalization = κ / (4π × sinh(κ))
    # For numerical stability, use sinh(x) = (exp(x) - exp(-x)) / 2
    # When κ is large, sinh(κ) ≈ exp(κ)/2, so norm ≈ 2κ / (4π × exp(κ)) = κ / (2π × exp(κ))
    kappa_clamped = dr.minimum(kappa_vmf, mi.Float(50.0))  # Prevent overflow
    sinh_kappa = (dr.exp(kappa_clamped) - dr.exp(-kappa_clamped)) / mi.Float(2.0)
    norm_factor = kappa_vmf / (mi.Float(4.0 * np.pi) * dr.maximum(sinh_kappa, mi.Float(1e-6)))
    f_SPM = f_vmf * norm_factor

    # Material modulation for SPM
    # The SPM directive lobe should be weighted by material properties.
    # We use permittivity contrast AND add a base scattering factor to ensure
    # reasonable energy at non-specular angles.
    #
    # For dielectric (eps_real=3.2): contrast = 2.2 + 0.14 = 2.34
    # eps_factor = clamp(2.34/10, 0.1, 1.0) = 0.234
    if eps_real is not None and eps_imag is not None:
        eps_contrast = dr.abs(eps_real - mi.Float(1.0)) + eps_imag
        eps_factor = dr.clamp(eps_contrast / mi.Float(5.0), mi.Float(0.2), mi.Float(1.0))
        f_SPM = f_SPM * eps_factor

    valid = (cos_theta_i > mi.Float(1e-6)) & (cos_theta_o > mi.Float(1e-6))
    return dr.select(valid, f_SPM, mi.Float(0.0))


# PHASE 7: INCOHERENT LOBES (Enabled early for validation)

def eval_lobe_directive(
    wo: 'mi.Vector3f',
    wi: 'mi.Vector3f',
    n: 'mi.Vector3f',
    sigma_h: 'mi.Float',
    l_c: 'mi.Float',
    wavelength: float = WAVELENGTH_77GHZ,
) -> 'mi.Float':
    """Evaluate directive incoherent lobe."""
    cos_theta_i = dr.dot(wi, n)
    cos_theta_o = dr.dot(wo, n)

    # Specular reflection direction
    wi_reflected = mi.Float(2.0) * dr.dot(wi, n) * n - wi
    cos_dev = dr.dot(wo, wi_reflected)

    # Directive lobe width coupled to surface slope RMS (sigma_h / l_c).
    # The RMS slope s = sqrt(2) * sigma_h / l_c determines the angular spread
    # of scattered energy. Rougher relative to correlation → broader lobe.
    l_c_wavelengths = l_c / mi.Float(wavelength)
    kappa_spm = dr.sqrt(l_c_wavelengths)  # SPM-equivalent concentration
    roughness_slope = sigma_h / dr.maximum(l_c, mi.Float(1e-8))
    broadening = mi.Float(1.0) + mi.Float(5.0) * roughness_slope  # 1x smooth → ~2-3x rough
    kappa_dir = kappa_spm / broadening
    kappa_dir = dr.clamp(kappa_dir, mi.Float(0.3), mi.Float(5.0))

    # vMF distribution
    f_vmf = dr.exp(kappa_dir * (cos_dev - mi.Float(1.0)))

    # Normalization: κ / (4π × sinh(κ))
    sinh_kappa = (dr.exp(kappa_dir) - dr.exp(-kappa_dir)) / mi.Float(2.0)
    norm_factor = kappa_dir / (mi.Float(4.0 * np.pi) * dr.maximum(sinh_kappa, mi.Float(1e-6)))
    f_dir = f_vmf * norm_factor

    valid = (cos_theta_i > mi.Float(1e-6)) & (cos_theta_o > mi.Float(1e-6))
    return dr.select(valid, f_dir, mi.Float(0.0))


def eval_lobe_broad(
    wo: 'mi.Vector3f',
    wi: 'mi.Vector3f',
    n: 'mi.Vector3f',
    albedo: 'mi.Float',
) -> 'mi.Float':
    """Evaluate broad incoherent lobe (Lambertian-like)."""
    cos_theta_i = dr.dot(wi, n)
    cos_theta_o = dr.dot(wo, n)

    # Scale by albedo squared for physical energy balance at mmWave
    # (albedo affects both absorption and scattering)
    f_broad = albedo / mi.Float(np.pi)

    valid = (cos_theta_i > mi.Float(1e-6)) & (cos_theta_o > mi.Float(1e-6))
    return dr.select(valid, f_broad, mi.Float(0.0))


def compute_cbs_factor(
    wo: 'mi.Vector3f',
    wi: 'mi.Vector3f',
    n: 'mi.Vector3f',
    l_c: 'mi.Float',
    wavelength: float = WAVELENGTH_77GHZ,
) -> 'mi.Float':
    """Coherent Backscatter Enhancement (CBS) factor."""
    k = mi.Float(2.0 * np.pi / wavelength)

    # Retroreflection direction: reflect wi through surface, then negate
    # If wo == -wi_reflected, we're at exact backscatter
    wi_reflected = mi.Float(2.0) * dr.dot(wi, n) * n - wi
    retro_dir = -wi_reflected  # wo should equal this for exact backscatter

    # Angular deviation from exact backscatter
    cos_bs = dr.clamp(dr.dot(wo, retro_dir), mi.Float(-1.0), mi.Float(1.0))
    sin_bs = dr.sqrt(dr.maximum(mi.Float(1.0) - cos_bs * cos_bs, mi.Float(1e-20)))

    # CBS kernel: sinc^2(k * l_c * sin_theta_bs)
    x = k * l_c * sin_bs
    sinc_x = dr.select(
        x < mi.Float(1e-6),
        mi.Float(1.0),
        dr.sin(x) / dr.maximum(x, mi.Float(1e-10))
    )
    cbs = mi.Float(1.0) + sinc_x * sinc_x  # Range: [1, 2]
    return cbs


def compute_cbs_hemispherical_mean(
    l_c: 'mi.Float',
    wavelength: float = WAVELENGTH_77GHZ,
) -> 'mi.Float':
    """Cosine-weighted hemispherical average of the CBS factor."""
    k = mi.Float(2.0 * np.pi / wavelength)
    alpha = k * l_c

    I_alpha = mi.Float(1.0) / (mi.Float(1.0) + alpha)
    cbs_mean = mi.Float(1.0) + I_alpha

    return cbs_mean


def compute_coherent_incoherent_blend(
    sigma_h: 'mi.Float',
    l_c: 'mi.Float',
    wavelength: float = WAVELENGTH_77GHZ,
    cos_theta_i: 'mi.Float' = None,
) -> tuple:
    """Compute η (coherent/incoherent) and γ (directive/broad) blend coefficients using the Rayleigh
    roughness factor.
    """
    k = mi.Float(2.0 * np.pi / wavelength)

    # Rayleigh factor for coherent power fraction

    g = (mi.Float(2.0) * k * sigma_h * cos_theta_i)
    g = g * g  # g = (2*k*sigma_h*cos_theta)^2
    eta = dr.exp(-g)
    eta = dr.clamp(eta, mi.Float(0.01), mi.Float(0.99))

    # Directive vs broad: based on roughness relative to wavelength
    # sigma_h/λ < 0.01: mostly directive
    # sigma_h/λ > 0.1: mostly broad
    roughness_ratio = sigma_h / mi.Float(wavelength)
    gamma = _sigmoid((mi.Float(0.05) - roughness_ratio) * mi.Float(50.0))
    gamma = dr.clamp(gamma, mi.Float(0.1), mi.Float(0.9))

    return eta, gamma


# MATERIAL DATABASE
#
# Full ITU-R P.2040-4 database is in itu_materials.py.
# Import here for backward compatibility.

from ..materials.itu_materials import (
    ITU_MATERIAL_PROPERTIES,
    get_material_properties,
)

# Legacy alias for code that references MATERIAL_DB_77GHZ directly
def _build_legacy_db():
    """Build 77 GHz snapshot for backward compatibility."""
    import warnings
    db = {}
    freq_hz = 77e9
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)  # out-of-band entries use the nearest ITU band
        for name in ITU_MATERIAL_PROPERTIES:
            try:
                props = get_material_properties(name, freq_hz)
                db[name] = {
                    'eps_real': props['eps_real'],
                    'eps_imag': props['eps_imag'],
                    'sigma_h': props['sigma_h'],
                    'l_c': props['l_c'],
                    'tau': 0.5,  # Default; actual tau computed by map_renderer_params_to_physical
                }
            except Exception:
                pass
    return db

MATERIAL_DB_77GHZ = _build_legacy_db()
