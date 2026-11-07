"""Polarization-aware Jones BSDF for 77 GHz mmWave radar."""

import drjit as dr
import mitsuba as mi
from typing import Tuple

from .base import BSDFBase

# Import shared utilities from scalar version
from .mmwave_scalar import (
    WAVELENGTH_77GHZ,
    K_77GHZ,
    permittivity_to_ior,
    enforce_spm_validity,
    compute_validity_aware_blend,
    map_renderer_params_to_physical,
    eval_lobe_KA,
    eval_lobe_SPM,
    eval_lobe_directive,
    eval_lobe_broad,
    compute_coherent_incoherent_blend,
    compute_cbs_factor,
    compute_cbs_hemispherical_mean,
    itu_slab_fresnel,
    compute_slab_energy_gate,
)


# JONES POLARIZATION UTILITIES


def compute_sp_basis_jones(wi: 'mi.Vector3f', n: 'mi.Vector3f') -> tuple:
    """Compute s and p polarization basis vectors (incident side)."""
    # s is perpendicular to plane of incidence
    s_unnorm = dr.cross(wi, n)
    s_len = dr.norm(s_unnorm)

    # Handle grazing incidence (s becomes undefined)
    valid = s_len > mi.Float(1e-6)

    # Fallback: use global x or y axis
    fallback_s = dr.cross(wi, mi.Vector3f(1, 0, 0))
    fallback_len = dr.norm(fallback_s)
    fallback_s2 = dr.cross(wi, mi.Vector3f(0, 1, 0))
    fallback_s2_len = dr.sqrt(dr.maximum(dr.dot(fallback_s2, fallback_s2), mi.Float(1e-20)))
    fallback_s = dr.select(
        fallback_len > mi.Float(1e-6),
        fallback_s / dr.maximum(fallback_len, mi.Float(1e-10)),
        fallback_s2 / fallback_s2_len
    )

    s = dr.select(valid, s_unnorm / dr.maximum(s_len, mi.Float(1e-10)), fallback_s)

    # p = s × wi gives vector in plane of incidence, perpendicular to wi
    p_unnorm = dr.cross(s, wi)
    p_len = dr.sqrt(dr.maximum(dr.dot(p_unnorm, p_unnorm), mi.Float(1e-20)))
    p = p_unnorm / p_len

    return s, p


def project_polarization_to_sp(
    pol_vec: 'mi.Vector3f',
    s: 'mi.Vector3f',
    p: 'mi.Vector3f',
) -> tuple:
    """Project a polarization vector onto the s/p basis."""
    amp_s = dr.dot(pol_vec, s)
    amp_p = dr.dot(pol_vec, p)
    return amp_s, amp_p


def apply_jones_reflection(
    E_s_in: 'mi.Float',
    E_p_in: 'mi.Float',
    r_s_real: 'mi.Float',
    r_s_imag: 'mi.Float',
    r_p_real: 'mi.Float',
    r_p_imag: 'mi.Float',
) -> tuple:
    """Apply Jones matrix for reflection."""
    # E_s_out = r_s × E_s_in = (r_s_real + j*r_s_imag) × E_s_in
    # For real input: E_out = E_in × r = E_in × (r_r + j*r_i)
    E_s_out_real = r_s_real * E_s_in
    E_s_out_imag = r_s_imag * E_s_in

    E_p_out_real = r_p_real * E_p_in
    E_p_out_imag = r_p_imag * E_p_in

    return E_s_out_real, E_s_out_imag, E_p_out_real, E_p_out_imag


def compute_received_field_complex(
    E_s_out_real: 'mi.Float',
    E_s_out_imag: 'mi.Float',
    E_p_out_real: 'mi.Float',
    E_p_out_imag: 'mi.Float',
    rx_amp_s: 'mi.Float',
    rx_amp_p: 'mi.Float',
) -> Tuple['mi.Float', 'mi.Float']:
    """Compute received complex field by projecting scattered field onto RX polarization."""
    # E_received = E_s × rx_s + E_p × rx_p (complex)
    E_rx_real = E_s_out_real * rx_amp_s + E_p_out_real * rx_amp_p
    E_rx_imag = E_s_out_imag * rx_amp_s + E_p_out_imag * rx_amp_p

    return E_rx_real, E_rx_imag


# PHASE 7: POLARIMETRIC SPM KERNELS


class BSDFmmWaveJones(BSDFBase):
    """Polarization-aware Jones BSDF for 77 GHz mmWave radar."""

    WAVELENGTH = WAVELENGTH_77GHZ
    K = K_77GHZ

    def __init__(
        self,
        tx_polarization: 'mi.Vector3f' = None,
        rx_polarization: 'mi.Vector3f' = None,
        enable_incoherent: bool = True,
        use_fresnel_phase: bool = False,
        enable_cbs: bool = True,
        enable_polarimetric_spm: bool = True,
        enable_slab_fresnel: bool = True,
        default_thickness: float = 0.1,
    ):
        """Initialize Jones BSDF."""

        self.tx_polarization = tx_polarization
        self.rx_polarization = rx_polarization
        self.enable_incoherent = enable_incoherent
        self.use_fresnel_phase = use_fresnel_phase
        self.enable_cbs = enable_cbs
        self.enable_polarimetric_spm = enable_polarimetric_spm
        self.enable_slab_fresnel = enable_slab_fresnel
        self.default_thickness = default_thickness

    def _compute_jones_fresnel_complex(
        self,
        wi: 'mi.Vector3f',
        wo: 'mi.Vector3f',
        n: 'mi.Vector3f',
        n_ior: 'mi.Float',
        kappa: 'mi.Float',
        cos_theta_i: 'mi.Float',
        eps_real: 'mi.Float' = None,
        eps_imag: 'mi.Float' = None,
        thickness: 'mi.Float' = None,
    ) -> Tuple['mi.Float', 'mi.Float']:
        """Compute polarization-weighted COMPLEX Fresnel field using Jones formalism."""
        # Step 1: Compute s/p basis for incident wave
        s_in, p_in = compute_sp_basis_jones(wi, n)

        # Step 2: Project TX polarization onto incident s/p basis
        tx_amp_s, tx_amp_p = project_polarization_to_sp(self.tx_polarization, s_in, p_in)

        # Step 3: Compute complex Fresnel coefficients
        R_TE_r, R_TE_i, R_TM_r, R_TM_i, _, _, _, _ = itu_slab_fresnel(
            eps_real, eps_imag, cos_theta_i, thickness, self.WAVELENGTH
        )
        r_s_real, r_s_imag = R_TE_r, R_TE_i
        r_p_real, r_p_imag = R_TM_r, R_TM_i

        # Step 4: Apply Jones reflection matrix
        E_s_out_real, E_s_out_imag, E_p_out_real, E_p_out_imag = apply_jones_reflection(
            tx_amp_s, tx_amp_p,
            r_s_real, r_s_imag,
            r_p_real, r_p_imag
        )

        # Step 5: Compute s/p basis for outgoing wave
        s_out = s_in  # Same as incident (perpendicular to plane of incidence)
        p_out_unnorm = dr.cross(s_out, wo)
        p_out_len = dr.sqrt(dr.maximum(dr.dot(p_out_unnorm, p_out_unnorm), mi.Float(1e-20)))
        p_out = p_out_unnorm / p_out_len

        # Step 6: Project RX polarization onto outgoing s/p basis
        rx_amp_s, rx_amp_p = project_polarization_to_sp(self.rx_polarization, s_out, p_out)

        # Step 7: Compute received COMPLEX field (preserves Fresnel phase!)
        E_rx_real, E_rx_imag = compute_received_field_complex(
            E_s_out_real, E_s_out_imag,
            E_p_out_real, E_p_out_imag,
            rx_amp_s, rx_amp_p
        )

        return E_rx_real, E_rx_imag

    def _compute_jones_fresnel_power(
        self,
        wi: 'mi.Vector3f',
        wo: 'mi.Vector3f',
        n: 'mi.Vector3f',
        n_ior: 'mi.Float',
        kappa: 'mi.Float',
        cos_theta_i: 'mi.Float',
        eps_real: 'mi.Float' = None,
        eps_imag: 'mi.Float' = None,
        thickness: 'mi.Float' = None,
    ) -> 'mi.Float':
        """Compute polarization-weighted Fresnel POWER reflectance using Jones formalism."""
        E_real, E_imag = self._compute_jones_fresnel_complex(
            wi, wo, n, n_ior, kappa, cos_theta_i,
            eps_real=eps_real, eps_imag=eps_imag, thickness=thickness
        )
        R_jones = E_real * E_real + E_imag * E_imag
        return dr.clamp(R_jones, mi.Float(0.0), mi.Float(1.0))

    # CORE EVALUATION (shared by legacy and physics paths)

    def _eval_f_core(
        self,
        wo: 'mi.Vector3f',
        wi: 'mi.Vector3f',
        n: 'mi.Vector3f',
        eps_real: 'mi.Float',
        eps_imag: 'mi.Float',
        sigma_h: 'mi.Float',
        l_c: 'mi.Float',
        tau_base: 'mi.Float',
        thickness: 'mi.Float',
        albedo_broad: 'mi.Float' = None,
    ) -> 'mi.Float':
        """Core Jones BSDF evaluation. Returns POWER (1/sr)."""
        sigma_h, l_c = enforce_spm_validity(sigma_h, l_c, self.WAVELENGTH)
        n_ior, kappa = permittivity_to_ior(eps_real, eps_imag)
        cos_theta_i = dr.maximum(dr.dot(wi, n), mi.Float(0.0))

        R_jones = self._compute_jones_fresnel_power(
            wi, wo, n, n_ior, kappa, cos_theta_i,
            eps_real=eps_real, eps_imag=eps_imag, thickness=thickness
        )

        tau_eff = compute_validity_aware_blend(
            tau_base, cos_theta_i, sigma_h, l_c, self.WAVELENGTH
        )

        f_KA = eval_lobe_KA(wo, wi, n, sigma_h, l_c, self.WAVELENGTH)
        f_SPM = eval_lobe_SPM(wo, wi, n, sigma_h, l_c, self.WAVELENGTH, eps_real, eps_imag)
        f_coh = tau_eff * f_KA + (mi.Float(1.0) - tau_eff) * f_SPM
        f_coh_gated = R_jones * f_coh

        f_dir = eval_lobe_directive(wo, wi, n, sigma_h, l_c, self.WAVELENGTH)
        broad_scale = albedo_broad if albedo_broad is not None else mi.Float(1.0)
        f_broad = eval_lobe_broad(wo, wi, n, broad_scale)

        eta, gamma = compute_coherent_incoherent_blend(
            sigma_h, l_c, self.WAVELENGTH, cos_theta_i
        )
        f_inc = gamma * f_dir + (mi.Float(1.0) - gamma) * f_broad

        if self.enable_cbs:
            cbs = compute_cbs_factor(wo, wi, n, l_c, self.WAVELENGTH)
            cbs_mean = compute_cbs_hemispherical_mean(l_c, self.WAVELENGTH)
            f_inc = f_inc * (cbs / cbs_mean)

        f_inc_gated = R_jones * f_inc
        f_total = eta * f_coh_gated + (mi.Float(1.0) - eta) * f_inc_gated

        return f_total

    def _eval_non_ka_core(
        self,
        wo: 'mi.Vector3f',
        wi: 'mi.Vector3f',
        n: 'mi.Vector3f',
        eps_real: 'mi.Float',
        eps_imag: 'mi.Float',
        sigma_h: 'mi.Float',
        l_c: 'mi.Float',
        tau_base: 'mi.Float',
        thickness: 'mi.Float',
        albedo_broad: 'mi.Float' = None,
    ) -> 'mi.Float':
        """Core non-KA Jones BSDF evaluation. Returns POWER (1/sr)."""
        sigma_h, l_c = enforce_spm_validity(sigma_h, l_c, self.WAVELENGTH)
        n_ior, kappa = permittivity_to_ior(eps_real, eps_imag)
        cos_theta_i = dr.maximum(dr.dot(wi, n), mi.Float(0.0))

        R_jones = self._compute_jones_fresnel_power(
            wi, wo, n, n_ior, kappa, cos_theta_i,
            eps_real=eps_real, eps_imag=eps_imag, thickness=thickness
        )

        tau_eff = compute_validity_aware_blend(
            tau_base, cos_theta_i, sigma_h, l_c, self.WAVELENGTH
        )
        eta, gamma = compute_coherent_incoherent_blend(
            sigma_h, l_c, self.WAVELENGTH, cos_theta_i
        )

        f_SPM = eval_lobe_SPM(wo, wi, n, sigma_h, l_c, self.WAVELENGTH, eps_real, eps_imag)
        f_coh_no_ka = eta * R_jones * (mi.Float(1.0) - tau_eff) * f_SPM

        f_dir = eval_lobe_directive(wo, wi, n, sigma_h, l_c, self.WAVELENGTH)
        broad_scale = albedo_broad if albedo_broad is not None else mi.Float(1.0)
        f_broad = eval_lobe_broad(wo, wi, n, broad_scale)
        f_inc = gamma * f_dir + (mi.Float(1.0) - gamma) * f_broad

        if self.enable_cbs:
            cbs = compute_cbs_factor(wo, wi, n, l_c, self.WAVELENGTH)
            cbs_mean = compute_cbs_hemispherical_mean(l_c, self.WAVELENGTH)
            f_inc = f_inc * (cbs / cbs_mean)

        f_inc_gated = (mi.Float(1.0) - eta) * R_jones * f_inc

        f_total = f_coh_no_ka + f_inc_gated

        return f_total

    def _eval_ka_only_core(
        self,
        wo: 'mi.Vector3f',
        wi: 'mi.Vector3f',
        n: 'mi.Vector3f',
        eps_real: 'mi.Float',
        eps_imag: 'mi.Float',
        sigma_h: 'mi.Float',
        l_c: 'mi.Float',
        tau_base: 'mi.Float',
        thickness: 'mi.Float',
    ) -> 'mi.Float':
        """KA-only Jones BSDF: R_jones × η × τ_eff × f_KA. Returns POWER (1/sr)."""
        sigma_h, l_c = enforce_spm_validity(sigma_h, l_c, self.WAVELENGTH)
        n_ior, kappa = permittivity_to_ior(eps_real, eps_imag)
        cos_theta_i = dr.maximum(dr.dot(wi, n), mi.Float(0.0))

        R_jones = self._compute_jones_fresnel_power(
            wi, wo, n, n_ior, kappa, cos_theta_i,
            eps_real=eps_real, eps_imag=eps_imag, thickness=thickness
        )

        tau_eff = compute_validity_aware_blend(
            tau_base, cos_theta_i, sigma_h, l_c, self.WAVELENGTH
        )
        eta, _ = compute_coherent_incoherent_blend(
            sigma_h, l_c, self.WAVELENGTH, cos_theta_i
        )

        f_KA = eval_lobe_KA(wo, wi, n, sigma_h, l_c, self.WAVELENGTH)

        return eta * R_jones * tau_eff * f_KA

    # LEGACY ENTRY POINTS (3-param: albedo, roughness, metallic)

    def eval_f(self, wo, wi, n, albedo, roughness, metallic):
        """Evaluate BSDF f(wo, wi) - returns POWER (1/sr). Legacy CG param path."""
        eps_real, eps_imag, sigma_h, l_c, tau = \
            map_renderer_params_to_physical(albedo, roughness, metallic, self.WAVELENGTH)
        thickness = mi.Float(self.default_thickness)
        return self._eval_f_core(wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                  thickness, albedo_broad=albedo)

    def eval_f_cos(self, wo, wi, n, albedo, roughness, metallic):
        """Evaluate BSDF × |cos(θ_i)|. Legacy CG param path."""
        cos_theta_i = dr.dot(wi, n)
        f = self.eval_f(wo, wi, n, albedo, roughness, metallic)
        return f * dr.maximum(cos_theta_i, mi.Float(0.0))

    def eval_f_cos_complex(self, wo, wi, n, albedo, roughness, metallic):
        """Evaluate complex coherent + power incoherent. Legacy CG param path."""
        eps_real, eps_imag, sigma_h, l_c, tau = \
            map_renderer_params_to_physical(albedo, roughness, metallic, self.WAVELENGTH)
        thickness = mi.Float(self.default_thickness)
        return self._eval_f_cos_complex_core(wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                              thickness, albedo_broad=albedo)

    def eval_non_ka_cos(self, wo, wi, n, albedo, roughness, metallic):
        """eval_non_ka × |cos(θ_i)|. Legacy CG param path."""
        cos_theta_i = dr.dot(wi, n)
        f = self.eval_non_ka(wo, wi, n, albedo, roughness, metallic)
        return f * dr.maximum(cos_theta_i, mi.Float(0.0))

    def eval_f_cos_components(self, wo, wi, n, albedo, roughness, metallic):
        """Evaluate BSDF × |cos(θ_i)| with per-lobe components. Legacy CG param path."""
        eps_real, eps_imag, sigma_h, l_c, tau = \
            map_renderer_params_to_physical(albedo, roughness, metallic, self.WAVELENGTH)
        thickness = mi.Float(self.default_thickness)
        return self._eval_f_cos_components_core(
            wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
            thickness, albedo_broad=albedo
        )

    # PHYSICS ENTRY POINTS (6-param: eps_real, eps_imag, sigma_h, l_c, tau,
    #                        thickness)

    def eval_f_physics(self, wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                        thickness):
        """Evaluate BSDF f(wo, wi) from direct physics params. Returns POWER."""
        return self._eval_f_core(wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                  thickness)

    def eval_f_cos_physics(self, wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                            thickness):
        """Evaluate BSDF × |cos(θ_i)| from direct physics params."""
        cos_theta_i = dr.dot(wi, n)
        f = self.eval_f_physics(wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                 thickness)
        return f * dr.maximum(cos_theta_i, mi.Float(0.0))

    def eval_f_cos_complex_physics(self, wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                    thickness):
        """Evaluate complex coherent + power incoherent from direct physics params."""
        return self._eval_f_cos_complex_core(wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                              thickness)

    def eval_non_ka_physics(self, wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                             thickness):
        """Evaluate BSDF minus KA lobe from direct physics params."""
        return self._eval_non_ka_core(wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                       thickness)

    def eval_non_ka_cos_physics(self, wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                 thickness):
        """eval_non_ka × |cos(θ_i)| from direct physics params."""
        cos_theta_i = dr.dot(wi, n)
        f = self.eval_non_ka_physics(wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                      thickness)
        return f * dr.maximum(cos_theta_i, mi.Float(0.0))

    def eval_ka_only_physics(self, wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                              thickness):
        """Evaluate KA-only BSDF: R_jones × η × τ_eff × f_KA from direct physics params."""
        return self._eval_ka_only_core(wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                        thickness)

    def eval_specular_reflectance(self, cos_theta_i, eps_real, eps_imag, sigma_h, l_c,
                                    tau, thickness):
        """Compute integrated specular reflectance R_specular = η × τ_eff × A."""
        sigma_h_v, l_c_v = enforce_spm_validity(sigma_h, l_c, self.WAVELENGTH)

        A = compute_slab_energy_gate(
            eps_real, eps_imag, cos_theta_i, thickness,
            mi.Float(0.5), mi.Float(0.5),
            wavelength=self.WAVELENGTH,
            sigma_h=sigma_h_v,
        )

        eta, _ = compute_coherent_incoherent_blend(
            sigma_h_v, l_c_v, self.WAVELENGTH, cos_theta_i
        )

        tau_eff = compute_validity_aware_blend(
            tau, cos_theta_i, sigma_h_v, l_c_v, self.WAVELENGTH
        )

        return eta * tau_eff * A

    def eval_f_cos_components_physics(self, wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                       thickness):
        """Evaluate BSDF × |cos(θ_i)| with per-lobe components from direct physics params."""
        return self._eval_f_cos_components_core(
            wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau, thickness
        )


# FACTORY FUNCTIONS


__all__ = [
    'BSDFmmWaveJones',
    'compute_sp_basis_jones',
    'project_polarization_to_sp',
    'apply_jones_reflection',
    'compute_received_field_complex',
]
