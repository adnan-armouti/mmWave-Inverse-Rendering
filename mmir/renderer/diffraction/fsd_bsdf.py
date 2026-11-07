"""Free-space diffraction BSDF evaluation."""

import numpy as np
from dataclasses import dataclass
from typing import List


# FsdBSDF class

class FsdBSDF:
    """Free-space diffraction BSDF."""

    def __init__(self, k: float, beam_sigma: float, beta_max: float = 0.5):
        """Args: k: Wavenumber 2π/λ. beam_sigma: Gaussian beam standard deviation in meters.
        beta_max: Maximum energy borrowing fraction.
        """
        self.k = k
        self.beam_sigma = beam_sigma
        self.beta_max = beta_max


# Flat edge data structure for vectorized batch evaluation

@dataclass
class FlatEdgeData:
    """All edges from all apertures packed into flat contiguous arrays (CSR format)."""
    # Per-edge arrays (E = total edges across all apertures)
    edge_ex: np.ndarray        # [E] float64 — edge vector x
    edge_ey: np.ndarray        # [E] float64 — edge vector y
    edge_vx: np.ndarray        # [E] float64 — midpoint x
    edge_vy: np.ndarray        # [E] float64 — midpoint y
    edge_a_real: np.ndarray    # [E] float64 — amplitude a, real part
    edge_a_imag: np.ndarray    # [E] float64 — amplitude a, imag part
    edge_b_real: np.ndarray    # [E] float64 — amplitude b, real part
    edge_b_imag: np.ndarray    # [E] float64 — amplitude b, imag part

    # Per-edge material data (for Phase B differentiable re-evaluation)
    edge_opacity: np.ndarray   # [E] float64 — Fresnel opacity from Phase A
    edge_face_idx: np.ndarray  # [E] int32 — global face index of owning triangle
    edge_cos_theta: np.ndarray # [E] float64 — cos(θ_i) at edge's triangle surface

    # Per-edge barycentric + vertex data (for per-vertex material interpolation in Phase B)
    edge_bary_u: np.ndarray    # [E] float64 — barycentric u at edge midpoint (-1 if not set)
    edge_bary_v: np.ndarray    # [E] float64 — barycentric v at edge midpoint (-1 if not set)
    edge_vert_idx_0: np.ndarray  # [E] int32 — vertex index 0 of owning triangle (-1 if not set)
    edge_vert_idx_1: np.ndarray  # [E] int32 — vertex index 1 of owning triangle
    edge_vert_idx_2: np.ndarray  # [E] int32 — vertex index 2 of owning triangle

    # Jones polarization data (for Phase B complex correction)
    edge_jones_real: np.ndarray   # [E] float64 — Re(E_jones) from Phase A
    edge_jones_imag: np.ndarray   # [E] float64 — Im(E_jones) from Phase A
    edge_tx_amp_s: np.ndarray     # [E] float64 — TX s-projection (frozen)
    edge_tx_amp_p: np.ndarray     # [E] float64 — TX p-projection (frozen)
    edge_rx_amp_s: np.ndarray     # [E] float64 — RX s-projection (frozen)
    edge_rx_amp_p: np.ndarray     # [E] float64 — RX p-projection (frozen)

    # Per-edge endpoint barycentrics (for AD vertex position re-projection in E2E)
    # Tracks actual tessellated endpoint positions within the owning triangle.
    edge_ba_u: np.ndarray      # [E] float64 — barycentric u of endpoint A
    edge_ba_v: np.ndarray      # [E] float64 — barycentric v of endpoint A
    edge_bb_u: np.ndarray      # [E] float64 — barycentric u of endpoint B
    edge_bb_v: np.ndarray      # [E] float64 — barycentric v of endpoint B

    # Per-aperture arrays (A = number of apertures with edges)
    ap_edge_start: np.ndarray  # [A] int32 — start index in edge arrays
    ap_edge_count: np.ndarray  # [A] int32 — number of edges
    ap_P_A_hat: np.ndarray     # [A] float64 — clamped diffracted power (opacity-weighted)
    ap_P_A_hat_bare: np.ndarray  # [A] float64 — bare geometric P_A_hat (no material modulation) — for beta
    ap_P_A_bare: np.ndarray      # [A] float64 — bare geometric P_A full (no material modulation) — for normalization
    ap_tangent: np.ndarray     # [A, 3] float32 — screen tangent
    ap_bitangent: np.ndarray   # [A, 3] float32 — screen bitangent
    ap_wo_dir: np.ndarray      # [A, 3] float32 — screen z-axis (wo)
    ap_beta: np.ndarray        # [A] float64 — energy borrowing fraction
    ap_hit_pos: np.ndarray     # [A, 3] float64 — aperture center in world space

    # Mapping: which flat path indices does each aperture correspond to?
    ap_to_path_indices: List[np.ndarray]  # [A] → variable-length arrays of flat indices

    n_edges: int
    n_apertures: int


def _fresnel_opacity_drjit(
    eps_real: 'mi.Float',
    eps_imag: 'mi.Float',
    cos_theta_i: 'mi.Float',
) -> 'mi.Float':
    """Differentiable Fresnel power reflectance averaged over polarization."""
    import drjit as dr
    import mitsuba as mi

    # Convert permittivity to complex IOR: ñ = n - jκ
    # |ε| = √(ε'² + ε''²)
    eps_mag = dr.sqrt(eps_real * eps_real + eps_imag * eps_imag)
    # n = √((|ε| + ε') / 2), κ = √((|ε| - ε') / 2)
    n = dr.sqrt((eps_mag + eps_real) * mi.Float(0.5))
    kappa = dr.sqrt(dr.maximum(mi.Float(0.0), (eps_mag - eps_real) * mi.Float(0.5)))

    sin2 = mi.Float(1.0) - cos_theta_i * cos_theta_i

    # ñ² = n² - κ² - 2jnκ
    n2_real = n * n - kappa * kappa
    n2_imag = mi.Float(-2.0) * n * kappa

    # ξ = ñ² - sin²θ
    xi_real = n2_real - sin2
    xi_imag = n2_imag

    # √ξ = a + jb
    xi_mag = dr.sqrt(xi_real * xi_real + xi_imag * xi_imag)
    xi_arg = dr.atan2(xi_imag, xi_real)
    a = dr.sqrt(xi_mag) * dr.cos(xi_arg * mi.Float(0.5))
    b = dr.sqrt(xi_mag) * dr.sin(xi_arg * mi.Float(0.5))

    # R_s = |(cosθ - (a+jb)) / (cosθ + (a+jb))|²
    rs_num = (cos_theta_i - a) * (cos_theta_i - a) + b * b
    rs_den = dr.maximum((cos_theta_i + a) * (cos_theta_i + a) + b * b, mi.Float(1e-10))
    R_s = rs_num / rs_den

    # R_p = |(ñ²cosθ - (a+jb)) / (ñ²cosθ + (a+jb))|²
    n2cos_r = n2_real * cos_theta_i
    n2cos_i = n2_imag * cos_theta_i
    rp_num = (n2cos_r - a) * (n2cos_r - a) + (n2cos_i - b) * (n2cos_i - b)
    rp_den = dr.maximum((n2cos_r + a) * (n2cos_r + a) + (n2cos_i + b) * (n2cos_i + b), mi.Float(1e-10))
    R_p = rp_num / rp_den

    opacity = (R_s + R_p) * mi.Float(0.5)
    return dr.clip(opacity, mi.Float(0.0), mi.Float(1.0))


def _jones_reflectance_drjit(
    eps_real: 'mi.Float',     # per-edge ε'
    eps_imag: 'mi.Float',     # per-edge ε''
    cos_theta_i: 'mi.Float',  # per-edge cos(θ_i)
    tx_amp_s: 'mi.Float',     # per-edge TX s-projection (frozen)
    tx_amp_p: 'mi.Float',     # per-edge TX p-projection (frozen)
    rx_amp_s: 'mi.Float',     # per-edge RX s-projection (frozen)
    rx_amp_p: 'mi.Float',     # per-edge RX p-projection (frozen)
):
    """Differentiable Jones reflectance coefficient per edge."""
    import drjit as dr
    import mitsuba as mi

    # Convert permittivity to complex IOR: ñ = n - jκ
    eps_mag = dr.sqrt(eps_real * eps_real + eps_imag * eps_imag)
    n = dr.sqrt((eps_mag + eps_real) * mi.Float(0.5))
    kappa = dr.sqrt(dr.maximum(mi.Float(0.0), (eps_mag - eps_real) * mi.Float(0.5)))

    sin2 = mi.Float(1.0) - cos_theta_i * cos_theta_i
    n2_real = n * n - kappa * kappa
    n2_imag = mi.Float(-2.0) * n * kappa

    xi_real = n2_real - sin2
    xi_imag = n2_imag
    xi_mag = dr.sqrt(xi_real * xi_real + xi_imag * xi_imag)
    xi_arg = dr.atan2(xi_imag, xi_real)
    a = dr.sqrt(xi_mag) * dr.cos(xi_arg * mi.Float(0.5))
    b = dr.sqrt(xi_mag) * dr.sin(xi_arg * mi.Float(0.5))

    # Complex division: (p_r + j*p_i) / (q_r + j*q_i)
    rs_num_r = cos_theta_i - a
    rs_num_i = -b
    rs_den_r = cos_theta_i + a
    rs_den_i = b
    rs_den_sq = dr.maximum(rs_den_r * rs_den_r + rs_den_i * rs_den_i, mi.Float(1e-10))
    r_s_r = (rs_num_r * rs_den_r + rs_num_i * rs_den_i) / rs_den_sq
    r_s_i = (rs_num_i * rs_den_r - rs_num_r * rs_den_i) / rs_den_sq

    # r_p = (ñ²cos_θ - (a+jb)) / (ñ²cos_θ + (a+jb))
    n2cos_r = n2_real * cos_theta_i
    n2cos_i = n2_imag * cos_theta_i
    rp_num_r = n2cos_r - a
    rp_num_i = n2cos_i - b
    rp_den_r = n2cos_r + a
    rp_den_i = n2cos_i + b
    rp_den_sq = dr.maximum(rp_den_r * rp_den_r + rp_den_i * rp_den_i, mi.Float(1e-10))
    r_p_r = (rp_num_r * rp_den_r + rp_num_i * rp_den_i) / rp_den_sq
    r_p_i = (rp_num_i * rp_den_r - rp_num_r * rp_den_i) / rp_den_sq

    # E_s_out = r_s * tx_s → (r_s_r*tx_s, r_s_i*tx_s)
    # E_p_out = r_p * tx_p → (r_p_r*tx_p, r_p_i*tx_p)
    # E_jones = E_s_out * rx_s + E_p_out * rx_p (complex addition)
    E_r = r_s_r * tx_amp_s * rx_amp_s + r_p_r * tx_amp_p * rx_amp_p
    E_i = r_s_i * tx_amp_s * rx_amp_s + r_p_i * tx_amp_p * rx_amp_p

    return E_r, E_i


def eval_batch_drjit(
    flat: FlatEdgeData,
    wi_world,   # np.ndarray [N, 3] OR tuple (mi.Float, mi.Float, mi.Float) [N]
    k: float,
    aperture_indices: np.ndarray,  # [N] int — which aperture each eval belongs to
    eps_real_all: 'mi.Float' = None,  # [n_faces] grad-enabled ε' for differentiable opacity
    eps_imag_all: 'mi.Float' = None,  # [n_faces] grad-enabled ε'' for differentiable opacity
    eps_real_vertex: 'mi.Float' = None,  # [n_vertices] grad-enabled ε' (per-vertex mode)
    eps_imag_vertex: 'mi.Float' = None,  # [n_vertices] grad-enabled ε'' (per-vertex mode)
    jones_mode: bool = False,  # If True, use Jones complex correction instead of scalar opacity
    # AD-attached geometry overrides (E2E pipeline)
    cos_theta_live: 'mi.Float' = None,  # [E_total] replaces flat.edge_cos_theta
    live_edge_ex: 'mi.Float' = None,    # [E_total] replaces flat.edge_ex
    live_edge_ey: 'mi.Float' = None,    # [E_total] replaces flat.edge_ey
    live_edge_vx: 'mi.Float' = None,    # [E_total] replaces flat.edge_vx
    live_edge_vy: 'mi.Float' = None,    # [E_total] replaces flat.edge_vy
):
    """GPU-accelerated eval_batch using DrJit CUDA kernels."""
    import drjit as dr
    import mitsuba as mi

    A = flat.n_apertures

    # Handle wi_world as either numpy or DrJit tuple
    _wi_is_drjit = isinstance(wi_world, tuple)
    if _wi_is_drjit:
        N = dr.width(wi_world[0])
    else:
        N = len(wi_world)

    zero_n = dr.zeros(mi.Float, N)
    if A == 0 or N == 0:
        return zero_n, zero_n, zero_n

    # Upload edge data to GPU (float32)
    _has_live_edges = (live_edge_ex is not None and live_edge_ey is not None
                       and live_edge_vx is not None and live_edge_vy is not None)
    if _has_live_edges:
        g_ex = live_edge_ex   # [E_total] AD-attached mi.Float
        g_ey = live_edge_ey
        g_vx = live_edge_vx
        g_vy = live_edge_vy
    else:
        g_ex = mi.Float(flat.edge_ex.astype(np.float32))
        g_ey = mi.Float(flat.edge_ey.astype(np.float32))
        g_vx = mi.Float(flat.edge_vx.astype(np.float32))
        g_vy = mi.Float(flat.edge_vy.astype(np.float32))
    g_ar = mi.Float(flat.edge_a_real.astype(np.float32))
    g_ai = mi.Float(flat.edge_a_imag.astype(np.float32))
    g_br = mi.Float(flat.edge_b_real.astype(np.float32))
    g_bi = mi.Float(flat.edge_b_imag.astype(np.float32))

    # Per-aperture metadata on GPU
    g_tang_x = mi.Float(flat.ap_tangent[:, 0])
    g_tang_y = mi.Float(flat.ap_tangent[:, 1])
    g_tang_z = mi.Float(flat.ap_tangent[:, 2])
    g_bitang_x = mi.Float(flat.ap_bitangent[:, 0])
    g_bitang_y = mi.Float(flat.ap_bitangent[:, 1])
    g_bitang_z = mi.Float(flat.ap_bitangent[:, 2])
    g_wo_x = mi.Float(flat.ap_wo_dir[:, 0])
    g_wo_y = mi.Float(flat.ap_wo_dir[:, 1])
    g_wo_z = mi.Float(flat.ap_wo_dir[:, 2])
    mi.Float(flat.ap_P_A_hat.astype(np.float32))
    g_P_A_hat_bare = mi.Float(flat.ap_P_A_hat_bare.astype(np.float32))
    mi.Float(flat.ap_P_A_bare.astype(np.float32))

    # Compute expansion indices on CPU (integer math)
    edge_counts_per_eval = flat.ap_edge_count[aperture_indices]  # [N]
    total_exp_edges = int(np.sum(edge_counts_per_eval))

    if total_exp_edges == 0:
        return zero_n, zero_n, zero_n

    exp_eval_idx_np = np.repeat(np.arange(N, dtype=np.int32), edge_counts_per_eval)
    eval_exp_starts = np.empty(N, dtype=np.int64)
    eval_exp_starts[0] = 0
    if N > 1:
        np.cumsum(edge_counts_per_eval[:-1], out=eval_exp_starts[1:])
    local_offset_np = np.arange(total_exp_edges, dtype=np.int64) - \
        np.repeat(eval_exp_starts, edge_counts_per_eval)
    ap_starts_per_exp_np = flat.ap_edge_start[aperture_indices[exp_eval_idx_np]]
    src_edge_idx_np = (ap_starts_per_exp_np + local_offset_np).astype(np.int32)

    # Upload indices to GPU
    src_edge_idx = mi.UInt32(src_edge_idx_np)
    exp_eval_idx = mi.UInt32(exp_eval_idx_np)
    ap_idx = mi.UInt32(aperture_indices)

    # Projection on GPU
    if _wi_is_drjit:
        # AD-attached: use DrJit arrays directly (preserves gradient tape)
        wi_x, wi_y, wi_z = wi_world
    else:
        # Detached: create new mi.Float from numpy (backward-compatible)
        wi_x = mi.Float(wi_world[:, 0].astype(np.float32))
        wi_y = mi.Float(wi_world[:, 1].astype(np.float32))
        wi_z = mi.Float(wi_world[:, 2].astype(np.float32))

    tang_x = dr.gather(mi.Float, g_tang_x, ap_idx)
    tang_y = dr.gather(mi.Float, g_tang_y, ap_idx)
    tang_z = dr.gather(mi.Float, g_tang_z, ap_idx)
    bitang_x = dr.gather(mi.Float, g_bitang_x, ap_idx)
    bitang_y = dr.gather(mi.Float, g_bitang_y, ap_idx)
    bitang_z = dr.gather(mi.Float, g_bitang_z, ap_idx)
    wo_x = dr.gather(mi.Float, g_wo_x, ap_idx)
    wo_y = dr.gather(mi.Float, g_wo_y, ap_idx)
    wo_z = dr.gather(mi.Float, g_wo_z, ap_idx)

    dx = tang_x * wi_x + tang_y * wi_y + tang_z * wi_z
    dy = bitang_x * wi_x + bitang_y * wi_y + bitang_z * wi_z
    dz = wo_x * wi_x + wo_y * wi_y + wo_z * wi_z
    dz = dr.select(dr.abs(dz) < 1e-8, mi.Float(1e-8), dz)

    xi_x = dx / dz  # [N]
    xi_y = dy / dz  # [N]

    # Gather expanded edge data on GPU
    exp_ex = dr.gather(mi.Float, g_ex, src_edge_idx)
    exp_ey = dr.gather(mi.Float, g_ey, src_edge_idx)
    exp_vx = dr.gather(mi.Float, g_vx, src_edge_idx)
    exp_vy = dr.gather(mi.Float, g_vy, src_edge_idx)
    exp_ar = dr.gather(mi.Float, g_ar, src_edge_idx)
    exp_ai = dr.gather(mi.Float, g_ai, src_edge_idx)
    exp_br = dr.gather(mi.Float, g_br, src_edge_idx)
    exp_bi = dr.gather(mi.Float, g_bi, src_edge_idx)
    exp_xi_x = dr.gather(mi.Float, xi_x, exp_eval_idx)
    exp_xi_y = dr.gather(mi.Float, xi_y, exp_eval_idx)

    # Differentiable material correction (Phase B)
    # When grad-enabled material params are provided, re-compute Fresnel reflectance
    # and apply correction to amplitudes.
    #
    # Jones mode: complex correction E_jones_drjit / E_jones_numpy
    # Scalar mode: real correction √(opacity_drjit) / √(opacity_numpy)
    #
    # Material gathering modes:
    #   1. Per-vertex (eps_real_vertex/eps_imag_vertex): Barycentric interpolation at
    #      the edge midpoint. More accurate — each edge gets material at its actual
    #      position within the triangle, not the centroid average.
    #   2. Per-face (eps_real_all/eps_imag_all): Gather by face index. Fallback when
    #      per-vertex data or barycentrics are unavailable.
    _has_vertex_data = (eps_real_vertex is not None and eps_imag_vertex is not None
                        and flat.edge_bary_u is not None
                        and len(flat.edge_bary_u) > 0
                        and flat.edge_bary_u[0] >= 0)  # -1 = not set
    _has_face_data = (eps_real_all is not None and eps_imag_all is not None
                      and flat.edge_face_idx is not None)

    if _has_vertex_data or _has_face_data:
        # Gather per-edge cos_theta: use AD-attached if provided, else frozen Phase A
        exp_cos = dr.gather(mi.Float, cos_theta_live, src_edge_idx)

        g_vi0 = mi.UInt32(flat.edge_vert_idx_0.astype(np.int32))
        g_vi1 = mi.UInt32(flat.edge_vert_idx_1.astype(np.int32))
        g_vi2 = mi.UInt32(flat.edge_vert_idx_2.astype(np.int32))
        g_bu = mi.Float(flat.edge_bary_u.astype(np.float32))
        g_bv = mi.Float(flat.edge_bary_v.astype(np.float32))

        exp_vi0 = dr.gather(mi.UInt32, g_vi0, src_edge_idx)
        exp_vi1 = dr.gather(mi.UInt32, g_vi1, src_edge_idx)
        exp_vi2 = dr.gather(mi.UInt32, g_vi2, src_edge_idx)
        exp_bu = dr.gather(mi.Float, g_bu, src_edge_idx)
        exp_bv = dr.gather(mi.Float, g_bv, src_edge_idx)
        exp_bw = mi.Float(1.0) - exp_bu - exp_bv

        eps_r_v0 = dr.gather(mi.Float, eps_real_vertex, exp_vi0)
        eps_r_v1 = dr.gather(mi.Float, eps_real_vertex, exp_vi1)
        eps_r_v2 = dr.gather(mi.Float, eps_real_vertex, exp_vi2)
        edge_eps_r = exp_bw * eps_r_v0 + exp_bu * eps_r_v1 + exp_bv * eps_r_v2

        eps_i_v0 = dr.gather(mi.Float, eps_imag_vertex, exp_vi0)
        eps_i_v1 = dr.gather(mi.Float, eps_imag_vertex, exp_vi1)
        eps_i_v2 = dr.gather(mi.Float, eps_imag_vertex, exp_vi2)
        edge_eps_i = exp_bw * eps_i_v0 + exp_bu * eps_i_v1 + exp_bv * eps_i_v2

        g_jones_r = mi.Float(flat.edge_jones_real.astype(np.float32))
        g_jones_i = mi.Float(flat.edge_jones_imag.astype(np.float32))
        g_txs = mi.Float(flat.edge_tx_amp_s.astype(np.float32))
        g_txp = mi.Float(flat.edge_tx_amp_p.astype(np.float32))
        g_rxs = mi.Float(flat.edge_rx_amp_s.astype(np.float32))
        g_rxp = mi.Float(flat.edge_rx_amp_p.astype(np.float32))

        exp_jones_r_np = dr.gather(mi.Float, g_jones_r, src_edge_idx)
        exp_jones_i_np = dr.gather(mi.Float, g_jones_i, src_edge_idx)
        exp_tx_s = dr.gather(mi.Float, g_txs, src_edge_idx)
        exp_tx_p = dr.gather(mi.Float, g_txp, src_edge_idx)
        exp_rx_s = dr.gather(mi.Float, g_rxs, src_edge_idx)
        exp_rx_p = dr.gather(mi.Float, g_rxp, src_edge_idx)

        # Compute differentiable Jones reflectance
        E_dr_r, E_dr_i = _jones_reflectance_drjit(
            edge_eps_r, edge_eps_i, exp_cos,
            exp_tx_s, exp_tx_p, exp_rx_s, exp_rx_p
        )

        # Complex correction: E_jones_drjit / E_jones_numpy
        # (a+jb)/(c+jd) = ((ac+bd) + j(bc-ad)) / (c²+d²)
        denom = dr.maximum(
            exp_jones_r_np * exp_jones_r_np + exp_jones_i_np * exp_jones_i_np,
            mi.Float(1e-20)
        )
        corr_r = (E_dr_r * exp_jones_r_np + E_dr_i * exp_jones_i_np) / denom
        corr_i = (E_dr_i * exp_jones_r_np - E_dr_r * exp_jones_i_np) / denom

        # Apply complex correction to edge amplitudes
        new_ar = exp_ar * corr_r - exp_ai * corr_i
        new_ai = exp_ar * corr_i + exp_ai * corr_r
        new_br = exp_br * corr_r - exp_bi * corr_i
        new_bi = exp_br * corr_i + exp_bi * corr_r
        exp_ar, exp_ai = new_ar, new_ai
        exp_br, exp_bi = new_br, new_bi

    # Psihat math on GPU
    ee = dr.sqrt(exp_ex * exp_ex + exp_ey * exp_ey)
    vxi = exp_vx * exp_xi_x + exp_vy * exp_xi_y
    mx = exp_ey
    my = -exp_ex

    kf = mi.Float(k)
    zeta_x = kf * (exp_ex * exp_xi_x + exp_ey * exp_xi_y)
    zeta_y = kf * (mx * exp_xi_x + my * exp_xi_y)
    zeta_sq = zeta_x * zeta_x + zeta_y * zeta_y

    chi_val = dr.sqrt(dr.maximum(mi.Float(0.0),
                                 mi.Float(1.0) - dr.exp(mi.Float(-0.5) * zeta_sq / mi.Float(3.0))))

    safe_zx = dr.select(dr.abs(zeta_x) < 1e-12, mi.Float(1e-12), zeta_x)
    safe_sq = dr.select(zeta_sq < 1e-20, mi.Float(1e-20), zeta_sq)
    half_zx = safe_zx * mi.Float(0.5)
    abs_half = dr.abs(half_zx)
    sinc_half = dr.select(abs_half < 1e-8, mi.Float(1.0), dr.sin(half_zx) / half_zx)
    cos_half = dr.cos(half_zx)
    inv_pi = mi.Float(1.0 / np.pi)

    a1 = zeta_y / safe_sq * inv_pi * (cos_half - sinc_half) / (mi.Float(2.0) * safe_zx)
    a2 = zeta_y / safe_sq * inv_pi * sinc_half / mi.Float(2.0)

    degen = (zeta_sq < 1e-20) | (dr.abs(zeta_x) < 1e-12)
    a1 = dr.select(degen, mi.Float(0.0), a1)
    a2 = dr.select(zeta_sq < 1e-20, mi.Float(0.0), a2)

    diff_r = (exp_ar - exp_br) * a1
    diff_i = (exp_ai - exp_bi) * a1
    sum_r_raw = (exp_ar + exp_br) * mi.Float(0.5) * a2
    sum_i_raw = (exp_ai + exp_bi) * mi.Float(0.5) * a2
    c_real = diff_r + (-sum_i_raw)
    c_imag = diff_i + sum_r_raw

    phase_arg = -kf * vxi
    phase_r = dr.cos(phase_arg)
    phase_i = dr.sin(phase_arg)

    scale = kf * ee * ee * chi_val
    pc_r = phase_r * c_real - phase_i * c_imag
    pc_i = phase_r * c_imag + phase_i * c_real
    psi_edge_r = scale * pc_r
    psi_edge_i = scale * pc_i

    # Segment reduction on GPU
    psi_real_sum = dr.zeros(mi.Float, N)
    psi_imag_sum = dr.zeros(mi.Float, N)
    dr.scatter_reduce(dr.ReduceOp.Add, psi_real_sum, psi_edge_r, exp_eval_idx)
    dr.scatter_reduce(dr.ReduceOp.Add, psi_imag_sum, psi_edge_i, exp_eval_idx)
    dr.eval(psi_real_sum, psi_imag_sum)  # CRITICAL: eval before using scatter-modified vars

    # Normalization (use bare geometric P_A_hat — no material modulation)
    cos_theta = mi.Float(1.0) / dr.sqrt(mi.Float(1.0) + xi_x * xi_x + xi_y * xi_y)
    P_A_hat_bare = dr.gather(mi.Float, g_P_A_hat_bare, ap_idx)
    denom = dr.maximum(cos_theta * P_A_hat_bare, mi.Float(1e-4))
    intensity = psi_real_sum * psi_real_sum + psi_imag_sum * psi_imag_sum
    f_diff = intensity / denom

    norm_val = dr.sqrt(denom)
    psi_norm_real = psi_real_sum / norm_val
    psi_norm_imag = psi_imag_sum / norm_val

    return f_diff, psi_norm_real, psi_norm_imag
