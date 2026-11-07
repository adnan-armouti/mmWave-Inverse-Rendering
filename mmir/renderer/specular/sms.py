"""Specular Manifold Sampling (SMS) for single-bounce specular path finding."""

from typing import Optional, Tuple
from dataclasses import dataclass
import drjit as dr
import mitsuba as mi
import numpy as np
import time

from .paths import SpecularPaths
from ..bsdf.mmwave_scalar import (
    WAVELENGTH_77GHZ, enforce_spm_validity, compute_slab_energy_gate, compute_validity_aware_blend,
    compute_coherent_incoherent_blend,
)
from ..utils.math import safe_normalize


@dataclass
class SpecularGradInfo:
    """Final-iteration state from SMS Newton solver needed for IFT gradient computation."""
    # Inverse of final 2×2 Newton Jacobian (∂C/∂(u,v))⁻¹
    inv_J00: 'mi.Float'  # [n_paths]
    inv_J01: 'mi.Float'  # [n_paths]
    inv_J10: 'mi.Float'  # [n_paths]
    inv_J11: 'mi.Float'  # [n_paths]
    # Surface tangent vectors at converged point
    dp_du: 'mi.Vector3f'  # [n_paths]
    dp_dv: 'mi.Vector3f'  # [n_paths]
    # TX/RX positions used in the forward solve (needed for constraint re-eval)
    tx_pos: 'mi.Point3f'  # [n_paths]
    rx_pos: 'mi.Point3f'  # [n_paths]


@dataclass
class MultibounceSpecularGradInfo:
    """Final-iteration state from multibounce SMS Newton solver for IFT."""
    k: int  # number of specular bounces in this chain

    # Per-vertex surface tangents at converged points: lists of length k
    dp_du: list  # [k] mi.Vector3f arrays, each [n_chains]
    dp_dv: list  # [k] mi.Vector3f arrays, each [n_chains]

    # Block-tridiagonal factorization from Thomas algorithm forward sweep:
    # D_inv[i] = 4-tuple of mi.Float (inverse of modified diagonal block)
    # U[i] = 4-tuple of mi.Float (upper coupling = D_inv[i] @ B[i])
    D_inv: list  # [k] 4-tuples (a00, a01, a10, a11)
    U: list      # [k-1] 4-tuples (for backward sweep)

    # TX/RX positions used in the forward solve
    tx_pos: 'mi.Point3f'  # [n_chains]
    rx_pos: 'mi.Point3f'  # [n_chains]


@dataclass
class MultibounceSpecularChain:
    """Result of the multibounce SMS Newton solver."""
    k: int  # number of specular bounces

    # Per-vertex arrays: lists of length k
    hit_P: list       # [k] mi.Point3f arrays, each [n_chains]
    hit_N: list       # [k] mi.Vector3f arrays, each [n_chains]
    prim_ids: list    # [k] mi.UInt32 arrays, each [n_chains]
    bary_u: list      # [k] mi.Float arrays, each [n_chains]
    bary_v: list      # [k] mi.Float arrays, each [n_chains]
    A_tri: list       # [k] mi.Float arrays (triangle area at each vertex)

    # Per-chain arrays [n_chains]
    tx_idx: 'mi.UInt32'
    rx_idx: 'mi.UInt32'
    valid: 'mi.Bool'
    n_chains: int

    # Gradient info for IFT
    grad_info: Optional[MultibounceSpecularGradInfo]

    # Per-segment distances: lists of length k+1
    # d_segments[0] = RX→S₁, d_segments[i] = Sᵢ→Sᵢ₊₁, d_segments[k] = Sₖ→TX
    d_segments: list   # [k+1] mi.Float arrays
    dir_segments: list  # [k+1] mi.Vector3f arrays (unit direction vectors)


class SpecularManifoldSampler:
    """Single-bounce Specular Manifold Sampling for mmWave radar."""

    def __init__(
        self,
        wavelength: float = WAVELENGTH_77GHZ,
        max_iterations: int = 20,
        solver_threshold: float = 1e-5,
        use_smooth_normals: bool = False,
        bsdf=None,
    ):
        self.wavelength = wavelength
        self.max_iterations = max_iterations
        self.solver_threshold = solver_threshold
        self.use_smooth_normals = use_smooth_normals
        self.bsdf = bsdf

        # Lazy-initialized mesh data for computing dn/du, dn/dv
        # (needed because Mitsuba 3 returns zero dn_du/dn_dv for PLY meshes)
        self._mesh_data_ready = False
        self._fi = None       # face indices, flat UInt32 [n_faces * 3]
        self._vn_x = None     # vertex normal x components [n_vertices]
        self._vn_y = None     # vertex normal y components [n_vertices]
        self._vn_z = None     # vertex normal z components [n_vertices]
        self._n_mesh_faces = 0

    # Block-tridiagonal Thomas algorithm for multibounce SMS

    @staticmethod
    def _inv2x2(a00, a01, a10, a11):
        """Invert a 2x2 block represented as 4 scalar DrJit arrays."""
        det = a00 * a11 - a01 * a10
        det_safe = dr.select(dr.abs(det) > mi.Float(1e-12), det, mi.Float(1e-12))
        inv_det = mi.Float(1.0) / det_safe
        return (a11 * inv_det, -a01 * inv_det, -a10 * inv_det, a00 * inv_det)

    @staticmethod
    def _mul2x2(a, b):
        """Multiply two 2x2 blocks (each a 4-tuple of DrJit arrays)."""
        return (a[0]*b[0] + a[1]*b[2], a[0]*b[1] + a[1]*b[3],
                a[2]*b[0] + a[3]*b[2], a[2]*b[1] + a[3]*b[3])

    @staticmethod
    def _mul2x2_vec(a, v):
        """Multiply 2x2 block by 2-vector (each a tuple of DrJit arrays)."""
        return (a[0]*v[0] + a[1]*v[1], a[2]*v[0] + a[3]*v[1])

    @staticmethod
    def _sub2x2(a, b):
        """Subtract two 2x2 blocks."""
        return (a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3])

    def _thomas_block_tridiagonal(
        self,
        A: list,        # [k] diagonal 2x2 blocks (4-tuples of mi.Float)
        B: list,        # [k-1] upper 2x2 blocks
        C_lower: list,  # [k-1] lower 2x2 blocks
        rhs: list,      # [k] right-hand side 2-vectors (2-tuples of mi.Float)
        active: 'mi.Bool',
    ) -> Tuple[list, list, list]:
        """Solve block-tridiagonal system via Thomas algorithm."""
        k = len(A)

        # Forward sweep
        D_inv = [None] * k
        y = [None] * k
        U = [None] * max(k - 1, 1)

        D_inv[0] = self._inv2x2(*A[0])
        y[0] = self._mul2x2_vec(D_inv[0], rhs[0])

        for i in range(1, k):
            temp = self._mul2x2(C_lower[i-1], D_inv[i-1])
            CDB = self._mul2x2(temp, B[i-1])
            D_i = self._sub2x2(A[i], CDB)
            D_inv[i] = self._inv2x2(*D_i)

            Cy = self._mul2x2_vec(C_lower[i-1], y[i-1])
            rhs_mod = (rhs[i][0] - Cy[0], rhs[i][1] - Cy[1])
            y[i] = self._mul2x2_vec(D_inv[i], rhs_mod)

        if k > 1:
            U[0] = self._mul2x2(D_inv[0], B[0])

        # Backward sweep
        x = [None] * k
        x[k-1] = y[k-1]

        for i in range(k-2, -1, -1):
            Ux = self._mul2x2_vec(U[i], x[i+1])
            x[i] = (y[i][0] - Ux[0], y[i][1] - Ux[1])

        return x, D_inv, U[:k-1] if k > 1 else []

    # Main entry point

    def find_specular_paths(
        self,
        unique_mask: 'mi.Bool',
        hit_P: 'mi.Point3f',
        hit_N: 'mi.Vector3f',
        hit_rho: np.ndarray,
        hit_ID: 'mi.Int32',
        tx_positions: 'mi.Point3f',
        rx_positions: 'mi.Point3f',
        n_hits_per_rx: int,
        scene: 'mi.Scene',
        verbose: bool = True,
        patch_data: Optional[object] = None,
        triangle_materials: Optional[np.ndarray] = None,
    ) -> SpecularPaths:
        """Find specular paths via Specular Manifold Sampling."""
        from ..utils.math import gather_point3f, gather_vector3f

        t_start = time.perf_counter()

        n_tx = dr.width(tx_positions)
        n_rx = dr.width(rx_positions)

        # Step 1: Extract unique seed positions from dedup mask
        unique_np = np.array(unique_mask)
        unique_indices = np.where(unique_np)[0]
        n_unique = len(unique_indices)

        if verbose:
            print("\n[SMS] Specular Manifold Sampling")
            print(f"  Unique seeds: {n_unique}")
            print(f"  TX elements: {n_tx}, RX elements: {n_rx}")
            print(f"  Max Newton iterations: {self.max_iterations}")
            print(f"  Convergence threshold: {self.solver_threshold}")

        unique_idx_dr = mi.UInt32(unique_indices)

        # Gather unique seed data
        seed_P = gather_point3f(hit_P, unique_idx_dr)
        gather_vector3f(hit_N, unique_idx_dr)
        seed_rho = hit_rho[unique_np]  # [n_unique, cols]
        dr.gather(mi.Int32, hit_ID, unique_idx_dr)
        seed_rx_idx_np = unique_indices // n_hits_per_rx

        # Step 2: Expand seeds × TX → (n_unique × n_tx) candidates
        n_paths = n_unique * n_tx
        if verbose:
            print(f"  SMS path candidates: {n_unique} × {n_tx} TX = {n_paths:,}")

        # Expand seed positions and normals
        exp_seed_P = mi.Point3f(
            dr.repeat(seed_P.x, n_tx), dr.repeat(seed_P.y, n_tx), dr.repeat(seed_P.z, n_tx)
        )

        # Expand TX/RX indices
        exp_tx_idx = mi.UInt32(np.tile(np.arange(n_tx), n_unique))
        exp_rx_idx = mi.UInt32(np.repeat(seed_rx_idx_np, n_tx))

        # Gather TX and RX positions for each candidate
        exp_tx_pos = mi.Point3f(
            dr.gather(mi.Float, tx_positions.x, exp_tx_idx),
            dr.gather(mi.Float, tx_positions.y, exp_tx_idx),
            dr.gather(mi.Float, tx_positions.z, exp_tx_idx),
        )
        exp_rx_pos = mi.Point3f(
            dr.gather(mi.Float, rx_positions.x, exp_rx_idx),
            dr.gather(mi.Float, rx_positions.y, exp_rx_idx),
            dr.gather(mi.Float, rx_positions.z, exp_rx_idx),
        )

        # Material info
        n_mat_cols = seed_rho.shape[1]
        physics_mode = (n_mat_cols == 6)

        # Step 3: Get surface parameterization at seed points
        init_result = self._get_surface_parameterization(
            exp_seed_P, exp_rx_pos, scene)
        dp_du, dp_dv, vtx_n, vtx_s, vtx_t, init_prim_ids, valid_init, \
            init_shape, init_dn_du, init_dn_dv = init_result

        # Step 4: Vectorized Newton solver
        t_newton_start = time.perf_counter()

        converged_P, converged_N, converged_mask, converged_prim_ids, n_iters_used, \
            newton_grad_info, converged_dp_du, converged_dp_dv, last_bary_u, last_bary_v = \
            self._newton_solve_vectorized(
                vtx_p=exp_seed_P,
                vtx_n=vtx_n,
                vtx_dp_du=dp_du,
                vtx_dp_dv=dp_dv,
                rx_pos=exp_rx_pos,
                tx_pos=exp_tx_pos,
                scene=scene,
                valid_init=valid_init,
                init_prim_ids=init_prim_ids,
                init_shape=init_shape,
                init_dn_du=init_dn_du,
                init_dn_dv=init_dn_dv,
            )

        t_newton = time.perf_counter() - t_newton_start
        n_converged = int(dr.sum(mi.UInt32(converged_mask))[0])

        if verbose:
            print(f"  Newton solver: {n_converged:,}/{n_paths:,} converged "
                  f"({100*n_converged/max(n_paths,1):.1f}%) in {n_iters_used} iterations, "
                  f"{t_newton:.3f}s")

        # Step 5: Post-Newton validation (geometry + visibility)
        delta_tx = mi.Vector3f(
            exp_tx_pos.x - converged_P.x,
            exp_tx_pos.y - converged_P.y,
            exp_tx_pos.z - converged_P.z,
        )
        delta_rx = mi.Vector3f(
            exp_rx_pos.x - converged_P.x,
            exp_rx_pos.y - converged_P.y,
            exp_rx_pos.z - converged_P.z,
        )
        d_tx = dr.norm(delta_tx)
        d_rx = dr.norm(delta_rx)

        dir_to_tx = mi.Vector3f(
            delta_tx.x / dr.maximum(d_tx, mi.Float(1e-10)),
            delta_tx.y / dr.maximum(d_tx, mi.Float(1e-10)),
            delta_tx.z / dr.maximum(d_tx, mi.Float(1e-10)),
        )
        dir_to_rx = mi.Vector3f(
            delta_rx.x / dr.maximum(d_rx, mi.Float(1e-10)),
            delta_rx.y / dr.maximum(d_rx, mi.Float(1e-10)),
            delta_rx.z / dr.maximum(d_rx, mi.Float(1e-10)),
        )

        cos_theta_i = dr.maximum(dr.dot(dir_to_tx, converged_N), mi.Float(0.0))
        cos_theta_r = dr.maximum(dr.dot(dir_to_rx, converged_N), mi.Float(0.0))

        # Geometry validation
        valid_dist = (d_tx > mi.Float(1e-4)) & (d_rx > mi.Float(1e-4))
        valid_angle = cos_theta_i > mi.Float(1e-6)
        valid_geom = converged_mask & valid_dist & valid_angle

        # Visibility check
        valid_vis = self._check_visibility(
            converged_P, dir_to_tx, dir_to_rx, d_tx, d_rx, scene,
            surface_normal=converged_N,
            skip_rx=True,  # RX visibility guaranteed by Newton reprojection
        )
        valid_all = valid_geom & valid_vis

        int(dr.sum(mi.UInt32(valid_geom))[0])
        n_valid_all = int(dr.sum(mi.UInt32(valid_all))[0])

        # Step 6: Compute triangle/patch area (from Newton solver state)
        # Use barycentrics and dp_du/dp_dv cached by Newton solver
        # (avoids an extra scene.ray_intersect call)
        spec_bary_u = last_bary_u
        spec_bary_v = last_bary_v

        # Triangle area from Newton solver's final dp_du, dp_dv
        cross_vec = dr.cross(converged_dp_du, converged_dp_dv)
        A_tri_raw = mi.Float(0.5) * dr.norm(cross_vec)

        # SMS operates at individual triangle level — use triangle area, not patch area
        A_tri = A_tri_raw
        exp_patch_id = None

        # Step 7: Compute specular weight using CONVERGED material
        n_tris_mat = triangle_materials.shape[0]
        prim_np = np.array(converged_prim_ids).astype(np.int64)
        prim_np = np.clip(prim_np, 0, n_tris_mat - 1)
        conv_rho = triangle_materials[prim_np]  # [n_paths, n_cols]

        R_specular = self._compute_specular_weight_physics(
            cos_theta_i,
            mi.Float(conv_rho[:, 0].astype(np.float32)),
            mi.Float(conv_rho[:, 1].astype(np.float32)),
            mi.Float(conv_rho[:, 2].astype(np.float32)),
            mi.Float(conv_rho[:, 3].astype(np.float32)),
            mi.Float(conv_rho[:, 4].astype(np.float32)),
            mi.Float(conv_rho[:, 5].astype(np.float32)),
        )
        R_specular = dr.select(valid_all, R_specular, mi.Float(0.0))

        t_total = time.perf_counter() - t_start
        print(f"  [SMS] Converged: {n_converged:,}/{n_paths:,} ({100*n_converged/max(n_paths,1):.1f}%) | "
              f"Valid specular paths: {n_valid_all:,}/{n_paths:,} | "
              f"Time: {t_total:.3f}s")

        return SpecularPaths(
            hit_P=converged_P,
            hit_N=converged_N,
            d_tx=d_tx,
            d_rx=d_rx,
            dir_to_tx=dir_to_tx,
            dir_to_rx=dir_to_rx,
            cos_theta_i=cos_theta_i,
            cos_theta_r=cos_theta_r,
            tx_idx=exp_tx_idx,
            rx_idx=exp_rx_idx,
            valid=valid_all,
            R_specular=R_specular,
            A_tri=A_tri,
            patch_id=exp_patch_id,
            prim_ids=converged_prim_ids,
            n_paths=n_paths,
            grad_info=newton_grad_info,
            bary_u=spec_bary_u,
            bary_v=spec_bary_v,
        )

    def find_specular_paths_from_seeds(
        self,
        seed_P: 'mi.Point3f',
        seed_N: 'mi.Vector3f',
        seed_prim_ids: 'mi.UInt32',
        seed_rx_idx: 'mi.UInt32',
        tx_positions: 'mi.Point3f',
        rx_positions: 'mi.Point3f',
        scene: 'mi.Scene',
        verbose: bool = True,
        # GPU-native material columns (preferred over triangle_materials)
        triangle_materials_gpu: Optional[list] = None,
        n_material_cols: int = 0,
        triangle_materials: Optional[np.ndarray] = None,
        patch_data: Optional[object] = None,
    ) -> SpecularPaths:
        """Find specular paths from pre-extracted seed positions."""

        t_start = time.perf_counter()

        n_tx = dr.width(tx_positions)
        n_rx = dr.width(rx_positions)
        n_unique = dr.width(seed_P)

        # Setup mesh normal data for smooth normals

        if verbose:
            print("\n[SMS-E2E] Specular Manifold Sampling (from seeds)")
            print(f"  Unique seeds: {n_unique}")
            print(f"  TX elements: {n_tx}, RX elements: {n_rx}")
            print(f"  Max Newton iterations: {self.max_iterations}")

        # Determine material source: GPU columns (fast) or numpy (legacy)
        _has_gpu_mats = triangle_materials_gpu is not None and len(triangle_materials_gpu) > 0
        _has_np_mats = triangle_materials is not None
        physics_mode = (n_material_cols == 6) if _has_gpu_mats else (
            _has_np_mats and triangle_materials.shape[1] == 6)
        n_mat_cols = n_material_cols if _has_gpu_mats else (
            triangle_materials.shape[1] if _has_np_mats else 0)

        # Step 2: Expand seeds × TX → (n_unique × n_tx) candidates
        n_paths = n_unique * n_tx
        if verbose:
            print(f"  SMS path candidates: {n_unique} × {n_tx} TX = {n_paths:,}")

        exp_seed_P = mi.Point3f(
            dr.repeat(seed_P.x, n_tx), dr.repeat(seed_P.y, n_tx), dr.repeat(seed_P.z, n_tx)
        )

        # PERF: GPU-native tiling instead of np.tile/np.repeat
        exp_tx_idx = dr.arange(mi.UInt32, n_paths) % mi.UInt32(n_tx)
        exp_rx_idx = mi.UInt32(dr.repeat(seed_rx_idx, n_tx))

        exp_tx_pos = mi.Point3f(
            dr.gather(mi.Float, tx_positions.x, exp_tx_idx),
            dr.gather(mi.Float, tx_positions.y, exp_tx_idx),
            dr.gather(mi.Float, tx_positions.z, exp_tx_idx),
        )
        exp_rx_pos = mi.Point3f(
            dr.gather(mi.Float, rx_positions.x, exp_rx_idx),
            dr.gather(mi.Float, rx_positions.y, exp_rx_idx),
            dr.gather(mi.Float, rx_positions.z, exp_rx_idx),
        )

        # Step 3: Get surface parameterization at seed points
        init_result = self._get_surface_parameterization(
            exp_seed_P, exp_rx_pos, scene)
        dp_du, dp_dv, vtx_n, vtx_s, vtx_t, init_prim_ids, valid_init, \
            init_shape, init_dn_du, init_dn_dv = init_result

        # Step 4: Vectorized Newton solver
        t_newton_start = time.perf_counter()

        converged_P, converged_N, converged_mask, converged_prim_ids, n_iters_used, \
            newton_grad_info, converged_dp_du, converged_dp_dv, last_bary_u, last_bary_v = \
            self._newton_solve_vectorized(
                vtx_p=exp_seed_P,
                vtx_n=vtx_n,
                vtx_dp_du=dp_du,
                vtx_dp_dv=dp_dv,
                rx_pos=exp_rx_pos,
                tx_pos=exp_tx_pos,
                scene=scene,
                valid_init=valid_init,
                init_prim_ids=init_prim_ids,
                init_shape=init_shape,
                init_dn_du=init_dn_du,
                init_dn_dv=init_dn_dv,
            )

        t_newton = time.perf_counter() - t_newton_start
        n_converged = int(dr.sum(mi.UInt32(converged_mask))[0])

        if verbose:
            print(f"  Newton solver: {n_converged:,}/{n_paths:,} converged "
                  f"({100*n_converged/max(n_paths,1):.1f}%) in {n_iters_used} iterations, "
                  f"{t_newton:.3f}s")

        # Step 5: Post-Newton validation (geometry + visibility)
        delta_tx = mi.Vector3f(
            exp_tx_pos.x - converged_P.x,
            exp_tx_pos.y - converged_P.y,
            exp_tx_pos.z - converged_P.z,
        )
        delta_rx = mi.Vector3f(
            exp_rx_pos.x - converged_P.x,
            exp_rx_pos.y - converged_P.y,
            exp_rx_pos.z - converged_P.z,
        )
        d_tx = dr.norm(delta_tx)
        d_rx = dr.norm(delta_rx)

        dir_to_tx = mi.Vector3f(
            delta_tx.x / dr.maximum(d_tx, mi.Float(1e-10)),
            delta_tx.y / dr.maximum(d_tx, mi.Float(1e-10)),
            delta_tx.z / dr.maximum(d_tx, mi.Float(1e-10)),
        )
        dir_to_rx = mi.Vector3f(
            delta_rx.x / dr.maximum(d_rx, mi.Float(1e-10)),
            delta_rx.y / dr.maximum(d_rx, mi.Float(1e-10)),
            delta_rx.z / dr.maximum(d_rx, mi.Float(1e-10)),
        )

        cos_theta_i = dr.maximum(dr.dot(dir_to_tx, converged_N), mi.Float(0.0))
        cos_theta_r = dr.maximum(dr.dot(dir_to_rx, converged_N), mi.Float(0.0))

        valid_dist = (d_tx > mi.Float(1e-4)) & (d_rx > mi.Float(1e-4))
        valid_angle = cos_theta_i > mi.Float(1e-6)
        valid_geom = converged_mask & valid_dist & valid_angle

        valid_vis = self._check_visibility(
            converged_P, dir_to_tx, dir_to_rx, d_tx, d_rx, scene,
            surface_normal=converged_N,
            skip_rx=True,
        )
        valid_all = valid_geom & valid_vis

        n_valid_all = int(dr.sum(mi.UInt32(valid_all))[0])

        if verbose:
            print(f"  Converged: {n_converged:,} | Valid: {n_valid_all:,}/{n_paths:,}")

        # Step 6: Triangle area
        spec_bary_u = last_bary_u
        spec_bary_v = last_bary_v

        cross_vec = dr.cross(converged_dp_du, converged_dp_dv)
        A_tri = mi.Float(0.5) * dr.norm(cross_vec)

        exp_patch_id = None

        # Step 7: Compute specular weight using CONVERGED material
        # GPU-native path: gather material columns directly on GPU (no CPU round trip)
        if _has_gpu_mats and n_valid_all > 0:
            n_tris_mat = dr.width(triangle_materials_gpu[0])
            safe_prim = dr.minimum(mi.UInt32(dr.maximum(mi.Int32(converged_prim_ids), mi.Int32(0))),
                                   mi.UInt32(n_tris_mat - 1))
            mat_cols = [dr.gather(mi.Float, triangle_materials_gpu[c], safe_prim)
                        for c in range(n_mat_cols)]
            R_specular = self._compute_specular_weight_physics(
                cos_theta_i, mat_cols[0], mat_cols[1], mat_cols[2],
                mat_cols[3], mat_cols[4], mat_cols[5])
        elif _has_np_mats and n_valid_all > 0:
            # Legacy numpy path (for non-e2e callers)
            n_tris_mat = triangle_materials.shape[0]
            prim_np = np.array(converged_prim_ids).astype(np.int64)
            prim_np = np.clip(prim_np, 0, n_tris_mat - 1)
            conv_rho = triangle_materials[prim_np]
            if physics_mode:
                R_specular = self._compute_specular_weight_physics(
                    cos_theta_i,
                    mi.Float(conv_rho[:, 0].astype(np.float32)),
                    mi.Float(conv_rho[:, 1].astype(np.float32)),
                    mi.Float(conv_rho[:, 2].astype(np.float32)),
                    mi.Float(conv_rho[:, 3].astype(np.float32)),
                    mi.Float(conv_rho[:, 4].astype(np.float32)),
                    mi.Float(conv_rho[:, 5].astype(np.float32)))
            else:
                R_specular = self._compute_specular_weight_legacy(
                    cos_theta_i,
                    mi.Float(conv_rho[:, 0].astype(np.float32)),
                    mi.Float(conv_rho[:, 1].astype(np.float32)),
                    mi.Float(conv_rho[:, 2].astype(np.float32)))
        else:
            R_specular = dr.zeros(mi.Float, n_paths)
        R_specular = dr.select(valid_all, R_specular, mi.Float(0.0))

        t_total = time.perf_counter() - t_start
        print(f"  [SMS-E2E] Converged: {n_converged:,}/{n_paths:,} ({100*n_converged/max(n_paths,1):.1f}%) | "
              f"Valid specular paths: {n_valid_all:,}/{n_paths:,} | "
              f"Time: {t_total:.3f}s")

        return SpecularPaths(
            hit_P=converged_P,
            hit_N=converged_N,
            d_tx=d_tx,
            d_rx=d_rx,
            dir_to_tx=dir_to_tx,
            dir_to_rx=dir_to_rx,
            cos_theta_i=cos_theta_i,
            cos_theta_r=cos_theta_r,
            tx_idx=exp_tx_idx,
            rx_idx=exp_rx_idx,
            valid=valid_all,
            R_specular=R_specular,
            A_tri=A_tri,
            patch_id=exp_patch_id,
            prim_ids=converged_prim_ids,
            n_paths=n_paths,
            grad_info=newton_grad_info,
            bary_u=spec_bary_u,
            bary_v=spec_bary_v,
        )

    # Surface parameterization

    def _get_surface_parameterization(
        self,
        seed_P: 'mi.Point3f',
        rx_pos: 'mi.Point3f',
        scene: 'mi.Scene',
    ):
        """Get surface parameterization at seed points by casting rays from RX."""
        ray_dir = safe_normalize(mi.Vector3f(
            seed_P.x - rx_pos.x,
            seed_P.y - rx_pos.y,
            seed_P.z - rx_pos.z,
        ))
        rays = mi.Ray3f(rx_pos, ray_dir)
        si = scene.ray_intersect(rays)

        dp_du = si.dp_du
        dp_dv = si.dp_dv

        n = si.n

        # Gram-Schmidt orthogonalization: s = normalize(dp_du - n * dot(n, dp_du))
        # Reference: compute_shading_frame (frame.h:197-200)
        n_dot_dpdu = dr.dot(n, dp_du)
        s_raw = mi.Vector3f(dp_du.x - n.x * n_dot_dpdu,
                            dp_du.y - n.y * n_dot_dpdu,
                            dp_du.z - n.z * n_dot_dpdu)
        s = safe_normalize(s_raw)
        t = dr.cross(n, s)

        prim_ids = mi.UInt32(si.prim_index)
        valid = si.is_valid()
        shape = si.shape

        # Normal derivatives
        n_paths = dr.width(seed_P)
        zero = mi.Vector3f(dr.zeros(mi.Float, n_paths),
                           dr.zeros(mi.Float, n_paths),
                           dr.zeros(mi.Float, n_paths))
        dn_du, dn_dv = zero, zero

        return dp_du, dp_dv, n, s, t, prim_ids, valid, shape, dn_du, dn_dv

    # Newton solver

    def _newton_solve_vectorized(
        self,
        vtx_p: 'mi.Point3f',
        vtx_n: 'mi.Vector3f',
        vtx_dp_du: 'mi.Vector3f',
        vtx_dp_dv: 'mi.Vector3f',
        rx_pos: 'mi.Point3f',
        tx_pos: 'mi.Point3f',
        scene: 'mi.Scene',
        valid_init: 'mi.Bool',
        init_prim_ids: 'mi.UInt32',
        init_shape,
        init_dn_du: 'mi.Vector3f',
        init_dn_dv: 'mi.Vector3f',
    ) -> Tuple['mi.Point3f', 'mi.Vector3f', 'mi.Bool', 'mi.UInt32', int, 'SpecularGradInfo',
               'mi.Vector3f', 'mi.Vector3f', 'mi.Float', 'mi.Float']:
        """Vectorized Newton solver for the specular manifold constraint."""
        n_paths = dr.width(vtx_p)

        active = mi.Bool(valid_init)
        converged = mi.Bool(False)
        beta = mi.Float(1.0)
        prim_ids = mi.UInt32(init_prim_ids)
        shape = init_shape

        # Mutable vertex state
        p = mi.Point3f(vtx_p)
        n = mi.Vector3f(vtx_n)
        dp_du = mi.Vector3f(vtx_dp_du)
        dp_dv = mi.Vector3f(vtx_dp_dv)
        mi.Vector3f(init_dn_du)
        mi.Vector3f(init_dn_dv)

        # Track final-iteration Jacobian and tangent frame for IFT (per-path)
        final_inv_J00 = dr.zeros(mi.Float, n_paths)
        final_inv_J01 = dr.zeros(mi.Float, n_paths)
        final_inv_J10 = dr.zeros(mi.Float, n_paths)
        final_inv_J11 = dr.zeros(mi.Float, n_paths)
        final_dp_du = mi.Vector3f(dp_du)
        final_dp_dv = mi.Vector3f(dp_dv)

        # Track barycentrics from last reprojection (eliminates Step 6 ray_intersect)
        last_bary_u = dr.zeros(mi.Float, n_paths)
        last_bary_v = dr.zeros(mi.Float, n_paths)

        iterations_used = 0

        for iteration in range(self.max_iterations):
            iterations_used = iteration + 1

            # Compute wi, wo directions
            wi_raw = mi.Vector3f(rx_pos.x - p.x, rx_pos.y - p.y, rx_pos.z - p.z)
            wo_raw = mi.Vector3f(tx_pos.x - p.x, tx_pos.y - p.y, tx_pos.z - p.z)

            d_wi = dr.norm(wi_raw)
            d_wo = dr.norm(wo_raw)

            valid_dist = (d_wi > mi.Float(1e-3)) & (d_wo > mi.Float(1e-3))
            active = active & valid_dist

            ili = mi.Float(1.0) / dr.maximum(d_wi, mi.Float(1e-10))
            ilo = mi.Float(1.0) / dr.maximum(d_wo, mi.Float(1e-10))

            wi = mi.Vector3f(wi_raw.x * ili, wi_raw.y * ili, wi_raw.z * ili)
            wo = mi.Vector3f(wo_raw.x * ilo, wo_raw.y * ilo, wo_raw.z * ilo)

            # Fix #5: Backfacing normal flip
            # Ref: manifold.h:112 - flip geometric normal when it faces away
            cos_n_wi = dr.dot(n, wi)
            need_flip = active & (cos_n_wi < mi.Float(0.0))
            n = mi.Vector3f(
                dr.select(need_flip, -n.x, n.x),
                dr.select(need_flip, -n.y, n.y),
                dr.select(need_flip, -n.z, n.z),
            )

            # Build tangent frame with Gram-Schmidt (Fix #2 prerequisite)
            n_dot_dpdu = dr.dot(n, dp_du)
            s_raw = mi.Vector3f(dp_du.x - n.x * n_dot_dpdu,
                                dp_du.y - n.y * n_dot_dpdu,
                                dp_du.z - n.z * n_dot_dpdu)
            s = safe_normalize(s_raw)
            t = dr.cross(n, s)

            # Compute half-vector h
            h_raw = mi.Vector3f(wi.x + wo.x, wi.y + wo.y, wi.z + wo.z)
            h_len = dr.norm(h_raw)
            ilh = mi.Float(1.0) / dr.maximum(h_len, mi.Float(1e-10))
            h = mi.Vector3f(h_raw.x * ilh, h_raw.y * ilh, h_raw.z * ilh)

            # Scaled inverse lengths for Jacobian
            ili_s = ili * ilh
            ilo_s = ilo * ilh

            # Evaluate constraint C = [s·h, t·h]
            C_s = dr.dot(s, h)
            C_t = dr.dot(t, h)
            C_norm = dr.sqrt(C_s * C_s + C_t * C_t)

            # Check convergence
            just_converged = active & (C_norm < mi.Float(self.solver_threshold))
            converged = converged | just_converged

            # Compute dh/du, dh/dv ONCE (used for both IFT save and Newton step)
            # Ref: manifold_ss.cpp:459-462
            dot_wi_du = dr.dot(wi, dp_du)
            dot_wo_du = dr.dot(wo, dp_du)
            scale_sum = ili_s + ilo_s

            dh_du = mi.Vector3f(
                -dp_du.x * scale_sum + wi.x * dot_wi_du * ili_s + wo.x * dot_wo_du * ilo_s,
                -dp_du.y * scale_sum + wi.y * dot_wi_du * ili_s + wo.y * dot_wo_du * ilo_s,
                -dp_du.z * scale_sum + wi.z * dot_wi_du * ili_s + wo.z * dot_wo_du * ilo_s,
            )
            proj_u = dr.dot(dh_du, h)
            dh_du = mi.Vector3f(dh_du.x - h.x * proj_u,
                                dh_du.y - h.y * proj_u,
                                dh_du.z - h.z * proj_u)

            dot_wi_dv = dr.dot(wi, dp_dv)
            dot_wo_dv = dr.dot(wo, dp_dv)

            dh_dv = mi.Vector3f(
                -dp_dv.x * scale_sum + wi.x * dot_wi_dv * ili_s + wo.x * dot_wo_dv * ilo_s,
                -dp_dv.y * scale_sum + wi.y * dot_wi_dv * ili_s + wo.y * dot_wo_dv * ilo_s,
                -dp_dv.z * scale_sum + wi.z * dot_wi_dv * ili_s + wo.z * dot_wo_dv * ilo_s,
            )
            proj_v = dr.dot(dh_dv, h)
            dh_dv = mi.Vector3f(dh_dv.x - h.x * proj_v,
                                dh_dv.y - h.y * proj_v,
                                dh_dv.z - h.z * proj_v)

            # Compute 2×2 Jacobian ONCE (used for both IFT save and Newton step)
            # Ref: manifold_ss.cpp:471-475
            J00 = dr.dot(s, dh_du)
            J10 = dr.dot(t, dh_du)
            J01 = dr.dot(s, dh_dv)
            J11 = dr.dot(t, dh_dv)

            det = J00 * J11 - J01 * J10
            valid_det = dr.abs(det) > mi.Float(1e-6)
            inv_det = dr.select(valid_det, mi.Float(1.0) / det, mi.Float(0.0))

            # Save IFT data for just-converged paths (NO guard, NO GPU sync)
            final_inv_J00 = dr.select(just_converged,  J11 * inv_det, final_inv_J00)
            final_inv_J01 = dr.select(just_converged, -J01 * inv_det, final_inv_J01)
            final_inv_J10 = dr.select(just_converged, -J10 * inv_det, final_inv_J10)
            final_inv_J11 = dr.select(just_converged,  J00 * inv_det, final_inv_J11)

            final_dp_du = mi.Vector3f(
                dr.select(just_converged, dp_du.x, final_dp_du.x),
                dr.select(just_converged, dp_du.y, final_dp_du.y),
                dr.select(just_converged, dp_du.z, final_dp_du.z))
            final_dp_dv = mi.Vector3f(
                dr.select(just_converged, dp_dv.x, final_dp_dv.x),
                dr.select(just_converged, dp_dv.y, final_dp_dv.y),
                dr.select(just_converged, dp_dv.z, final_dp_dv.z))

            active = active & ~just_converged
            # Also deactivate paths with singular Jacobian
            active = active & valid_det

            # PERF: Periodically evaluate all state to prevent lazy JIT graph
            # explosion from chained ray_intersect calls across iterations.
            if iteration % 3 == 2 or iteration == self.max_iterations - 1:
                dr.eval(active, converged, beta, prim_ids,
                        p.x, p.y, p.z, n.x, n.y, n.z,
                        dp_du.x, dp_du.y, dp_du.z,
                        dp_dv.x, dp_dv.y, dp_dv.z,
                        last_bary_u, last_bary_v,
                        final_inv_J00, final_inv_J01, final_inv_J10, final_inv_J11,
                        final_dp_du.x, final_dp_du.y, final_dp_du.z,
                        final_dp_dv.x, final_dp_dv.y, final_dp_dv.z)
                if dr.none(active):
                    break

            # Solve 2×2 Newton step (reuse J and inv_det from above)
            dX_u = inv_det * (J11 * C_s - J01 * C_t)
            dX_v = inv_det * (-J10 * C_s + J00 * C_t)

            # Propose new position
            step_x = dp_du.x * dX_u + dp_dv.x * dX_v
            step_y = dp_du.y * dX_u + dp_dv.y * dX_v
            step_z = dp_du.z * dX_u + dp_dv.z * dX_v

            p_prop_x = p.x - dr.select(active, beta * step_x, mi.Float(0.0))
            p_prop_y = p.y - dr.select(active, beta * step_y, mi.Float(0.0))
            p_prop_z = p.z - dr.select(active, beta * step_z, mi.Float(0.0))

            # Re-project onto mesh via ray tracing
            ray_dir_x = p_prop_x - rx_pos.x
            ray_dir_y = p_prop_y - rx_pos.y
            ray_dir_z = p_prop_z - rx_pos.z
            ray_dir_len = dr.sqrt(ray_dir_x * ray_dir_x + ray_dir_y * ray_dir_y + ray_dir_z * ray_dir_z)
            ray_dir_ilen = mi.Float(1.0) / dr.maximum(ray_dir_len, mi.Float(1e-10))
            ray_dir = mi.Vector3f(
                ray_dir_x * ray_dir_ilen,
                ray_dir_y * ray_dir_ilen,
                ray_dir_z * ray_dir_ilen,
            )

            ray = mi.Ray3f(rx_pos, ray_dir)
            si_new = scene.ray_intersect(ray, active)
            hit_valid = si_new.is_valid() & active

            # Fix #4: Shape check
            # Reject re-projection hits on different shapes (manifold_ss.cpp:373)
            try:
                same_shape = (si_new.shape == shape)
                hit_valid = hit_valid & same_shape
            except Exception:
                pass  # Shape comparison not supported, skip

            # Update vertex state where ray hit
            p = mi.Point3f(
                dr.select(hit_valid, si_new.p.x, p.x),
                dr.select(hit_valid, si_new.p.y, p.y),
                dr.select(hit_valid, si_new.p.z, p.z),
            )
            dp_du = mi.Vector3f(
                dr.select(hit_valid, si_new.dp_du.x, dp_du.x),
                dr.select(hit_valid, si_new.dp_du.y, dp_du.y),
                dr.select(hit_valid, si_new.dp_du.z, dp_du.z),
            )
            dp_dv = mi.Vector3f(
                dr.select(hit_valid, si_new.dp_dv.x, dp_dv.x),
                dr.select(hit_valid, si_new.dp_dv.y, dp_dv.y),
                dr.select(hit_valid, si_new.dp_dv.z, dp_dv.z),
            )

            # Update normal
            new_n = si_new.n
            n = mi.Vector3f(
                dr.select(hit_valid, new_n.x, n.x),
                dr.select(hit_valid, new_n.y, n.y),
                dr.select(hit_valid, new_n.z, n.z),
            )

            # Track barycentrics from reprojection
            last_bary_u = dr.select(hit_valid, si_new.uv.x, last_bary_u)
            last_bary_v = dr.select(hit_valid, si_new.uv.y, last_bary_v)

            # Track primitive IDs and shape
            prim_ids = dr.select(hit_valid, mi.UInt32(si_new.prim_index), prim_ids)
            # Update shape to track the current shape (matching reference behavior)
            try:
                shape = dr.select(hit_valid, si_new.shape, shape)
            except Exception:
                pass  # dr.select on ShapePtr not supported

            # Adaptive step size
            beta = dr.select(hit_valid,
                             dr.minimum(mi.Float(1.0), mi.Float(2.0) * beta),
                             beta)
            missed = active & ~hit_valid
            beta = dr.select(missed, mi.Float(0.5) * beta, beta)
            active = active & (beta > mi.Float(1e-8))

        grad_info = SpecularGradInfo(
            inv_J00=final_inv_J00,
            inv_J01=final_inv_J01,
            inv_J10=final_inv_J10,
            inv_J11=final_inv_J11,
            dp_du=final_dp_du,
            dp_dv=final_dp_dv,
            tx_pos=tx_pos,
            rx_pos=rx_pos,
        )

        return p, n, converged, prim_ids, iterations_used, grad_info, dp_du, dp_dv, last_bary_u, last_bary_v

    # Visibility and weight computation (unchanged)

    @staticmethod
    def _check_visibility(P_spec, dir_to_tx, dir_to_rx, d_tx, d_rx, scene,
                          surface_normal=None, skip_rx=False):
        """Shadow ray visibility check ."""
        normal_offset = mi.Float(1e-3)
        P_offset = mi.Point3f(
            P_spec.x + normal_offset * surface_normal.x,
            P_spec.y + normal_offset * surface_normal.y,
            P_spec.z + normal_offset * surface_normal.z,
        )

        epsilon = 1e-4
        shadow_origin_tx = mi.Point3f(
            P_offset.x + mi.Float(epsilon) * dir_to_tx.x,
            P_offset.y + mi.Float(epsilon) * dir_to_tx.y,
            P_offset.z + mi.Float(epsilon) * dir_to_tx.z,
        )
        shadow_rays_tx = mi.Ray3f(shadow_origin_tx, dir_to_tx)
        shadow_rays_tx.maxt = d_tx - 2.0 * epsilon
        occluded_tx = scene.ray_test(shadow_rays_tx)

        if skip_rx:
            return ~occluded_tx


    def _compute_specular_weight_physics(
        self,
        cos_theta_i: 'mi.Float',
        eps_real: 'mi.Float',
        eps_imag: 'mi.Float',
        sigma_h: 'mi.Float',
        l_c: 'mi.Float',
        tau: 'mi.Float',
        thickness: 'mi.Float',
    ) -> 'mi.Float':
        """Compute R_specular for physics-mode materials ."""
        sigma_h_v, l_c_v = enforce_spm_validity(sigma_h, l_c, self.wavelength)

        A = compute_slab_energy_gate(
            eps_real, eps_imag, cos_theta_i, thickness,
            mi.Float(0.5), mi.Float(0.5),
            wavelength=self.wavelength,
            sigma_h=sigma_h_v,
        )

        eta, _ = compute_coherent_incoherent_blend(
            sigma_h_v, l_c_v, self.wavelength, cos_theta_i
        )

        tau_eff = compute_validity_aware_blend(
            tau, cos_theta_i, sigma_h_v, l_c_v, self.wavelength
        )

        return eta * tau_eff * A

    # Multibounce SMS: find_specular_chains()

    def find_specular_chains(
        self,
        seed_P_list: list,           # [k] mi.Point3f seed positions per vertex
        seed_prim_ids_list: list,    # [k] mi.UInt32 seed prim IDs per vertex
        tx_positions: 'mi.Point3f',
        rx_positions: 'mi.Point3f',
        seed_tx_idx: 'mi.UInt32',    # TX index per chain
        seed_rx_idx: 'mi.UInt32',    # RX index per chain
        scene: 'mi.Scene',
        k: int = 2,
        max_iterations: int = 20,
        threshold: float = 1e-5,
        verbose: bool = False,
        triangle_materials: Optional[np.ndarray] = None,
        triangle_materials_gpu: Optional[list] = None,
        n_material_cols: int = 0,
    ) -> MultibounceSpecularChain:
        """Find k-bounce specular chains via block-tridiagonal Newton iteration."""
        from ..utils.math import gather_point3f

        t_start = time.perf_counter()
        n_chains = dr.width(seed_P_list[0])

        if verbose:
            print(f"\n[SMS-MB] Multibounce Specular Manifold Sampling (k={k})")
            print(f"  Chains: {n_chains:,}, Max iter: {max_iterations}")

        if n_chains == 0:
            return self._empty_multibounce_chain(k)

        # Expand TX/RX positions per chain
        exp_tx_pos = gather_point3f(tx_positions, seed_tx_idx)
        exp_rx_pos = gather_point3f(rx_positions, seed_rx_idx)

        # Initialize per-vertex state from seeds
        # Get surface parameterization at each seed via ray intersect
        p = []       # current positions [k]
        n_vec = []   # current normals [k]
        dp_du = []   # surface tangent u [k]
        dp_dv = []   # surface tangent v [k]
        dn_du_arr = []
        dn_dv_arr = []
        prim_ids = []  # current prim IDs [k]
        bary_u = [dr.zeros(mi.Float, n_chains) for _ in range(k)]
        bary_v = [dr.zeros(mi.Float, n_chains) for _ in range(k)]

        active = mi.Bool(True)

        for i in range(k):
            # Cast ray from the previous vertex (or RX for i=0) toward seed
            if i == 0:
                ray_origin = exp_rx_pos
            else:
                ray_origin = p[i-1]

            ray_dir = safe_normalize(mi.Vector3f(
                seed_P_list[i].x - ray_origin.x,
                seed_P_list[i].y - ray_origin.y,
                seed_P_list[i].z - ray_origin.z,
            ))
            ray = mi.Ray3f(ray_origin, ray_dir)
            si = scene.ray_intersect(ray)

            valid_hit = si.is_valid()
            active = active & valid_hit

            p.append(mi.Point3f(si.p))
            dp_du.append(mi.Vector3f(si.dp_du))
            dp_dv.append(mi.Vector3f(si.dp_dv))
            prim_ids.append(mi.UInt32(si.prim_index))
            bary_u[i] = si.uv.x
            bary_v[i] = si.uv.y

            n_vec.append(mi.Vector3f(si.n))
            zero3 = mi.Vector3f(dr.zeros(mi.Float, n_chains),
                                dr.zeros(mi.Float, n_chains),
                                dr.zeros(mi.Float, n_chains))
            dn_du_arr.append(zero3)
            dn_dv_arr.append(zero3)

        # Newton iteration on block-tridiagonal system
        converged = mi.Bool(False)
        beta = mi.Float(1.0)  # adaptive step size

        # Storage for IFT factorization (saved at convergence)
        final_D_inv = [None] * k
        final_U = [None] * max(k - 1, 1)
        final_dp_du = [mi.Vector3f(dp_du[i]) for i in range(k)]
        final_dp_dv = [mi.Vector3f(dp_dv[i]) for i in range(k)]

        iterations_used = 0

        for iteration in range(max_iterations):
            iterations_used = iteration + 1

            # Compute directions at each vertex
            wi_list = []  # incoming direction (from previous vertex / RX)
            wo_list = []  # outgoing direction (toward next vertex / TX)
            d_wi_list = []
            d_wo_list = []

            for i in range(k):
                # wi: direction FROM previous vertex (or RX)
                if i == 0:
                    wi_raw = mi.Vector3f(
                        exp_rx_pos.x - p[i].x,
                        exp_rx_pos.y - p[i].y,
                        exp_rx_pos.z - p[i].z)
                else:
                    wi_raw = mi.Vector3f(
                        p[i-1].x - p[i].x,
                        p[i-1].y - p[i].y,
                        p[i-1].z - p[i].z)

                # wo: direction TO next vertex (or TX)
                if i == k - 1:
                    wo_raw = mi.Vector3f(
                        exp_tx_pos.x - p[i].x,
                        exp_tx_pos.y - p[i].y,
                        exp_tx_pos.z - p[i].z)
                else:
                    wo_raw = mi.Vector3f(
                        p[i+1].x - p[i].x,
                        p[i+1].y - p[i].y,
                        p[i+1].z - p[i].z)

                d_wi = dr.norm(wi_raw)
                d_wo = dr.norm(wo_raw)
                d_wi_list.append(d_wi)
                d_wo_list.append(d_wo)

                valid_dist = (d_wi > mi.Float(1e-3)) & (d_wo > mi.Float(1e-3))
                active = active & valid_dist

                ili = mi.Float(1.0) / dr.maximum(d_wi, mi.Float(1e-10))
                ilo = mi.Float(1.0) / dr.maximum(d_wo, mi.Float(1e-10))
                wi_list.append((mi.Vector3f(wi_raw.x*ili, wi_raw.y*ili, wi_raw.z*ili), ili))
                wo_list.append((mi.Vector3f(wo_raw.x*ilo, wo_raw.y*ilo, wo_raw.z*ilo), ilo))

            # Backfacing normal flip
            for i in range(k):
                wi, _ = wi_list[i]
                cos_n_wi = dr.dot(n_vec[i], wi)
                need_flip = active & (cos_n_wi < mi.Float(0.0))
                n_vec[i] = mi.Vector3f(
                    dr.select(need_flip, -n_vec[i].x, n_vec[i].x),
                    dr.select(need_flip, -n_vec[i].y, n_vec[i].y),
                    dr.select(need_flip, -n_vec[i].z, n_vec[i].z))

            # Build tangent frames and evaluate constraints
            C_list = []       # [k] constraint 2-vectors
            C_norm_sum = dr.zeros(mi.Float, n_chains)

            # Jacobian blocks
            A_diag = []       # [k] diagonal 2x2 blocks
            B_upper = []      # [k-1] upper 2x2 blocks
            C_lower_blocks = []  # [k-1] lower 2x2 blocks

            for i in range(k):
                wi, ili = wi_list[i]
                wo, ilo = wo_list[i]
                ni = n_vec[i]

                # Tangent frame
                n_dot_dpdu = dr.dot(ni, dp_du[i])
                s_raw = mi.Vector3f(
                    dp_du[i].x - ni.x * n_dot_dpdu,
                    dp_du[i].y - ni.y * n_dot_dpdu,
                    dp_du[i].z - ni.z * n_dot_dpdu)
                s = safe_normalize(s_raw)
                t = dr.cross(ni, s)

                # Half-vector
                h_raw = mi.Vector3f(wi.x + wo.x, wi.y + wo.y, wi.z + wo.z)
                h_len = dr.norm(h_raw)
                ilh = mi.Float(1.0) / dr.maximum(h_len, mi.Float(1e-10))
                h = mi.Vector3f(h_raw.x * ilh, h_raw.y * ilh, h_raw.z * ilh)

                # Constraint
                C_s = dr.dot(s, h)
                C_t = dr.dot(t, h)
                C_list.append((C_s, C_t))
                C_norm_sum = C_norm_sum + C_s * C_s + C_t * C_t

                # Jacobian: dC_i/d(u_i, v_i) = diagonal block A_i
                ili_s = ili * ilh
                ilo_s = ilo * ilh
                scale_sum = ili_s + ilo_s

                # dh/du_i (effect of moving S_i on its own half-vector)
                dot_wi_du = dr.dot(wi, dp_du[i])
                dot_wo_du = dr.dot(wo, dp_du[i])
                dh_du = mi.Vector3f(
                    -dp_du[i].x * scale_sum + wi.x * dot_wi_du * ili_s + wo.x * dot_wo_du * ilo_s,
                    -dp_du[i].y * scale_sum + wi.y * dot_wi_du * ili_s + wo.y * dot_wo_du * ilo_s,
                    -dp_du[i].z * scale_sum + wi.z * dot_wi_du * ili_s + wo.z * dot_wo_du * ilo_s)
                proj_u = dr.dot(dh_du, h)
                dh_du = mi.Vector3f(dh_du.x - h.x*proj_u, dh_du.y - h.y*proj_u, dh_du.z - h.z*proj_u)

                dot_wi_dv = dr.dot(wi, dp_dv[i])
                dot_wo_dv = dr.dot(wo, dp_dv[i])
                dh_dv = mi.Vector3f(
                    -dp_dv[i].x * scale_sum + wi.x * dot_wi_dv * ili_s + wo.x * dot_wo_dv * ilo_s,
                    -dp_dv[i].y * scale_sum + wi.y * dot_wi_dv * ili_s + wo.y * dot_wo_dv * ilo_s,
                    -dp_dv[i].z * scale_sum + wi.z * dot_wi_dv * ili_s + wo.z * dot_wo_dv * ilo_s)
                proj_v = dr.dot(dh_dv, h)
                dh_dv = mi.Vector3f(dh_dv.x - h.x*proj_v, dh_dv.y - h.y*proj_v, dh_dv.z - h.z*proj_v)

                A00 = dr.dot(s, dh_du)
                A10 = dr.dot(t, dh_du)
                A01 = dr.dot(s, dh_dv)
                A11 = dr.dot(t, dh_dv)

                A_diag.append((A00, A01, A10, A11))

                # Off-diagonal blocks: coupling to adjacent vertices
                # Upper block B_i: dC_i/d(u_{i+1}, v_{i+1})
                # Moving S_{i+1} changes wo at vertex i
                if i < k - 1:
                    # dh_i / d(u_{i+1}) via change in wo direction
                    # wo = (p[i+1] - p[i]) / ||...||
                    # d(wo)/d(u_{i+1}) = ilo * (dp_du[i+1] - wo * dot(wo, dp_du[i+1]))
                    # but projected through ilh
                    dwo_du_next = mi.Vector3f(
                        ilo * (dp_du[i+1].x - wo.x * dr.dot(wo, dp_du[i+1])),
                        ilo * (dp_du[i+1].y - wo.y * dr.dot(wo, dp_du[i+1])),
                        ilo * (dp_du[i+1].z - wo.z * dr.dot(wo, dp_du[i+1])))
                    dh_du_next = mi.Vector3f(
                        ilh * (dwo_du_next.x - h.x * dr.dot(dwo_du_next, h)),
                        ilh * (dwo_du_next.y - h.y * dr.dot(dwo_du_next, h)),
                        ilh * (dwo_du_next.z - h.z * dr.dot(dwo_du_next, h)))

                    dwo_dv_next = mi.Vector3f(
                        ilo * (dp_dv[i+1].x - wo.x * dr.dot(wo, dp_dv[i+1])),
                        ilo * (dp_dv[i+1].y - wo.y * dr.dot(wo, dp_dv[i+1])),
                        ilo * (dp_dv[i+1].z - wo.z * dr.dot(wo, dp_dv[i+1])))
                    dh_dv_next_dot = dr.dot(dwo_dv_next, h)
                    dh_dv_next = mi.Vector3f(
                        ilh * (dwo_dv_next.x - h.x * dh_dv_next_dot),
                        ilh * (dwo_dv_next.y - h.y * dh_dv_next_dot),
                        ilh * (dwo_dv_next.z - h.z * dh_dv_next_dot))

                    B00 = dr.dot(s, dh_du_next)
                    B10 = dr.dot(t, dh_du_next)
                    B01 = dr.dot(s, dh_dv_next)
                    B11 = dr.dot(t, dh_dv_next)
                    B_upper.append((B00, B01, B10, B11))

                # Lower block C_i: dC_i/d(u_{i-1}, v_{i-1})
                # Moving S_{i-1} changes wi at vertex i
                if i > 0:
                    # d(wi)/d(u_{i-1}) via change in wi direction
                    dwi_du_prev = mi.Vector3f(
                        ili * (dp_du[i-1].x - wi.x * dr.dot(wi, dp_du[i-1])),
                        ili * (dp_du[i-1].y - wi.y * dr.dot(wi, dp_du[i-1])),
                        ili * (dp_du[i-1].z - wi.z * dr.dot(wi, dp_du[i-1])))
                    dh_du_prev = mi.Vector3f(
                        ilh * (dwi_du_prev.x - h.x * dr.dot(dwi_du_prev, h)),
                        ilh * (dwi_du_prev.y - h.y * dr.dot(dwi_du_prev, h)),
                        ilh * (dwi_du_prev.z - h.z * dr.dot(dwi_du_prev, h)))

                    dwi_dv_prev = mi.Vector3f(
                        ili * (dp_dv[i-1].x - wi.x * dr.dot(wi, dp_dv[i-1])),
                        ili * (dp_dv[i-1].y - wi.y * dr.dot(wi, dp_dv[i-1])),
                        ili * (dp_dv[i-1].z - wi.z * dr.dot(wi, dp_dv[i-1])))
                    dh_dv_prev = mi.Vector3f(
                        ilh * (dwi_dv_prev.x - h.x * dr.dot(dwi_dv_prev, h)),
                        ilh * (dwi_dv_prev.y - h.y * dr.dot(dwi_dv_prev, h)),
                        ilh * (dwi_dv_prev.z - h.z * dr.dot(dwi_dv_prev, h)))

                    CL00 = dr.dot(s, dh_du_prev)
                    CL10 = dr.dot(t, dh_du_prev)
                    CL01 = dr.dot(s, dh_dv_prev)
                    CL11 = dr.dot(t, dh_dv_prev)
                    C_lower_blocks.append((CL00, CL01, CL10, CL11))

            # Check convergence
            C_norm_total = dr.sqrt(dr.maximum(C_norm_sum, mi.Float(0.0)))
            just_converged = active & (C_norm_total < mi.Float(threshold))
            converged = converged | just_converged

            # Save IFT factorization for just-converged chains
            if len(A_diag) == k and len(B_upper) == k - 1 and len(C_lower_blocks) == k - 1:
                _, D_inv_iter, U_iter = self._thomas_block_tridiagonal(
                    A_diag, B_upper, C_lower_blocks, C_list, active)

                for vi in range(k):
                    if final_D_inv[vi] is None:
                        final_D_inv[vi] = tuple(
                            dr.select(just_converged, D_inv_iter[vi][j], dr.zeros(mi.Float, n_chains))
                            for j in range(4))
                    else:
                        final_D_inv[vi] = tuple(
                            dr.select(just_converged, D_inv_iter[vi][j], final_D_inv[vi][j])
                            for j in range(4))

                    final_dp_du[vi] = mi.Vector3f(
                        dr.select(just_converged, dp_du[vi].x, final_dp_du[vi].x),
                        dr.select(just_converged, dp_du[vi].y, final_dp_du[vi].y),
                        dr.select(just_converged, dp_du[vi].z, final_dp_du[vi].z))
                    final_dp_dv[vi] = mi.Vector3f(
                        dr.select(just_converged, dp_dv[vi].x, final_dp_dv[vi].x),
                        dr.select(just_converged, dp_dv[vi].y, final_dp_dv[vi].y),
                        dr.select(just_converged, dp_dv[vi].z, final_dp_dv[vi].z))

                for ui in range(len(U_iter)):
                    if final_U[ui] is None:
                        final_U[ui] = tuple(
                            dr.select(just_converged, U_iter[ui][j], dr.zeros(mi.Float, n_chains))
                            for j in range(4))
                    else:
                        final_U[ui] = tuple(
                            dr.select(just_converged, U_iter[ui][j], final_U[ui][j])
                            for j in range(4))

            active = active & ~just_converged

            # PERF: Periodically evaluate all per-vertex state AND IFT
            # accumulator to prevent lazy JIT graph explosion. Without this,
            # 20 iterations of ray_intersect calls build an enormous graph
            # that takes tens of seconds to JIT-compile. Evaluating every 3
            # iterations trades a few small kernel launches for one massive
            # compilation.
            if iteration % 3 == 2 or iteration == max_iterations - 1:
                _eval_args = [active, converged, beta]
                for i in range(k):
                    _eval_args.extend([
                        p[i].x, p[i].y, p[i].z,
                        n_vec[i].x, n_vec[i].y, n_vec[i].z,
                        dp_du[i].x, dp_du[i].y, dp_du[i].z,
                        dp_dv[i].x, dp_dv[i].y, dp_dv[i].z,
                        prim_ids[i], bary_u[i], bary_v[i],
                    ])
                    # IFT accumulator state
                    if final_D_inv[i] is not None:
                        _eval_args.extend(final_D_inv[i])
                    _eval_args.extend([
                        final_dp_du[i].x, final_dp_du[i].y, final_dp_du[i].z,
                        final_dp_dv[i].x, final_dp_dv[i].y, final_dp_dv[i].z,
                    ])
                for ui in range(len(final_U)):
                    if final_U[ui] is not None:
                        _eval_args.extend(final_U[ui])
                dr.eval(*_eval_args)

                # Check if all done
                if dr.none(active):
                    break

            # Solve block-tridiagonal for Newton step
            if len(A_diag) == k and len(B_upper) == k - 1 and len(C_lower_blocks) == k - 1:
                deltas, _, _ = self._thomas_block_tridiagonal(
                    A_diag, B_upper, C_lower_blocks, C_list, active)
            else:
                # Fallback for malformed system: no update
                break

            # Update all vertices
            for i in range(k):
                dX_u, dX_v = deltas[i]

                step_x = dp_du[i].x * dX_u + dp_dv[i].x * dX_v
                step_y = dp_du[i].y * dX_u + dp_dv[i].y * dX_v
                step_z = dp_du[i].z * dX_u + dp_dv[i].z * dX_v

                p_prop = mi.Point3f(
                    p[i].x - dr.select(active, beta * step_x, mi.Float(0.0)),
                    p[i].y - dr.select(active, beta * step_y, mi.Float(0.0)),
                    p[i].z - dr.select(active, beta * step_z, mi.Float(0.0)))

                # Re-project onto mesh
                if i == 0:
                    ray_origin = exp_rx_pos
                else:
                    ray_origin = p[i-1]

                ray_dir = safe_normalize(mi.Vector3f(
                    p_prop.x - ray_origin.x,
                    p_prop.y - ray_origin.y,
                    p_prop.z - ray_origin.z))
                ray = mi.Ray3f(ray_origin, ray_dir)
                si_new = scene.ray_intersect(ray, active)
                hit_valid = si_new.is_valid() & active

                p[i] = mi.Point3f(
                    dr.select(hit_valid, si_new.p.x, p[i].x),
                    dr.select(hit_valid, si_new.p.y, p[i].y),
                    dr.select(hit_valid, si_new.p.z, p[i].z))
                dp_du[i] = mi.Vector3f(
                    dr.select(hit_valid, si_new.dp_du.x, dp_du[i].x),
                    dr.select(hit_valid, si_new.dp_du.y, dp_du[i].y),
                    dr.select(hit_valid, si_new.dp_du.z, dp_du[i].z))
                dp_dv[i] = mi.Vector3f(
                    dr.select(hit_valid, si_new.dp_dv.x, dp_dv[i].x),
                    dr.select(hit_valid, si_new.dp_dv.y, dp_dv[i].y),
                    dr.select(hit_valid, si_new.dp_dv.z, dp_dv[i].z))

                new_n = si_new.sh_frame.n if self.use_smooth_normals else si_new.n
                n_vec[i] = mi.Vector3f(
                    dr.select(hit_valid, new_n.x, n_vec[i].x),
                    dr.select(hit_valid, new_n.y, n_vec[i].y),
                    dr.select(hit_valid, new_n.z, n_vec[i].z))

                prim_ids[i] = dr.select(hit_valid, mi.UInt32(si_new.prim_index), prim_ids[i])
                bary_u[i] = dr.select(hit_valid, si_new.uv.x, bary_u[i])
                bary_v[i] = dr.select(hit_valid, si_new.uv.y, bary_v[i])

            # Adaptive step size
            # (use first vertex hit_valid as proxy — if it misses, reduce step)
            beta = dr.select(active & hit_valid,
                             dr.minimum(mi.Float(1.0), mi.Float(2.0) * beta),
                             beta)
            missed = active & ~hit_valid
            beta = dr.select(missed, mi.Float(0.5) * beta, beta)
            active = active & (beta > mi.Float(1e-8))

        # Post-Newton: convergence stats
        n_converged = int(dr.sum(mi.UInt32(converged))[0])
        t_newton = time.perf_counter() - t_start

        if verbose:
            print(f"  Newton solver: {n_converged:,}/{n_chains:,} converged "
                  f"({100*n_converged/max(n_chains,1):.1f}%) in {iterations_used} iter, "
                  f"{t_newton:.3f}s")

        if n_converged == 0:
            return self._empty_multibounce_chain(k)

        # Visibility check for all k+1 segments
        valid_all = mi.Bool(converged)
        epsilon = 1e-4

        for i in range(k + 1):
            if i == 0:
                seg_from = exp_rx_pos
                seg_to = p[0]
            elif i == k:
                seg_from = p[k-1]
                seg_to = exp_tx_pos
            else:
                seg_from = p[i-1]
                seg_to = p[i]

            seg_dir = mi.Vector3f(
                seg_to.x - seg_from.x,
                seg_to.y - seg_from.y,
                seg_to.z - seg_from.z)
            seg_len = dr.norm(seg_dir)
            seg_dir_n = mi.Vector3f(
                seg_dir.x / dr.maximum(seg_len, mi.Float(1e-10)),
                seg_dir.y / dr.maximum(seg_len, mi.Float(1e-10)),
                seg_dir.z / dr.maximum(seg_len, mi.Float(1e-10)))

            ray_o = mi.Point3f(
                seg_from.x + mi.Float(epsilon) * seg_dir_n.x,
                seg_from.y + mi.Float(epsilon) * seg_dir_n.y,
                seg_from.z + mi.Float(epsilon) * seg_dir_n.z)
            shadow_ray = mi.Ray3f(ray_o, seg_dir_n)
            shadow_ray.maxt = seg_len - 2.0 * epsilon
            occluded = scene.ray_test(shadow_ray)
            valid_all = valid_all & ~occluded

        n_valid_all = int(dr.sum(mi.UInt32(valid_all))[0])
        if verbose:
            print(f"  Valid after visibility: {n_valid_all:,}/{n_chains:,}")

        # Compute per-vertex triangle area and segment distances
        A_tri_list = []
        for i in range(k):
            cross_vec = dr.cross(dp_du[i], dp_dv[i])
            A_tri_list.append(mi.Float(0.5) * dr.norm(cross_vec))

        d_segments = []
        dir_segments = []
        for i in range(k + 1):
            if i == 0:
                seg_from = exp_rx_pos
                seg_to = p[0]
            elif i == k:
                seg_from = p[k-1]
                seg_to = exp_tx_pos
            else:
                seg_from = p[i-1]
                seg_to = p[i]

            delta = mi.Vector3f(
                seg_to.x - seg_from.x,
                seg_to.y - seg_from.y,
                seg_to.z - seg_from.z)
            d = dr.norm(delta)
            d_segments.append(d)
            dir_segments.append(mi.Vector3f(
                delta.x / dr.maximum(d, mi.Float(1e-10)),
                delta.y / dr.maximum(d, mi.Float(1e-10)),
                delta.z / dr.maximum(d, mi.Float(1e-10))))

        # Build grad info
        grad_info = MultibounceSpecularGradInfo(
            k=k,
            dp_du=final_dp_du,
            dp_dv=final_dp_dv,
            D_inv=[final_D_inv[i] for i in range(k)],
            U=[final_U[i] for i in range(k-1)] if k > 1 else [],
            tx_pos=exp_tx_pos,
            rx_pos=exp_rx_pos,
        )

        t_total = time.perf_counter() - t_start
        print(f"  [SMS-MB] k={k}: Converged {n_converged:,}/{n_chains:,} | "
              f"Valid {n_valid_all:,} | Time: {t_total:.3f}s")

        return MultibounceSpecularChain(
            k=k,
            hit_P=p,
            hit_N=n_vec,
            prim_ids=prim_ids,
            bary_u=bary_u,
            bary_v=bary_v,
            A_tri=A_tri_list,
            tx_idx=seed_tx_idx,
            rx_idx=seed_rx_idx,
            valid=valid_all,
            n_chains=n_chains,
            grad_info=grad_info,
            d_segments=d_segments,
            dir_segments=dir_segments,
        )

    def _empty_multibounce_chain(self, k: int) -> MultibounceSpecularChain:
        """Return an empty MultibounceSpecularChain."""
        return MultibounceSpecularChain(
            k=k,
            hit_P=[mi.Point3f() for _ in range(k)],
            hit_N=[mi.Vector3f() for _ in range(k)],
            prim_ids=[mi.UInt32() for _ in range(k)],
            bary_u=[mi.Float() for _ in range(k)],
            bary_v=[mi.Float() for _ in range(k)],
            A_tri=[mi.Float() for _ in range(k)],
            tx_idx=mi.UInt32(),
            rx_idx=mi.UInt32(),
            valid=mi.Bool(),
            n_chains=0,
            grad_info=None,
            d_segments=[mi.Float() for _ in range(k + 1)],
            dir_segments=[mi.Vector3f() for _ in range(k + 1)],
        )


__all__ = [
    'SpecularManifoldSampler',
    'SpecularGradInfo',
    'MultibounceSpecularChain',
    'MultibounceSpecularGradInfo',
]
