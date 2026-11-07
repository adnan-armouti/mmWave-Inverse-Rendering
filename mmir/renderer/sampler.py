"""RX-centric reservoir sampling matching RRTS."""

from dataclasses import dataclass
from typing import Optional
import drjit as dr
import mitsuba as mi
import numpy as np

from .utils.math import (sample_cosine_hemisphere_concentric, to_global, safe_normalize)


@dataclass
class ReservoirHitsDrJit:
    """Reservoir sampling results with all arrays in DrJit (AD-compatible)."""
    hit_P: 'mi.Point3f'             # [n_valid] AD-attached hit positions
    hit_N: 'mi.Vector3f'            # [n_valid] AD-attached hit normals
    hit_prim_ids: 'mi.UInt32'       # [n_valid] triangle primitive IDs
    hit_bary_u: 'mi.Float'          # [n_valid] barycentric u
    hit_bary_v: 'mi.Float'          # [n_valid] barycentric v
    hit_pdf: 'mi.Float'             # [n_valid] sampling PDF (non-diff)
    rx_element_idx: 'mi.UInt32'     # [n_valid] which RX element owns this hit
    n_valid: int                     # total valid hits
    n_attempted_per_rx: 'mi.UInt32' # [n_rx] rays attempted per RX
    n_rx: int                        # number of RX elements

    # Vertex indices for barycentric interpolation (populated from mesh)
    vertex_ids_0: Optional['mi.UInt32'] = None  # [n_valid]
    vertex_ids_1: Optional['mi.UInt32'] = None  # [n_valid]
    vertex_ids_2: Optional['mi.UInt32'] = None  # [n_valid]


@dataclass
class ReservoirHits:
    """Result of reservoir sampling for all RX elements."""
    hit_P: 'mi.Point3f'
    hit_N: 'mi.Vector3f'
    hit_ID: 'mi.Int32'
    ray_count: 'mi.Int32'
    hit_rho: np.ndarray  # [n_total, N] - N=3 legacy or N=6 physics params
    n_rx: int
    n_hits_per_rx: int
    valid: 'mi.Bool'

    # Monte Carlo probability chain fields
    hit_pdf: np.ndarray = None  # [n_total] - sampling PDF for each hit
    n_attempted_per_rx: np.ndarray = None  # [n_rx] - total rays attempted per RX
    n_hits_per_rx_actual: np.ndarray = None  # [n_rx] - actual valid hits per RX

    # Triangle vertex data (for image method specular path refinement)
    hit_tri_v0: Optional['mi.Point3f'] = None
    hit_tri_v1: Optional['mi.Point3f'] = None
    hit_tri_v2: Optional['mi.Point3f'] = None

    # Barycentric coordinates (for per-vertex material parameterization)
    hit_bary_u: Optional['mi.Float'] = None  # [n_total] si.uv.x
    hit_bary_v: Optional['mi.Float'] = None  # [n_total] si.uv.y


class ReservoirSampler:
    """RX-centric reservoir sampling matching RRTS InitialResampling."""

    def __init__(
        self,
        n_hits_per_rx: int = 500,
        n_rays_per_res: int = 16,
        max_distance: float = 1e10,
        hemisphere_sampling: str = 'cosine',
    ):
        """Initialize reservoir sampler."""
        self.n_hits_per_rx = n_hits_per_rx
        self.n_rays_per_res = n_rays_per_res
        self.max_distance = max_distance
        self.hemisphere_sampling = hemisphere_sampling

    def sample_vectorized(
        self,
        scene: 'mi.Scene',
        rx_positions: 'mi.Point3f',
        rx_boresights: 'mi.Vector3f',
        seed: int = 0,
        verbose: bool = True,
        triangle_materials: 'Optional[np.ndarray]' = None,
        use_vertex_normals: bool = True,
    ) -> ReservoirHits:
        """Vectorized reservoir sampling (more efficient for large RX counts)."""
        n_rx = dr.width(rx_positions)

        if verbose:
            print(f"\n[ReservoirSampler] Vectorized sampling from {n_rx} RX elements")
            print(f"  n_hits_per_rx: {self.n_hits_per_rx}")

        # Total rays to sample per batch (all RX elements simultaneously)
        rays_per_rx_batch = self.n_hits_per_rx * 2  # Oversample to ensure fill
        total_rays = n_rx * rays_per_rx_batch

        # Create RNG
        rng = mi.PCG32(size=total_rays, initseq=seed)
        u1 = rng.next_float32()
        u2 = rng.next_float32()
        u = mi.Point2f(u1, u2)

        # Sample directions in local frame for all rays
        # IMPORTANT: Capture PDF for MC probability chain
        local_dirs, pdf = sample_cosine_hemisphere_concentric(u)

        # Expand RX positions and boresights for all rays
        # Each RX gets rays_per_rx_batch rays
        rx_idx = dr.arange(mi.UInt32, total_rays) // rays_per_rx_batch

        expanded_pos = mi.Point3f(
            dr.gather(mi.Float, rx_positions.x, rx_idx),
            dr.gather(mi.Float, rx_positions.y, rx_idx),
            dr.gather(mi.Float, rx_positions.z, rx_idx)
        )

        expanded_bore = mi.Vector3f(
            dr.gather(mi.Float, rx_boresights.x, rx_idx),
            dr.gather(mi.Float, rx_boresights.y, rx_idx),
            dr.gather(mi.Float, rx_boresights.z, rx_idx)
        )
        expanded_bore = safe_normalize(expanded_bore)

        # Transform directions to world frame
        world_dirs = to_global(local_dirs, expanded_bore)

        # Create and trace rays
        rays = mi.Ray3f(expanded_pos, world_dirs)
        rays.maxt = mi.Float(self.max_distance)

        si = scene.ray_intersect(rays)
        hit_mask = si.is_valid()

        # Get mesh for triangle vertex lookups
        mesh = scene.shapes()[0] if len(scene.shapes()) > 0 else None

        # Gather triangle vertices for all rays (needed for image method)
        face_idx = mesh.face_indices(si.prim_index)
        all_v0 = mesh.vertex_position(face_idx[0])
        all_v1 = mesh.vertex_position(face_idx[1])
        all_v2 = mesh.vertex_position(face_idx[2])

        if verbose:
            n_hits = int(dr.sum(mi.UInt32(hit_mask))[0])
            print(f"  Total rays: {total_rays:,}, hits: {n_hits:,}")

        # Reservoir filling using DrJit (fully GPU, no Python loops)
        #
        # Strategy: Since rays are contiguous per RX (each RX owns
        # rays_per_rx_batch consecutive rays), we use prefix_sum on
        # the hit indicator to compute within-group rank per hit.
        # Only hits with rank < n_hits_per_rx are scattered into slots.
        total_slots = n_rx * self.n_hits_per_rx

        # Step 1: Compute within-RX hit rank via segmented prefix sum
        hit_indicator = mi.UInt32(hit_mask)  # 1 for hit, 0 for miss
        inclusive_cumsum = dr.prefix_sum(hit_indicator)  # [total_rays], inclusive
        cumsum = inclusive_cumsum - hit_indicator  # exclusive prefix sum

        # Cumulative count at start of each RX group
        group_start_idx = rx_idx * mi.UInt32(rays_per_rx_batch)
        cumsum_at_start = dr.gather(mi.UInt32, cumsum, group_start_idx)
        within_rank = cumsum - cumsum_at_start  # per-ray rank within its RX

        # Step 2: Active mask = is a hit AND rank fits in reservoir
        active = hit_mask & (within_rank < mi.UInt32(self.n_hits_per_rx))

        # Step 3: Destination slot = rx_idx * n_hits_per_rx + within_rank
        dest_slot = rx_idx * mi.UInt32(self.n_hits_per_rx) + within_rank

        # Step 4: Allocate result arrays in DrJit
        result_P_x = dr.zeros(mi.Float, total_slots)
        result_P_y = dr.zeros(mi.Float, total_slots)
        result_P_z = dr.zeros(mi.Float, total_slots)
        result_N_x = dr.zeros(mi.Float, total_slots)
        result_N_y = dr.zeros(mi.Float, total_slots)
        result_N_z = dr.zeros(mi.Float, total_slots)
        result_ID = dr.full(mi.Int32, -1, total_slots)
        result_ray_count = dr.zeros(mi.Int32, total_slots)
        result_valid = dr.zeros(mi.Bool, total_slots)
        result_pdf_dr = dr.zeros(mi.Float, total_slots)

        # Step 5: Scatter hit geometry into reservoir slots
        dr.scatter(result_P_x, si.p.x, dest_slot, active)
        dr.scatter(result_P_y, si.p.y, dest_slot, active)
        dr.scatter(result_P_z, si.p.z, dest_slot, active)
        # Select normal source: vertex (smooth) or face (flat)
        si_normal = si.sh_frame.n if use_vertex_normals else si.n
        dr.scatter(result_N_x, si_normal.x, dest_slot, active)
        dr.scatter(result_N_y, si_normal.y, dest_slot, active)
        dr.scatter(result_N_z, si_normal.z, dest_slot, active)
        dr.scatter(result_ID, mi.Int32(si.prim_index), dest_slot, active)
        ray_indices = mi.Int32(dr.arange(mi.UInt32, total_rays)) + mi.Int32(1)
        dr.scatter(result_ray_count, ray_indices, dest_slot, active)
        dr.scatter(result_valid, dr.full(mi.Bool, True, total_rays), dest_slot, active)
        dr.scatter(result_pdf_dr, pdf, dest_slot, active)

        # Step 5b: Scatter triangle vertex data
        result_v0_x = dr.zeros(mi.Float, total_slots)
        result_v0_y = dr.zeros(mi.Float, total_slots)
        result_v0_z = dr.zeros(mi.Float, total_slots)
        result_v1_x = dr.zeros(mi.Float, total_slots)
        result_v1_y = dr.zeros(mi.Float, total_slots)
        result_v1_z = dr.zeros(mi.Float, total_slots)
        result_v2_x = dr.zeros(mi.Float, total_slots)
        result_v2_y = dr.zeros(mi.Float, total_slots)
        result_v2_z = dr.zeros(mi.Float, total_slots)

        if mesh is not None:
            dr.scatter(result_v0_x, all_v0.x, dest_slot, active)
            dr.scatter(result_v0_y, all_v0.y, dest_slot, active)
            dr.scatter(result_v0_z, all_v0.z, dest_slot, active)
            dr.scatter(result_v1_x, all_v1.x, dest_slot, active)
            dr.scatter(result_v1_y, all_v1.y, dest_slot, active)
            dr.scatter(result_v1_z, all_v1.z, dest_slot, active)
            dr.scatter(result_v2_x, all_v2.x, dest_slot, active)
            dr.scatter(result_v2_y, all_v2.y, dest_slot, active)
            dr.scatter(result_v2_z, all_v2.z, dest_slot, active)

        # Step 5c: Scatter barycentric coordinates for per-vertex interpolation
        result_bary_u_dr = dr.zeros(mi.Float, total_slots)
        result_bary_v_dr = dr.zeros(mi.Float, total_slots)
        dr.scatter(result_bary_u_dr, si.uv.x, dest_slot, active)
        dr.scatter(result_bary_v_dr, si.uv.y, dest_slot, active)

        # Step 6: Material handling (all in DrJit)
        # Determine material column count from triangle_materials
        n_mat_cols = triangle_materials.shape[1] if triangle_materials is not None else 3

        # Default values per column
        if n_mat_cols == 6:
            # Physics mode defaults: [eps_r=4, eps_i=0.1, sigma_h=3e-5, l_c=0.01, tau=0.5, d=0.1]
            default_vals = [4.0, 0.1, 3e-5, 0.01, 0.5, 0.1]
        else:
            # Legacy defaults: [albedo=1, roughness=0.5, metallic=0]
            default_vals = [1.0, 0.5, 0.0]

        result_rho_cols = [dr.full(mi.Float, default_vals[c], total_slots) for c in range(n_mat_cols)]

        if triangle_materials is not None:
            n_tris = len(triangle_materials)
            tm_cols = [mi.Float(triangle_materials[:, c].astype(np.float32)) for c in range(n_mat_cols)]

            # Clamp prim index to valid range for gather
            prim_clamped = dr.clamp(
                mi.UInt32(dr.maximum(si.prim_index, mi.Int32(0))),
                mi.UInt32(0), mi.UInt32(n_tris - 1)
            )
            for c in range(n_mat_cols):
                dr.scatter(result_rho_cols[c], dr.gather(mi.Float, tm_cols[c], prim_clamped), dest_slot, active)

        # Step 7: Fill counts per RX
        rx_fill_dr = dr.zeros(mi.UInt32, n_rx)
        dr.scatter_add(rx_fill_dr, mi.UInt32(1), rx_idx, active)

        if verbose:
            rx_fill_np = np.array(rx_fill_dr)
            avg_fill = float(np.mean(rx_fill_np))
            total_hits_count = int(np.sum(rx_fill_np))
            print(f"  Reservoir fill: {total_hits_count:,} total, {avg_fill:.1f} avg per RX")
            print(f"  Rays per RX: {rays_per_rx_batch}")

        # Assemble DrJit result arrays
        hit_P = mi.Point3f(result_P_x, result_P_y, result_P_z)
        hit_N = mi.Vector3f(result_N_x, result_N_y, result_N_z)

        # Convert to numpy only for ReservoirHits interface fields that require it
        rx_fill_np = np.array(rx_fill_dr).astype(np.int32)
        n_attempted_per_rx = np.full(n_rx, rays_per_rx_batch, dtype=np.int32)
        result_rho = np.stack([np.array(col) for col in result_rho_cols], axis=1).astype(np.float32)

        # Build triangle vertex Point3f arrays
        hit_tri_v0 = mi.Point3f(result_v0_x, result_v0_y, result_v0_z)
        hit_tri_v1 = mi.Point3f(result_v1_x, result_v1_y, result_v1_z)
        hit_tri_v2 = mi.Point3f(result_v2_x, result_v2_y, result_v2_z)

        return ReservoirHits(
            hit_P=hit_P,
            hit_N=hit_N,
            hit_ID=result_ID,
            ray_count=result_ray_count,
            hit_rho=result_rho,
            n_rx=n_rx,
            n_hits_per_rx=self.n_hits_per_rx,
            valid=result_valid,
            hit_pdf=np.array(result_pdf_dr),
            n_attempted_per_rx=n_attempted_per_rx,
            n_hits_per_rx_actual=rx_fill_np,
            # Triangle vertex data for image method
            hit_tri_v0=hit_tri_v0,
            hit_tri_v1=hit_tri_v1,
            hit_tri_v2=hit_tri_v2,
            # Barycentric coordinates for per-vertex interpolation
            hit_bary_u=result_bary_u_dr,
            hit_bary_v=result_bary_v_dr,
        )

    def sample_reservoir_drjit(
        self,
        scene: 'mi.Scene',
        rx_positions: 'mi.Point3f',
        rx_boresights: 'mi.Vector3f',
        seed: int = 0,
        verbose: bool = False,
        use_vertex_normals: bool = True,
    ) -> ReservoirHitsDrJit:
        """Reservoir sampling returning compact DrJit arrays for end-to-end AD."""
        n_rx = dr.width(rx_positions)

        # Reuse vectorized sampling logic
        # We inline a simplified version to avoid the numpy conversions in
        # the original sample_vectorized and to skip material lookup (handled
        # by the integrator in end-to-end mode via dr.gather on prim_ids).

        rays_per_rx_batch = self.n_hits_per_rx * 2
        total_rays = n_rx * rays_per_rx_batch

        # Sample directions
        rng = mi.PCG32(size=total_rays, initseq=seed)
        u1 = rng.next_float32()
        u2 = rng.next_float32()
        u = mi.Point2f(u1, u2)

        local_dirs, pdf = sample_cosine_hemisphere_concentric(u)

        # Expand RX for all rays
        rx_idx = dr.arange(mi.UInt32, total_rays) // rays_per_rx_batch
        expanded_pos = mi.Point3f(
            dr.gather(mi.Float, rx_positions.x, rx_idx),
            dr.gather(mi.Float, rx_positions.y, rx_idx),
            dr.gather(mi.Float, rx_positions.z, rx_idx),
        )
        expanded_bore = mi.Vector3f(
            dr.gather(mi.Float, rx_boresights.x, rx_idx),
            dr.gather(mi.Float, rx_boresights.y, rx_idx),
            dr.gather(mi.Float, rx_boresights.z, rx_idx),
        )
        expanded_bore = safe_normalize(expanded_bore)
        world_dirs = to_global(local_dirs, expanded_bore)

        # Trace rays (AD-attached if vertex positions registered in scene)
        rays = mi.Ray3f(expanded_pos, world_dirs)
        rays.maxt = mi.Float(self.max_distance)
        si = scene.ray_intersect(rays)
        hit_mask = si.is_valid()

        # Reservoir filling via prefix sum (same logic as sample_vectorized)
        hit_indicator = mi.UInt32(hit_mask)
        inclusive_cumsum = dr.prefix_sum(hit_indicator)
        cumsum = inclusive_cumsum - hit_indicator
        group_start_idx = rx_idx * mi.UInt32(rays_per_rx_batch)
        cumsum_at_start = dr.gather(mi.UInt32, cumsum, group_start_idx)
        within_rank = cumsum - cumsum_at_start
        active = hit_mask & (within_rank < mi.UInt32(self.n_hits_per_rx))

        # Compress active hits to valid-only indices into the ORIGINAL ray arrays.
        # Using direct gather from si.p (not scatter→gather) to preserve AD chain
        # for vertex position gradients. DrJit's scatter→gather breaks AD.
        valid_indices = dr.compress(active)
        n_valid_total = dr.width(valid_indices)

        si_normal = si.sh_frame.n if use_vertex_normals else si.n

        # Gather directly from ray-intersection results (AD preserved for si.p, si.n)
        hit_P = mi.Point3f(
            dr.gather(mi.Float, si.p.x, valid_indices),
            dr.gather(mi.Float, si.p.y, valid_indices),
            dr.gather(mi.Float, si.p.z, valid_indices),
        )
        hit_N = mi.Vector3f(
            dr.gather(mi.Float, si_normal.x, valid_indices),
            dr.gather(mi.Float, si_normal.y, valid_indices),
            dr.gather(mi.Float, si_normal.z, valid_indices),
        )
        hit_prim_ids = dr.gather(mi.UInt32,
            mi.UInt32(dr.maximum(si.prim_index, mi.Int32(0))), valid_indices)
        hit_bary_u = dr.gather(mi.Float, si.uv.x, valid_indices)
        hit_bary_v = dr.gather(mi.Float, si.uv.y, valid_indices)
        hit_pdf = dr.gather(mi.Float, pdf, valid_indices)
        hit_rx_idx = dr.gather(mi.UInt32, rx_idx, valid_indices)

        # Get vertex indices from mesh face connectivity
        vertex_ids_0 = vertex_ids_1 = vertex_ids_2 = None
        mesh = scene.shapes()[0] if len(scene.shapes()) > 0 else None
        if mesh is not None:
            face_idx = mesh.face_indices(hit_prim_ids)
            vertex_ids_0 = face_idx[0]
            vertex_ids_1 = face_idx[1]
            vertex_ids_2 = face_idx[2]

        n_attempted = dr.full(mi.UInt32, rays_per_rx_batch, n_rx)

        if verbose:
            print(f"  [sample_reservoir_drjit] {n_valid_total} valid hits from "
                  f"{n_rx} RX × {rays_per_rx_batch} rays")

        return ReservoirHitsDrJit(
            hit_P=hit_P,
            hit_N=hit_N,
            hit_prim_ids=hit_prim_ids,
            hit_bary_u=hit_bary_u,
            hit_bary_v=hit_bary_v,
            hit_pdf=hit_pdf,
            rx_element_idx=hit_rx_idx,
            n_valid=n_valid_total,
            n_attempted_per_rx=n_attempted,
            n_rx=n_rx,
            vertex_ids_0=vertex_ids_0,
            vertex_ids_1=vertex_ids_1,
            vertex_ids_2=vertex_ids_2,
        )


__all__ = [
    'ReservoirSampler',
    'ReservoirHits',
    'ReservoirHitsDrJit',
]
