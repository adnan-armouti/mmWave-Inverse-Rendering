"""Full GPU-accelerated batch aperture construction (Steps 1-14)."""

import numpy as np
import drjit as dr
import mitsuba as mi
from dataclasses import dataclass
from typing import Optional, Tuple, List

from .fsd_bsdf import FlatEdgeData

# FSD power integral constants (from fsd_aperture.py)
_INTEGRAL_1 = 0.0045255085   # ∫(1-χ)|α₁|² for Phat
_INTEGRAL_2 = 0.114875434    # ∫(1-χ)|α₂|² for Phat


# GPU-resident spatial hash data (uploaded once at mesh load)

@dataclass
class GPUSpatialHashData:
    """All mesh + hash-grid data needed for full GPU aperture construction."""

    # Per-face centroids [F] (Structure-of-Arrays)
    centroid_x: 'mi.Float'
    centroid_y: 'mi.Float'
    centroid_z: 'mi.Float'

    # Per-face cell coordinates [F] — for exact cell matching (no hash collisions)
    cell_ix: 'mi.Int32'
    cell_iy: 'mi.Int32'
    cell_iz: 'mi.Int32'

    # Hash grid (faces sorted by cell hash)
    sorted_face_idx: 'mi.UInt32'   # [F]
    cell_start: 'mi.UInt32'        # [T] start offset per bucket
    cell_count: 'mi.UInt32'        # [T] face count per bucket

    # Per-face geometry [F] — for back-face culling, projection, material
    fn_x: 'mi.Float'              # face normal x
    fn_y: 'mi.Float'              # face normal y
    fn_z: 'mi.Float'              # face normal z
    v0_x: 'mi.Float'              # vertex 0 x
    v0_y: 'mi.Float'              # vertex 0 y
    v0_z: 'mi.Float'              # vertex 0 z
    v1_x: 'mi.Float'              # vertex 1 x
    v1_y: 'mi.Float'              # vertex 1 y
    v1_z: 'mi.Float'              # vertex 1 z
    v2_x: 'mi.Float'              # vertex 2 x
    v2_y: 'mi.Float'              # vertex 2 y
    v2_z: 'mi.Float'              # vertex 2 z

    # Edge detection data [F*3] — neighbor normals and suppression flags
    # Indexed as face_idx * 3 + edge_j (j=0,1,2)
    nn_x: 'mi.Float'              # neighbor face normal x
    nn_y: 'mi.Float'              # neighbor face normal y
    nn_z: 'mi.Float'              # neighbor face normal z
    edge_suppressed: 'mi.Bool'    # pre-computed suppression flags

    # Mesh connectivity [F] — vertex indices per face
    face_vi0: 'mi.UInt32'
    face_vi1: 'mi.UInt32'
    face_vi2: 'mi.UInt32'

    # Scalars
    n_faces: int
    table_size: int
    inv_cell_size: float
    max_per_cell: int


def _spatial_hash_cpu(ix, iy, iz, table_size):
    """Hash (ix, iy, iz) integer cell coords → bucket index."""
    h = (ix.astype(np.uint32) * np.uint32(73856093)) ^ \
        (iy.astype(np.uint32) * np.uint32(19349669)) ^ \
        (iz.astype(np.uint32) * np.uint32(83492791))
    return (h % np.uint32(table_size)).astype(np.int32)


def _spatial_hash_gpu(ix: 'mi.Int32', iy: 'mi.Int32', iz: 'mi.Int32',
                      table_size: int) -> 'mi.UInt32':
    """Hash (ix, iy, iz) DrJit integer cell coords → bucket index on GPU."""
    hx = dr.reinterpret_array(mi.UInt32, ix) * mi.UInt32(73856093)
    hy = dr.reinterpret_array(mi.UInt32, iy) * mi.UInt32(19349669)
    hz = dr.reinterpret_array(mi.UInt32, iz) * mi.UInt32(83492791)
    h = hx ^ hy ^ hz
    return h % mi.UInt32(table_size)


def build_gpu_spatial_hash(tri_hash) -> GPUSpatialHashData:
    """Upload a TriangleSpatialHash to GPU as persistent DrJit arrays."""
    F = tri_hash.n_faces

    # Per-face centroids
    cx = tri_hash.centroids[:, 0].astype(np.float32)
    cy = tri_hash.centroids[:, 1].astype(np.float32)
    cz = tri_hash.centroids[:, 2].astype(np.float32)

    # Build sorted hash grid on CPU
    inv_cs = tri_hash.inv_cell_size
    ix = np.floor(cx * inv_cs).astype(np.int32)
    iy = np.floor(cy * inv_cs).astype(np.int32)
    iz = np.floor(cz * inv_cs).astype(np.int32)

    # Choose table size: next prime > 2 × #occupied cells
    n_occupied = len(tri_hash._grid)
    table_size = max(int(2.5 * n_occupied), 1024)
    if table_size % 2 == 0:
        table_size += 1

    cell_hashes = _spatial_hash_cpu(ix, iy, iz, table_size)

    # Sort faces by cell hash
    sort_order = np.argsort(cell_hashes, kind='stable')
    sorted_hashes = cell_hashes[sort_order]
    sorted_face = sort_order.astype(np.int32)

    # Build cell_start / cell_count
    cell_count_np = np.bincount(sorted_hashes.clip(0), minlength=table_size).astype(np.int32)
    cell_start_np = np.zeros(table_size, dtype=np.int32)
    if table_size > 1:
        np.cumsum(cell_count_np[:-1], out=cell_start_np[1:])

    max_per_cell = int(cell_count_np.max()) if F > 0 else 0
    max_per_cell = min(max_per_cell, 512)

    # Per-face geometry
    fn = tri_hash.face_normals.astype(np.float32)  # [F, 3]
    v0 = tri_hash.tri_v0.astype(np.float32)        # [F, 3]
    v1 = tri_hash.tri_v1.astype(np.float32)
    v2 = tri_hash.tri_v2.astype(np.float32)

    # Edge detection data [F*3]
    nn = tri_hash.neighbor_normals.astype(np.float32)  # [F, 3, 3]
    # Flatten to [F*3, 3]: for face f, edge j → index f*3 + j
    nn_flat = nn.reshape(-1, 3)  # [F*3, 3]
    es = tri_hash.edge_is_suppressed.ravel()  # [F*3] bool

    # Mesh connectivity
    faces = tri_hash.faces.astype(np.int32)  # [F, 3]

    print(f"[GPUSpatialHash] table_size={table_size}, n_occupied={n_occupied}, "
          f"max_per_cell={max_per_cell}, F={F}")

    return GPUSpatialHashData(
        centroid_x=mi.Float(cx), centroid_y=mi.Float(cy), centroid_z=mi.Float(cz),
        cell_ix=mi.Int32(ix), cell_iy=mi.Int32(iy), cell_iz=mi.Int32(iz),
        sorted_face_idx=mi.UInt32(sorted_face),
        cell_start=mi.UInt32(cell_start_np), cell_count=mi.UInt32(cell_count_np),
        # Per-face geometry
        fn_x=mi.Float(fn[:, 0]), fn_y=mi.Float(fn[:, 1]), fn_z=mi.Float(fn[:, 2]),
        v0_x=mi.Float(v0[:, 0]), v0_y=mi.Float(v0[:, 1]), v0_z=mi.Float(v0[:, 2]),
        v1_x=mi.Float(v1[:, 0]), v1_y=mi.Float(v1[:, 1]), v1_z=mi.Float(v1[:, 2]),
        v2_x=mi.Float(v2[:, 0]), v2_y=mi.Float(v2[:, 1]), v2_z=mi.Float(v2[:, 2]),
        # Edge detection
        nn_x=mi.Float(nn_flat[:, 0]), nn_y=mi.Float(nn_flat[:, 1]), nn_z=mi.Float(nn_flat[:, 2]),
        edge_suppressed=mi.Bool(es),
        # Mesh connectivity
        face_vi0=mi.UInt32(faces[:, 0]), face_vi1=mi.UInt32(faces[:, 1]), face_vi2=mi.UInt32(faces[:, 2]),
        # Scalars
        n_faces=F, table_size=table_size, inv_cell_size=inv_cs,
        max_per_cell=max_per_cell,
    )


# Helper: DrJit Sigmat computation (port of _batch_Sigmat)

def _batch_Sigmat_drjit(u1x, u1y, u2x, u2y, u3x, u3y, ph1, ph2, ph3):
    """GPU Sigmat: weighted second moments for [M] triangles."""
    area = dr.abs(
        -u1y * u2x + u1x * u2y + u1y * u3x - u2y * u3x - u1x * u3y + u2x * u3y
    )

    inv60 = mi.Float(1.0 / 60.0)
    inv120 = mi.Float(1.0 / 120.0)

    # Sigma_xx
    Sxx = area * (
        (mi.Float(3.0) * ph1 + ph2 + ph3) * u1x * u1x
        + (ph1 + mi.Float(3.0) * ph2 + ph3) * u2x * u2x
        + (ph1 + mi.Float(2.0) * (ph2 + ph3)) * u2x * u3x
        + (ph1 + ph2 + mi.Float(3.0) * ph3) * u3x * u3x
        + u1x * ((mi.Float(2.0) * (ph1 + ph2) + ph3) * u2x + (mi.Float(2.0) * ph1 + ph2 + mi.Float(2.0) * ph3) * u3x)
    ) * inv60

    # Sigma_xy
    Sxy = area * (
        u1x * (mi.Float(2.0) * (mi.Float(3.0) * ph1 + ph2 + ph3) * u1y + (mi.Float(2.0) * (ph1 + ph2) + ph3) * u2y + (mi.Float(2.0) * ph1 + ph2 + mi.Float(2.0) * ph3) * u3y)
        + u3x * ((mi.Float(2.0) * ph1 + ph2 + mi.Float(2.0) * ph3) * u1y + (ph1 + mi.Float(2.0) * (ph2 + ph3)) * u2y + mi.Float(2.0) * (ph1 + ph2 + mi.Float(3.0) * ph3) * u3y)
        + u2x * ((mi.Float(2.0) * (ph1 + ph2) + ph3) * u1y + mi.Float(2.0) * (ph1 + mi.Float(3.0) * ph2 + ph3) * u2y + (ph1 + mi.Float(2.0) * (ph2 + ph3)) * u3y)
    ) * inv120

    # Sigma_yy
    Syy = area * (
        (mi.Float(3.0) * ph1 + ph2 + ph3) * u1y * u1y
        + (ph1 + mi.Float(3.0) * ph2 + ph3) * u2y * u2y
        + (ph1 + mi.Float(2.0) * (ph2 + ph3)) * u2y * u3y
        + (ph1 + ph2 + mi.Float(3.0) * ph3) * u3y * u3y
        + u1y * ((mi.Float(2.0) * (ph1 + ph2) + ph3) * u2y + (mi.Float(2.0) * ph1 + ph2 + mi.Float(2.0) * ph3) * u3y)
    ) * inv60

    return Sxx, Sxy, Syy


# Helper: gather and re-index arrays after dr.compress

def _gather_sel(arr, sel):
    """Gather mi.Float elements at indices sel."""
    return dr.gather(mi.Float, arr, sel)

def _gather_sel_uint(arr, sel):
    """Gather mi.UInt32 elements at indices sel."""
    return dr.gather(mi.UInt32, arr, sel)

def _gather_sel_bool(arr, sel):
    """Gather mi.Bool elements at indices sel."""
    return dr.gather(mi.Bool, arr, sel)


# Main GPU aperture construction (Steps 1-14)

def construct_apertures_batch_gpu(
    hit_pos_all: np.ndarray,         # [N, 3]
    wo_all: np.ndarray,              # [N, 3]
    gpu_hash: GPUSpatialHashData,
    k: float,
    beam_sigma: float,
    max_tessellation_depth: int = 5,
    max_edges_per_hit: int = 256,
    fill_min: float = 1e-6,
    fill_max: float = 1.0 - 1e-6,
    tri_eps_real: Optional[np.ndarray] = None,   # [F]
    tri_eps_imag: Optional[np.ndarray] = None,   # [F]
    tx_polarization: Optional[np.ndarray] = None,
    rx_polarization: Optional[np.ndarray] = None,
    jones_mode: bool = False,
    beta_max: float = 0.5,
    verbose: bool = False,
    edge_angle_threshold_deg: float = 15.0,
    tri_hash=None,
) -> Optional[Tuple[FlatEdgeData, np.ndarray]]:
    """Full GPU-accelerated batch aperture construction (Steps 1-14)."""
    from .fsd_bsdf import _fresnel_opacity_drjit

    N = len(hit_pos_all)
    if N == 0:
        return None

    search_radius = 3.0 * beam_sigma
    r_sq = float(search_radius ** 2)
    circle_area = float(np.pi * search_radius ** 2)
    max_edge_len_sq = float(beam_sigma ** 2)
    inv_cs = gpu_hash.inv_cell_size
    use_material_opacity = (tri_eps_real is not None and tri_eps_imag is not None)
    jones_mode = jones_mode and use_material_opacity

    # Upload per-iteration hit positions and directions to GPU
    hx = mi.Float(hit_pos_all[:, 0].astype(np.float32))
    hy = mi.Float(hit_pos_all[:, 1].astype(np.float32))
    hz = mi.Float(hit_pos_all[:, 2].astype(np.float32))

    # Step 1: Frame setup (tangent/bitangent for all N hits) on GPU
    wo_x = mi.Float(wo_all[:, 0].astype(np.float32))
    wo_y = mi.Float(wo_all[:, 1].astype(np.float32))
    wo_z = mi.Float(wo_all[:, 2].astype(np.float32))

    wo_norm = dr.sqrt(dr.maximum(wo_x * wo_x + wo_y * wo_y + wo_z * wo_z, mi.Float(1e-24)))
    wo_x = wo_x / wo_norm; wo_y = wo_y / wo_norm; wo_z = wo_z / wo_norm

    # tangent = up - (wo · up) * wo, up = (0, 0, 1)
    dot_wo_up = wo_z  # wo · (0,0,1)
    tang_x = -dot_wo_up * wo_x
    tang_y = -dot_wo_up * wo_y
    tang_z = mi.Float(1.0) - dot_wo_up * wo_z

    # Handle near-parallel case: use alt_up = (0, 1, 0)
    near_parallel = dr.abs(dot_wo_up) > mi.Float(0.999)
    dot_wo_alt = wo_y  # wo · (0,1,0)
    alt_tang_x = -dot_wo_alt * wo_x
    alt_tang_y = mi.Float(1.0) - dot_wo_alt * wo_y
    alt_tang_z = -dot_wo_alt * wo_z

    tang_x = dr.select(near_parallel, alt_tang_x, tang_x)
    tang_y = dr.select(near_parallel, alt_tang_y, tang_y)
    tang_z = dr.select(near_parallel, alt_tang_z, tang_z)

    t_norm = dr.sqrt(dr.maximum(tang_x * tang_x + tang_y * tang_y + tang_z * tang_z, mi.Float(1e-24)))
    tang_x = tang_x / t_norm; tang_y = tang_y / t_norm; tang_z = tang_z / t_norm

    # bitangent = wo × tangent
    btang_x = wo_y * tang_z - wo_z * tang_y
    btang_y = wo_z * tang_x - wo_x * tang_z
    btang_z = wo_x * tang_y - wo_y * tang_x

    # Step 2: GPU Spatial query – 27-pass hash lookup
    hit_ix = mi.Int32(dr.floor(hx * mi.Float(inv_cs)))
    hit_iy = mi.Int32(dr.floor(hy * mi.Float(inv_cs)))
    hit_iz = mi.Int32(dr.floor(hz * mi.Float(inv_cs)))

    K_MAX = 128  # max faces per hit
    pair_buf = dr.full(mi.UInt32, 0xFFFFFFFF, N * K_MAX)
    hit_counts = dr.zeros(mi.UInt32, N)

    tbl = gpu_hash.table_size
    mpc = gpu_hash.max_per_cell

    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                total_work = N * mpc
                if total_work == 0:
                    continue
                wi = dr.arange(mi.UInt32, total_work)
                w_hit = wi // mi.UInt32(mpc)
                w_rank = wi % mi.UInt32(mpc)

                nb_ix = dr.gather(mi.Int32, hit_ix, w_hit) + mi.Int32(di)
                nb_iy = dr.gather(mi.Int32, hit_iy, w_hit) + mi.Int32(dj)
                nb_iz = dr.gather(mi.Int32, hit_iz, w_hit) + mi.Int32(dk)
                nb_hash = _spatial_hash_gpu(nb_ix, nb_iy, nb_iz, tbl)

                c_count = dr.gather(mi.UInt32, gpu_hash.cell_count, nb_hash)
                valid_rank = w_rank < c_count

                c_start = dr.gather(mi.UInt32, gpu_hash.cell_start, nb_hash)
                src_idx = c_start + w_rank
                src_idx = dr.minimum(src_idx, mi.UInt32(gpu_hash.n_faces - 1))
                face_j = dr.gather(mi.UInt32, gpu_hash.sorted_face_idx, src_idx, valid_rank)

                # Exact cell match
                f_cix = dr.gather(mi.Int32, gpu_hash.cell_ix, face_j, valid_rank)
                f_ciy = dr.gather(mi.Int32, gpu_hash.cell_iy, face_j, valid_rank)
                f_ciz = dr.gather(mi.Int32, gpu_hash.cell_iz, face_j, valid_rank)
                cell_match = (f_cix == nb_ix) & (f_ciy == nb_iy) & (f_ciz == nb_iz)

                # Distance check
                f_cx = dr.gather(mi.Float, gpu_hash.centroid_x, face_j, valid_rank)
                f_cy = dr.gather(mi.Float, gpu_hash.centroid_y, face_j, valid_rank)
                f_cz = dr.gather(mi.Float, gpu_hash.centroid_z, face_j, valid_rank)
                h_x = dr.gather(mi.Float, hx, w_hit)
                h_y = dr.gather(mi.Float, hy, w_hit)
                h_z = dr.gather(mi.Float, hz, w_hit)
                dsq = (h_x - f_cx) ** 2 + (h_y - f_cy) ** 2 + (h_z - f_cz) ** 2
                within = valid_rank & cell_match & (dsq < mi.Float(r_sq))

                slot = dr.scatter_inc(hit_counts, w_hit, within)
                can_store = within & (slot < mi.UInt32(K_MAX))
                dest = w_hit * mi.UInt32(K_MAX) + slot
                dr.scatter(pair_buf, face_j, dest, can_store)

    dr.eval(pair_buf, hit_counts)

    # Flatten valid pairs
    all_idx = dr.arange(mi.UInt32, N * K_MAX)
    pair_hit_idx = all_idx // mi.UInt32(K_MAX)
    pair_local = all_idx % mi.UInt32(K_MAX)
    pair_face = dr.gather(mi.UInt32, pair_buf, all_idx)
    pair_limit = dr.gather(mi.UInt32, hit_counts, pair_hit_idx)
    pair_valid = (pair_local < pair_limit) & (pair_face != mi.UInt32(0xFFFFFFFF))

    valid_sel = dr.compress(pair_valid)
    M = dr.width(valid_sel)
    if M == 0:
        return None

    hit_idx = dr.gather(mi.UInt32, pair_hit_idx, valid_sel)
    face_idx = dr.gather(mi.UInt32, pair_face, valid_sel)

    # Step 3: Back-face culling
    fn_x_m = dr.gather(mi.Float, gpu_hash.fn_x, face_idx)
    fn_y_m = dr.gather(mi.Float, gpu_hash.fn_y, face_idx)
    fn_z_m = dr.gather(mi.Float, gpu_hash.fn_z, face_idx)

    wo_x_m = dr.gather(mi.Float, wo_x, hit_idx)
    wo_y_m = dr.gather(mi.Float, wo_y, hit_idx)
    wo_z_m = dr.gather(mi.Float, wo_z, hit_idx)

    dots = wo_x_m * fn_x_m + wo_y_m * fn_y_m + wo_z_m * fn_z_m
    front_sel = dr.compress(dots > mi.Float(0.0))
    M_f = dr.width(front_sel)
    if M_f == 0:
        return None

    hit_idx_f = dr.gather(mi.UInt32, hit_idx, front_sel)
    face_idx_f = dr.gather(mi.UInt32, face_idx, front_sel)

    # Step 4: Vertex projection onto per-hit virtual screens
    # Gather hit positions
    hp_x = dr.gather(mi.Float, hx, hit_idx_f)
    hp_y = dr.gather(mi.Float, hy, hit_idx_f)
    hp_z = dr.gather(mi.Float, hz, hit_idx_f)

    # Gather vertices
    vv0_x = dr.gather(mi.Float, gpu_hash.v0_x, face_idx_f)
    vv0_y = dr.gather(mi.Float, gpu_hash.v0_y, face_idx_f)
    vv0_z = dr.gather(mi.Float, gpu_hash.v0_z, face_idx_f)
    vv1_x = dr.gather(mi.Float, gpu_hash.v1_x, face_idx_f)
    vv1_y = dr.gather(mi.Float, gpu_hash.v1_y, face_idx_f)
    vv1_z = dr.gather(mi.Float, gpu_hash.v1_z, face_idx_f)
    vv2_x = dr.gather(mi.Float, gpu_hash.v2_x, face_idx_f)
    vv2_y = dr.gather(mi.Float, gpu_hash.v2_y, face_idx_f)
    vv2_z = dr.gather(mi.Float, gpu_hash.v2_z, face_idx_f)

    # Displacement vectors
    d0_x = vv0_x - hp_x; d0_y = vv0_y - hp_y; d0_z = vv0_z - hp_z
    d1_x = vv1_x - hp_x; d1_y = vv1_y - hp_y; d1_z = vv1_z - hp_z
    d2_x = vv2_x - hp_x; d2_y = vv2_y - hp_y; d2_z = vv2_z - hp_z

    # Gather per-hit frame
    tf_x = dr.gather(mi.Float, tang_x, hit_idx_f)
    tf_y = dr.gather(mi.Float, tang_y, hit_idx_f)
    tf_z = dr.gather(mi.Float, tang_z, hit_idx_f)
    bf_x = dr.gather(mi.Float, btang_x, hit_idx_f)
    bf_y = dr.gather(mi.Float, btang_y, hit_idx_f)
    bf_z = dr.gather(mi.Float, btang_z, hit_idx_f)
    wf_x = dr.gather(mi.Float, wo_x, hit_idx_f)
    wf_y = dr.gather(mi.Float, wo_y, hit_idx_f)
    wf_z = dr.gather(mi.Float, wo_z, hit_idx_f)

    # Screen coordinates
    u1x = d0_x * tf_x + d0_y * tf_y + d0_z * tf_z
    u1y = d0_x * bf_x + d0_y * bf_y + d0_z * bf_z
    u2x = d1_x * tf_x + d1_y * tf_y + d1_z * tf_z
    u2y = d1_x * bf_x + d1_y * bf_y + d1_z * bf_z
    u3x = d2_x * tf_x + d2_y * tf_y + d2_z * tf_z
    u3y = d2_x * bf_x + d2_y * bf_y + d2_z * bf_z

    # Z-depths along -wo direction
    z1 = -(d0_x * wf_x + d0_y * wf_y + d0_z * wf_z)
    z2 = -(d1_x * wf_x + d1_y * wf_y + d1_z * wf_z)
    z3 = -(d2_x * wf_x + d2_y * wf_y + d2_z * wf_z)

    # Step 5: Fill fraction filter
    area_2x = dr.abs((u2x - u1x) * (u3y - u1y) - (u3x - u1x) * (u2y - u1y))

    fill_per_hit = dr.zeros(mi.Float, N)
    dr.scatter_reduce(dr.ReduceOp.Add, fill_per_hit, area_2x * mi.Float(0.5), hit_idx_f)
    dr.eval(fill_per_hit)

    fill_per_hit = fill_per_hit / mi.Float(circle_area)
    skip = (fill_per_hit > mi.Float(fill_max)) | (fill_per_hit < mi.Float(fill_min))
    skip_f = dr.gather(mi.Bool, skip, hit_idx_f)
    keep_sel = dr.compress(~skip_f)
    M_f = dr.width(keep_sel)
    if M_f == 0:
        return None

    hit_idx_f = _gather_sel_uint(hit_idx_f, keep_sel)
    face_idx_f = _gather_sel_uint(face_idx_f, keep_sel)
    u1x = _gather_sel(u1x, keep_sel); u1y = _gather_sel(u1y, keep_sel)
    u2x = _gather_sel(u2x, keep_sel); u2y = _gather_sel(u2y, keep_sel)
    u3x = _gather_sel(u3x, keep_sel); u3y = _gather_sel(u3y, keep_sel)
    z1 = _gather_sel(z1, keep_sel); z2 = _gather_sel(z2, keep_sel); z3 = _gather_sel(z3, keep_sel)

    # Step 6: Edge detection with dihedral angle filtering
    # For each face, 3 edges: j=0,1,2. Neighbor normal at face_idx_f * 3 + j
    base_idx = face_idx_f * mi.UInt32(3)  # [M_f]

    # Gather wo for each pair
    wo_fx = dr.gather(mi.Float, wo_x, hit_idx_f)
    wo_fy = dr.gather(mi.Float, wo_y, hit_idx_f)
    wo_fz = dr.gather(mi.Float, wo_z, hit_idx_f)

    # Gather face normals for dihedral computation
    fnx_f = dr.gather(mi.Float, gpu_hash.fn_x, face_idx_f)
    fny_f = dr.gather(mi.Float, gpu_hash.fn_y, face_idx_f)
    fnz_f = dr.gather(mi.Float, gpu_hash.fn_z, face_idx_f)

    cos_threshold = float(np.cos(np.radians(edge_angle_threshold_deg)))

    # Process 3 edges per face
    e_diffracting = []
    for j in range(3):
        idx_j = base_idx + mi.UInt32(j)
        nnx_j = dr.gather(mi.Float, gpu_hash.nn_x, idx_j)
        nny_j = dr.gather(mi.Float, gpu_hash.nn_y, idx_j)
        nnz_j = dr.gather(mi.Float, gpu_hash.nn_z, idx_j)

        # Neighbor normal magnitude
        nn_norm = dr.sqrt(nnx_j * nnx_j + nny_j * nny_j + nnz_j * nnz_j)
        has_neighbor = nn_norm > mi.Float(0.5)

        # wo · neighbor_normal
        wo_dot_nn = wo_fx * nnx_j + wo_fy * nny_j + wo_fz * nnz_j
        back_facing = wo_dot_nn <= mi.Float(0.0)

        # Dihedral: cos_dihedral = dot(fn, nn) / |nn|
        cos_dih = (fnx_f * nnx_j + fny_f * nny_j + fnz_f * nnz_j) / dr.maximum(nn_norm, mi.Float(1e-8))
        is_significant = cos_dih < mi.Float(cos_threshold)

        # Suppression flag
        suppressed = dr.gather(mi.Bool, gpu_hash.edge_suppressed, idx_j)

        is_diffracting = has_neighbor & back_facing & is_significant & ~suppressed
        e_diffracting.append(is_diffracting)

    e12 = e_diffracting[0]  # edge between v0-v1 (matches CPU edge type 0)
    e13 = e_diffracting[1]  # edge between v1-v2 (matches CPU edge type 1)
    e23 = e_diffracting[2]  # edge between v2-v0 (matches CPU edge type 2)

    # Step 7: Material pre-computation (Fresnel / Jones)
    fn_norm = dr.sqrt(dr.maximum(fnx_f * fnx_f + fny_f * fny_f + fnz_f * fnz_f, mi.Float(1e-24)))
    fn_ux = fnx_f / fn_norm; fn_uy = fny_f / fn_norm; fn_uz = fnz_f / fn_norm
    cos_theta_f = dr.abs(wo_fx * fn_ux + wo_fy * fn_uy + wo_fz * fn_uz)

    gpu_eps_r = mi.Float(tri_eps_real.astype(np.float32))
    gpu_eps_i = mi.Float(tri_eps_imag.astype(np.float32))
    eps_r_f = dr.gather(mi.Float, gpu_eps_r, face_idx_f)
    eps_i_f = dr.gather(mi.Float, gpu_eps_i, face_idx_f)
    opacity_f = _fresnel_opacity_drjit(eps_r_f, eps_i_f, cos_theta_f)

    face_idx_f_np = np.array(face_idx_f).astype(np.int32)
    cos_theta_f_np = np.array(cos_theta_f).astype(np.float64)
    fn_unit_np = np.stack([np.array(fn_ux), np.array(fn_uy), np.array(fn_uz)], axis=1).astype(np.float64)
    wo_f_np = np.stack([np.array(wo_fx), np.array(wo_fy), np.array(wo_fz)], axis=1).astype(np.float64)
    eps_r_np = tri_eps_real[face_idx_f_np].astype(np.float64)
    eps_i_np = tri_eps_imag[face_idx_f_np].astype(np.float64)

    from .fsd_aperture import _jones_reflectance_numpy_batch
    jr_np, ji_np, jpow_np, ts_np, tp_np, rs_np, rp_np = \
        _jones_reflectance_numpy_batch(
            eps_r_np, eps_i_np, cos_theta_f_np,
            wo_f_np[0],  # Use first hit's wo (small variation)
            fn_unit_np,
            tx_polarization, rx_polarization,
        )
    jr_f = mi.Float(jr_np.astype(np.float32))
    ji_f = mi.Float(ji_np.astype(np.float32))
    jpow_f = mi.Float(jpow_np.astype(np.float32))
    ts_f = mi.Float(ts_np.astype(np.float32))
    tp_f = mi.Float(tp_np.astype(np.float32))
    rs_f = mi.Float(rs_np.astype(np.float32))
    rp_f = mi.Float(rp_np.astype(np.float32))

    # Step 8: Tessellation (longest-edge bisection, up to 5 levels)
    cur_u1x = u1x; cur_u1y = u1y
    cur_u2x = u2x; cur_u2y = u2y
    cur_u3x = u3x; cur_u3y = u3y
    cur_z1 = z1; cur_z2 = z2; cur_z3 = z3
    cur_e12 = e12; cur_e13 = e13; cur_e23 = e23
    cur_hit = hit_idx_f; cur_face = face_idx_f
    cur_pair = dr.arange(mi.UInt32, M_f)  # pair index into material arrays

    # Barycentric coordinates: u1=v0→(0,0), u2=v1→(1,0), u3=v2→(0,1)
    cur_bu1 = dr.zeros(mi.Float, M_f)
    cur_bv1 = dr.zeros(mi.Float, M_f)
    cur_bu2 = dr.full(mi.Float, 1.0, M_f)
    cur_bv2 = dr.zeros(mi.Float, M_f)
    cur_bu3 = dr.zeros(mi.Float, M_f)
    cur_bv3 = dr.full(mi.Float, 1.0, M_f)

    # Leaf collectors
    leaf_arrays: List[dict] = []

    max_sq = mi.Float(max_edge_len_sq)

    for _depth in range(max_tessellation_depth):
        T = dr.width(cur_u1x)
        if T == 0:
            break

        # Screen-space edge lengths²
        l12 = (cur_u2x - cur_u1x) ** 2 + (cur_u2y - cur_u1y) ** 2
        l13 = (cur_u3x - cur_u1x) ** 2 + (cur_u3y - cur_u1y) ** 2
        l23 = (cur_u3x - cur_u2x) ** 2 + (cur_u3y - cur_u2y) ** 2

        needs_sub = (l12 > max_sq) | (l13 > max_sq) | (l23 > max_sq)

        # Collect leaves
        leaf_sel = dr.compress(~needs_sub)
        if dr.width(leaf_sel) > 0:
            leaf_arrays.append({
                'u1x': _gather_sel(cur_u1x, leaf_sel), 'u1y': _gather_sel(cur_u1y, leaf_sel),
                'u2x': _gather_sel(cur_u2x, leaf_sel), 'u2y': _gather_sel(cur_u2y, leaf_sel),
                'u3x': _gather_sel(cur_u3x, leaf_sel), 'u3y': _gather_sel(cur_u3y, leaf_sel),
                'z1': _gather_sel(cur_z1, leaf_sel), 'z2': _gather_sel(cur_z2, leaf_sel), 'z3': _gather_sel(cur_z3, leaf_sel),
                'e12': _gather_sel_bool(cur_e12, leaf_sel), 'e13': _gather_sel_bool(cur_e13, leaf_sel), 'e23': _gather_sel_bool(cur_e23, leaf_sel),
                'hit': _gather_sel_uint(cur_hit, leaf_sel), 'face': _gather_sel_uint(cur_face, leaf_sel), 'pair': _gather_sel_uint(cur_pair, leaf_sel),
                'bu1': _gather_sel(cur_bu1, leaf_sel), 'bv1': _gather_sel(cur_bv1, leaf_sel),
                'bu2': _gather_sel(cur_bu2, leaf_sel), 'bv2': _gather_sel(cur_bv2, leaf_sel),
                'bu3': _gather_sel(cur_bu3, leaf_sel), 'bv3': _gather_sel(cur_bv3, leaf_sel),
            })

        sub_sel = dr.compress(needs_sub)
        T_sub = dr.width(sub_sel)
        if T_sub == 0:
            break

        # Gather active sub-triangles
        su1x = _gather_sel(cur_u1x, sub_sel); su1y = _gather_sel(cur_u1y, sub_sel)
        su2x = _gather_sel(cur_u2x, sub_sel); su2y = _gather_sel(cur_u2y, sub_sel)
        su3x = _gather_sel(cur_u3x, sub_sel); su3y = _gather_sel(cur_u3y, sub_sel)
        sz1 = _gather_sel(cur_z1, sub_sel); sz2 = _gather_sel(cur_z2, sub_sel); sz3 = _gather_sel(cur_z3, sub_sel)
        se12 = _gather_sel_bool(cur_e12, sub_sel); se13 = _gather_sel_bool(cur_e13, sub_sel); se23 = _gather_sel_bool(cur_e23, sub_sel)
        s_hit = _gather_sel_uint(cur_hit, sub_sel); s_face = _gather_sel_uint(cur_face, sub_sel); s_pair = _gather_sel_uint(cur_pair, sub_sel)
        sbu1 = _gather_sel(cur_bu1, sub_sel); sbv1 = _gather_sel(cur_bv1, sub_sel)
        sbu2 = _gather_sel(cur_bu2, sub_sel); sbv2 = _gather_sel(cur_bv2, sub_sel)
        sbu3 = _gather_sel(cur_bu3, sub_sel); sbv3 = _gather_sel(cur_bv3, sub_sel)

        # Edge lengths² for sub-triangles
        sl12 = _gather_sel(l12, sub_sel)
        sl13 = _gather_sel(l13, sub_sel)
        sl23 = _gather_sel(l23, sub_sel)

        # Find longest edge
        split_12 = (sl12 >= sl13) & (sl12 >= sl23)
        split_13 = (~split_12) & (sl13 >= sl23)

        # Midpoint of longest edge
        half = mi.Float(0.5)
        mid_x = dr.select(split_12, (su1x + su2x) * half,
                    dr.select(split_13, (su1x + su3x) * half,
                                        (su2x + su3x) * half))
        mid_y = dr.select(split_12, (su1y + su2y) * half,
                    dr.select(split_13, (su1y + su3y) * half,
                                        (su2y + su3y) * half))
        mid_z = dr.select(split_12, (sz1 + sz2) * half,
                    dr.select(split_13, (sz1 + sz3) * half,
                                        (sz2 + sz3) * half))

        # Midpoint barycentrics
        mid_bu = dr.select(split_12, (sbu1 + sbu2) * half,
                     dr.select(split_13, (sbu1 + sbu3) * half,
                                         (sbu2 + sbu3) * half))
        mid_bv = dr.select(split_12, (sbv1 + sbv2) * half,
                     dr.select(split_13, (sbv1 + sbv3) * half,
                                         (sbv2 + sbv3) * half))

        # Sub-triangle A
        # edge12 → A=(u1, mid, u3)
        # edge13 → A=(u1, u2, mid)
        # edge23 → A=(u1, mid, u3)
        a_u1x = su1x; a_u1y = su1y; a_z1 = sz1
        a_u2x = dr.select(split_13, su2x, mid_x)
        a_u2y = dr.select(split_13, su2y, mid_y)
        a_z2 = dr.select(split_13, sz2, mid_z)
        a_u3x = dr.select(split_13, mid_x, su3x)
        a_u3y = dr.select(split_13, mid_y, su3y)
        a_z3 = dr.select(split_13, mid_z, sz3)
        # Edge flags for A
        split_23 = ~split_12 & ~split_13
        a_e12 = se12 & ~split_23
        a_e13 = se13
        a_e23 = se23 & split_23
        # Barycentrics for A
        a_bu1 = sbu1; a_bv1 = sbv1
        a_bu2 = dr.select(split_13, sbu2, mid_bu)
        a_bv2 = dr.select(split_13, sbv2, mid_bv)
        a_bu3 = dr.select(split_13, mid_bu, sbu3)
        a_bv3 = dr.select(split_13, mid_bv, sbv3)

        # Sub-triangle B
        # edge12 → B=(mid, u2, u3)
        # edge13 → B=(mid, u2, u3)
        # edge23 → B=(u1, u2, mid)
        b_u1x = dr.select(split_23, su1x, mid_x)
        b_u1y = dr.select(split_23, su1y, mid_y)
        b_z1 = dr.select(split_23, sz1, mid_z)
        b_u2x = su2x; b_u2y = su2y; b_z2 = sz2
        b_u3x = dr.select(split_23, mid_x, su3x)
        b_u3y = dr.select(split_23, mid_y, su3y)
        b_z3 = dr.select(split_23, mid_z, sz3)
        # Edge flags for B
        b_e12 = se12 & ~split_13
        b_e13 = se13 & split_13
        b_e23 = se23
        # Barycentrics for B
        b_bu1 = dr.select(split_23, sbu1, mid_bu)
        b_bv1 = dr.select(split_23, sbv1, mid_bv)
        b_bu2 = sbu2; b_bv2 = sbv2
        b_bu3 = dr.select(split_23, mid_bu, sbu3)
        b_bv3 = dr.select(split_23, mid_bv, sbv3)

        # Concatenate children → next level's working set
        cur_u1x = dr.concat([a_u1x, b_u1x]); cur_u1y = dr.concat([a_u1y, b_u1y])
        cur_u2x = dr.concat([a_u2x, b_u2x]); cur_u2y = dr.concat([a_u2y, b_u2y])
        cur_u3x = dr.concat([a_u3x, b_u3x]); cur_u3y = dr.concat([a_u3y, b_u3y])
        cur_z1 = dr.concat([a_z1, b_z1]); cur_z2 = dr.concat([a_z2, b_z2]); cur_z3 = dr.concat([a_z3, b_z3])
        cur_e12 = dr.concat([a_e12, b_e12]); cur_e13 = dr.concat([a_e13, b_e13]); cur_e23 = dr.concat([a_e23, b_e23])
        cur_hit = dr.concat([s_hit, s_hit]); cur_face = dr.concat([s_face, s_face]); cur_pair = dr.concat([s_pair, s_pair])
        cur_bu1 = dr.concat([a_bu1, b_bu1]); cur_bv1 = dr.concat([a_bv1, b_bv1])
        cur_bu2 = dr.concat([a_bu2, b_bu2]); cur_bv2 = dr.concat([a_bv2, b_bv2])
        cur_bu3 = dr.concat([a_bu3, b_bu3]); cur_bv3 = dr.concat([a_bv3, b_bv3])

        # CRITICAL: force evaluation to prevent graph explosion
        dr.eval(cur_u1x, cur_u1y, cur_u2x, cur_u2y, cur_u3x, cur_u3y,
                cur_z1, cur_z2, cur_z3,
                cur_e12, cur_e13, cur_e23,
                cur_hit, cur_face, cur_pair,
                cur_bu1, cur_bv1, cur_bu2, cur_bv2, cur_bu3, cur_bv3)

    # Remaining at max depth → leaves
    if dr.width(cur_u1x) > 0:
        leaf_arrays.append({
            'u1x': cur_u1x, 'u1y': cur_u1y,
            'u2x': cur_u2x, 'u2y': cur_u2y,
            'u3x': cur_u3x, 'u3y': cur_u3y,
            'z1': cur_z1, 'z2': cur_z2, 'z3': cur_z3,
            'e12': cur_e12, 'e13': cur_e13, 'e23': cur_e23,
            'hit': cur_hit, 'face': cur_face, 'pair': cur_pair,
            'bu1': cur_bu1, 'bv1': cur_bv1,
            'bu2': cur_bu2, 'bv2': cur_bv2,
            'bu3': cur_bu3, 'bv3': cur_bv3,
        })

    if not leaf_arrays:
        return None

    # Concatenate all leaf sub-triangles
    u1x = dr.concat([la['u1x'] for la in leaf_arrays])
    u1y = dr.concat([la['u1y'] for la in leaf_arrays])
    u2x = dr.concat([la['u2x'] for la in leaf_arrays])
    u2y = dr.concat([la['u2y'] for la in leaf_arrays])
    u3x = dr.concat([la['u3x'] for la in leaf_arrays])
    u3y = dr.concat([la['u3y'] for la in leaf_arrays])
    z1 = dr.concat([la['z1'] for la in leaf_arrays])
    z2 = dr.concat([la['z2'] for la in leaf_arrays])
    z3 = dr.concat([la['z3'] for la in leaf_arrays])
    e12 = dr.concat([la['e12'] for la in leaf_arrays])
    e13 = dr.concat([la['e13'] for la in leaf_arrays])
    e23 = dr.concat([la['e23'] for la in leaf_arrays])
    hit_idx_f = dr.concat([la['hit'] for la in leaf_arrays])
    face_idx_f = dr.concat([la['face'] for la in leaf_arrays])
    leaf_pair_idx = dr.concat([la['pair'] for la in leaf_arrays])
    leaf_bu1 = dr.concat([la['bu1'] for la in leaf_arrays])
    leaf_bv1 = dr.concat([la['bv1'] for la in leaf_arrays])
    leaf_bu2 = dr.concat([la['bu2'] for la in leaf_arrays])
    leaf_bv2 = dr.concat([la['bv2'] for la in leaf_arrays])
    leaf_bu3 = dr.concat([la['bu3'] for la in leaf_arrays])
    leaf_bv3 = dr.concat([la['bv3'] for la in leaf_arrays])

    dr.eval(u1x, u1y, u2x, u2y, u3x, u3y, z1, z2, z3, hit_idx_f, face_idx_f, leaf_pair_idx)

    M_f = dr.width(u1x)

    # Reindex material data for leaf sub-triangles
    opacity_f = dr.gather(mi.Float, opacity_f, leaf_pair_idx)
    jr_f = dr.gather(mi.Float, jr_f, leaf_pair_idx)
    ji_f = dr.gather(mi.Float, ji_f, leaf_pair_idx)
    jpow_f = dr.gather(mi.Float, jpow_f, leaf_pair_idx)
    ts_f = dr.gather(mi.Float, ts_f, leaf_pair_idx)
    tp_f = dr.gather(mi.Float, tp_f, leaf_pair_idx)
    rs_f = dr.gather(mi.Float, rs_f, leaf_pair_idx)
    rp_f = dr.gather(mi.Float, rp_f, leaf_pair_idx)
    cos_theta_f = dr.gather(mi.Float, cos_theta_f, leaf_pair_idx)

    if M_f == 0:
        return None

    # Step 9: Beam amplitudes
    inv_4s2 = mi.Float(-0.25 / (beam_sigma ** 2))
    norm_phi = mi.Float(1.0 / (np.sqrt(2.0 * np.pi) * beam_sigma))

    r2_1 = u1x * u1x + u1y * u1y + z1 * z1
    r2_2 = u2x * u2x + u2y * u2y + z2 * z2
    r2_3 = u3x * u3x + u3y * u3y + z3 * z3

    ph1 = dr.exp(inv_4s2 * r2_1) * norm_phi
    ph2 = dr.exp(inv_4s2 * r2_2) * norm_phi
    ph3 = dr.exp(inv_4s2 * r2_3) * norm_phi

    # Step 10: Power integrals per leaf sub-triangle
    area_2x_f = dr.abs((u2x - u1x) * (u3y - u1y) - (u3x - u1x) * (u2y - u1y))

    Pt_f = area_2x_f * (
        ph1 * ph1 + ph2 * ph2 + ph3 * ph3 + ph1 * ph2 + ph1 * ph3 + ph2 * ph3
    ) / mi.Float(12.0)

    Psi0t_f = (ph1 + ph2 + ph3) * area_2x_f / mi.Float(6.0)

    Sigmat_xx_f, Sigmat_xy_f, Sigmat_yy_f = _batch_Sigmat_drjit(
        u1x, u1y, u2x, u2y, u3x, u3y, ph1, ph2, ph3
    )

    # Effective opacity
    eff_opacity_f = jpow_f

    sqrt_eff_f = dr.sqrt(dr.maximum(eff_opacity_f, mi.Float(1e-20)))

    # Step 11: Edge expansion — 3 edges per leaf → filter valid
    L = M_f  # number of leaf sub-triangles

    # Expand to 3×L potential edges
    parent_idx = dr.repeat(dr.arange(mi.UInt32, L), 3)  # [3L]: 0,0,0,1,1,1,...
    edge_type = dr.tile(mi.UInt32([0, 1, 2]), L)         # [3L]: 0,1,2,0,1,2,...

    # Gather diffracting flags per edge type
    e12_exp = dr.gather(mi.Bool, e12, parent_idx)
    e13_exp = dr.gather(mi.Bool, e13, parent_idx)
    e23_exp = dr.gather(mi.Bool, e23, parent_idx)
    is_type0 = edge_type == mi.UInt32(0)
    is_type1 = edge_type == mi.UInt32(1)
    e_valid = dr.select(is_type0, e12_exp, dr.select(is_type1, e13_exp, e23_exp))

    valid_sel_e = dr.compress(e_valid)
    E = dr.width(valid_sel_e)
    if E == 0:
        return None

    par = dr.gather(mi.UInt32, parent_idx, valid_sel_e)
    etype = dr.gather(mi.UInt32, edge_type, valid_sel_e)
    is_t0 = etype == mi.UInt32(0)
    is_t1 = etype == mi.UInt32(1)

    # Edge endpoint screen coordinates by edge type:
    # type0 (e12): a=u2, b=u1
    # type1 (e13): a=u1, b=u3
    # type2 (e23): a=u3, b=u2
    u1x_p = dr.gather(mi.Float, u1x, par); u1y_p = dr.gather(mi.Float, u1y, par)
    u2x_p = dr.gather(mi.Float, u2x, par); u2y_p = dr.gather(mi.Float, u2y, par)
    u3x_p = dr.gather(mi.Float, u3x, par); u3y_p = dr.gather(mi.Float, u3y, par)
    z1_p = dr.gather(mi.Float, z1, par); z2_p = dr.gather(mi.Float, z2, par); z3_p = dr.gather(mi.Float, z3, par)
    ph1_p = dr.gather(mi.Float, ph1, par); ph2_p = dr.gather(mi.Float, ph2, par); ph3_p = dr.gather(mi.Float, ph3, par)

    uax = dr.select(is_t0, u2x_p, dr.select(is_t1, u1x_p, u3x_p))
    uay = dr.select(is_t0, u2y_p, dr.select(is_t1, u1y_p, u3y_p))
    ubx = dr.select(is_t0, u1x_p, dr.select(is_t1, u3x_p, u2x_p))
    uby = dr.select(is_t0, u1y_p, dr.select(is_t1, u3y_p, u2y_p))
    za_e = dr.select(is_t0, z2_p, dr.select(is_t1, z1_p, z3_p))
    zb_e = dr.select(is_t0, z1_p, dr.select(is_t1, z3_p, z2_p))
    pha_e = dr.select(is_t0, ph2_p, dr.select(is_t1, ph1_p, ph3_p))
    phb_e = dr.select(is_t0, ph1_p, dr.select(is_t1, ph3_p, ph2_p))

    # Barycentrics of endpoints
    bu1_p = dr.gather(mi.Float, leaf_bu1, par); bv1_p = dr.gather(mi.Float, leaf_bv1, par)
    bu2_p = dr.gather(mi.Float, leaf_bu2, par); bv2_p = dr.gather(mi.Float, leaf_bv2, par)
    bu3_p = dr.gather(mi.Float, leaf_bu3, par); bv3_p = dr.gather(mi.Float, leaf_bv3, par)

    ba_u_e = dr.select(is_t0, bu2_p, dr.select(is_t1, bu1_p, bu3_p))
    ba_v_e = dr.select(is_t0, bv2_p, dr.select(is_t1, bv1_p, bv3_p))
    bb_u_e = dr.select(is_t0, bu1_p, dr.select(is_t1, bu3_p, bu2_p))
    bb_v_e = dr.select(is_t0, bv1_p, dr.select(is_t1, bv3_p, bv2_p))

    # Edge vector and midpoint
    ex_e = ubx - uax
    ey_e = uby - uay
    vmx_e = (uax + ubx) * mi.Float(0.5)
    vmy_e = (uay + uby) * mi.Float(0.5)

    # Triangle centroid for winding check
    cx_e = (u1x_p + u2x_p + u3x_p) / mi.Float(3.0)
    cy_e = (u1y_p + u2y_p + u3y_p) / mi.Float(3.0)

    # Winding correction
    mx = ey_e; my = -ex_e
    winding_dot = mx * (vmx_e - cx_e) + my * (vmy_e - cy_e)
    flip = winding_dot < mi.Float(0.0)

    ex_e = dr.select(flip, -ex_e, ex_e)
    ey_e = dr.select(flip, -ey_e, ey_e)
    pha_e_w = dr.select(flip, phb_e, pha_e)
    phb_e_w = dr.select(flip, pha_e, phb_e)
    za_e_w = dr.select(flip, zb_e, za_e)
    zb_e_w = dr.select(flip, za_e, zb_e)
    ba_u_e_w = dr.select(flip, bb_u_e, ba_u_e)
    ba_v_e_w = dr.select(flip, bb_v_e, ba_v_e)
    bb_u_e_w = dr.select(flip, ba_u_e, bb_u_e)
    bb_v_e_w = dr.select(flip, ba_v_e, bb_v_e)

    # Complex amplitudes: ca = phi_a * exp(i*k*z_a) * material_mod
    # DrJit has no complex type — split into real/imag
    k_f = mi.Float(k)
    phase_a_r = dr.cos(k_f * za_e_w); phase_a_i = dr.sin(k_f * za_e_w)
    phase_b_r = dr.cos(k_f * zb_e_w); phase_b_i = dr.sin(k_f * zb_e_w)
    ca_r = pha_e_w * phase_a_r; ca_i = pha_e_w * phase_a_i
    cb_r = phb_e_w * phase_b_r; cb_i = phb_e_w * phase_b_i

    # Material modulation
    edge_opacity_e = dr.gather(mi.Float, opacity_f, par)
    if jones_mode and use_material_opacity:
        ej_r = dr.gather(mi.Float, jr_f, par)
        ej_i = dr.gather(mi.Float, ji_f, par)
        # Complex multiply: ca * E_jones
        ca_r2 = ca_r * ej_r - ca_i * ej_i
        ca_i2 = ca_r * ej_i + ca_i * ej_r
        cb_r2 = cb_r * ej_r - cb_i * ej_i
        cb_i2 = cb_r * ej_i + cb_i * ej_r
        ca_r = ca_r2; ca_i = ca_i2; cb_r = cb_r2; cb_i = cb_i2
    elif use_material_opacity:
        sqrt_op = dr.sqrt(dr.maximum(edge_opacity_e, mi.Float(1e-20)))
        ca_r = ca_r * sqrt_op; ca_i = ca_i * sqrt_op
        cb_r = cb_r * sqrt_op; cb_i = cb_i * sqrt_op

    # Pjhat: edge-diffracted power
    e_len_sq = ex_e * ex_e + ey_e * ey_e
    # |ca - cb|² = (ca_r-cb_r)² + (ca_i-cb_i)²
    diff_r = ca_r - cb_r; diff_i = ca_i - cb_i
    abs_diff_sq = diff_r * diff_r + diff_i * diff_i
    # |ca + cb|² = (ca_r+cb_r)² + (ca_i+cb_i)²
    sum_r = ca_r + cb_r; sum_i = ca_i + cb_i
    abs_sum_sq = sum_r * sum_r + sum_i * sum_i

    Pjhat = e_len_sq * (
        abs_diff_sq * mi.Float(_INTEGRAL_1) +
        abs_sum_sq * mi.Float(0.25 * _INTEGRAL_2)
    )

    # Filter zero-power edges
    power_valid = Pjhat > mi.Float(0.0)
    pv_sel = dr.compress(power_valid)
    E = dr.width(pv_sel)
    if E == 0:
        return None

    par = _gather_sel_uint(par, pv_sel)
    ex_e = _gather_sel(ex_e, pv_sel); ey_e = _gather_sel(ey_e, pv_sel)
    vmx_e = _gather_sel(vmx_e, pv_sel); vmy_e = _gather_sel(vmy_e, pv_sel)
    ca_r = _gather_sel(ca_r, pv_sel); ca_i = _gather_sel(ca_i, pv_sel)
    cb_r = _gather_sel(cb_r, pv_sel); cb_i = _gather_sel(cb_i, pv_sel)
    Pjhat = _gather_sel(Pjhat, pv_sel)
    e_len_sq = _gather_sel(e_len_sq, pv_sel)
    edge_opacity_e = _gather_sel(edge_opacity_e, pv_sel)
    ba_u_e_w = _gather_sel(ba_u_e_w, pv_sel); ba_v_e_w = _gather_sel(ba_v_e_w, pv_sel)
    bb_u_e_w = _gather_sel(bb_u_e_w, pv_sel); bb_v_e_w = _gather_sel(bb_v_e_w, pv_sel)

    # Step 12: Per-hit power integral reduction
    edge_hit_idx = dr.gather(mi.UInt32, hit_idx_f, par)

    # P_A per hit = sum(eff_opacity * Pt)
    PA_contrib = eff_opacity_f * Pt_f
    P_A_per_hit = dr.zeros(mi.Float, N)
    dr.scatter_reduce(dr.ReduceOp.Add, P_A_per_hit, PA_contrib, hit_idx_f)

    # P_A_geom per hit = sum(opacity * Pt)
    PA_geom_contrib = opacity_f * Pt_f
    P_A_geom_per_hit = dr.zeros(mi.Float, N)
    dr.scatter_reduce(dr.ReduceOp.Add, P_A_geom_per_hit, PA_geom_contrib, hit_idx_f)

    # P_A_bare per hit = sum(Pt)
    P_A_bare_per_hit = dr.zeros(mi.Float, N)
    dr.scatter_reduce(dr.ReduceOp.Add, P_A_bare_per_hit, Pt_f, hit_idx_f)

    # psi_0 per hit = sum(sqrt_eff * Psi0t)
    psi0_contrib = sqrt_eff_f * Psi0t_f
    psi_0_per_hit = dr.zeros(mi.Float, N)
    dr.scatter_reduce(dr.ReduceOp.Add, psi_0_per_hit, psi0_contrib, hit_idx_f)

    # psi_0_bare per hit = sum(Psi0t)
    psi_0_bare_per_hit = dr.zeros(mi.Float, N)
    dr.scatter_reduce(dr.ReduceOp.Add, psi_0_bare_per_hit, Psi0t_f, hit_idx_f)

    # Sigmat per hit
    Sigmat_xx_ph = dr.zeros(mi.Float, N); Sigmat_xy_ph = dr.zeros(mi.Float, N); Sigmat_yy_ph = dr.zeros(mi.Float, N)
    dr.scatter_reduce(dr.ReduceOp.Add, Sigmat_xx_ph, sqrt_eff_f * Sigmat_xx_f, hit_idx_f)
    dr.scatter_reduce(dr.ReduceOp.Add, Sigmat_xy_ph, sqrt_eff_f * Sigmat_xy_f, hit_idx_f)
    dr.scatter_reduce(dr.ReduceOp.Add, Sigmat_yy_ph, sqrt_eff_f * Sigmat_yy_f, hit_idx_f)

    # Sigmat_bare per hit
    Sigmat_xx_bare_ph = dr.zeros(mi.Float, N); Sigmat_xy_bare_ph = dr.zeros(mi.Float, N); Sigmat_yy_bare_ph = dr.zeros(mi.Float, N)
    dr.scatter_reduce(dr.ReduceOp.Add, Sigmat_xx_bare_ph, Sigmat_xx_f, hit_idx_f)
    dr.scatter_reduce(dr.ReduceOp.Add, Sigmat_xy_bare_ph, Sigmat_xy_f, hit_idx_f)
    dr.scatter_reduce(dr.ReduceOp.Add, Sigmat_yy_bare_ph, Sigmat_yy_f, hit_idx_f)

    # sum_Phat per hit (from edges)
    sum_Phat_per_hit = dr.zeros(mi.Float, N)
    dr.scatter_reduce(dr.ReduceOp.Add, sum_Phat_per_hit, Pjhat, edge_hit_idx)

    # e_avg: power-weighted average edge length
    e_len = dr.sqrt(dr.maximum(e_len_sq, mi.Float(1e-20)))
    e_avg_num = dr.zeros(mi.Float, N)
    dr.scatter_reduce(dr.ReduceOp.Add, e_avg_num, Pjhat * e_len, edge_hit_idx)

    # Edge count per hit
    edge_count_per_hit = dr.zeros(mi.UInt32, N)
    dr.scatter_reduce(dr.ReduceOp.Add, edge_count_per_hit, mi.UInt32(1), edge_hit_idx)

    # CRITICAL: eval all scatter-reduced arrays before reading
    dr.eval(P_A_per_hit, P_A_geom_per_hit, P_A_bare_per_hit,
            psi_0_per_hit, psi_0_bare_per_hit,
            Sigmat_xx_ph, Sigmat_xy_ph, Sigmat_yy_ph,
            Sigmat_xx_bare_ph, Sigmat_xy_bare_ph, Sigmat_yy_bare_ph,
            sum_Phat_per_hit, e_avg_num, edge_count_per_hit)

    # Step 13: Per-hit finalization (P_central, P_A_hat, beta)
    has_edges = edge_count_per_hit > mi.UInt32(0)
    above_threshold = P_A_geom_per_hit >= mi.Float(1e-2)
    active_mask = has_edges & above_threshold
    active_hits_sel = dr.compress(active_mask)
    A = dr.width(active_hits_sel)
    if A == 0:
        return None

    # Gather per-aperture data
    sum_Pj = dr.gather(mi.Float, sum_Phat_per_hit, active_hits_sel)
    e_avg_n = dr.gather(mi.Float, e_avg_num, active_hits_sel)
    e_avg_a = dr.select(sum_Pj > mi.Float(0.0), e_avg_n / sum_Pj, mi.Float(0.01))

    k_sq = mi.Float(k * k)
    sigma_xi = mi.Float(np.sqrt(3.0)) / (mi.Float(k) * dr.maximum(e_avg_a, mi.Float(1e-6)))
    inv_sigma_xi_sq = mi.Float(1.0) / (sigma_xi * sigma_xi)

    psi0_a = dr.gather(mi.Float, psi_0_per_hit, active_hits_sel)
    Sxx_a = dr.gather(mi.Float, Sigmat_xx_ph, active_hits_sel)
    Sxy_a = dr.gather(mi.Float, Sigmat_xy_ph, active_hits_sel)
    Syy_a = dr.gather(mi.Float, Sigmat_yy_ph, active_hits_sel)

    scale_f = dr.select(dr.abs(psi0_a) > mi.Float(1e-12),
                        mi.Float(6.0) * k_sq / psi0_a, mi.Float(0.0))
    S0_xx = scale_f * Sxx_a + inv_sigma_xi_sq
    S0_xy = scale_f * Sxy_a
    S0_yy = scale_f * Syy_a + inv_sigma_xi_sq

    det_S0 = dr.maximum(mi.Float(0.0), S0_xx * S0_yy - S0_xy * S0_xy)

    P_central = dr.select(det_S0 > mi.Float(0.0),
                          k_sq / mi.Float(18.0 * np.pi) / dr.sqrt(dr.maximum(det_S0, mi.Float(1e-30))) * psi0_a * psi0_a,
                          mi.Float(0.0))

    P_A_hat = dr.maximum(mi.Float(0.0),
                         dr.gather(mi.Float, P_A_per_hit, active_hits_sel) - P_central)

    # Bare P_central and P_A_hat
    psi0_bare_a = dr.gather(mi.Float, psi_0_bare_per_hit, active_hits_sel)
    Sxx_bare_a = dr.gather(mi.Float, Sigmat_xx_bare_ph, active_hits_sel)
    Sxy_bare_a = dr.gather(mi.Float, Sigmat_xy_bare_ph, active_hits_sel)
    Syy_bare_a = dr.gather(mi.Float, Sigmat_yy_bare_ph, active_hits_sel)

    scale_f_bare = dr.select(dr.abs(psi0_bare_a) > mi.Float(1e-12),
                             mi.Float(6.0) * k_sq / psi0_bare_a, mi.Float(0.0))
    S0_xx_bare = scale_f_bare * Sxx_bare_a + inv_sigma_xi_sq
    S0_xy_bare = scale_f_bare * Sxy_bare_a
    S0_yy_bare = scale_f_bare * Syy_bare_a + inv_sigma_xi_sq
    det_S0_bare = dr.maximum(mi.Float(0.0), S0_xx_bare * S0_yy_bare - S0_xy_bare * S0_xy_bare)

    P_central_bare = dr.select(det_S0_bare > mi.Float(0.0),
                               k_sq / mi.Float(18.0 * np.pi) / dr.sqrt(dr.maximum(det_S0_bare, mi.Float(1e-30))) * psi0_bare_a * psi0_bare_a,
                               mi.Float(0.0))
    P_A_hat_bare = dr.maximum(mi.Float(0.0),
                              dr.gather(mi.Float, P_A_bare_per_hit, active_hits_sel) - P_central_bare)
    P_A_bare_a = dr.gather(mi.Float, P_A_bare_per_hit, active_hits_sel)

    # Re-filter: P_A_hat_bare > 0
    has_diff = P_A_hat_bare > mi.Float(0.0)
    diff_sel = dr.compress(has_diff)
    A = dr.width(diff_sel)
    if A == 0:
        return None

    active_hits_sel = dr.gather(mi.UInt32, active_hits_sel, diff_sel)
    P_A_hat = _gather_sel(P_A_hat, diff_sel)
    P_A_hat_bare = _gather_sel(P_A_hat_bare, diff_sel)
    P_A_bare_a = _gather_sel(P_A_bare_a, diff_sel)

    # Beta (energy borrowing)
    beta = dr.minimum(P_A_hat_bare, mi.Float(beta_max))

    # Step 14: CSR packing
    # Map hit index → aperture index (-1 if inactive)
    hit_to_ap = dr.full(mi.Int32, -1, N)
    ap_indices = mi.Int32(dr.arange(mi.UInt32, A))
    dr.scatter(hit_to_ap, ap_indices, active_hits_sel)
    dr.eval(hit_to_ap)

    # Map edges to aperture indices
    edge_ap_idx = dr.gather(mi.Int32, hit_to_ap, edge_hit_idx)
    edge_active = edge_ap_idx >= mi.Int32(0)
    ea_sel = dr.compress(edge_active)
    E_final = dr.width(ea_sel)
    if E_final == 0:
        return None

    # Gather all edge arrays for active edges
    par_a = _gather_sel_uint(par, ea_sel)
    ex_e_a = _gather_sel(ex_e, ea_sel); ey_e_a = _gather_sel(ey_e, ea_sel)
    vmx_e_a = _gather_sel(vmx_e, ea_sel); vmy_e_a = _gather_sel(vmy_e, ea_sel)
    ca_r_a = _gather_sel(ca_r, ea_sel); ca_i_a = _gather_sel(ca_i, ea_sel)
    cb_r_a = _gather_sel(cb_r, ea_sel); cb_i_a = _gather_sel(cb_i, ea_sel)
    _gather_sel(Pjhat, ea_sel)
    edge_opacity_a = _gather_sel(edge_opacity_e, ea_sel)
    ba_u_a = _gather_sel(ba_u_e_w, ea_sel); ba_v_a = _gather_sel(ba_v_e_w, ea_sel)
    bb_u_a = _gather_sel(bb_u_e_w, ea_sel); bb_v_a = _gather_sel(bb_v_e_w, ea_sel)
    edge_ap_a = dr.gather(mi.Int32, edge_ap_idx, ea_sel)
    edge_ap_u = dr.reinterpret_array(mi.UInt32, edge_ap_a)

    # CSR: count edges per aperture
    ap_edge_count_g = dr.zeros(mi.UInt32, A)
    dr.scatter_reduce(dr.ReduceOp.Add, ap_edge_count_g, mi.UInt32(1), edge_ap_u)
    dr.eval(ap_edge_count_g)

    # Enforce max_edges_per_hit: clamp counts
    ap_edge_count_clamped = dr.minimum(ap_edge_count_g, mi.UInt32(max_edges_per_hit))

    # Exclusive prefix sum for CSR start offsets
    # dr.prefix_sum already computes exclusive prefix sum
    ap_edge_start_g = dr.prefix_sum(ap_edge_count_clamped)
    dr.eval(ap_edge_start_g)

    # Place edges at CSR positions via scatter_inc
    ap_counter = dr.zeros(mi.UInt32, A)
    local_rank = dr.scatter_inc(ap_counter, edge_ap_u)
    dr.eval(local_rank)

    # Only keep edges within max_edges_per_hit
    max_for_edge = dr.gather(mi.UInt32, ap_edge_count_clamped, edge_ap_u)
    keep_edge = local_rank < max_for_edge
    ke_sel = dr.compress(keep_edge)
    E_final = dr.width(ke_sel)
    if E_final == 0:
        return None

    # Re-gather after truncation
    par_a = _gather_sel_uint(par_a, ke_sel)
    ex_e_a = _gather_sel(ex_e_a, ke_sel); ey_e_a = _gather_sel(ey_e_a, ke_sel)
    vmx_e_a = _gather_sel(vmx_e_a, ke_sel); vmy_e_a = _gather_sel(vmy_e_a, ke_sel)
    ca_r_a = _gather_sel(ca_r_a, ke_sel); ca_i_a = _gather_sel(ca_i_a, ke_sel)
    cb_r_a = _gather_sel(cb_r_a, ke_sel); cb_i_a = _gather_sel(cb_i_a, ke_sel)
    edge_opacity_a = _gather_sel(edge_opacity_a, ke_sel)
    ba_u_a = _gather_sel(ba_u_a, ke_sel); ba_v_a = _gather_sel(ba_v_a, ke_sel)
    bb_u_a = _gather_sel(bb_u_a, ke_sel); bb_v_a = _gather_sel(bb_v_a, ke_sel)
    edge_ap_u2 = _gather_sel_uint(edge_ap_u, ke_sel)
    local_rank2 = _gather_sel_uint(local_rank, ke_sel)

    # Compute destination slot in CSR array
    dest_slot = dr.gather(mi.UInt32, ap_edge_start_g, edge_ap_u2) + local_rank2

    # Create sorted edge arrays via scatter
    sorted_ex = dr.zeros(mi.Float, E_final)
    sorted_ey = dr.zeros(mi.Float, E_final)
    sorted_vmx = dr.zeros(mi.Float, E_final)
    sorted_vmy = dr.zeros(mi.Float, E_final)
    sorted_ca_r = dr.zeros(mi.Float, E_final)
    sorted_ca_i = dr.zeros(mi.Float, E_final)
    sorted_cb_r = dr.zeros(mi.Float, E_final)
    sorted_cb_i = dr.zeros(mi.Float, E_final)
    sorted_opacity = dr.zeros(mi.Float, E_final)
    sorted_ba_u = dr.zeros(mi.Float, E_final)
    sorted_ba_v = dr.zeros(mi.Float, E_final)
    sorted_bb_u = dr.zeros(mi.Float, E_final)
    sorted_bb_v = dr.zeros(mi.Float, E_final)
    sorted_par = dr.zeros(mi.UInt32, E_final)

    dr.scatter(sorted_ex, ex_e_a, dest_slot)
    dr.scatter(sorted_ey, ey_e_a, dest_slot)
    dr.scatter(sorted_vmx, vmx_e_a, dest_slot)
    dr.scatter(sorted_vmy, vmy_e_a, dest_slot)
    dr.scatter(sorted_ca_r, ca_r_a, dest_slot)
    dr.scatter(sorted_ca_i, ca_i_a, dest_slot)
    dr.scatter(sorted_cb_r, cb_r_a, dest_slot)
    dr.scatter(sorted_cb_i, cb_i_a, dest_slot)
    dr.scatter(sorted_opacity, edge_opacity_a, dest_slot)
    dr.scatter(sorted_ba_u, ba_u_a, dest_slot)
    dr.scatter(sorted_ba_v, ba_v_a, dest_slot)
    dr.scatter(sorted_bb_u, bb_u_a, dest_slot)
    dr.scatter(sorted_bb_v, bb_v_a, dest_slot)
    dr.scatter(sorted_par, par_a, dest_slot)

    dr.eval(sorted_ex, sorted_ey, sorted_vmx, sorted_vmy,
            sorted_ca_r, sorted_ca_i, sorted_cb_r, sorted_cb_i,
            sorted_opacity, sorted_ba_u, sorted_ba_v, sorted_bb_u, sorted_bb_v,
            sorted_par)

    # Per-edge face/vertex data and material data
    sorted_face_idx = dr.gather(mi.UInt32, face_idx_f, sorted_par)
    sorted_cos_theta = dr.gather(mi.Float, cos_theta_f, sorted_par)
    sorted_jr = dr.gather(mi.Float, jr_f, sorted_par)
    sorted_ji = dr.gather(mi.Float, ji_f, sorted_par)
    sorted_ts = dr.gather(mi.Float, ts_f, sorted_par)
    sorted_tp = dr.gather(mi.Float, tp_f, sorted_par)
    sorted_rs = dr.gather(mi.Float, rs_f, sorted_par)
    sorted_rp = dr.gather(mi.Float, rp_f, sorted_par)

    # Vertex indices
    sorted_vi0 = dr.gather(mi.UInt32, gpu_hash.face_vi0, sorted_face_idx)
    sorted_vi1 = dr.gather(mi.UInt32, gpu_hash.face_vi1, sorted_face_idx)
    sorted_vi2 = dr.gather(mi.UInt32, gpu_hash.face_vi2, sorted_face_idx)

    # Per-aperture frame data
    ap_tang_x = dr.gather(mi.Float, tang_x, active_hits_sel)
    ap_tang_y = dr.gather(mi.Float, tang_y, active_hits_sel)
    ap_tang_z = dr.gather(mi.Float, tang_z, active_hits_sel)
    ap_btang_x = dr.gather(mi.Float, btang_x, active_hits_sel)
    ap_btang_y = dr.gather(mi.Float, btang_y, active_hits_sel)
    ap_btang_z = dr.gather(mi.Float, btang_z, active_hits_sel)
    ap_wo_x = dr.gather(mi.Float, wo_x, active_hits_sel)
    ap_wo_y = dr.gather(mi.Float, wo_y, active_hits_sel)
    ap_wo_z = dr.gather(mi.Float, wo_z, active_hits_sel)
    ap_hp_x = dr.gather(mi.Float, hx, active_hits_sel)
    ap_hp_y = dr.gather(mi.Float, hy, active_hits_sel)
    ap_hp_z = dr.gather(mi.Float, hz, active_hits_sel)

    # Edge midpoint barycentrics
    sorted_bary_u = (sorted_ba_u + sorted_bb_u) * mi.Float(0.5)
    sorted_bary_v = (sorted_ba_v + sorted_bb_v) * mi.Float(0.5)

    # Download to FlatEdgeData (GPU → CPU, single transfer)
    # Force all GPU arrays to be evaluated
    dr.eval(sorted_face_idx, sorted_cos_theta,
            sorted_jr, sorted_ji, sorted_ts, sorted_tp, sorted_rs, sorted_rp,
            sorted_vi0, sorted_vi1, sorted_vi2,
            ap_tang_x, ap_tang_y, ap_tang_z,
            ap_btang_x, ap_btang_y, ap_btang_z,
            ap_wo_x, ap_wo_y, ap_wo_z,
            ap_hp_x, ap_hp_y, ap_hp_z,
            sorted_bary_u, sorted_bary_v,
            P_A_hat, P_A_hat_bare, P_A_bare_a, beta,
            ap_edge_start_g, ap_edge_count_clamped)

    # Download numpy arrays
    ap_tangent_np = np.stack([np.array(ap_tang_x), np.array(ap_tang_y), np.array(ap_tang_z)], axis=1).astype(np.float32)
    ap_bitangent_np = np.stack([np.array(ap_btang_x), np.array(ap_btang_y), np.array(ap_btang_z)], axis=1).astype(np.float32)
    ap_wo_dir_np = np.stack([np.array(ap_wo_x), np.array(ap_wo_y), np.array(ap_wo_z)], axis=1).astype(np.float32)
    ap_hit_pos_np = np.stack([np.array(ap_hp_x), np.array(ap_hp_y), np.array(ap_hp_z)], axis=1).astype(np.float64)

    active_hits_np = np.array(active_hits_sel).astype(np.int32)

    if verbose:
        print(f"  [GPU Aperture] {A} apertures, {E_final} edges "
              f"(from {N} hits, all-GPU pipeline)")

    flat = FlatEdgeData(
        edge_ex=np.array(sorted_ex).astype(np.float64),
        edge_ey=np.array(sorted_ey).astype(np.float64),
        edge_vx=np.array(sorted_vmx).astype(np.float64),
        edge_vy=np.array(sorted_vmy).astype(np.float64),
        edge_a_real=np.array(sorted_ca_r).astype(np.float64),
        edge_a_imag=np.array(sorted_ca_i).astype(np.float64),
        edge_b_real=np.array(sorted_cb_r).astype(np.float64),
        edge_b_imag=np.array(sorted_cb_i).astype(np.float64),
        edge_opacity=np.array(sorted_opacity).astype(np.float64),
        edge_face_idx=np.array(sorted_face_idx).astype(np.int32),
        edge_cos_theta=np.array(sorted_cos_theta).astype(np.float64),
        edge_bary_u=np.array(sorted_bary_u).astype(np.float64),
        edge_bary_v=np.array(sorted_bary_v).astype(np.float64),
        edge_vert_idx_0=np.array(sorted_vi0).astype(np.int32),
        edge_vert_idx_1=np.array(sorted_vi1).astype(np.int32),
        edge_vert_idx_2=np.array(sorted_vi2).astype(np.int32),
        edge_jones_real=np.array(sorted_jr).astype(np.float64),
        edge_jones_imag=np.array(sorted_ji).astype(np.float64),
        edge_tx_amp_s=np.array(sorted_ts).astype(np.float64),
        edge_tx_amp_p=np.array(sorted_tp).astype(np.float64),
        edge_rx_amp_s=np.array(sorted_rs).astype(np.float64),
        edge_rx_amp_p=np.array(sorted_rp).astype(np.float64),
        edge_ba_u=np.array(sorted_ba_u).astype(np.float64),
        edge_ba_v=np.array(sorted_ba_v).astype(np.float64),
        edge_bb_u=np.array(sorted_bb_u).astype(np.float64),
        edge_bb_v=np.array(sorted_bb_v).astype(np.float64),
        ap_edge_start=np.array(ap_edge_start_g).astype(np.int32),
        ap_edge_count=np.array(ap_edge_count_clamped).astype(np.int32),
        ap_P_A_hat=np.array(P_A_hat).astype(np.float64),
        ap_P_A_hat_bare=np.array(P_A_hat_bare).astype(np.float64),
        ap_P_A_bare=np.array(P_A_bare_a).astype(np.float64),
        ap_tangent=ap_tangent_np,
        ap_bitangent=ap_bitangent_np,
        ap_wo_dir=ap_wo_dir_np,
        ap_beta=np.array(beta).astype(np.float64),
        ap_hit_pos=ap_hit_pos_np,
        ap_to_path_indices=[np.empty(0, dtype=np.int32)] * A,
        n_edges=E_final,
        n_apertures=A,
    )

    return (flat, active_hits_np)
