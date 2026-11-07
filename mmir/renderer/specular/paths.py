"""Specular reflection paths found by the SMS solver."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SpecularPaths:
    """Result of image method refinement."""
    hit_P: 'mi.Point3f'         # Exact specular reflection points
    hit_N: 'mi.Vector3f'        # Surface normals
    d_tx: 'mi.Float'            # Distance TX → P_spec
    d_rx: 'mi.Float'            # Distance P_spec → RX
    dir_to_tx: 'mi.Vector3f'    # Direction P_spec → TX
    dir_to_rx: 'mi.Vector3f'    # Direction P_spec → RX
    cos_theta_i: 'mi.Float'     # cos(angle of incidence) = dot(dir_to_tx, N)
    cos_theta_r: 'mi.Float'     # cos(angle of reflection) = dot(dir_to_rx, N)
    tx_idx: 'mi.UInt32'         # TX index
    rx_idx: 'mi.UInt32'         # RX index
    valid: 'mi.Bool'            # Validity mask
    R_specular: 'mi.Float'      # η × τ × A  (specular reflectance)
    A_tri: 'mi.Float'           # Triangle area
    patch_id: Optional['mi.Int32']  # Patch ID per path (None if not using patches)
    prim_ids: Optional['mi.UInt32']  # Triangle primitive IDs per path (for Phase B material gather)
    n_paths: int                # Total number of potentially valid specular paths
    # IFT gradient support (populated by SMS, not image method)
    grad_info: Optional[object] = None     # SpecularGradInfo from SMS Newton solver
    bary_u: Optional['mi.Float'] = None    # Barycentric u at converged point
    bary_v: Optional['mi.Float'] = None    # Barycentric v at converged point
