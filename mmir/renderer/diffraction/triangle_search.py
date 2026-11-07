"""Spatial hash grid for fast triangle sphere queries."""

import numpy as np
from typing import Optional
from collections import defaultdict


class TriangleSpatialHash:
    """3D spatial hash over triangle centroids for sphere-range queries."""

    def __init__(
        self,
        vertices: np.ndarray,       # [V, 3] float32
        faces: np.ndarray,          # [F, 3] int32
        face_normals: np.ndarray,   # [F, 3] float32
        cell_size: float,
        neighbor_face_normals: Optional[np.ndarray] = None,  # [F, 3, 3] normals of neighbor faces
        edge_angle_threshold_deg: float = 20.0,  # min dihedral angle for diffracting edges
        min_edge_length: float = 0.0,            # meters (0 = disabled)
        boundary_erosion_hops: int = 0,          # hops (0 = disabled)
        region_angle_threshold_deg: float = 0.0, # degrees (0 = disabled)
        region_dihedral_cap_deg: float = 45.0,   # degrees
        min_region_faces: int = 10,              # faces
        inter_region_min_faces: int = 0,         # faces (0 = disabled)
        min_chain_length: float = 0.0,           # meters (0 = disabled)
        chain_collinearity_threshold_deg: float = 0.0,  # degrees (0 = disabled)
    ):
        """Build spatial hash from mesh geometry."""
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        self.faces = np.ascontiguousarray(faces, dtype=np.int32)
        self.face_normals = np.ascontiguousarray(face_normals, dtype=np.float32)
        self.cell_size = float(cell_size)
        self.inv_cell_size = 1.0 / self.cell_size
        self.n_faces = len(faces)

        # Precompute triangle vertex positions [F, 3, 3]
        self.tri_v0 = vertices[faces[:, 0]]  # [F, 3]
        self.tri_v1 = vertices[faces[:, 1]]  # [F, 3]
        self.tri_v2 = vertices[faces[:, 2]]  # [F, 3]

        # Triangle centroids [F, 3]
        self.centroids = (self.tri_v0 + self.tri_v1 + self.tri_v2) / 3.0

        # Build edge-to-face adjacency (reused by neighbor normals and filters)
        self._edge_to_faces = self._build_edge_to_faces()

        # Build neighbor normals if not provided
        self.neighbor_normals = self._build_neighbor_normals()

        # Per-edge suppression filters (view-independent, computed once)
        self.edge_is_suppressed = np.zeros((self.n_faces, 3), dtype=bool)
        total_edge_slots = self.n_faces * 3
        filter_stats = {}

        # Filter E: Minimum edge length
        if min_edge_length > 0:
            mask_E = self._compute_short_edge_mask(min_edge_length)
            n_E = int(np.sum(mask_E & ~self.edge_is_suppressed))
            filter_stats['E_short_edge'] = n_E
            self.edge_is_suppressed |= mask_E

        # Filter C: Mesh boundary erosion
        if boundary_erosion_hops > 0:
            mask_C = self._compute_boundary_erosion_mask(boundary_erosion_hops)
            n_C = int(np.sum(mask_C & ~self.edge_is_suppressed))
            filter_stats['C_boundary_erosion'] = n_C
            self.edge_is_suppressed |= mask_C

        # Filter R: Region-based edge classification (replaces old Filter D)
        if region_angle_threshold_deg > 0:
            mask_R = self._compute_region_edge_mask(
                region_angle_threshold_deg, region_dihedral_cap_deg,
                min_region_faces, edge_angle_threshold_deg,
                inter_region_min_faces,
            )
            n_R = int(np.sum(mask_R & ~self.edge_is_suppressed))
            filter_stats['R_region'] = n_R
            self.edge_is_suppressed |= mask_R

        # Filter F: Connected chain length (depends on E/C/R + dihedral angle)
        if min_chain_length > 0:
            mask_F = self._compute_chain_length_mask(
                min_chain_length, edge_angle_threshold_deg,
                chain_collinearity_threshold_deg,
            )
            n_F = int(np.sum(mask_F & ~self.edge_is_suppressed))
            filter_stats['F_chain_length'] = n_F
            self.edge_is_suppressed |= mask_F

        n_suppressed = int(np.sum(self.edge_is_suppressed))
        if filter_stats:
            print(f"[EdgeFilters] {n_suppressed}/{total_edge_slots} edge-slots suppressed "
                  f"({100.0 * n_suppressed / max(total_edge_slots, 1):.1f}%)")
            for name, count in filter_stats.items():
                print(f"  Filter {name}: {count} new edge-slots")

        # Pre-compute per-face flag: does this face have potentially diffracting edges?
        self.has_potential_boundary = self._compute_boundary_flags(edge_angle_threshold_deg)

        # Build hash grid
        self._grid = defaultdict(list)
        self._build_grid()

    def _build_grid(self):
        """Insert all triangle centroids into the hash grid."""
        cx = self.centroids[:, 0]
        cy = self.centroids[:, 1]
        cz = self.centroids[:, 2]

        ix = np.floor(cx * self.inv_cell_size).astype(np.int32)
        iy = np.floor(cy * self.inv_cell_size).astype(np.int32)
        iz = np.floor(cz * self.inv_cell_size).astype(np.int32)

        for f_idx in range(self.n_faces):
            key = (int(ix[f_idx]), int(iy[f_idx]), int(iz[f_idx]))
            self._grid[key].append(f_idx)

        # Convert lists to numpy arrays for faster access
        for key in self._grid:
            self._grid[key] = np.array(self._grid[key], dtype=np.int32)

    def _build_edge_to_faces(self) -> dict:
        """Build edge-to-face adjacency: canonical edge (min_v, max_v) -> [(f_idx, j), ...]."""
        edge_to_faces = defaultdict(list)
        for f_idx in range(self.n_faces):
            face = self.faces[f_idx]
            for j in range(3):
                v0, v1 = int(face[j]), int(face[(j + 1) % 3])
                edge_key = (min(v0, v1), max(v0, v1))
                edge_to_faces[edge_key].append((f_idx, j))
        return edge_to_faces

    def _build_neighbor_normals(self) -> np.ndarray:
        """Build per-face neighbor normals [F, 3, 3] using stored edge-to-face adjacency."""
        neighbor_normals = np.zeros((self.n_faces, 3, 3), dtype=np.float32)
        for edge_key, face_edge_list in self._edge_to_faces.items():
            if len(face_edge_list) == 2:
                (f0, j0), (f1, j1) = face_edge_list
                neighbor_normals[f0, j0] = self.face_normals[f1]
                neighbor_normals[f1, j1] = self.face_normals[f0]
            # Boundary edges: neighbor normal stays [0,0,0]
        return neighbor_normals

    # Edge suppression filters (all produce [F, 3] bool masks)

    def _compute_short_edge_mask(self, min_edge_length: float) -> np.ndarray:
        """Filter E: suppress edges shorter than min_edge_length (meters)."""
        v = self.vertices
        f = self.faces
        # Edge j connects faces[f, j] -> faces[f, (j+1)%3]
        len0 = np.linalg.norm(v[f[:, 1]] - v[f[:, 0]], axis=1)  # edge 0
        len1 = np.linalg.norm(v[f[:, 2]] - v[f[:, 1]], axis=1)  # edge 1
        len2 = np.linalg.norm(v[f[:, 0]] - v[f[:, 2]], axis=1)  # edge 2
        return np.stack([len0 < min_edge_length,
                         len1 < min_edge_length,
                         len2 < min_edge_length], axis=1)  # [F, 3]

    def _compute_boundary_erosion_mask(self, n_hops: int) -> np.ndarray:
        """Filter C: suppress all edges on faces within N hops of mesh boundary."""
        # Step 1: identify boundary faces
        boundary_faces = set()
        for edge_key, fel in self._edge_to_faces.items():
            if len(fel) == 1:
                boundary_faces.add(fel[0][0])

        # Step 2: build face-to-face adjacency for BFS
        face_neighbors = defaultdict(set)
        for edge_key, fel in self._edge_to_faces.items():
            if len(fel) == 2:
                f0, f1 = fel[0][0], fel[1][0]
                face_neighbors[f0].add(f1)
                face_neighbors[f1].add(f0)

        # Step 3: BFS propagation
        eroded = set(boundary_faces)
        frontier = set(boundary_faces)
        for _ in range(n_hops):
            next_frontier = set()
            for f_idx in frontier:
                for nb in face_neighbors[f_idx]:
                    if nb not in eroded:
                        eroded.add(nb)
                        next_frontier.add(nb)
            frontier = next_frontier
            if not frontier:
                break

        # Step 4: convert to [F, 3] mask (all 3 edges of eroded faces)
        mask = np.zeros((self.n_faces, 3), dtype=bool)
        if eroded:
            eroded_arr = np.array(list(eroded), dtype=np.int32)
            mask[eroded_arr, :] = True
        return mask

    def _segment_planar_regions(
        self,
        angle_threshold_deg: float,
        min_region_faces: int,
    ) -> np.ndarray:
        """Segment mesh faces into approximately planar regions."""
        from collections import deque

        cos_thresh = np.cos(np.radians(angle_threshold_deg))
        fn = self.face_normals

        # Build face-to-face adjacency
        face_adj = defaultdict(list)
        for edge_key, fel in self._edge_to_faces.items():
            if len(fel) == 2:
                f0, f1 = fel[0][0], fel[1][0]
                face_adj[f0].append(f1)
                face_adj[f1].append(f0)

        # Phase 1: Flood-fill region growing
        region_id = np.full(self.n_faces, -1, dtype=np.int32)
        region_sizes = []
        rid = 0

        for seed in range(self.n_faces):
            if region_id[seed] >= 0:
                continue

            seed_normal = fn[seed].astype(np.float64)
            queue = deque([seed])
            region_id[seed] = rid
            count = 1

            while queue:
                cur = queue.popleft()
                for nb in face_adj[cur]:
                    if region_id[nb] >= 0:
                        continue
                    if np.dot(seed_normal, fn[nb]) < cos_thresh:
                        continue
                    region_id[nb] = rid
                    queue.append(nb)
                    count += 1

            region_sizes.append(count)
            rid += 1

        n_regions_phase1 = rid
        sizes = np.array(region_sizes, dtype=np.int64)

        # Phase 2: Merge small regions into largest neighbor (union-find)
        if min_region_faces > 1:
            # Build region adjacency
            region_adj = defaultdict(lambda: defaultdict(int))
            for edge_key, fel in self._edge_to_faces.items():
                if len(fel) != 2:
                    continue
                r0, r1 = region_id[fel[0][0]], region_id[fel[1][0]]
                if r0 != r1:
                    region_adj[r0][r1] += 1
                    region_adj[r1][r0] += 1

            # Union-find with path compression
            parent = np.arange(rid, dtype=np.int32)

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for r in range(rid):
                if sizes[r] >= min_region_faces:
                    continue
                # Find largest neighbor region (by root size)
                best_nb = -1
                best_size = 0
                for nb_r in region_adj[r]:
                    nb_root = find(nb_r)
                    if sizes[nb_root] > best_size:
                        best_size = sizes[nb_root]
                        best_nb = nb_root
                if best_nb >= 0:
                    r_root = find(r)
                    parent[r_root] = best_nb
                    sizes[best_nb] += sizes[r_root]

            # Apply merging
            for f in range(self.n_faces):
                region_id[f] = find(int(region_id[f]))

        unique_regions = np.unique(region_id)
        n_regions_final = len(unique_regions)

        # Region size stats for diagnostics
        final_sizes = np.array([np.sum(region_id == r) for r in unique_regions])
        n_large = int(np.sum(final_sizes >= 100))
        n_medium = int(np.sum((final_sizes >= 10) & (final_sizes < 100)))
        n_small = int(np.sum(final_sizes < 10))

        print(f"  Filter R regions: {n_regions_phase1} → {n_regions_final} "
              f"(after merging <{min_region_faces}), "
              f"large={n_large}, medium={n_medium}, small={n_small}, "
              f"largest={int(final_sizes.max())}")

        return region_id

    def _compute_region_edge_mask(
        self,
        region_angle_threshold_deg: float,
        region_dihedral_cap_deg: float,
        min_region_faces: int,
        edge_angle_threshold_deg: float,
        inter_region_min_faces: int = 0,
    ) -> np.ndarray:
        """Filter R: suppress edges that are intra-region noise."""
        region_id = self._segment_planar_regions(
            region_angle_threshold_deg, min_region_faces)
        self.region_id = region_id  # Store for downstream use

        # Precompute region sizes
        region_size = np.bincount(region_id.clip(min=0),
                                   minlength=int(region_id.max()) + 1)

        cos_dihedral_threshold = np.cos(np.radians(edge_angle_threshold_deg))
        cos_cap = np.cos(np.radians(region_dihedral_cap_deg))
        fn = self.face_normals
        mask = np.zeros((self.n_faces, 3), dtype=bool)

        n_intra_suppressed = 0
        n_tiny_suppressed = 0
        n_small_region_suppressed = 0

        for edge_key, fel in self._edge_to_faces.items():
            if len(fel) != 2:
                continue

            (f0, j0), (f1, j1) = fel

            # Skip edges already suppressed by earlier filters
            both_suppressed = (self.edge_is_suppressed[f0, j0] and
                               self.edge_is_suppressed[f1, j1])
            if both_suppressed:
                continue

            # Only consider edges with dihedral above the edge angle threshold
            cos_d = float(np.dot(fn[f0], fn[f1]))
            if cos_d >= cos_dihedral_threshold:
                continue

            r0, r1 = int(region_id[f0]), int(region_id[f1])

            if r0 == r1:
                # Intra-region: suppress if dihedral below safety cap
                # cos_d > cos_cap means dihedral < cap (smaller angle)
                if cos_d > cos_cap:
                    mask[f0, j0] = True
                    mask[f1, j1] = True
                    n_intra_suppressed += 1
            else:
                # Inter-region: suppress if BOTH regions are tiny
                s0, s1 = int(region_size[r0]), int(region_size[r1])
                if s0 < min_region_faces and s1 < min_region_faces:
                    mask[f0, j0] = True
                    mask[f1, j1] = True
                    n_tiny_suppressed += 1
                # Low-dihedral inter-region: suppress if smaller region is
                # below inter_region_min_faces (noise patch next to large surface).
                # Only applies below the safety cap (cos_d > cos_cap => dih < cap).
                elif inter_region_min_faces > 0 and cos_d > cos_cap:
                    smaller = min(s0, s1)
                    if smaller < inter_region_min_faces:
                        mask[f0, j0] = True
                        mask[f1, j1] = True
                        n_small_region_suppressed += 1

        print(f"  Filter R edges: {n_intra_suppressed} intra-region suppressed "
              f"(dih<{region_dihedral_cap_deg}°), "
              f"{n_tiny_suppressed} tiny-boundary, "
              f"{n_small_region_suppressed} small-region inter-region suppressed "
              f"(dih<{region_dihedral_cap_deg}°, region<{inter_region_min_faces})")

        return mask

    def _compute_chain_length_mask(
        self,
        min_chain_length: float,
        edge_angle_threshold_deg: float,
        collinearity_threshold_deg: float = 0.0,
    ) -> np.ndarray:
        """Filter F: suppress diffracting edges in short disconnected chains."""
        cos_angle_threshold = np.cos(np.radians(edge_angle_threshold_deg))
        collinearity_threshold_deg > 0

        v = self.vertices

        # Step 1: Identify candidate diffracting edges
        # An edge is a candidate if: 2 adjacent faces, dihedral > threshold,
        # and not fully suppressed by earlier filters.
        active_edges = {}  # edge_key -> (va, vb, length, edge_vec, dihedral_deg, face_edge_list)

        for edge_key, face_edge_list in self._edge_to_faces.items():
            if len(face_edge_list) != 2:
                continue

            # Skip if all slots already suppressed
            if all(self.edge_is_suppressed[f_idx, j] for f_idx, j in face_edge_list):
                continue

            # Check dihedral angle
            (f0, j0), (f1, j1) = face_edge_list
            cos_dihedral = float(np.dot(self.face_normals[f0], self.face_normals[f1]))
            if cos_dihedral >= cos_angle_threshold:
                continue

            dihedral_deg = float(np.degrees(np.arccos(np.clip(cos_dihedral, -1.0, 1.0))))
            va, vb = edge_key
            edge_vec = v[vb].astype(np.float64) - v[va].astype(np.float64)
            length = float(np.linalg.norm(edge_vec))
            active_edges[edge_key] = (va, vb, length, edge_vec, dihedral_deg, face_edge_list)

        if not active_edges:
            return np.zeros((self.n_faces, 3), dtype=bool)

        # Step 2: Build vertex-to-active-edge adjacency
        vertex_to_edges = defaultdict(set)
        for edge_key, (va, vb, _, _, _, _) in active_edges.items():
            vertex_to_edges[va].add(edge_key)
            vertex_to_edges[vb].add(edge_key)

        # Step 3: BFS connected components
        visited = set()
        components = []  # list of (set_of_edge_keys, total_length)

        for start_edge in active_edges:
            if start_edge in visited:
                continue

            component = set()
            total_length = 0.0
            queue = [start_edge]
            visited.add(start_edge)

            max_dihedral = 0.0

            while queue:
                current = queue.pop()
                component.add(current)
                va_c, vb_c, len_c, vec_c, dih_c, _ = active_edges[current]
                total_length += len_c
                if dih_c > max_dihedral:
                    max_dihedral = dih_c

                for vtx in (va_c, vb_c):
                    for neighbor_edge in vertex_to_edges[vtx]:
                        if neighbor_edge in visited:
                            continue

                        visited.add(neighbor_edge)
                        queue.append(neighbor_edge)

            components.append((component, total_length, max_dihedral))

        # Step 4: Suppress edges in short components
        # Exempt chains whose max dihedral > 90° — these are almost certainly
        # real geometric features (right angles, acute edges) even if short.
        HIGH_DIHEDRAL_EXEMPT_DEG = 90.0
        mask = np.zeros((self.n_faces, 3), dtype=bool)
        n_short_components = 0
        n_short_edges = 0
        n_exempt = 0

        for component, total_length, max_dihedral in components:
            if total_length < min_chain_length:
                if max_dihedral > HIGH_DIHEDRAL_EXEMPT_DEG:
                    n_exempt += 1
                    continue  # preserve high-dihedral short chain
                n_short_components += 1
                n_short_edges += len(component)
                for edge_key in component:
                    _, _, _, _, _, face_edge_list = active_edges[edge_key]
                    for f_idx, j in face_edge_list:
                        mask[f_idx, j] = True

        n_active = len(active_edges)
        n_components = len(components)
        n_long = n_components - n_short_components - n_exempt
        print(f"  Filter F chain details: {n_active} active diffracting edges → "
              f"{n_components} components ({n_long} long, {n_short_components} short, "
              f"{n_exempt} short-but-exempt[dih>{HIGH_DIHEDRAL_EXEMPT_DEG:.0f}°]), "
              f"{n_short_edges} edges suppressed (threshold={min_chain_length*1e3:.1f}mm)")

        return mask

    # Boundary flag pre-filter (uses suppression masks)

    def _compute_boundary_flags(self, edge_angle_threshold_deg: float = 20.0) -> np.ndarray:
        """Pre-compute per-face flag: does this face have potentially diffracting edges?"""
        nn = self.neighbor_normals  # [F, 3, 3]
        fn = self.face_normals      # [F, 3]

        nn_norms = np.linalg.norm(nn, axis=2)  # [F, 3]
        has_neighbor = nn_norms >= 0.5           # [F, 3]

        dots = np.einsum('ij,ikj->ik', fn, nn)  # [F, 3]
        cos_angle = dots / np.maximum(nn_norms, 1e-8)
        cos_threshold = np.cos(np.radians(edge_angle_threshold_deg))
        has_significant_edge = cos_angle < cos_threshold  # [F, 3]

        not_suppressed = ~self.edge_is_suppressed  # [F, 3]

        return np.any(has_neighbor & has_significant_edge & not_suppressed, axis=1)  # [F]


def build_triangle_hash_from_scene(
    scene,
    cell_size: float,
    edge_angle_threshold_deg: float = 20.0,
    wavelength: float = 0.0,
    min_edge_length_wavelengths: float = 0.0,
    boundary_erosion_hops: int = 0,
    region_angle_threshold_deg: float = 0.0,
    region_dihedral_cap_deg: float = 45.0,
    min_region_faces: int = 10,
    inter_region_min_faces: int = 0,
    min_chain_length_wavelengths: float = 0.0,
    chain_collinearity_threshold_deg: float = 0.0,
) -> TriangleSpatialHash:
    """Build a TriangleSpatialHash from a Mitsuba scene."""
    import mitsuba as mi

    # Get the mesh from the scene
    shapes = scene.shapes()
    mesh = None
    for shape in shapes:
        if hasattr(shape, 'vertex_count'):
            mesh = shape
            break

    if mesh is None:
        raise RuntimeError("No mesh found in scene")

    # Extract vertices
    mesh.vertex_count()
    face_count = mesh.face_count()

    # Get vertex positions
    params = mi.traverse(mesh)
    vertex_positions = np.array(params['vertex_positions']).reshape(-1, 3)

    # Get face indices
    face_indices = np.array(params['faces']).reshape(-1, 3).astype(np.int32)

    # Compute face normals
    v0 = vertex_positions[face_indices[:, 0]]
    v1 = vertex_positions[face_indices[:, 1]]
    v2 = vertex_positions[face_indices[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    normals = np.cross(e1, e2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-12)

    # Compute length thresholds in meters
    min_edge_length = min_edge_length_wavelengths * wavelength if wavelength > 0 else 0.0
    min_chain_length = min_chain_length_wavelengths * wavelength if wavelength > 0 else 0.0

    print(f"[TriangleSpatialHash] Building hash for {face_count} triangles, "
          f"cell_size={cell_size:.4f}m")

    tri_hash = TriangleSpatialHash(
        vertices=vertex_positions.astype(np.float32),
        faces=face_indices,
        face_normals=normals.astype(np.float32),
        cell_size=cell_size,
        edge_angle_threshold_deg=edge_angle_threshold_deg,
        min_edge_length=min_edge_length,
        boundary_erosion_hops=boundary_erosion_hops,
        region_angle_threshold_deg=region_angle_threshold_deg,
        region_dihedral_cap_deg=region_dihedral_cap_deg,
        min_region_faces=min_region_faces,
        inter_region_min_faces=inter_region_min_faces,
        min_chain_length=min_chain_length,
        chain_collinearity_threshold_deg=chain_collinearity_threshold_deg,
    )

    n_cells = len(tri_hash._grid)
    avg_per_cell = face_count / max(n_cells, 1)
    n_boundary = int(np.sum(tri_hash.has_potential_boundary))
    n_suppressed = int(np.sum(tri_hash.edge_is_suppressed))
    print(f"[TriangleSpatialHash] {n_cells} occupied cells, "
          f"{avg_per_cell:.1f} triangles/cell avg, "
          f"{n_boundary}/{face_count} faces with potential boundary edges, "
          f"{n_suppressed}/{face_count * 3} edge-slots suppressed")

    return tri_hash
