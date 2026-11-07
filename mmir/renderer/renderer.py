"""RRTS Reference Renderer - Main orchestrator class."""

from typing import Optional
import drjit as dr
import mitsuba as mi
import numpy as np

# Scene context and logging (inlined)
from .scene_context import SceneContext, RenderLogger

from .config import RenderConfigRef, RenderResultRef
from .sampler import ReservoirSampler
from .integrator import SBRIntegratorRef
from .specular.hashing import SpecularDeduplicator
from .specular.sms import SpecularManifoldSampler
from .materials.parameterization import MaterialParameterization

# Import diffraction modules (optional)
try:
    from .diffraction import FsdBSDF
    from .diffraction.triangle_search import build_triangle_hash_from_scene
    _HAS_DIFFRACTION = True
except ImportError:
    _HAS_DIFFRACTION = False

# Import RRTS-style pattern loading and GPU-native pattern evaluation
try:
    from mmir.sensor.element_patterns import load_pattern_rrts_format, AntennaPatternLoader
except ImportError:
    load_pattern_rrts_format = None
    AntennaPatternLoader = None


class FMCWRendererRef:
    """RRTS Reference Renderer."""

    def __init__(
        self,
        scene_ctx: SceneContext,
        config: Optional[RenderConfigRef] = None,
        verbose: bool = True,
        tx_pattern_file: Optional[str] = None,
        rx_pattern_file: Optional[str] = None
    ):
        """Initialize RRTS reference renderer."""
        self.scene_ctx = scene_ctx
        self.config = config or RenderConfigRef()
        self.verbose = verbose
        self.logger = RenderLogger(verbose)

        # Store pattern file paths for RRTS-style loading
        self.tx_pattern_file = tx_pattern_file
        self.rx_pattern_file = rx_pattern_file

        # Optional per-triangle materials for iterative optimization
        # 3 columns: (albedo, roughness, metallic) - legacy mode
        # 6 columns: (eps_real, eps_imag, sigma_h, l_c, tau, thickness) - physics mode
        # If None, sampler uses default materials
        self.triangle_materials: Optional[np.ndarray] = None
        # GPU-resident copy: list of mi.Float, one per material column
        self._triangle_materials_gpu: Optional[list] = None
        self._triangle_materials_gpu_src = None  # identity tracking for cache invalidation

        # Load RRTS-style patterns if available
        self.tx_pattern_rrts = None
        self.rx_pattern_rrts = None
        if load_pattern_rrts_format is not None:
            if tx_pattern_file is not None:
                try:
                    self.tx_pattern_rrts = load_pattern_rrts_format(tx_pattern_file)
                except Exception as e:
                    if verbose:
                        print(f"[Warning] Could not load TX RRTS pattern: {e}")
            if rx_pattern_file is not None:
                try:
                    self.rx_pattern_rrts = load_pattern_rrts_format(rx_pattern_file)
                except Exception as e:
                    if verbose:
                        print(f"[Warning] Could not load RX RRTS pattern: {e}")

        # Load GPU-native pattern loaders (for fast antenna evaluation in DrJit)
        self.tx_pattern_loader = None
        self.rx_pattern_loader = None
        if AntennaPatternLoader is not None:
            if tx_pattern_file is not None:
                try:
                    self.tx_pattern_loader = AntennaPatternLoader(tx_pattern_file)
                except Exception as e:
                    if verbose:
                        print(f"[Warning] Could not create TX AntennaPatternLoader: {e}")
            if rx_pattern_file is not None:
                try:
                    self.rx_pattern_loader = AntennaPatternLoader(rx_pattern_file)
                except Exception as e:
                    if verbose:
                        print(f"[Warning] Could not create RX AntennaPatternLoader: {e}")


        # Initialize components
        self.sampler = ReservoirSampler(
            n_hits_per_rx=self.config.n_hits_per_rx,
            n_rays_per_res=self.config.n_rays_per_res,
            max_distance=self.config.max_distance,
            hemisphere_sampling=getattr(self.config, 'hemisphere_sampling', 'cosine'),
        )

        self.integrator = SBRIntegratorRef(scene_ctx.config, self.config)

        # Material parameterization strategy (default: None = per-triangle fallback)
        self._material_param: Optional[MaterialParameterization] = None

        # Initialize SMS solver (if enabled)
        self.sms_solver = None
        if self.config.enable_sms:
            self.sms_solver = SpecularManifoldSampler(
                wavelength=self.integrator.wavelength,
                bsdf=self.integrator.bsdf,
                max_iterations=self.config.sms_max_iterations,
                solver_threshold=self.config.sms_solver_threshold,
                use_smooth_normals=self.config.sms_use_smooth_normals,
            )
            if verbose:
                print("[SMS] Specular Manifold Sampling enabled "
                      f"(max_iter={self.config.sms_max_iterations}, "
                      f"thresh={self.config.sms_solver_threshold:.1e})")

        # Initialize image method components (if enabled)
        self.deduplicator = None
        self.patch_data = None
        if self.config.enable_sms:
            self.deduplicator = SpecularDeduplicator(
                counter_size=self.config.specular_dedup_counter_size,
                num_hash_functions=self.config.num_hash_functions,
            )
            if verbose:
                print(f"[SMS] Deduplicator: counter_size={self.config.specular_dedup_counter_size}, "
                      f"num_hashes={self.config.num_hash_functions}")

        # Initialize free-space diffraction (if enabled)
        if (_HAS_DIFFRACTION and self.config.diffraction_config is not None
                and self.config.diffraction_config.enabled):
            diff_cfg = self.config.diffraction_config
            wavelength = self.integrator.wavelength
            k = 2.0 * np.pi / wavelength
            beam_sigma = diff_cfg.beam_sigma_wavelengths * wavelength

            # Auto-compute hash cell size if not specified
            cell_size = diff_cfg.hash_cell_size
            if cell_size <= 0:
                cell_size = 3.0 * beam_sigma  # search radius

            # Build spatial hash from scene mesh
            tri_hash = build_triangle_hash_from_scene(
                scene_ctx.scene, cell_size,
                edge_angle_threshold_deg=diff_cfg.edge_angle_threshold_deg,
                wavelength=wavelength,
                min_edge_length_wavelengths=diff_cfg.min_edge_length_wavelengths,
                boundary_erosion_hops=diff_cfg.boundary_erosion_hops,
                region_angle_threshold_deg=diff_cfg.region_angle_threshold_deg,
                region_dihedral_cap_deg=diff_cfg.region_dihedral_cap_deg,
                min_region_faces=diff_cfg.min_region_faces,
                inter_region_min_faces=diff_cfg.inter_region_min_faces,
                min_chain_length_wavelengths=diff_cfg.min_chain_length_wavelengths,
                chain_collinearity_threshold_deg=diff_cfg.chain_collinearity_threshold_deg,
            )

            # Create FsdBSDF evaluator
            fsd_bsdf = FsdBSDF(k=k, beam_sigma=beam_sigma, beta_max=diff_cfg.beta_max)

            # Wire into the integrator
            self.integrator.tri_hash = tri_hash
            self.integrator.fsd_bsdf = fsd_bsdf
            self.integrator.diffraction_config = diff_cfg

            # Build GPU spatial hash (always used; CPU path is deprecated)
            from .diffraction.fsd_aperture_gpu import build_gpu_spatial_hash
            self.integrator._gpu_spatial_hash = build_gpu_spatial_hash(tri_hash)
            if verbose:
                print("  [FSD] GPU aperture construction enabled")


        # Store last render result
        self._last_result: Optional[RenderResultRef] = None

    def _update_diffraction_materials(self):
        """Propagate per-triangle eps_real/eps_imag to integrator for material-dependent diffraction."""
        diff_cfg = self.config.diffraction_config
        if (diff_cfg is not None and diff_cfg.enabled and diff_cfg.material_opacity
                and self.triangle_materials is not None
                and self.triangle_materials.shape[1] >= 2):
            self.integrator.diffraction_tri_eps = (
                self.triangle_materials[:, 0].astype(np.float32),  # eps_real
                self.triangle_materials[:, 1].astype(np.float32),  # eps_imag
            )
        else:
            self.integrator.diffraction_tri_eps = (None, None)

    @property
    def material_param(self) -> Optional[MaterialParameterization]:
        """Material parameterization strategy."""
        return self._material_param

    @material_param.setter
    def material_param(self, value: Optional[MaterialParameterization]):
        """Set material parameterization and propagate to integrator."""
        self._material_param = value
        self.integrator.material_param = value

    @classmethod
    def from_files(
        cls,
        mesh_file: str,
        config_file: str,
        pattern_file: Optional[str] = None,
        tx_pattern_file: Optional[str] = None,
        rx_pattern_file: Optional[str] = None,
        material_type: str = "metal",
        render_config: Optional[RenderConfigRef] = None,
        verbose: bool = True
    ) -> 'FMCWRendererRef':
        """Create renderer from files."""
        # Use SceneContext.from_files() to load everything
        # SceneContext.from_files() expects: config_file, scene_file, pattern_file, etc.
        scene_ctx = SceneContext.from_files(
            config_file=config_file,
            scene_file=mesh_file,  # SceneContext calls it scene_file
            pattern_file=pattern_file,
            tx_pattern_file=tx_pattern_file,
            rx_pattern_file=rx_pattern_file,
            material_type=material_type,
            enable_gradients=False,  # forward pass only
            verbose=verbose
        )

        # Determine pattern files for RRTS-style loading
        actual_tx_pattern = tx_pattern_file or pattern_file
        actual_rx_pattern = rx_pattern_file or pattern_file

        return cls(
            scene_ctx,
            render_config,
            verbose,
            tx_pattern_file=actual_tx_pattern,
            rx_pattern_file=actual_rx_pattern
        )

    def render(self, seed: Optional[int] = None) -> RenderResultRef:
        """Execute RRTS-style rendering."""
        seed = seed or self.config.seed

        # Release previous result to free DrJit arrays
        if self._last_result is not None:
            self._last_result = None

        if self.verbose:
            print("\n" + "=" * 60)
            print("RRTS Reference Renderer")
            print("=" * 60)
            print(f"Seed: {seed}")
            print(f"n_hits_per_rx: {self.config.n_hits_per_rx}")
            print(f"n_rays_per_res: {self.config.n_rays_per_res}")

        # Get RX positions and boresights
        rx_array = self.scene_ctx.rx_array
        n_rx = rx_array.num_elements

        rx_positions = mi.Point3f(
            mi.Float(rx_array.positions[:, 0]),
            mi.Float(rx_array.positions[:, 1]),
            mi.Float(rx_array.positions[:, 2])
        )

        # Use orientations (boresight directions) from RxArray
        # RxArray.orientations is a mi.Vector3f with per-element boresight directions
        rx_boresights = rx_array.orientations

        # Propagate material arrays for material-dependent diffraction
        self._update_diffraction_materials()

        # Step 1: Reservoir sampling
        if self.verbose:
            print("\n[Step 1] Reservoir Sampling")

        reservoir_hits = self.sampler.sample_vectorized(
            scene=self.scene_ctx.scene,
            rx_positions=rx_positions,
            rx_boresights=rx_boresights,
            seed=seed,
            verbose=self.verbose,
            triangle_materials=self.triangle_materials,
            use_vertex_normals=self.config.use_vertex_normals,
        )

        # Get TX positions and boresights
        tx_array = self.scene_ctx.tx_array
        tx_array.num_elements

        tx_positions = mi.Point3f(
            mi.Float(tx_array.positions[:, 0]),
            mi.Float(tx_array.positions[:, 1]),
            mi.Float(tx_array.positions[:, 2])
        )

        # Use orientations (boresight directions) from TxArray
        tx_boresights = tx_array.orientations

        # Step 2: Specular path pipeline (SMS or image method, if enabled)
        specular_paths = None
        use_non_ka = False
        patch_has_specular = None
        tri_to_patch = None

        _specular_enabled = self.config.enable_sms and self.deduplicator is not None
        if _specular_enabled:
            use_patches = self.patch_data is not None

            # Step 2a: Deduplicate specular paths
            if self.verbose:
                print("  [Step 2a] Specular path deduplication")

            unique_mask, dedup_stats = self.deduplicator.deduplicate(
                hit_N=reservoir_hits.hit_N,
                hit_P=reservoir_hits.hit_P,
                valid=reservoir_hits.valid,
                n_rx=n_rx,
                use_patches=use_patches,
                hit_ID=reservoir_hits.hit_ID if use_patches else None,
                patch_data=self.patch_data,
            )

            dedup_label = "patches" if dedup_stats.get('use_patches') else "triangles"
            if self.verbose:
                print(f"    Valid hits: {dedup_stats['n_valid_hits']:,}")
                print(f"    Unique {dedup_label}: {dedup_stats['n_unique_triangles']:,}")
                print(f"    Duplicates removed: {dedup_stats['n_duplicates_removed']:,}")
                print(f"    Dedup ratio: {dedup_stats['dedup_ratio']:.3f}")

            # Step 2b: Specular path finding (SMS or image method)
            if self.config.enable_sms and self.sms_solver is not None:
                # SMS path
                if self.verbose:
                    print("  [Step 2b] SMS Newton solver")

                specular_paths = self.sms_solver.find_specular_paths(
                    unique_mask=unique_mask,
                    hit_P=reservoir_hits.hit_P,
                    hit_N=reservoir_hits.hit_N,
                    hit_rho=reservoir_hits.hit_rho,
                    hit_ID=reservoir_hits.hit_ID,
                    tx_positions=tx_positions,
                    rx_positions=rx_positions,
                    n_hits_per_rx=self.config.n_hits_per_rx,
                    scene=self.scene_ctx.scene,
                    verbose=self.verbose,
                    patch_data=self.patch_data,
                    triangle_materials=self.triangle_materials,
                )
                n_valid_spec = int(dr.sum(mi.UInt32(specular_paths.valid))[0]) if specular_paths.n_paths > 0 else 0
                n_total_hits = dedup_stats['n_valid_hits']
                n_unique = dedup_stats['n_unique_triangles']
                n_deduped = dedup_stats['n_duplicates_removed']
                print(f"  [SMS] Diffuse hits: {n_total_hits:,} | "
                      f"Unique {dedup_label}: {n_unique:,} (dedup removed {n_deduped:,}) | "
                      f"Valid specular paths: {n_valid_spec:,}/{specular_paths.n_paths:,}")

                # Per-patch adaptive BSDF: build mask of which patches have specular coverage
                use_non_ka = True


        # Step 3: ADC synthesis
        if self.verbose:
            synthesis_mode = "(per-patch dual-path)" if patch_has_specular is not None else \
                             "(dual-path)" if use_non_ka else "(standard)"
            print(f"\n[Step 3] ADC Synthesis {synthesis_mode}")

        # Sync triangle materials to integrator for Phase A specular BSDF evaluation
        self.integrator.triangle_materials = self.triangle_materials

        adc_result = self.integrator.synthesize_shared_hits_vectorized(
            tx_positions=tx_positions,
            rx_positions=rx_positions,
            tx_boresights=tx_boresights,  # For antenna pattern evaluation
            rx_boresights=rx_boresights,  # For antenna pattern evaluation + sampling PDF
            reservoir_hits=reservoir_hits,
            scene=self.scene_ctx.scene,
            tx_pattern_rrts=self.tx_pattern_rrts,
            rx_pattern_rrts=self.rx_pattern_rrts,
            tx_pattern_loader=self.tx_pattern_loader,
            rx_pattern_loader=self.rx_pattern_loader,
            antenna_gain_linear=self.config.antenna_gain_linear,
            verbose=self.verbose,
            use_non_ka_bsdf=use_non_ka,
            specular_paths=specular_paths,
            patch_has_specular=patch_has_specular,
            tri_to_patch=tri_to_patch,
        )

        # Store result
        self._last_result = RenderResultRef(
            adc_result=adc_result,
            reservoir_hits=reservoir_hits,
            config=self.config
        )

        if self.verbose:
            print("\n" + "=" * 60)
            print("Rendering Complete")
            print(f"  Total paths: {adc_result.total_paths:,}")
            print(f"  ADC shape: ({adc_result.n_tx}, {adc_result.n_rx}, {adc_result.n_samples})")
            print("=" * 60)

        return self._last_result

    def _get_triangle_materials_gpu(self):
        """Return GPU-resident per-triangle material columns (lazy upload)."""
        tm = self.triangle_materials
        if tm is None:
            self._triangle_materials_gpu = None
            return None
        # Re-upload when the numpy array changes (identity check is cheap)
        if (self._triangle_materials_gpu is None
                or self._triangle_materials_gpu_src is not tm):
            cols = []
            for c in range(tm.shape[1]):
                cols.append(mi.Float(tm[:, c].astype(np.float32)))
            dr.eval(*cols)  # single JIT compilation for all columns
            self._triangle_materials_gpu = cols
            self._triangle_materials_gpu_src = tm
        return self._triangle_materials_gpu

    def render_end_to_end(
        self,
        raw_params_drjit: list,
        pose_params: Optional[dict] = None,
        normal_params: Optional[list] = None,
        vertex_offset_params: Optional[list] = None,
        pattern_loaders: Optional[dict] = None,
        seed: Optional[int] = None,
        include_diffuse: bool = True,
        include_specular: bool = True,
        include_diffraction: bool = True,
        include_energy_borrowing: bool = True,
        return_total_distance: bool = False,
    ) -> tuple:
        """Single-pass end-to-end differentiable render."""
        from mmir.renderer.bsdf.reparameterization import reparameterize_physics_params_drjit

        seed = seed or self.config.seed

        # Profiling support
        getattr(self.integrator, '_profile', None)
        def _tic(label):
            pass
        def _toc(label):
            pass

        _tic('material_reparam')
        # 0. Reparameterize materials (diff)
        physics_params = reparameterize_physics_params_drjit(raw_params_drjit)
        _toc('material_reparam')

        _tic('antenna_pose')
        # 1. Apply pose to TX/RX positions (diff if pose_params provided)
        tx_positions, rx_positions, tx_boresights, rx_boresights = \
            self._get_antenna_arrays(pose_params)
        _toc('antenna_pose')

        # 2. Update scene vertex positions if vertex offsets provided

        _tic('ray_sampling')
        # 3. Sample rays per RX (non-diff directions, AD-attached origins from RX)
        reservoir_hits = self.sampler.sample_reservoir_drjit(
            scene=self.scene_ctx.scene,
            rx_positions=rx_positions,
            rx_boresights=rx_boresights,
            seed=seed,
            verbose=self.verbose,
            use_vertex_normals=self.config.use_vertex_normals,
        )
        _toc('ray_sampling')

        _tic('sms_specular_cache')
        # 3b. Find SMS specular paths on the (possibly deformed) mesh.
        # SMS paths are cached to avoid re-running the expensive Newton solver
        # every iteration. IFT in synthesis handles geometry perturbations.
        #
        # Caching strategy:
        # - Static geometry (no pose/vtx learning): cache forever
        # - Dynamic geometry with sms_cache_interval > 0: cache for N iters
        # - Dynamic geometry with sms_cache_interval == 0: recompute every iter
        _geom_static = (pose_params is None and vertex_offset_params is None)
        _cache_interval = self.config.sms_cache_interval

        # Initialize cache state
        if not hasattr(self, '_sms_cache_iter'):
            self._sms_cache_iter = 0
            self._cached_specular_paths = None
            self._cached_specular_chains = None

        # Determine if SMS needs refresh
        _need_refresh = False
        if _geom_static:
            _need_refresh = (self._cached_specular_paths is None)
        elif _cache_interval > 0:
            _need_refresh = (self._cached_specular_paths is None
                             or self._sms_cache_iter >= _cache_interval)
        else:
            _need_refresh = True  # recompute every iteration

        specular_paths = None
        if include_specular and self.sms_solver is not None:
            if _need_refresh:
                self._cached_specular_paths = self._find_sms_specular_e2e(
                    reservoir_hits, tx_positions, rx_positions,
                    self.scene_ctx.scene)
                self._sms_cache_iter = 0
            specular_paths = self._cached_specular_paths

        # 3c. Find multibounce specular chains (k >= 2)
        specular_chains = None
        if (include_specular and self.sms_solver is not None
                and self.config.max_bounces > 1):
            if _need_refresh:
                self._cached_specular_chains = self._find_sms_specular_chains_e2e(
                    reservoir_hits, tx_positions, rx_positions,
                    self.scene_ctx.scene,
                    k=min(self.config.max_bounces, 2),
                    seed=seed,
                )
            specular_chains = self._cached_specular_chains

        self._sms_cache_iter += 1

        _toc('sms_specular_cache')

        # 5. End-to-end synthesis (single-bounce or multibounce)
        synth_result = self.integrator.synthesize_end_to_end_multibounce(
            reservoir_hits=reservoir_hits,
            scene=self.scene_ctx.scene,
            tx_positions=tx_positions,
            rx_positions=rx_positions,
            tx_boresights=tx_boresights,
            rx_boresights=rx_boresights,
            physics_params=physics_params,
            normal_params=normal_params,
            pattern_loaders=pattern_loaders,
            tx_pattern_rrts=self.tx_pattern_rrts,
            rx_pattern_rrts=self.rx_pattern_rrts,
            tx_pattern_loader=self.tx_pattern_loader,
            rx_pattern_loader=self.rx_pattern_loader,
            antenna_gain_linear=self.config.antenna_gain_linear,
            verbose=self.verbose,
            specular_paths=specular_paths,
            specular_chains=specular_chains,
            include_diffuse=include_diffuse,
            include_specular=include_specular,
            include_diffraction=include_diffraction,
            include_energy_borrowing=include_energy_borrowing,
            vertex_offset_params=vertex_offset_params,
            return_total_distance=return_total_distance,
            max_bounces=self.config.max_bounces,
            nee_every_bounce=self.config.nee_every_bounce,
            rr_start_bounce=self.config.rr_start_bounce,
            rr_prob=self.config.rr_prob,
            compute_boundary_viewpoints=(
                self.config.enable_boundary_gradients
                and (vertex_offset_params is not None or pose_params is not None)
            ),
        )

        adc_real, adc_imag = synth_result

        # Note: Do NOT auto-restore scene vertices here. The AD tape for
        # vertex position gradients requires the modified positions to remain
        # in the scene until dr.backward() completes. Callers should call
        # _restore_scene_vertices() after backward if needed (or the next
        # _update_scene_vertices call will overwrite them anyway).

        return adc_real, adc_imag

    def _find_sms_specular_e2e(self, reservoir_hits, tx_positions, rx_positions, scene):
        """Find SMS specular paths from e2e reservoir hits."""
        n_valid = reservoir_hits.n_valid
        if n_valid == 0:
            return None

        # Dedup on compressed hits (per-RX). Uses patch dedup when available
        # (matches the reference renderer), else plane-hash fallback.
        unique_mask, dedup_stats = self.deduplicator.deduplicate_compressed(
            hit_N=reservoir_hits.hit_N,
            hit_P=reservoir_hits.hit_P,
            rx_idx=reservoir_hits.rx_element_idx,
            n_rx=reservoir_hits.n_rx,
            hit_prim_ids=reservoir_hits.hit_prim_ids,
            patch_data=self.patch_data,
        )
        dedup_mode = "Patch" if dedup_stats.get('use_patches') else "Plane-hash"
        if self.verbose:
            print(f"  [SMS-E2E] {dedup_mode} dedup: {dedup_stats['n_unique_triangles']:,} unique "
                  f"/ {dedup_stats['n_valid_hits']:,} valid "
                  f"({dedup_stats['n_duplicates_removed']:,} duplicates removed)")
        # Compress unique hits to seed indices
        unique_indices = dr.compress(unique_mask)
        n_unique = dr.width(unique_indices)

        if n_unique == 0:
            return None

        # Gather seed data at unique occurrences.
        # Detach from AD — SMS solver runs without AD (IFT attaches grads later).
        # PERF: Gather all components first, then evaluate in a single dr.eval()
        # call to avoid 6 separate JIT compilations of the AD-attached ray
        # intersection graph (each dr.detach() on a lazy AD var forces eval).
        _px = dr.gather(mi.Float, reservoir_hits.hit_P.x, unique_indices)
        _py = dr.gather(mi.Float, reservoir_hits.hit_P.y, unique_indices)
        _pz = dr.gather(mi.Float, reservoir_hits.hit_P.z, unique_indices)
        _nx = dr.gather(mi.Float, reservoir_hits.hit_N.x, unique_indices)
        _ny = dr.gather(mi.Float, reservoir_hits.hit_N.y, unique_indices)
        _nz = dr.gather(mi.Float, reservoir_hits.hit_N.z, unique_indices)
        dr.eval(_px, _py, _pz, _nx, _ny, _nz)
        seed_P = mi.Point3f(mi.Float(dr.detach(_px)), mi.Float(dr.detach(_py)), mi.Float(dr.detach(_pz)))
        seed_N = mi.Vector3f(mi.Float(dr.detach(_nx)), mi.Float(dr.detach(_ny)), mi.Float(dr.detach(_nz)))
        seed_prim_ids = dr.gather(mi.UInt32, reservoir_hits.hit_prim_ids, unique_indices)
        seed_rx_idx = dr.gather(mi.UInt32, reservoir_hits.rx_element_idx, unique_indices)

        # Detach TX/RX positions from AD graph — SMS Newton solver runs
        # without AD. Gradients are attached retroactively via IFT in
        # _synthesize_specular_e2e(). Without detach, the Newton iterations
        # create an enormous AD tape that hangs backward traversal.
        tx_pos_detached = mi.Point3f(
            mi.Float(dr.detach(tx_positions.x)),
            mi.Float(dr.detach(tx_positions.y)),
            mi.Float(dr.detach(tx_positions.z)),
        )
        rx_pos_detached = mi.Point3f(
            mi.Float(dr.detach(rx_positions.x)),
            mi.Float(dr.detach(rx_positions.y)),
            mi.Float(dr.detach(rx_positions.z)),
        )

        # Wrap in dr.suspend_grad() to prevent the SMS Newton solver from
        # recording AD operations through the scene's vertex buffer. When
        # vertex_offset_params are used, the scene's vertices are AD-attached,
        # and the solver's ray_test() / surface_parameterization() calls would
        # create an enormous AD tape that hangs backward traversal.
        # IFT in _synthesize_specular_e2e() attaches gradients retroactively.
        with dr.suspend_grad():
            return self.sms_solver.find_specular_paths_from_seeds(
                seed_P=seed_P,
                seed_N=seed_N,
                seed_prim_ids=seed_prim_ids,
                seed_rx_idx=seed_rx_idx,
                tx_positions=tx_pos_detached,
                rx_positions=rx_pos_detached,
                scene=scene,
                verbose=self.verbose,
                triangle_materials_gpu=self._get_triangle_materials_gpu(),
                n_material_cols=self.triangle_materials.shape[1] if self.triangle_materials is not None else 0,
                patch_data=self.patch_data,
            )

    def _find_sms_specular_chains_e2e(
        self, reservoir_hits, tx_positions, rx_positions, scene, k=2, seed=42,
    ):
        """Find k-bounce specular chains via multibounce SMS."""
        from mmir.renderer.integrator.mixture_sampling import _sample_cosine_hemisphere_drjit

        n_valid = reservoir_hits.n_valid
        if n_valid == 0:
            return None

        n_tx = dr.width(tx_positions)

        # Dedup: use GPU deduplicator if available, else GPU-native fallback
        unique_mask, _dedup_stats = self.deduplicator.deduplicate_compressed(
            hit_N=reservoir_hits.hit_N,
            hit_P=reservoir_hits.hit_P,
            rx_idx=reservoir_hits.rx_element_idx,
            n_rx=reservoir_hits.n_rx,
            hit_prim_ids=reservoir_hits.hit_prim_ids,
            patch_data=self.patch_data,
        )
        idx_dr = dr.compress(unique_mask)
        n_unique = dr.width(idx_dr)

        if n_unique == 0:
            return None

        # Gather S₁ seed data (detached from AD)
        # PERF: batch all gathers, single dr.eval(), then detach
        _px = dr.gather(mi.Float, reservoir_hits.hit_P.x, idx_dr)
        _py = dr.gather(mi.Float, reservoir_hits.hit_P.y, idx_dr)
        _pz = dr.gather(mi.Float, reservoir_hits.hit_P.z, idx_dr)
        _nx = dr.gather(mi.Float, reservoir_hits.hit_N.x, idx_dr)
        _ny = dr.gather(mi.Float, reservoir_hits.hit_N.y, idx_dr)
        _nz = dr.gather(mi.Float, reservoir_hits.hit_N.z, idx_dr)
        s1_prim_ids = dr.gather(mi.UInt32, reservoir_hits.hit_prim_ids, idx_dr)
        s1_rx_idx = dr.gather(mi.UInt32, reservoir_hits.rx_element_idx, idx_dr)
        dr.eval(_px, _py, _pz, _nx, _ny, _nz, s1_prim_ids, s1_rx_idx)
        s1_P = mi.Point3f(mi.Float(dr.detach(_px)), mi.Float(dr.detach(_py)), mi.Float(dr.detach(_pz)))
        s1_N = mi.Vector3f(mi.Float(dr.detach(_nx)), mi.Float(dr.detach(_ny)), mi.Float(dr.detach(_nz)))
        s1_prim_ids = mi.UInt32(dr.detach(s1_prim_ids))
        s1_rx_idx = mi.UInt32(dr.detach(s1_rx_idx))

        # Expand × TX: n_chains = n_unique × n_tx
        n_chains = n_unique * n_tx

        s1_P_exp = mi.Point3f(
            dr.repeat(s1_P.x, n_tx), dr.repeat(s1_P.y, n_tx), dr.repeat(s1_P.z, n_tx))
        s1_N_exp = mi.Vector3f(
            dr.repeat(s1_N.x, n_tx), dr.repeat(s1_N.y, n_tx), dr.repeat(s1_N.z, n_tx))
        s1_prim_exp = mi.UInt32(dr.repeat(s1_prim_ids, n_tx))

        # PERF: GPU-native tiling instead of np.tile/np.repeat
        # chain_tx_idx: [0,1,...,n_tx-1, 0,1,...,n_tx-1, ...] repeated n_unique times
        chain_tx_idx = dr.arange(mi.UInt32, n_chains) % mi.UInt32(n_tx)
        # chain_rx_idx: each s1_rx_idx repeated n_tx times
        chain_rx_idx = mi.UInt32(dr.repeat(s1_rx_idx, n_tx))

        # Generate S₂ seeds: trace random ray from S₁ (cosine-weighted)
        with dr.suspend_grad():
            s2_dirs, _s2_pdf, _s2_sampler = _sample_cosine_hemisphere_drjit(s1_N_exp, n_chains, seed + 12345)

            # Offset origin along normal to avoid self-intersection
            offset = mi.Float(1e-3)
            ray_o = mi.Point3f(
                s1_P_exp.x + offset * s1_N_exp.x,
                s1_P_exp.y + offset * s1_N_exp.y,
                s1_P_exp.z + offset * s1_N_exp.z)
            ray = mi.Ray3f(ray_o, s2_dirs)
            si = scene.ray_intersect(ray)

            s2_P = mi.Point3f(si.p)
            s2_prim_ids = mi.UInt32(si.prim_index)
            s2_valid = si.is_valid()

            # Filter out chains where S₂ didn't hit anything
            n_s2_valid = int(dr.sum(mi.UInt32(s2_valid))[0])

            if self.verbose:
                print(f"  [SMS-MB] S₂ seeding: {n_s2_valid}/{n_chains} hit surface")

            if n_s2_valid == 0:
                return None

            # Build seed lists for find_specular_chains
            seed_P_list = [s1_P_exp, s2_P]
            seed_prim_list = [s1_prim_exp, s2_prim_ids]

            # For k > 2, would add more bounce seeds here (future extension)

            result = self.sms_solver.find_specular_chains(
                seed_P_list=seed_P_list,
                seed_prim_ids_list=seed_prim_list,
                tx_positions=mi.Point3f(
                    mi.Float(dr.detach(tx_positions.x)),
                    mi.Float(dr.detach(tx_positions.y)),
                    mi.Float(dr.detach(tx_positions.z))),
                rx_positions=mi.Point3f(
                    mi.Float(dr.detach(rx_positions.x)),
                    mi.Float(dr.detach(rx_positions.y)),
                    mi.Float(dr.detach(rx_positions.z))),
                seed_tx_idx=chain_tx_idx,
                seed_rx_idx=chain_rx_idx,
                scene=scene,
                k=k,
                max_iterations=self.config.sms_max_iterations,
                threshold=self.config.sms_solver_threshold,
                verbose=self.verbose,
                triangle_materials_gpu=self._get_triangle_materials_gpu(),
                n_material_cols=self.triangle_materials.shape[1] if self.triangle_materials is not None else 0,
            )

        return result

    def _get_antenna_arrays(self, pose_params=None):
        """Return TX/RX positions and boresights, optionally transformed by pose."""
        tx_array = self.scene_ctx.tx_array
        rx_array = self.scene_ctx.rx_array

        tx_positions = mi.Point3f(
            mi.Float(tx_array.positions[:, 0]),
            mi.Float(tx_array.positions[:, 1]),
            mi.Float(tx_array.positions[:, 2]),
        )
        rx_positions = mi.Point3f(
            mi.Float(rx_array.positions[:, 0]),
            mi.Float(rx_array.positions[:, 1]),
            mi.Float(rx_array.positions[:, 2]),
        )

        tx_array.num_elements
        rx_array.num_elements

        tx_boresights = tx_array.orientations

        rx_boresights = rx_array.orientations

        return tx_positions, rx_positions, tx_boresights, rx_boresights

    def get_adc_numpy(self) -> np.ndarray:
        """Get ADC data as numpy array."""
        if self._last_result is None:
            raise RuntimeError("No render result available. Call render() first.")

        return self._last_result.adc_result.get_ri_array()


__all__ = [
    'FMCWRendererRef',
    'RenderConfigRef',
    'RenderResultRef',
]
