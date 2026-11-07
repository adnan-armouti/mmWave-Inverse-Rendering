import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional

import numpy as np


def _ensure_poisson_binaries(root_dir: str) -> tuple[str, str]:
    """Ensure PoissonRecon and SurfaceTrimmer binaries exist under the submodule. Builds them via
    Makefile if missing. Returns absolute paths to binaries.
    """
    repo_root = Path(root_dir).resolve()
    poisson_dir = repo_root / "third_party" / "PoissonRecon"
    if not poisson_dir.exists():
        raise FileNotFoundError(f"PoissonRecon submodule not found at: {poisson_dir}")

    bin_dir = poisson_dir / "Bin" / "Linux"
    poisson_bin = bin_dir / "PoissonRecon"
    trimmer_bin = bin_dir / "SurfaceTrimmer"

    if poisson_bin.exists() and trimmer_bin.exists():
        return str(poisson_bin), str(trimmer_bin)

    # Attempt build via Makefile (Linux)
    makefile_path = poisson_dir / "Makefile"
    if not makefile_path.exists():
        raise FileNotFoundError("PoissonRecon Makefile not found; cannot build binaries.")

    # Build only required targets
    subprocess.run(["make", "poissonrecon", "surfacetrimmer"], cwd=str(poisson_dir), check=True)

    if not poisson_bin.exists():
        raise FileNotFoundError("PoissonRecon binary missing after build.")
    if not trimmer_bin.exists():
        raise FileNotFoundError("SurfaceTrimmer binary missing after build.")

    return str(poisson_bin), str(trimmer_bin)


def _save_ply(xyz: np.ndarray, normals: Optional[np.ndarray], colors: Optional[np.ndarray], out_path: str) -> str:
    """Save point cloud to ASCII PLY. Supports optional normals and colors (uint8 RGB). xyz: (N,3),
    normals: (N,3) or None, colors: (N,3) or None
    """
    N = int(xyz.shape[0])
    has_normals = normals is not None and normals.shape[0] == N
    has_colors = colors is not None and colors.shape[0] == N

    header_elems = [
        "ply",
        "format ascii 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_normals:
        header_elems += [
            "property float nx",
            "property float ny",
            "property float nz",
        ]
    header_elems.append("end_header")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header_elems) + "\n")
        if has_normals:
            for i in range(N):
                x, y, z = xyz[i]
                nx, ny, nz = normals[i]
                f.write(f"{x} {y} {z} {nx} {ny} {nz}\n")
        elif has_colors:
            for i in range(N):
                x, y, z = xyz[i]
                r, g, b = colors[i]
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
        else:
            for i in range(N):
                x, y, z = xyz[i]
                f.write(f"{x} {y} {z}\n")

    return out_path


def _load_scene_pcl(scene_npy_path: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Load scene array saved by build_and_save_lidar_scene: columns [x,y,z,nx,ny,nz,intensity].
    Returns (xyz, normals_or_none).
    """
    arr = np.load(scene_npy_path)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("scene npy must be (N, >=3)")
    xyz = arr[:, :3].astype(np.float32)
    normals = None
    if arr.shape[1] >= 6:
        normals = arr[:, 3:6].astype(np.float32)
    return xyz, normals


def reconstruct_mesh_from_scene(scene_npy_path: str,
                                out_mesh_path: str,
                                depth: int = 10,
                                point_weight: float = 2.0,
                                trim_threshold: float = 8.0,
                                colorize: bool = False,
                                temp_dir: Optional[str] = None,
                                repo_root: Optional[str] = None) -> str:
    """Run Poisson surface reconstruction from scene npy to trimmed mesh (PLY/OBJ)."""
    if repo_root is None:
        repo_root = str(Path(__file__).resolve().parents[2])

    poisson_bin, trimmer_bin = _ensure_poisson_binaries(repo_root)

    work_dir_cm = tempfile.mkdtemp(prefix="poisson_work_", dir=temp_dir)
    work_dir = Path(work_dir_cm)
    try:
        xyz, normals = _load_scene_pcl(scene_npy_path)
        colors = None

        in_ply = work_dir / "points.ply"
        _save_ply(xyz, normals, colors, str(in_ply))

        rec_ply = work_dir / "recon.ply"
        cmd = [
            poisson_bin,
            "--in", str(in_ply),
            "--out", str(rec_ply),
            "--depth", str(int(depth)),
            "--pointWeight", str(float(point_weight)),
            "--density",
        ]
        subprocess.run(cmd, check=True)

        # Trim
        out_mesh = Path(out_mesh_path)
        out_mesh.parent.mkdir(parents=True, exist_ok=True)
        cmd_trim = [
            trimmer_bin,
            "--in", str(rec_ply),
            "--out", str(out_mesh),
            "--trim", str(float(trim_threshold)),
        ]
        subprocess.run(cmd_trim, check=True)
        return str(out_mesh)
    finally:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass
