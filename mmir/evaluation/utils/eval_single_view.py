import numpy as np
from scipy.spatial import cKDTree
import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def compute_distance_based_occupancy_metrics(radar_pts: np.ndarray,
                                             lidar_pts: np.ndarray,
                                             distance_threshold: float = 0.5) -> dict:
    """Compute precision, recall, and accuracy based on nearest neighbor distances."""
    metrics = {}

    if len(radar_pts) == 0 or len(lidar_pts) == 0:
        print("  [warn] Empty point cloud, setting metrics to 0.0")
        return {
            'distance_precision': 0.0,
            'distance_recall': 0.0,
            'distance_specificity': 0.0,
            'distance_accuracy': 0.0,
            'distance_threshold_m': distance_threshold
        }

    # Build KD-trees for nearest neighbor search
    radar_tree = cKDTree(radar_pts)
    lidar_tree = cKDTree(lidar_pts)

    # Compute d(p, Q) = min distance from each radar point to LiDAR
    dist_radar_to_lidar, _ = lidar_tree.query(radar_pts, k=1)

    # Compute d(q, P) = min distance from each LiDAR point to radar
    dist_lidar_to_radar, _ = radar_tree.query(lidar_pts, k=1)

    # Count matches within threshold
    matched_radar = np.sum(dist_radar_to_lidar < distance_threshold)
    matched_lidar = np.sum(dist_lidar_to_radar < distance_threshold)

    # Compute metrics
    precision = matched_radar / len(radar_pts) if len(radar_pts) > 0 else 0.0
    recall = matched_lidar / len(lidar_pts) if len(lidar_pts) > 0 else 0.0
    accuracy = (matched_radar + matched_lidar) / (len(radar_pts) + len(lidar_pts))

    # Compute specificity as: (unmatched_lidar) / (unmatched_lidar + unmatched_radar)
    # This measures "what fraction of unmatched points are from LiDAR"
    # High specificity means radar has few false positives relative to LiDAR's coverage
    unmatched_radar = len(radar_pts) - matched_radar
    unmatched_lidar = len(lidar_pts) - matched_lidar
    specificity = unmatched_lidar / (unmatched_lidar + unmatched_radar) if (unmatched_lidar + unmatched_radar) > 0 else 0.0

    metrics['distance_precision'] = precision
    metrics['distance_recall'] = recall
    metrics['distance_specificity'] = specificity
    metrics['distance_accuracy'] = accuracy
    metrics['distance_threshold_m'] = distance_threshold
    metrics['matched_radar_points'] = int(matched_radar)
    metrics['matched_lidar_points'] = int(matched_lidar)
    metrics['total_radar_points'] = len(radar_pts)
    metrics['total_lidar_points'] = len(lidar_pts)

    return metrics
# NEW COMPOSABLE EVALUATION HELPERS


def compute_point_cloud_rmse_and_chamfer(radar_pts: np.ndarray,
                                          lidar_pts: np.ndarray) -> dict:
    """Compute RMSE and Relative Chamfer Distance between radar and lidar point clouds."""
    metrics: dict = {}
    print("\n3) POINT CLOUD RMSE AND CHAMFER DISTANCE")
    print("-" * 40)

    if radar_pts is None or lidar_pts is None or radar_pts.shape[0] == 0 or lidar_pts.shape[0] == 0:
        print("[warn] Insufficient points for RMSE/Chamfer; skipping")
        metrics.update({
            'pc_rmse': float('inf'),
            'chamfer_distance': float('inf'),
            'relative_chamfer_distance': float('inf')
        })
        return metrics

    try:
        # Build KD-trees for efficient nearest neighbor search
        from scipy.spatial import cKDTree
        radar_tree = cKDTree(radar_pts)
        lidar_tree = cKDTree(lidar_pts)

        # Compute nearest neighbor distances: radar -> lidar
        dist_r2l, _ = lidar_tree.query(radar_pts, k=1)

        # Compute nearest neighbor distances: lidar -> radar
        dist_l2r, _ = radar_tree.query(lidar_pts, k=1)

        # RMSE: average of bidirectional RMSE
        rmse_r2l = float(np.sqrt(np.mean(dist_r2l**2)))
        rmse_l2r = float(np.sqrt(np.mean(dist_l2r**2)))
        rmse = (rmse_r2l + rmse_l2r) / 2.0

        # Chamfer Distance (bidirectional)
        chamfer_r2l = float(np.mean(dist_r2l))
        chamfer_l2r = float(np.mean(dist_l2r))
        chamfer = chamfer_r2l + chamfer_l2r

        # Relative Chamfer Distance (normalized by scene scale)
        # Scene scale = max extent of lidar point cloud
        lidar_extent = float(np.max(np.ptp(lidar_pts, axis=0)))
        relative_chamfer = chamfer / lidar_extent if lidar_extent > 0 else float('inf')

        metrics['pc_rmse'] = rmse
        metrics['pc_rmse_r2l'] = rmse_r2l
        metrics['pc_rmse_l2r'] = rmse_l2r
        metrics['chamfer_distance'] = chamfer
        metrics['chamfer_r2l'] = chamfer_r2l
        metrics['chamfer_l2r'] = chamfer_l2r
        metrics['relative_chamfer_distance'] = relative_chamfer
        metrics['lidar_scene_extent'] = lidar_extent

        print(f"RMSE (bidirectional avg): {rmse:.4f} m")
        print(f"  RMSE radar->lidar: {rmse_r2l:.4f} m")
        print(f"  RMSE lidar->radar: {rmse_l2r:.4f} m")
        print(f"Chamfer Distance: {chamfer:.4f} m")
        print(f"  Chamfer radar->lidar: {chamfer_r2l:.4f} m")
        print(f"  Chamfer lidar->radar: {chamfer_l2r:.4f} m")
        print(f"Relative Chamfer Distance: {relative_chamfer:.4f}")
        print(f"LiDAR scene extent: {lidar_extent:.4f} m")

    except Exception as e:
        print(f"[error] Failed to compute RMSE/Chamfer: {e}")
        metrics.update({
            'pc_rmse': float('inf'),
            'chamfer_distance': float('inf'),
            'relative_chamfer_distance': float('inf')
        })

    return metrics
