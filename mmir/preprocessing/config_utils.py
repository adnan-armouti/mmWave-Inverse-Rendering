import os
import json
import numpy as np

from scipy.spatial.transform import Rotation

from mmir.preprocessing.ColoRadar_tools.dataset_loaders import (
    get_cascade_params,
    get_single_chip_params,
    get_groundtruth_params,
    get_timestamps,
    get_groundtruth,
)
from mmir.preprocessing.ColoRadar_tools.plot_pointclouds import interpolate_poses

from mmir.preprocessing.radar_utils import get_base_dir, get_subdir


c = 299792458.0
lambda_m = c / (77.0 * 1e9)
lambda_mm = lambda_m * 1000.0


def compute_world_pose_and_boresight(sensor: str,
                                     seq_idx: int,
                                     frame_idx: int,
                                     dataset_dir: str,
                                     calib_path: str):
    seq_dir = dataset_dir + str(seq_idx)
    if sensor == "cascade":
        radar_params = get_cascade_params(calib_path)["heatmap"]
    elif sensor == "single":
        radar_params = get_single_chip_params(calib_path)["heatmap"]
    else:
        raise ValueError("sensor must be 'cascade' or 'single'")

    radar_T_bs = np.eye(4)
    radar_T_bs[:3, 3] = np.asarray(radar_params["translation"], dtype=float)
    radar_T_bs[:3, :3] = Rotation.from_quat(np.asarray(radar_params["rotation"], dtype=float)).as_matrix()

    try:
        gt_params = get_groundtruth_params()
        gt_poses = get_groundtruth(seq_dir)
        gt_stamps = get_timestamps(seq_dir, gt_params)
        radar_stamps = get_timestamps(seq_dir, radar_params)
        if gt_stamps is None or radar_stamps is None:
            raise RuntimeError("timestamps not available")
        radar_gt, _ = interpolate_poses(gt_poses, gt_stamps, radar_stamps)
        radar_pose_T_wb = np.array(radar_gt[frame_idx])
        radar_T_ws = np.dot(radar_pose_T_wb, radar_T_bs)
        radar_position = radar_T_ws[:3, 3]
        radar_boresight = radar_T_ws[:3, 0]
        return radar_position.astype(float), radar_boresight.astype(float)
    except Exception:
        radar_position = radar_T_bs[:3, 3]
        radar_boresight = radar_T_bs[:3, 0]
        return radar_position.astype(float), radar_boresight.astype(float)


def antenna_layout_cascade():
    A1, A3, A4, A5, A6 = 0.5, 4.0, 16.0, 7.75, 2.0
    B1, B2, B3, B4 = 19.0, 0.5, 1.5, 1.0
    rx_array_B = [(i * A1, 0.0) for i in range(4)]
    start_C = 4 * A1 + A3
    rx_array_C = [(start_C + i * A1, 0.0) for i in range(4)]
    start_A = start_C + (4 * A1) + A4
    rx_array_A = [(start_A + i * A1, 0.0) for i in range(8)]
    rx = np.array(rx_array_B + rx_array_C + rx_array_A, dtype=float)
    tx = [(A5 + i * A6, -B1) for i in range(9)]
    start_tx_x = A5 + 2 * A6; start_tx_y = -B1
    tx += [(start_tx_x + A1, start_tx_y + B2), (start_tx_x + 2 * A1, start_tx_y + B2 + B3), (start_tx_x + 3 * A1, start_tx_y + B2 + B3 + B4)]
    tx = np.array(tx, dtype=float)
    rx[:, 1] *= -1.0; tx[:, 1] *= -1.0
    allp = np.vstack([rx, tx])
    cx, cy = (allp[:, 0].min() + allp[:, 0].max()) / 2.0, (allp[:, 1].min() + allp[:, 1].max()) / 2.0
    rel = allp - np.array([cx, cy], dtype=float)
    return rel[:rx.shape[0]], rel[rx.shape[0]:]


def antenna_layout_single_chip():
    A1, C1 = 0.5, 1.0
    B1 = 3.724 / lambda_mm
    rx = np.array([(i * A1, 0.0) for i in range(4)], dtype=float)
    start_x = 4 * A1 + B1
    tx = np.array([(start_x + i * C1, 0.0) for i in range(3)], dtype=float)
    tx[1, 1] += A1
    allp = np.vstack([rx, tx])
    cx, cy = (allp[:, 0].min() + allp[:, 0].max()) / 2.0, (allp[:, 1].min() + allp[:, 1].max()) / 2.0
    rel = allp - np.array([cx, cy], dtype=float)
    return rel[:rx.shape[0]], rel[rx.shape[0]:]


def basis_from_boresight(boresight: np.ndarray):
    y_hat = boresight / np.linalg.norm(boresight)
    z_hint = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(y_hat, z_hint))) > 0.99:
        z_hint = np.array([0.0, 1.0, 0.0], dtype=float)
    x_hat = np.cross(y_hat, z_hint); x_hat /= np.linalg.norm(x_hat)
    z_hat = np.cross(x_hat, y_hat);  z_hat /= np.linalg.norm(z_hat)
    return x_hat, y_hat, z_hat


def to_world_coords(rel_pos_lambda: np.ndarray,
                    radar_position: np.ndarray,
                    x_hat: np.ndarray, y_hat: np.ndarray, z_hat: np.ndarray,
                    shift_vec: np.ndarray | None = None) -> np.ndarray:
    off = np.zeros(3, dtype=float) if shift_vec is None else shift_vec
    return np.array([
        radar_position + off + (p[0] * lambda_m) * x_hat + (p[1] * lambda_m) * z_hat
        for p in rel_pos_lambda
    ], dtype=float)


def serialize_single_chip_config(tx_world: np.ndarray,
                     rx_world: np.ndarray,
                     boresight: np.ndarray,
                     waveform_params: dict) -> dict:
    return {
        "modulation_scheme": "FMCW",
        "carrierFrequency": 76999999488.0,
        "sampleRate": 10666000.0,
        "freqSlope": 100000000377000.0,
        "numAdcSamples": 128,
        "adcStartTime": 7.00000009601e-06,
        "rampEndTime": 1.99999994948e-05,
        "tx_array": [
            {"name": f"TX{i+1}", "pos_mm": list((tx_world[i] * 1000.0).round(6)), "boresight": list(np.asarray(boresight, float).round(8)), "radius": 1.0, "polarization": "V"}
            for i in range(len(tx_world))
        ],
        "rx_array": [
            {"name": f"RX{i+1}", "pos_mm": list((rx_world[i] * 1000.0).round(6)), "boresight": list(np.asarray(boresight, float).round(8)), "power_dBm": 10.0, "polarization": "V"}
            for i in range(len(rx_world))
        ],
    }

def serialize_cascade_config(tx_world: np.ndarray,
                     rx_world: np.ndarray,
                     boresight: np.ndarray,
                     waveform_params: dict) -> dict:
    return {
        "modulation_scheme": "FMCW",
        "carrierFrequency": 76999999488.0,
        "sampleRate": 8000000.0,
        "freqSlope": 79000001052700.0,
        "numAdcSamples": 256,
        "adcStartTime": 2e-06,
        "rampEndTime": 34e-06,
        "tx_array": [
            {"name": f"TX{i+1}", "pos_mm": list((tx_world[i] * 1000.0).round(6)), "boresight": list(np.asarray(boresight, float).round(8)), "radius": 1.0, "polarization": "V"}
            for i in range(len(tx_world))
        ],
        "rx_array": [
            {"name": f"RX{i+1}", "pos_mm": list((rx_world[i] * 1000.0).round(6)), "boresight": list(np.asarray(boresight, float).round(8)), "power_dBm": 10.0, "polarization": "V"}
            for i in range(len(rx_world))
        ],
    }


def save_config(base_dir: str, kind: str, frame_idx: int, cfg: dict) -> str:
    cfg_dir = get_subdir(base_dir, "configs")
    path = os.path.join(cfg_dir, f"{kind}_frame_{frame_idx}.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def generate_config_window(*,
                           seq_idx: int,
                           center_frame_idx: int,
                           num_radar_frames: int,
                           dataset_dir: str,
                           calib_path: str,
                           out_root: str,
                           run_cascade: bool,
                           run_single: bool,
                           apply_adjust: bool = False,
                           verbose: bool = False) -> dict:
    if num_radar_frames < 1 or (num_radar_frames % 2) != 1:
        raise ValueError("num_radar_frames must be odd and >= 1")

    base_dir = get_base_dir(seq_idx, center_frame_idx, out_root)
    half = num_radar_frames // 2
    cascade_indices = list(range(center_frame_idx - half, center_frame_idx + half + 1))

    results = {"cascade": [], "single_chip": []}

    def _basis(boresight):
        return basis_from_boresight(boresight)

    # rotation/shift params (match prior defaults)
    dy_px, dx_px = (0, 14)

    for c_idx in cascade_indices:
        if run_cascade:
            pos, bore = compute_world_pose_and_boresight("cascade", seq_idx, c_idx, dataset_dir, calib_path)
            x_hat, y_hat, z_hat = _basis(bore)
            shift_vec = np.zeros(3, dtype=float)
            rel_rx, rel_tx = antenna_layout_cascade()
            rx_w = to_world_coords(rel_rx, pos, x_hat, y_hat, z_hat, shift_vec)
            tx_w = to_world_coords(rel_tx, pos, x_hat, y_hat, z_hat, shift_vec)
            cfg = serialize_cascade_config(tx_w, rx_w, bore, {})
            results["cascade"].append(save_config(base_dir, "cascaded", c_idx, cfg))

        if run_single:
            # map cascade->single idx
            seq_path = dataset_dir + str(seq_idx)
            cas_params = get_cascade_params(calib_path)["heatmap"]
            sc_params = get_single_chip_params(calib_path)["heatmap"]
            cas_ts = np.asarray(get_timestamps(seq_path, cas_params), dtype=float)
            sc_ts = np.asarray(get_timestamps(seq_path, sc_params), dtype=float)
            ref = cas_ts[c_idx]; sc_idx = int(np.argmin(np.abs(sc_ts - ref)))

            pos, bore = compute_world_pose_and_boresight("single", seq_idx, sc_idx, dataset_dir, calib_path)
            x_hat, y_hat, z_hat = _basis(bore)
            shift_vec = np.zeros(3, dtype=float)
            rel_rx, rel_tx = antenna_layout_single_chip()
            rx_w = to_world_coords(rel_rx, pos, x_hat, y_hat, z_hat, shift_vec)
            tx_w = to_world_coords(rel_tx, pos, x_hat, y_hat, z_hat, shift_vec)
            cfg = serialize_single_chip_config(tx_w, rx_w, bore, {})
            results["single_chip"].append(save_config(base_dir, "single_chip", sc_idx, cfg))

    if verbose:
        print(f"[configs] wrote {len(results['cascade'])} cascade and {len(results['single_chip'])} single-chip configs")
    return results
