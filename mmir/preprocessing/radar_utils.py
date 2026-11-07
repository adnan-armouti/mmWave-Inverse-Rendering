import os
import numpy as np

from mmir.preprocessing.ColoRadar_tools.dataset_loaders import (
    get_cascade_params,
    get_single_chip_params,
    get_adc_frame,
    get_timestamps,
)

from typing import Optional
import math


def map_cascade_to_single_chip(seq_path: str,
                               calib_path: str,
                               cascade_frame_idx: int,
                               max_time_diff: float | None = 0.5) -> int:
    cas_params = get_cascade_params(calib_path)['heatmap']
    sc_params  = get_single_chip_params(calib_path)['heatmap']
    cas_ts = np.asarray(get_timestamps(seq_path, cas_params), dtype=float)
    sc_ts  = np.asarray(get_timestamps(seq_path, sc_params), dtype=float)
    if cascade_frame_idx < 0 or cascade_frame_idx >= len(cas_ts):
        raise IndexError("cascade_frame_idx out of range")
    ref = cas_ts[cascade_frame_idx]
    diffs = np.abs(sc_ts - ref)
    sc_idx = int(np.argmin(diffs))
    if (max_time_diff is not None) and (diffs[sc_idx] > max_time_diff):
        raise ValueError(
            f"No single\u2011chip frame within {max_time_diff:.3f}s (closest \u0394t {float(diffs[sc_idx]):.3f}s)"
        )
    return sc_idx


def _save_adc(base_dir: str, kind: str, frame_idx: int, adc: np.ndarray) -> str:
    radar_dir = get_subdir(base_dir, "radar")
    fname = f"{kind}_frame_{frame_idx}.npy"
    path = os.path.join(radar_dir, fname)
    np.save(path, adc)
    return path


def generate_adc_window(*,
                        seq_idx: int,
                        center_frame_idx: int,
                        num_radar_frames: int,
                        dataset_dir: str,
                        calib_path: str,
                        out_root: str,
                        run_cascade: bool,
                        run_single: bool,
                        verbose: bool = False) -> dict:
    if num_radar_frames < 1 or (num_radar_frames % 2) != 1:
        raise ValueError("num_radar_frames must be odd and >= 1")

    seq_path = dataset_dir + str(seq_idx)
    base_dir = get_base_dir(seq_idx, center_frame_idx, out_root)

    half = num_radar_frames // 2
    cascade_indices = list(range(center_frame_idx - half, center_frame_idx + half + 1))

    results = {"cascade_adc": [], "single_chip_adc": []}

    if run_cascade:
        all_params = get_cascade_params(calib_path)
        wf = all_params["waveform"]
        for c_idx in cascade_indices:
            adc = get_adc_frame(c_idx, seq_path, wf)
            adc = apply_frequency_calibration(adc, all_params["frequency"], wf)  # freq only (matches CoIR)
            adc = adc.transpose(2, 1, 0, 3)  # (N_C, N_Rx, N_Tx, N_ADC)
            if verbose:
                print(f"[adc] cascade idx={c_idx} shape={adc.shape}")
            results["cascade_adc"].append(_save_adc(base_dir, "cascaded", c_idx, adc))

    if run_single:
        sc_params = get_single_chip_params(calib_path)
        wf_sc = sc_params["waveform"]
        for c_idx in cascade_indices:
            sc_idx = map_cascade_to_single_chip(seq_path, calib_path, c_idx, max_time_diff=0.5)
            adc = get_adc_frame(sc_idx, seq_path, wf_sc)
            adc = adc.transpose(2, 1, 0, 3)
            if verbose:
                print(f"[adc] single idx={sc_idx} (from cascade {c_idx}) shape={adc.shape}")
            results["single_chip_adc"].append(_save_adc(base_dir, "single_chip", sc_idx, adc))

    return results


def get_base_dir(seq_idx: int, frame_idx: int, out_root: str) -> str:
    return os.path.join(out_root, f"seq_{seq_idx}_frame_{frame_idx}")


def get_subdir(base_dir: str, name: str) -> str:
    path = os.path.join(base_dir, name)
    os.makedirs(path, exist_ok=True)
    return path


def find_lowest_cascaded_config(base_dir: str) -> Optional[str]:
    cfg_dir = os.path.join(base_dir, "configs")
    if not os.path.isdir(cfg_dir):
        return None
    try:
        entries = os.listdir(cfg_dir)
        candidates: list[tuple[int, str]] = []
        for name in entries:
            if name.startswith("cascaded_frame_") and name.endswith(".json"):
                try:
                    idx = int(name[len("cascaded_frame_"):-len(".json")])
                    candidates.append((idx, os.path.join(cfg_dir, name)))
                except Exception:
                    continue
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    except Exception:
        return None


def apply_frequency_calibration(adc_samples, freq_calib, wave_config):

  num_tx = freq_calib['num_tx']
  num_rx = freq_calib['num_rx']
  cal_frequency_slope = freq_calib['frequency_slope']
  cal_sampling_rate = freq_calib['sampling_rate']
  frequency_slope = wave_config['frequency_slope']
  sampling_rate = wave_config['adc_sample_frequency']
  num_adc_samples_per_chirp = wave_config['num_adc_samples_per_chirp']
  freq_cal_mat = freq_calib['cal_matrix']

  # create frequency calibration matrix
  delta_p = freq_cal_mat - freq_cal_mat[0,0]
  freq_calib = (2.0 * math.pi * delta_p
                * (frequency_slope / cal_frequency_slope)
                * (cal_sampling_rate / sampling_rate))
  freq_calib /= num_adc_samples_per_chirp
  freq_calib_vec = np.expand_dims(freq_calib, -1) * np.arange(num_adc_samples_per_chirp)
  freq_calib_vec = np.exp(-1j * freq_calib_vec)

  # apply frequency calibration
  freq_calibrated_adc_samples = freq_calib_vec.reshape(num_tx,num_rx,1,num_adc_samples_per_chirp) * adc_samples

  return freq_calibrated_adc_samples
