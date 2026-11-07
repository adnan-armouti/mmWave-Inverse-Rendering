"""Figure generation utilities for paper publication."""

from typing import Dict, List, Optional, Tuple


def format_metric(val: Optional[float], fmt: str = ".3f") -> str:
    """Format a single metric value for LaTeX."""
    if val is None:
        return "---"
    return f"{val:{fmt}}"


_RA_COLUMNS = [("corr", ".3f", r"RA Correlation $\uparrow$"), ("psnr", ".1f", r"RA PSNR $\uparrow$"),
               ("ssim", ".3f", r"RA SSIM $\uparrow$"), ("rmse", ".4f", r"RA RMSE $\downarrow$"),
               ("adc_log_mag_mse", ".4f", r"ADC Log-Mag MSE $\downarrow$"),
               ("adc_minmax_mag_mse", ".4f", r"ADC MM-Mag MSE $\downarrow$")]


def _per_scene_table(scene_metrics, aggregate, output_path, keys, caption, label):
    """One row per scene plus a mean and standard deviation row, for the metric keys in ``keys``."""
    lines = [
        r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l" + " c" * len(keys) + "}", r"\toprule",
        "Scene & " + " & ".join(title for _, _, title in _RA_COLUMNS) + r" \\", r"\midrule",
    ]
    for scene_name, m in sorted(scene_metrics.items()):
        short = scene_name.replace("seq_", "S").replace("_frame_", "F")
        cells = [format_metric(m.get(k), fmt) for k, (_, fmt, _) in zip(keys, _RA_COLUMNS)]
        lines.append(f"  {short} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    cells = []
    for k, (_, fmt, _) in zip(keys, _RA_COLUMNS):
        a = aggregate.get(k)
        cells.append(f"{a['mean']:{fmt}}$\\pm${a['std']:{fmt}}" if a else "---")
    lines.append("  Mean & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    return output_path


def generate_training_ra_table(scene_metrics: Dict[str, dict], aggregate: dict, output_path: str):
    """Training RA metrics per scene (paper Table 1)."""
    keys = ["ours_corr", "ours_psnr", "ours_ssim", "ours_rmse", "ours_adc_log_mag_mse", "ours_adc_minmax_mag_mse"]
    return _per_scene_table(scene_metrics, aggregate, output_path, keys,
                            "Training-view RA and ADC metrics.", "tab:training_ra")


def generate_radar_transfer_table(scene_metrics: Dict[str, dict], aggregate: dict, output_path: str):
    """Cascade-to-single-chip transfer metrics per scene (paper Table 2)."""
    keys = ["transfer_cart_corr", "transfer_cart_psnr", "transfer_cart_ssim", "transfer_cart_rmse",
            "transfer_adc_log_mag_mse", "transfer_adc_minmax_mag_mse"]
    return _per_scene_table(
        scene_metrics, aggregate, output_path, keys,
        r"Cross-sensor transfer evaluation. Scene parameters optimized on the cascaded radar "
        r"(12TX$\times$16RX) are frozen and used to render ADC for the co-located single-chip radar "
        r"(3TX$\times$4RX) without re-training.", "tab:radar_transfer")


def bold_best(values: List[Optional[float]], fmt: str = ".3f", higher_better: bool = True) -> List[str]:
    """Format a list of values, bolding the best one."""
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        return [format_metric(v, fmt) for v in values]

    best_idx = max(valid, key=lambda x: x[1] if higher_better else -x[1])[0]
    result = []
    for i, v in enumerate(values):
        s = format_metric(v, fmt)
        if i == best_idx and v is not None:
            s = r"\textbf{" + s + "}"
        result.append(s)
    return result


def bold_best_agg(
    agg: dict, key_a: str, key_b: str, fmt: str = ".3f", higher_better: bool = True,
) -> Tuple[str, str]:
    """Format two aggregate mean±std values, bolding the better one (by mean)."""
    def _fmt(key):
        if key in agg:
            return f"{agg[key]['mean']:{fmt}}$\\pm${agg[key]['std']:{fmt}}"

    a_mean = agg.get(key_a, {}).get("mean")
    b_mean = agg.get(key_b, {}).get("mean")

    a_str = _fmt(key_a)
    b_str = _fmt(key_b)

    if a_mean is not None and b_mean is not None:
        if higher_better:
            a_str = r"\textbf{" + a_str + "}"
        else:
            a_str = r"\textbf{" + a_str + "}"

    return a_str, b_str


def generate_3d_occupancy_table(
    scene_metrics: Dict[str, dict],
    aggregate: dict,
    output_path: str,
):
    """Generate Table 3: 3D occupancy metrics (Ours vs ColoRadar CASCADE)."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{3D occupancy evaluation against LiDAR ground truth. Baseline: ColoRadar-provided 3D point cloud from the cascaded radar's near-1D virtual aperture, where elevation is synthesized by extruding range--azimuth detections along the vertical axis due to minimal elevation resolution. Ours: dense virtual array (100TX$\times$100RX) with full 2D aperture enabling range--azimuth--elevation (RAE) imaging. Occupancy metrics use KD-tree nearest neighbors ($\tau{=}0.5$\,m). Best in \textbf{bold}.}",
        r"\label{tab:3d_occupancy}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{l cc cc cc cc cc cc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Precision $\uparrow$}"
        r" & \multicolumn{2}{c}{Recall $\uparrow$}"
        r" & \multicolumn{2}{c}{F1 $\uparrow$}"
        r" & \multicolumn{2}{c}{Accuracy $\uparrow$}"
        r" & \multicolumn{2}{c}{RMSE $\downarrow$}"
        r" & \multicolumn{2}{c}{R-CD $\downarrow$} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}"
        r" \cmidrule(lr){8-9} \cmidrule(lr){10-11} \cmidrule(lr){12-13}",
        r"Scene & Ours & ColoR & Ours & ColoR & Ours & ColoR"
        r" & Ours & ColoR & Ours & ColoR & Ours & ColoR \\",
        r"\midrule",
    ]

    # Collect per-scene F1 values for aggregate computation
    ours_f1_vals = []
    color_f1_vals = []

    for scene_name, m in sorted(scene_metrics.items()):
        short = scene_name.replace("seq_", "S").replace("_frame_", "F")

        # Ours
        o_prec = m.get("distance_precision")
        o_rec = m.get("distance_recall")
        o_acc = m.get("distance_accuracy")
        o_rmse = m.get("pc_rmse")
        o_rcd = m.get("relative_chamfer_distance")
        o_p = o_prec if o_prec is not None else 0.0
        o_r = o_rec if o_rec is not None else 0.0
        o_f1 = 2 * o_p * o_r / (o_p + o_r) if (o_p + o_r) > 0 else None

        # ColoRadar
        c_prec = m.get("coloradar_distance_precision")
        c_rec = m.get("coloradar_distance_recall")
        c_acc = m.get("coloradar_distance_accuracy")
        c_rmse = m.get("coloradar_pc_rmse")
        c_rcd = m.get("coloradar_relative_chamfer_distance")
        c_p = c_prec if c_prec is not None else 0.0
        c_r = c_rec if c_rec is not None else 0.0
        c_f1 = 2 * c_p * c_r / (c_p + c_r) if (c_p + c_r) > 0 else None

        if o_f1 is not None:
            ours_f1_vals.append(o_f1)
        if c_f1 is not None:
            color_f1_vals.append(c_f1)

        prec_o, prec_c = bold_best([o_prec, c_prec], ".3f", higher_better=True)
        rec_o, rec_c = bold_best([o_rec, c_rec], ".3f", higher_better=True)
        f1_o, f1_c = bold_best([o_f1, c_f1], ".3f", higher_better=True)
        acc_o, acc_c = bold_best([o_acc, c_acc], ".3f", higher_better=True)
        rmse_o, rmse_c = bold_best([o_rmse, c_rmse], ".3f", higher_better=False)
        rcd_o, rcd_c = bold_best([o_rcd, c_rcd], ".3f", higher_better=False)

        lines.append(
            f"  {short} & {prec_o} & {prec_c}"
            f" & {rec_o} & {rec_c}"
            f" & {f1_o} & {f1_c}"
            f" & {acc_o} & {acc_c}"
            f" & {rmse_o} & {rmse_c}"
            f" & {rcd_o} & {rcd_c} \\\\"
        )

    # Bolded Mean±Std row
    lines.append(r"\midrule")

    a_prec, b_prec = bold_best_agg(aggregate, "distance_precision", "coloradar_distance_precision", ".3f", higher_better=True)
    a_rec, b_rec = bold_best_agg(aggregate, "distance_recall", "coloradar_distance_recall", ".3f", higher_better=True)
    a_acc, b_acc = bold_best_agg(aggregate, "distance_accuracy", "coloradar_distance_accuracy", ".3f", higher_better=True)
    a_rmse, b_rmse = bold_best_agg(aggregate, "pc_rmse", "coloradar_pc_rmse", ".3f", higher_better=False)
    a_rcd, b_rcd = bold_best_agg(aggregate, "relative_chamfer_distance", "coloradar_relative_chamfer_distance", ".3f", higher_better=False)

    # Compute F1 aggregate with bolding
    import numpy as _np
    ours_f1_mean = float(_np.mean(ours_f1_vals))
    ours_f1_str = f"{ours_f1_mean:.3f}$\\pm${float(_np.std(ours_f1_vals)):.3f}"
    color_f1_mean = float(_np.mean(color_f1_vals))
    color_f1_str = f"{color_f1_mean:.3f}$\\pm${float(_np.std(color_f1_vals)):.3f}"
    if ours_f1_mean is not None and color_f1_mean is not None:
        ours_f1_str = r"\textbf{" + ours_f1_str + "}"

    lines.append(
        f"  Mean & {a_prec} & {b_prec}"
        f" & {a_rec} & {b_rec}"
        f" & {ours_f1_str} & {color_f1_str}"
        f" & {a_acc} & {b_acc}"
        f" & {a_rmse} & {b_rmse}"
        f" & {a_rcd} & {b_rcd} \\\\"
    )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    return output_path
