import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_plot_style() -> None:
    """Configures clean, publication-ready Matplotlib aesthetics."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "axes.edgecolor": "#cccccc",
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def extract_std_series(
    df: pd.DataFrame,
    mean_col: str,
    std_col: str,
    warn_context: str = ""
) -> np.ndarray:
    """
    Retrieves the standard deviation column `std_col` from `df` if available; otherwise
    falls back to a 5-step rolling standard deviation of `mean_col` so existing CSVs
    without explicit std columns can still be plotted.

    Args:
        df (pd.DataFrame): Input metrics dataframe.
        mean_col (str): Name of the mean metric column.
        std_col (str): Name of the desired standard deviation column.
        warn_context (str): Optional context string printed once if fallback is triggered.

    Returns:
        np.ndarray: 1D float array of standard deviation values matching `len(df)`.
    """
    if std_col in df.columns and df[std_col].notna().any():
        return df[std_col].fillna(0.0).to_numpy(dtype=float)

    if warn_context:
        print(
            f"  [Note] '{std_col}' not found in {warn_context}; "
            f"approximating SD via 5-step rolling std of '{mean_col}'."
        )
    return df[mean_col].rolling(window=5, min_periods=1, center=True).std().fillna(0.0).to_numpy(dtype=float)


def get_power_of_two_indices(steps: np.ndarray, min_step: int = 2) -> np.ndarray:
    """
    Finds array indices where `steps` is a power of two (>= min_step) or the final step.

    Args:
        steps (np.ndarray): 1D array of integer/float step numbers.
        min_step (int): Minimum power-of-two step to include.

    Returns:
        np.ndarray: 1D integer array of matching indices into `steps`.
    """
    int_steps = np.rint(steps).astype(int)
    max_s = int(int_steps.max()) if len(int_steps) > 0 else 0
    mask = ((int_steps >= min_step) & ((int_steps & (int_steps - 1)) == 0)) | (int_steps == max_s)
    return np.nonzero(mask)[0]


def create_time_secondary_xaxis(
    ax: plt.Axes,
    steps: np.ndarray,
    times_sec: np.ndarray,
    label: str = "Elapsed Optimization Time (s)"
) -> Optional[plt.Axes]:
    """
    Attaches a secondary top x-axis that monotonically maps optimization steps to elapsed seconds.

    Args:
        ax (plt.Axes): Primary matplotlib axis.
        steps (np.ndarray): 1D array of step numbers.
        times_sec (np.ndarray): 1D array of corresponding cumulative elapsed seconds.
        label (str): Label for the secondary top x-axis.

    Returns:
        Optional[plt.Axes]: The secondary axis, or None if arrays have fewer than 2 distinct points.
    """
    valid_mask = (~np.isnan(steps)) & (~np.isnan(times_sec))
    s_clean = steps[valid_mask]
    t_clean = times_sec[valid_mask]

    if len(s_clean) < 2 or (t_clean[-1] - t_clean[0]) <= 0:
        return None

    s_uniq, idx = np.unique(s_clean, return_index=True)
    t_uniq = t_clean[idx]

    def forward_step_to_time(x: np.ndarray) -> np.ndarray:
        return np.interp(x, s_uniq, t_uniq)

    def inverse_time_to_step(t: np.ndarray) -> np.ndarray:
        return np.interp(t, t_uniq, s_uniq)

    sec_ax = ax.secondary_xaxis("top", functions=(forward_step_to_time, inverse_time_to_step))
    sec_ax.set_xlabel(label, color="#444444", labelpad=8)
    sec_ax.tick_params(colors="#444444")
    return sec_ax


def plot_local_stage2a(
    df_2a: pd.DataFrame,
    s1_stem: str,
    output_dir: Path
) -> Path:
    """
    Generates the local Phase 2A reconstruction trajectory for a single Stage 1 checkpoint,
    displaying both batch PSNR +/- 1 SD and full-volume PSNR +/- 1 SD at checkpoints.

    Args:
        df_2a (pd.DataFrame): Dataframe loaded from `stage2a_metrics_log.csv`.
        s1_stem (str): Name of the Stage 1 checkpoint (e.g. 'stage1_loop_0064').
        output_dir (Path): Output directory for saved plots.

    Returns:
        Path: Path to the generated PNG plot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "stage2a_recon_trajectory.png"

    steps = df_2a["fit_step"].to_numpy(dtype=float)
    times = df_2a["elapsed_stage2a_opt_time_sec"].to_numpy(dtype=float)
    mean_psnr = df_2a["mean_batch_psnr_db"].to_numpy(dtype=float)
    std_psnr = extract_std_series(df_2a, "mean_batch_psnr_db", "std_batch_psnr_db")

    fig, ax1 = plt.subplots(figsize=(9.5, 5.5))

    ax1.plot(
        steps, mean_psnr,
        color="#1f77b4", linewidth=2.2,
        label=f"Mean Batch PSNR (Final: {mean_psnr[-1]:.2f}±{std_psnr[-1]:.2f} dB)"
    )
    ax1.fill_between(
        steps, mean_psnr - std_psnr, mean_psnr + std_psnr,
        color="#1f77b4", alpha=0.18, label=r"Batch PSNR $\pm 1\sigma$"
    )

    if "mean_full_vol_psnr_db" in df_2a.columns:
        valid_full = df_2a["mean_full_vol_psnr_db"].dropna()
        if not valid_full.empty:
            full_steps = df_2a.loc[valid_full.index, "fit_step"].to_numpy(dtype=float)
            full_vals = valid_full.to_numpy(dtype=float)
            if "std_full_vol_psnr_db" in df_2a.columns:
                full_stds = df_2a.loc[valid_full.index, "std_full_vol_psnr_db"].fillna(0.0).to_numpy(dtype=float)
            else:
                full_stds = np.zeros_like(full_vals)

            ax1.errorbar(
                full_steps, full_vals, yerr=full_stds,
                fmt="D", color="#d62728", markersize=5.5, capsize=3.0, elinewidth=1.2,
                zorder=5, label=r"Full-Volume Eval $\pm 1\sigma$ (dB)"
            )
            for s_val, f_val, f_std in zip(full_steps, full_vals, full_stds):
                ax1.annotate(
                    f"{f_val:.1f}±{f_std:.1f}", (s_val, f_val),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8.0, color="#990000", weight="bold"
                )

    ax1.set_xlabel("Reconstruction Fitting Steps ($T_f$)")
    ax1.set_ylabel("Reconstruction PSNR (dB)")
    ax1.set_title(f"Phase 2A Pixel-Fitting Trajectory\n[Stage 1: {s1_stem}]", pad=28)
    ax1.legend(loc="lower right", framealpha=0.9)

    create_time_secondary_xaxis(ax1, steps, times, label="Elapsed Phase 2A Time (s)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=250)
    plt.close(fig)
    return save_path


def plot_local_stage2b(
    s2b_runs: Dict[str, pd.DataFrame],
    s1_stem: str,
    output_dir: Path,
    dice_ylim: Tuple[float, float] = (0.60, 1.00)
) -> Path:
    """
    Generates the local Phase 2B segmentation convergence plot comparing different Phase 2A stops,
    with narrowed y-limits, staggered power-of-two error bars, and peak Mean +/- SD in the legend.

    Args:
        s2b_runs (Dict[str, pd.DataFrame]): Mapping from 2A checkpoint stem (e.g. 'stage2a_step_0064')
            to its `stage2b_metrics_log.csv` dataframe.
        s1_stem (str): Name of the Stage 1 checkpoint.
        output_dir (Path): Output directory for saved plots.
        dice_ylim (Tuple[float, float]): Y-axis limits `(ymin, ymax)` for Dice scores.

    Returns:
        Path: Path to the generated PNG plot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "stage2b_dice_trajectories.png"

    fig, ax1 = plt.subplots(figsize=(10.5, 6))
    num_curves = max(1, len(s2b_runs))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, num_curves))

    max_time_series: Optional[np.ndarray] = None
    max_loops_series: Optional[np.ndarray] = None

    # Horizontal stagger offsets so error bars at power-of-two loops do not overlap
    stagger_offsets = np.linspace(-0.8, 0.8, num_curves)

    for idx_c, ((s2a_name, df_sub), color) in enumerate(zip(sorted(s2b_runs.items()), colors)):
        fit_steps_val = df_sub["stage2a_fit_steps"].iloc[0] if "stage2a_fit_steps" in df_sub.columns else s2a_name
        loops = df_sub["head_loop"].to_numpy(dtype=float)
        dice_all = df_sub["mean_batch_dice_all"].to_numpy(dtype=float)
        dice_fg = df_sub["mean_batch_dice_fg"].to_numpy(dtype=float)
        std_fg = extract_std_series(
            df_sub, "mean_batch_dice_fg", "std_batch_dice_fg",
            warn_context=f"{s1_stem}/{s2a_name}" if idx_c == 0 else ""
        )

        best_idx = int(np.argmax(dice_fg))
        peak_mean = dice_fg[best_idx]
        peak_std = std_fg[best_idx]

        ax1.plot(
            loops, dice_fg,
            color=color, linewidth=2.0,
            label=f"$T_f={fit_steps_val}$ (Peak FG: {peak_mean:.3f}±{peak_std:.3f})"
        )
        ax1.fill_between(
            loops, dice_fg - std_fg, dice_fg + std_fg,
            color=color, alpha=0.08
        )

        # Plot staggered error bars at power-of-two loops (2, 4, 8, 16, 32, 64, 128)
        pow2_idx = get_power_of_two_indices(loops, min_step=2)
        if len(pow2_idx) > 0:
            ax1.errorbar(
                loops[pow2_idx] + stagger_offsets[idx_c],
                dice_fg[pow2_idx],
                yerr=std_fg[pow2_idx],
                fmt="o", color=color, markersize=3.5, capsize=2.5,
                elinewidth=1.0, alpha=0.80
            )

        ax1.plot(
            loops, dice_all,
            color=color, linewidth=1.1, linestyle=":", alpha=0.55
        )

        times = df_sub["elapsed_stage2b_opt_time_sec"].to_numpy(dtype=float)
        if max_time_series is None or len(times) > len(max_time_series):
            max_time_series = times
            max_loops_series = loops

    ax1.set_xlabel("Head Optimization Loops ($E_{\mathrm{head}}$)")
    ax1.set_ylabel("Macro Dice Score")
    ax1.set_ylim(dice_ylim[0], dice_ylim[1])
    ax1.set_title(
        f"Phase 2B Segmentation Head Convergence vs. 2A Fitting Steps ($\pm 1\sigma$)\n[Stage 1: {s1_stem}]",
        pad=28
    )

    ax1.plot([], [], color="#555555", linewidth=2.0, label="Foreground Dice (Solid)")
    ax1.plot([], [], color="#555555", linewidth=1.1, linestyle=":", label="All-Class Dice (Dotted)")

    ax1.legend(loc="lower right", framealpha=0.92, ncol=2, fontsize=8.5)

    if max_loops_series is not None and max_time_series is not None:
        create_time_secondary_xaxis(ax1, max_loops_series, max_time_series, label="Elapsed Head Optimization Time (s)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=250)
    plt.close(fig)
    return save_path


def plot_global_stage2a(
    all_2a_dfs: Dict[str, pd.DataFrame],
    output_dir: Path
) -> Path:
    """
    Plots Phase 2A pixel reconstruction trajectories jointly across all Stage 1 checkpoints
    with light +/- 1 SD ribbons and staggered error bars at power-of-two steps.

    Args:
        all_2a_dfs (Dict[str, pd.DataFrame]): Mapping from Stage 1 checkpoint stem to its 2A dataframe.
        output_dir (Path): Output directory for global plots.

    Returns:
        Path: Path to the generated PNG plot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "global_stage2a_recon_comparison.png"

    fig, ax1 = plt.subplots(figsize=(10.5, 6))

    def sort_key(item: Tuple[str, pd.DataFrame]) -> int:
        df = item[1]
        return int(df["stage1_dataset_loops"].iloc[0]) if "stage1_dataset_loops" in df.columns else 0

    sorted_items = sorted(all_2a_dfs.items(), key=sort_key)
    num_curves = max(1, len(sorted_items))
    colors = plt.cm.viridis(np.linspace(0.1, 0.95, num_curves))
    stagger_offsets = np.linspace(-1.2, 1.2, num_curves)

    ref_steps: Optional[np.ndarray] = None
    all_times: List[np.ndarray] = []

    for idx_c, ((s1_stem, df_2a), color) in enumerate(zip(sorted_items, colors)):
        s1_loop = df_2a["stage1_dataset_loops"].iloc[0] if "stage1_dataset_loops" in df_2a.columns else s1_stem
        steps = df_2a["fit_step"].to_numpy(dtype=float)
        psnr = df_2a["mean_batch_psnr_db"].to_numpy(dtype=float)
        std_psnr = extract_std_series(df_2a, "mean_batch_psnr_db", "std_batch_psnr_db")
        times = df_2a["elapsed_stage2a_opt_time_sec"].to_numpy(dtype=float)

        ax1.plot(
            steps, psnr, color=color, linewidth=2.0,
            label=f"Stage 1 Loop {s1_loop} (End: {psnr[-1]:.1f}±{std_psnr[-1]:.1f} dB)"
        )
        ax1.fill_between(steps, psnr - std_psnr, psnr + std_psnr, color=color, alpha=0.08)

        pow2_idx = get_power_of_two_indices(steps, min_step=2)
        if len(pow2_idx) > 0:
            ax1.errorbar(
                steps[pow2_idx] + stagger_offsets[idx_c],
                psnr[pow2_idx],
                yerr=std_psnr[pow2_idx],
                fmt="o", color=color, markersize=3.5, capsize=2.5,
                elinewidth=1.0, alpha=0.80
            )

        if ref_steps is None or len(steps) > len(ref_steps):
            ref_steps = steps
        all_times.append(times)

    ax1.set_xlabel("Phase 2A Reconstruction Steps ($T_f$)")
    ax1.set_ylabel("Reconstruction PSNR (dB)")
    ax1.set_title("Global Phase 2A Adaptation across Stage 1 Checkpoints ($\pm 1\sigma$)", pad=28)
    ax1.legend(loc="lower right", framealpha=0.92, title="Meta-Training Horizon (Final PSNR ± SD)")

    if ref_steps is not None and all_times:
        min_len = min(len(t) for t in all_times)
        mean_times = np.mean([t[:min_len] for t in all_times], axis=0)
        create_time_secondary_xaxis(ax1, ref_steps[:min_len], mean_times, label="Mean Elapsed Fitting Time (s)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=250)
    plt.close(fig)
    return save_path


def plot_global_stage2b(
    all_2b_runs: Dict[str, Dict[str, pd.DataFrame]],
    output_dir: Path,
    dice_ylim: Tuple[float, float] = (0.60, 1.00)
) -> Path:
    """
    Plots a multi-panel grid comparing Phase 2B Dice scores across all Stage 1 checkpoints
    with a narrowed y-axis range [0.60, 1.00], staggered power-of-two error bars, and
    peak Mean +/- SD in each subplot's legend.

    Args:
        all_2b_runs (Dict[str, Dict[str, pd.DataFrame]]): Nested dict mapping
            `s1_stem -> {s2a_stem -> df_2b}`.
        output_dir (Path): Output directory for global plots.
        dice_ylim (Tuple[float, float]): Y-axis limits `(ymin, ymax)` for Dice scores.

    Returns:
        Path: Path to the generated PNG plot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "global_stage2b_dice_comparison.png"

    s1_keys = sorted(all_2b_runs.keys())
    n_s1 = len(s1_keys)
    if n_s1 == 0:
        return save_path

    cols = min(3, n_s1)
    rows = math.ceil(n_s1 / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5.8 * cols, 4.5 * rows), sharey=True, squeeze=False)
    fig.suptitle(
        "Global Phase 2B Head Convergence across Stage 1 Checkpoints ($\pm 1\sigma$)",
        fontsize=14, y=0.99
    )

    for idx, s1_key in enumerate(s1_keys):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        runs = all_2b_runs[s1_key]

        num_curves = max(1, len(runs))
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, num_curves))
        stagger_offsets = np.linspace(-0.8, 0.8, num_curves)

        for idx_c, ((s2a_stem, df_sub), color) in enumerate(zip(sorted(runs.items()), colors)):
            fit_steps = df_sub["stage2a_fit_steps"].iloc[0] if "stage2a_fit_steps" in df_sub.columns else s2a_stem
            loops = df_sub["head_loop"].to_numpy(dtype=float)
            dice_fg = df_sub["mean_batch_dice_fg"].to_numpy(dtype=float)
            std_fg = extract_std_series(df_sub, "mean_batch_dice_fg", "std_batch_dice_fg")

            best_i = int(np.argmax(dice_fg))
            peak_m, peak_s = dice_fg[best_i], std_fg[best_i]

            ax.plot(
                loops, dice_fg, color=color, linewidth=1.7,
                label=f"$T_f={fit_steps}$ ({peak_m:.3f}±{peak_s:.3f})"
            )
            ax.fill_between(loops, dice_fg - std_fg, dice_fg + std_fg, color=color, alpha=0.08)

            pow2_idx = get_power_of_two_indices(loops, min_step=2)
            if len(pow2_idx) > 0:
                ax.errorbar(
                    loops[pow2_idx] + stagger_offsets[idx_c],
                    dice_fg[pow2_idx],
                    yerr=std_fg[pow2_idx],
                    fmt="o", color=color, markersize=3.0, capsize=2.0,
                    elinewidth=0.9, alpha=0.80
                )

        ax.set_title(f"Stage 1: {s1_key}", fontsize=11)
        ax.set_xlabel("Head Loops ($E_{\mathrm{head}}$)")
        if c == 0:
            ax.set_ylabel("Foreground Dice")
        ax.set_ylim(dice_ylim[0], dice_ylim[1])
        ax.legend(fontsize=7.2, loc="lower right", framealpha=0.88)

    for idx in range(n_s1, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=250)
    plt.close(fig)
    return save_path


def plot_global_phase2_heatmap(
    summary_csv_path: Path,
    all_2b_runs: Dict[str, Dict[str, pd.DataFrame]],
    output_dir: Path
) -> Path:
    """
    Builds a 2D parameter heatmap of (Stage 1 Loops x Stage 2A Fitting Steps),
    displaying the peak Phase 2B Foreground Dice score (bold) and its standard
    deviation (smaller font underneath) inside each cell.

    Args:
        summary_csv_path (Path): Path to `stage2b_all_runs_summary.csv` if it exists.
        all_2b_runs (Dict[str, Dict[str, pd.DataFrame]]): Nested metric dataframes.
        output_dir (Path): Output directory for global plots.

    Returns:
        Path: Path to the generated PNG heatmap.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "global_phase2_dice_heatmap.png"

    records: List[Dict[str, Any]] = []

    # Prefer extracting peak (mean, std) directly from the per-run dataframes
    if all_2b_runs:
        for s1_stem, runs in all_2b_runs.items():
            for s2a_stem, df_sub in runs.items():
                s1_loops = int(df_sub["stage1_dataset_loops"].iloc[0]) if "stage1_dataset_loops" in df_sub.columns else 0
                s2a_steps = int(df_sub["stage2a_fit_steps"].iloc[0]) if "stage2a_fit_steps" in df_sub.columns else 0

                if "mean_full_vol_dice_fg" in df_sub.columns and df_sub["mean_full_vol_dice_fg"].notna().any():
                    valid_df = df_sub[df_sub["mean_full_vol_dice_fg"].notna()]
                    best_idx = valid_df["mean_full_vol_dice_fg"].idxmax()
                    best_mean = float(valid_df.loc[best_idx, "mean_full_vol_dice_fg"])
                    if "std_full_vol_dice_fg" in valid_df.columns and pd.notna(valid_df.loc[best_idx, "std_full_vol_dice_fg"]):
                        best_std = float(valid_df.loc[best_idx, "std_full_vol_dice_fg"])
                    else:
                        std_arr = extract_std_series(df_sub, "mean_batch_dice_fg", "std_batch_dice_fg")
                        best_std = float(std_arr[df_sub.index.get_loc(best_idx)])
                else:
                    dice_arr = df_sub["mean_batch_dice_fg"].to_numpy(dtype=float)
                    std_arr = extract_std_series(df_sub, "mean_batch_dice_fg", "std_batch_dice_fg")
                    best_pos = int(np.argmax(dice_arr))
                    best_mean = float(dice_arr[best_pos])
                    best_std = float(std_arr[best_pos])

                records.append({
                    "stage1_loops": s1_loops,
                    "stage2a_steps": s2a_steps,
                    "dice_mean": best_mean,
                    "dice_std": best_std
                })
    elif summary_csv_path.exists():
        df_sum = pd.read_csv(summary_csv_path)
        for (s1_l, s2a_s), grp in df_sum.groupby(["stage1_dataset_loops", "stage2a_fit_steps"]):
            use_full = "mean_full_vol_dice_fg" in grp.columns and grp["mean_full_vol_dice_fg"].notna().any()
            m_col = "mean_full_vol_dice_fg" if use_full else "mean_batch_dice_fg"
            s_col = "std_full_vol_dice_fg" if use_full else "std_batch_dice_fg"

            best_idx = grp[m_col].idxmax()
            best_mean = float(grp.loc[best_idx, m_col])
            best_std = float(grp.loc[best_idx, s_col]) if s_col in grp.columns and pd.notna(grp.loc[best_idx, s_col]) else float(grp[m_col].std(ddof=0))
            records.append({
                "stage1_loops": int(s1_l),
                "stage2a_steps": int(s2a_s),
                "dice_mean": best_mean,
                "dice_std": 0.0 if np.isnan(best_std) else best_std
            })

    if not records:
        return save_path

    df_heat = pd.DataFrame(records).dropna(subset=["dice_mean"])
    pivot_mean = df_heat.pivot_table(
        index="stage1_loops",
        columns="stage2a_steps",
        values="dice_mean",
        aggfunc="max"
    ).sort_index(ascending=True)

    pivot_std = df_heat.pivot_table(
        index="stage1_loops",
        columns="stage2a_steps",
        values="dice_std",
        aggfunc="first"
    ).reindex(index=pivot_mean.index, columns=pivot_mean.columns)

    if pivot_mean.empty:
        return save_path

    fig, ax = plt.subplots(
        figsize=(max(6.5, 1.15 * len(pivot_mean.columns)), max(5.2, 0.85 * len(pivot_mean.index)))
    )

    im = ax.imshow(pivot_mean.values, cmap="YlGnBu", vmin=0.7, vmax=1.0, aspect="auto", origin="lower")

    ax.set_xticks(range(len(pivot_mean.columns)))
    ax.set_yticks(range(len(pivot_mean.index)))
    ax.set_xticklabels([str(c) for c in pivot_mean.columns])
    ax.set_yticklabels([str(r) for r in pivot_mean.index])

    ax.set_xlabel("Stage 2A Pixel-Fitting Steps ($T_f$)")
    ax.set_ylabel("Stage 1 Meta-Learning Loops ($T_o$)")
    ax.set_title("Peak Foreground Dice Score Grid ($T_o$ vs. $T_f$)", pad=14)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Foreground Dice", rotation=270, labelpad=16)

    # Annotate Mean (bold, upper half of cell) and +/- SD (smaller font, lower half of cell)
    for r in range(len(pivot_mean.index)):
        for c in range(len(pivot_mean.columns)):
            val = pivot_mean.values[r, c]
            std_val = pivot_std.values[r, c]
            if not np.isnan(val):
                text_color = "white" if val > 0.85 else "black"
                ax.text(
                    c, r + 0.13, f"{val:.3f}",
                    ha="center", va="center", color=text_color, fontsize=9.5, weight="bold"
                )
                if not np.isnan(std_val):
                    ax.text(
                        c, r - 0.17, f"±{std_val:.3f}",
                        ha="center", va="center", color=text_color, fontsize=7.5, alpha=0.92
                    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=250)
    plt.close(fig)
    return save_path


def execute_plotting_pipeline(
    run_dir: Path,
    dice_ylim: Tuple[float, float] = (0.60, 1.00)
) -> None:
    """
    Parses `<run_dir>/phase2/` and generates all local and global Phase 2 plots.

    Args:
        run_dir (Path): Root timestamped run directory containing `phase2/`.
        dice_ylim (Tuple[float, float]): Y-axis `(ymin, ymax)` range for Phase 2B Dice line plots.
    """
    phase2_dir = run_dir / "phase2"
    if not phase2_dir.exists():
        raise FileNotFoundError(f"Phase 2 directory not found: {phase2_dir}")

    set_plot_style()
    global_plot_dir = phase2_dir / "plots"
    global_plot_dir.mkdir(parents=True, exist_ok=True)

    stage1_dirs = sorted([
        d for d in phase2_dir.iterdir()
        if d.is_dir() and d.name.startswith("stage1_")
    ])

    if not stage1_dirs:
        print(f"No 'stage1_*' subdirectories found in {phase2_dir}.")
        return

    all_2a_dfs: Dict[str, pd.DataFrame] = {}
    all_2b_runs: Dict[str, Dict[str, pd.DataFrame]] = {}

    print(f"Discovered {len(stage1_dirs)} Stage 1 directories under {phase2_dir.name}:")

    for s1_dir in stage1_dirs:
        s1_stem = s1_dir.name
        local_plot_dir = s1_dir / "plots"

        csv_2a = s1_dir / "stage2a_metrics_log.csv"
        if csv_2a.exists():
            df_2a = pd.read_csv(csv_2a)
            all_2a_dfs[s1_stem] = df_2a
            p2a_path = plot_local_stage2a(df_2a, s1_stem, local_plot_dir)
            print(f"  [{s1_stem}] Saved local Phase 2A plot -> {p2a_path.relative_to(run_dir)}")

        s2b_root = s1_dir / "stage2b"
        if s2b_root.is_dir():
            s2b_runs_local: Dict[str, pd.DataFrame] = {}
            for s2a_sub in sorted(s2b_root.iterdir()):
                if not s2a_sub.is_dir():
                    continue
                csv_2b = s2a_sub / "stage2b_metrics_log.csv"
                if csv_2b.exists():
                    df_2b = pd.read_csv(csv_2b)
                    s2b_runs_local[s2a_sub.name] = df_2b

            if s2b_runs_local:
                all_2b_runs[s1_stem] = s2b_runs_local
                p2b_path = plot_local_stage2b(
                    s2b_runs_local, s1_stem, local_plot_dir, dice_ylim=dice_ylim
                )
                print(f"  [{s1_stem}] Saved local Phase 2B plot -> {p2b_path.relative_to(run_dir)}")

    print("\nGenerating Global Aggregate Plots...")
    if all_2a_dfs:
        g2a_path = plot_global_stage2a(all_2a_dfs, global_plot_dir)
        print(f"  Saved Global Phase 2A comparison -> {g2a_path.relative_to(run_dir)}")

    if all_2b_runs:
        g2b_path = plot_global_stage2b(all_2b_runs, global_plot_dir, dice_ylim=dice_ylim)
        print(f"  Saved Global Phase 2B comparison -> {g2b_path.relative_to(run_dir)}")

    summary_csv = phase2_dir / "stage2b_all_runs_summary.csv"
    heatmap_path = plot_global_phase2_heatmap(summary_csv, all_2b_runs, global_plot_dir)
    if heatmap_path.exists():
        print(f"  Saved Global Phase 2 Heatmap    -> {heatmap_path.relative_to(run_dir)}")

    print("\nAll plotting tasks completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot local and global Phase 2A and Phase 2B metrics (with SD) for MetaSeg."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the timestamped run directory (e.g., .../OASIS_METASEG/cf_10/2026-10-21_18-00-00)."
    )
    parser.add_argument(
        "--dice_ylim_min",
        type=float,
        default=0.50,
        help="Lower y-axis bound for Phase 2B Dice line plots (default 0.60)."
    )
    parser.add_argument(
        "--dice_ylim_max",
        type=float,
        default=1.00,
        help="Upper y-axis bound for Phase 2B Dice line plots (default 1.00)."
    )

    args = parser.parse_args()
    run_dir_path = Path(args.run_dir)

    if not run_dir_path.exists():
        raise FileNotFoundError(f"Specified run directory does not exist: {run_dir_path}")

    execute_plotting_pipeline(
        run_dir_path,
        dice_ylim=(args.dice_ylim_min, args.dice_ylim_max)
    )