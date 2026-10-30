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
        "legend.fontsize": 9.0,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def create_time_secondary_xaxis(
    ax: plt.Axes,
    steps: np.ndarray,
    times_sec: np.ndarray,
    label: str = "Elapsed Optimization Time (s)"
) -> Optional[plt.Axes]:
    """
    Attaches a secondary top x-axis that monotonically maps primary x-axis steps/loops
    to elapsed optimization time in seconds.

    Args:
        ax (plt.Axes): Primary matplotlib axis.
        steps (np.ndarray): 1D array of primary x-axis values (steps or loops).
        times_sec (np.ndarray): 1D array of corresponding cumulative elapsed seconds.
        label (str): Label for the secondary top x-axis.

    Returns:
        Optional[plt.Axes]: The secondary axis, or None if fewer than 2 distinct points exist.
    """
    valid_mask = (~np.isnan(steps)) & (~np.isnan(times_sec))
    s_clean = steps[valid_mask]
    t_clean = times_sec[valid_mask]

    if len(s_clean) < 2:
        return None

    s_uniq, idx = np.unique(s_clean, return_index=True)
    t_uniq = np.maximum.accumulate(t_clean[idx])

    if len(s_uniq) < 2 or (t_uniq[-1] - t_uniq[0]) <= 0.0:
        return None

    def forward_step_to_time(x: np.ndarray) -> np.ndarray:
        return np.interp(x, s_uniq, t_uniq)

    def inverse_time_to_step(t: np.ndarray) -> np.ndarray:
        return np.interp(t, t_uniq, s_uniq)

    sec_ax = ax.secondary_xaxis("top", functions=(forward_step_to_time, inverse_time_to_step))
    sec_ax.set_xlabel(label, color="#444444", labelpad=8)
    sec_ax.tick_params(colors="#444444")
    return sec_ax


def load_or_build_stage3_summary(phase3_dir: Path) -> pd.DataFrame:
    """
    Loads `<run_dir>/phase3/stage3_checkpoint_summary.csv` if available, or aggregates it
    from the per-Stage-1 `stage3_checkpoint_summary.csv` / `stage3_subject_evaluations.csv` files.

    Args:
        phase3_dir (Path): Path to `<run_dir>/phase3/`.

    Returns:
        pd.DataFrame: Master summary dataframe containing mean and std PSNR/Dice per checkpoint.
    """
    master_sum_path = phase3_dir / "stage3_checkpoint_summary.csv"
    if master_sum_path.exists():
        return pd.read_csv(master_sum_path)

    sub_dfs: List[pd.DataFrame] = []
    for s1_dir in sorted(phase3_dir.iterdir()):
        if not s1_dir.is_dir() or not s1_dir.name.startswith("stage1_"):
            continue
        s1_sum = s1_dir / "stage3_checkpoint_summary.csv"
        if s1_sum.exists():
            sub_dfs.append(pd.read_csv(s1_sum))

    if sub_dfs:
        return pd.concat(sub_dfs, ignore_index=True)

    raise FileNotFoundError(
        f"No 'stage3_checkpoint_summary.csv' found in {phase3_dir} or its 'stage1_*' subdirectories."
    )


def select_head_loop_slice(
    df_summary: pd.DataFrame,
    head_selection: str = "best",
    fixed_head_loop: Optional[int] = None
) -> pd.DataFrame:
    """
    Reduces the Stage 2B head-loop dimension (`stage2b_head_loops`) so each
    `(stage1_dataset_loops, stage2a_fit_steps)` pair maps to a single representative row.

    Args:
        df_summary (pd.DataFrame): Master Stage 3 summary dataframe.
        head_selection (str): Selection rule across `stage2b_head_loops`:
            - 'best': Picks the head loop maximizing `mean_full_vol_dice_fg`.
            - 'last': Picks the maximum `stage2b_head_loops` available for that pair.
            - 'fixed': Picks the head loop closest to `fixed_head_loop`.
        fixed_head_loop (Optional[int]): Target head loop index when `head_selection == 'fixed'`.

    Returns:
        pd.DataFrame: Filtered dataframe with one row per `(stage1_dataset_loops, stage2a_fit_steps)`.
    """
    selected_rows: List[pd.Series] = []
    group_cols = ["stage1_dataset_loops", "stage2a_fit_steps"]

    for _, grp in df_summary.groupby(group_cols, sort=True):
        if head_selection == "best":
            idx = grp["mean_full_vol_dice_fg"].idxmax()
        elif head_selection == "last":
            idx = grp["stage2b_head_loops"].idxmax()
        elif head_selection == "fixed":
            target = fixed_head_loop if fixed_head_loop is not None else int(grp["stage2b_head_loops"].max())
            idx = (grp["stage2b_head_loops"] - target).abs().idxmin()
        else:
            raise ValueError(f"Unknown head_selection mode: {head_selection}")
        selected_rows.append(grp.loc[idx])

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def plot_curves_by_fit_steps(
    df_reduced: pd.DataFrame,
    output_dir: Path,
    metric_mean_col: str,
    metric_std_col: str,
    ylabel: str,
    title: str,
    filename: str,
    ylim: Optional[Tuple[float, float]] = None,
    is_dice: bool = True
) -> Path:
    """
    Perspective 1: Plots validation performance with Test-Time Fine-Tuning Steps (T_f) on the
    primary bottom x-axis, Mean Test-Time Fitting Time (s) on the top x-axis, and separate
    lines for each Stage 1 checkpoint (T_o). Every second line is dashed.

    Args:
        df_reduced (pd.DataFrame): Reduced dataframe with one row per (T_o, T_f).
        output_dir (Path): Directory where the plot PNG will be saved.
        metric_mean_col (str): Column name for the mean metric.
        metric_std_col (str): Column name for the std metric.
        ylabel (str): Y-axis label string.
        title (str): Plot title string.
        filename (str): Output filename for the PNG.
        ylim (Optional[Tuple[float, float]]): Optional `(ymin, ymax)` bounds.
        is_dice (bool): Formatting toggle (3 decimal places for Dice, 2 for PSNR).

    Returns:
        Path: Path to the saved PNG file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / filename

    fig, ax1 = plt.subplots(figsize=(10.5, 6.2))

    s1_loops_list = sorted(df_reduced["stage1_dataset_loops"].unique())
    num_curves = max(1, len(s1_loops_list))
    colors = plt.cm.viridis(np.linspace(0.08, 0.95, num_curves))

    all_x = df_reduced["stage2a_fit_steps"].to_numpy(dtype=float)
    x_span = float(np.max(all_x) - np.min(all_x)) if len(all_x) > 1 else 1.0
    stagger_offsets = np.linspace(-0.012 * x_span, 0.012 * x_span, num_curves)

    ref_steps: Optional[np.ndarray] = None
    time_curves: List[np.ndarray] = []

    for idx_c, (s1_loop, color) in enumerate(zip(s1_loops_list, colors)):
        sub = df_reduced[df_reduced["stage1_dataset_loops"] == s1_loop].sort_values("stage2a_fit_steps")
        steps = sub["stage2a_fit_steps"].to_numpy(dtype=float)
        means = sub[metric_mean_col].to_numpy(dtype=float)
        stds = sub[metric_std_col].fillna(0.0).to_numpy(dtype=float)
        times = sub["mean_test_fit_opt_time_sec"].to_numpy(dtype=float)

        best_i = int(np.argmax(means))
        peak_m, peak_s, peak_tf = means[best_i], stds[best_i], int(steps[best_i])

        if is_dice:
            label_str = f"Stage 1 Loop {s1_loop} (Peak@{peak_tf}: {peak_m:.3f}±{peak_s:.3f})"
        else:
            label_str = f"Stage 1 Loop {s1_loop} (Peak@{peak_tf}: {peak_m:.2f}±{peak_s:.2f} dB)"

        linestyle = "--" if (idx_c % 2 == 1) else "-"
        ax1.plot(steps, means, color=color, linewidth=2.0, linestyle=linestyle, label=label_str)
        ax1.fill_between(steps, means - stds, means + stds, color=color, alpha=0.08)

        ax1.errorbar(
            steps + stagger_offsets[idx_c],
            means,
            yerr=stds,
            fmt="o", color=color, markersize=4.0, capsize=2.5,
            elinewidth=1.0, alpha=0.85
        )

        if ref_steps is None or len(steps) > len(ref_steps):
            ref_steps = steps
        time_curves.append(times)

    ax1.set_xlabel("Test-Time Pixel-Fitting Steps ($T_f$)")
    ax1.set_ylabel(ylabel)
    if ylim is not None:
        ax1.set_ylim(ylim[0], ylim[1])
    ax1.set_title(title, pad=28)
    ax1.legend(loc="lower right", framealpha=0.92, title="Stage 1 Meta-Learning Horizon")

    if ref_steps is not None and time_curves:
        min_len = min(len(t) for t in time_curves)
        mean_times = np.mean([t[:min_len] for t in time_curves], axis=0)
        create_time_secondary_xaxis(
            ax1, ref_steps[:min_len], mean_times,
            label="Mean Test-Time Fitting Duration per Subject (s)"
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=250)
    plt.close(fig)
    return save_path


def plot_curves_by_stage1_loops(
    df_reduced: pd.DataFrame,
    output_dir: Path,
    metric_mean_col: str,
    metric_std_col: str,
    ylabel: str,
    title: str,
    filename: str,
    ylim: Optional[Tuple[float, float]] = None,
    is_dice: bool = True
) -> Path:
    """
    Perspective 2: Plots validation performance with Stage 1 Meta-Learning Loops (T_o) on the
    primary bottom x-axis, Cumulative Stage 1 Optimization Time (s) on the top x-axis, and
    separate lines for each Test-Time Fine-Tuning Step horizon (T_f). Every second line is dashed.

    Args:
        df_reduced (pd.DataFrame): Reduced dataframe with one row per (T_o, T_f).
        output_dir (Path): Directory where the plot PNG will be saved.
        metric_mean_col (str): Column name for the mean metric.
        metric_std_col (str): Column name for the std metric.
        ylabel (str): Y-axis label string.
        title (str): Plot title string.
        filename (str): Output filename for the PNG.
        ylim (Optional[Tuple[float, float]]): Optional `(ymin, ymax)` bounds.
        is_dice (bool): Formatting toggle (3 decimal places for Dice, 2 for PSNR).

    Returns:
        Path: Path to the saved PNG file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / filename

    fig, ax1 = plt.subplots(figsize=(10.5, 6.2))

    tf_steps_list = sorted(df_reduced["stage2a_fit_steps"].unique())
    num_curves = max(1, len(tf_steps_list))
    colors = plt.cm.plasma(np.linspace(0.08, 0.92, num_curves))

    all_x = df_reduced["stage1_dataset_loops"].to_numpy(dtype=float)
    x_span = float(np.max(all_x) - np.min(all_x)) if len(all_x) > 1 else 1.0
    stagger_offsets = np.linspace(-0.012 * x_span, 0.012 * x_span, num_curves)

    ref_s1_loops: Optional[np.ndarray] = None
    ref_s1_times: Optional[np.ndarray] = None

    for idx_c, (tf_step, color) in enumerate(zip(tf_steps_list, colors)):
        sub = df_reduced[df_reduced["stage2a_fit_steps"] == tf_step].sort_values("stage1_dataset_loops")
        s1_loops = sub["stage1_dataset_loops"].to_numpy(dtype=float)
        means = sub[metric_mean_col].to_numpy(dtype=float)
        stds = sub[metric_std_col].fillna(0.0).to_numpy(dtype=float)
        s1_times = sub["stage1_opt_time_sec"].to_numpy(dtype=float)

        best_i = int(np.argmax(means))
        peak_m, peak_s, peak_to = means[best_i], stds[best_i], int(s1_loops[best_i])

        if is_dice:
            label_str = f"$T_f={tf_step}$ steps (Peak@Loop {peak_to}: {peak_m:.3f}±{peak_s:.3f})"
        else:
            label_str = f"$T_f={tf_step}$ steps (Peak@Loop {peak_to}: {peak_m:.2f}±{peak_s:.2f} dB)"

        linestyle = "--" if (idx_c % 2 == 1) else "-"
        ax1.plot(s1_loops, means, color=color, linewidth=2.0, linestyle=linestyle, label=label_str)
        ax1.fill_between(s1_loops, means - stds, means + stds, color=color, alpha=0.08)

        ax1.errorbar(
            s1_loops + stagger_offsets[idx_c],
            means,
            yerr=stds,
            fmt="s", color=color, markersize=4.0, capsize=2.5,
            elinewidth=1.0, alpha=0.85
        )

        if ref_s1_loops is None or len(s1_loops) > len(ref_s1_loops):
            ref_s1_loops = s1_loops
            ref_s1_times = s1_times

    ax1.set_xlabel("Stage 1 Meta-Learning Loops ($T_o$)")
    ax1.set_ylabel(ylabel)
    if ylim is not None:
        ax1.set_ylim(ylim[0], ylim[1])
    ax1.set_title(title, pad=28)
    ax1.legend(loc="lower right", framealpha=0.92, title="Test-Time Fitting Horizon ($T_f$)")

    if ref_s1_loops is not None and ref_s1_times is not None:
        create_time_secondary_xaxis(
            ax1, ref_s1_loops, ref_s1_times,
            label="Cumulative Stage 1 Meta-Optimization Time (s)"
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=250)
    plt.close(fig)
    return save_path


def plot_global_stage3_heatmap(
    df_reduced: pd.DataFrame,
    output_dir: Path,
    metric_mean_col: str,
    metric_std_col: str,
    title: str,
    cbar_label: str,
    filename: str,
    vmin: float = 0.7,
    vmax: float = 1.0,
    is_dice: bool = True
) -> Path:
    """
    Generates a 2D heatmap where Stage 1 Meta-Learning Loops (T_o) are on the x-axis and
    Test-Time Fine-Tuning Steps (T_f) are on the y-axis (origin='lower'), displaying the mean
    value in bold and +/- SD underneath inside every cell.

    Args:
        df_reduced (pd.DataFrame): Reduced dataframe with one row per (T_o, T_f).
        output_dir (Path): Directory where the heatmap PNG will be saved.
        metric_mean_col (str): Column name for the mean metric.
        metric_std_col (str): Column name for the std metric.
        title (str): Plot title.
        cbar_label (str): Vertical colorbar label.
        filename (str): Output PNG filename.
        vmin (float): Minimum value for colormap scaling.
        vmax (float): Maximum value for colormap scaling.
        is_dice (bool): Formatting toggle.

    Returns:
        Path: Path to the saved PNG file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / filename

    # SWAPPED: index = stage2a_fit_steps (y-axis), columns = stage1_dataset_loops (x-axis)
    pivot_mean = df_reduced.pivot_table(
        index="stage2a_fit_steps",
        columns="stage1_dataset_loops",
        values=metric_mean_col,
        aggfunc="max"
    ).sort_index(ascending=True)

    pivot_std = df_reduced.pivot_table(
        index="stage2a_fit_steps",
        columns="stage1_dataset_loops",
        values=metric_std_col,
        aggfunc="first"
    ).reindex(index=pivot_mean.index, columns=pivot_mean.columns)

    if pivot_mean.empty:
        return save_path

    fig, ax = plt.subplots(
        figsize=(max(6.5, 1.15 * len(pivot_mean.columns)), max(5.2, 0.85 * len(pivot_mean.index)))
    )

    im = ax.imshow(
        pivot_mean.values, cmap="YlGnBu",
        vmin=vmin, vmax=vmax, aspect="auto", origin="lower"
    )

    ax.set_xticks(range(len(pivot_mean.columns)))
    ax.set_yticks(range(len(pivot_mean.index)))
    ax.set_xticklabels([str(c) for c in pivot_mean.columns])
    ax.set_yticklabels([str(r) for r in pivot_mean.index])

    ax.set_xlabel("Stage 1 Meta-Learning Loops ($T_o$)")
    ax.set_ylabel("Test-Time Pixel-Fitting Steps ($T_f$)")
    ax.set_title(title, pad=14)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, rotation=270, labelpad=16)

    midpoint = vmin + 0.52 * (vmax - vmin)
    fmt_str = ".3f" if is_dice else ".2f"

    for r in range(len(pivot_mean.index)):
        for c in range(len(pivot_mean.columns)):
            val = pivot_mean.values[r, c]
            std_val = pivot_std.values[r, c]
            if not np.isnan(val):
                text_color = "white" if val > midpoint else "black"
                ax.text(
                    c, r + 0.13, f"{val:{fmt_str}}",
                    ha="center", va="center", color=text_color, fontsize=9.5, weight="bold"
                )
                if not np.isnan(std_val):
                    ax.text(
                        c, r - 0.17, f"±{std_val:{fmt_str}}",
                        ha="center", va="center", color=text_color, fontsize=7.5, alpha=0.92
                    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=250)
    plt.close(fig)
    return save_path


def plot_global_stage3_head_grid(
    df_summary: pd.DataFrame,
    output_dir: Path,
    dice_ylim: Tuple[float, float] = (0.60, 1.00)
) -> Path:
    """
    Plots a multi-panel grid across all Stage 1 checkpoints showing Validation Foreground Dice
    as a function of Stage 2B Head Optimization Loops (E_head), with alternating dashed lines per T_f.

    Args:
        df_summary (pd.DataFrame): Full Stage 3 summary dataframe.
        output_dir (Path): Directory where the plot PNG will be saved.
        dice_ylim (Tuple[float, float]): Y-axis bounds.

    Returns:
        Path: Path to the saved PNG file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "global_stage3_head_convergence_grid.png"

    s1_loops_list = sorted(df_summary["stage1_dataset_loops"].unique())
    n_s1 = len(s1_loops_list)
    if n_s1 == 0:
        return save_path

    cols = min(3, n_s1)
    rows = math.ceil(n_s1 / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5.8 * cols, 4.5 * rows), sharey=True, squeeze=False)
    fig.suptitle(
        "Stage 3 Validation Foreground Dice vs. Stage 2B Head Loops ($\pm 1\sigma$)",
        fontsize=14, y=0.99
    )

    for idx, s1_loop in enumerate(s1_loops_list):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        df_s1 = df_summary[df_summary["stage1_dataset_loops"] == s1_loop]

        tf_list = sorted(df_s1["stage2a_fit_steps"].unique())
        num_curves = max(1, len(tf_list))
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, num_curves))
        stagger_offsets = np.linspace(-0.8, 0.8, num_curves)

        for idx_c, (tf_val, color) in enumerate(zip(tf_list, colors)):
            sub = df_s1[df_s1["stage2a_fit_steps"] == tf_val].sort_values("stage2b_head_loops")
            h_loops = sub["stage2b_head_loops"].to_numpy(dtype=float)
            dice_fg = sub["mean_full_vol_dice_fg"].to_numpy(dtype=float)
            std_fg = sub["std_full_vol_dice_fg"].fillna(0.0).to_numpy(dtype=float)

            best_i = int(np.argmax(dice_fg))
            peak_m, peak_s = dice_fg[best_i], std_fg[best_i]

            linestyle = "--" if (idx_c % 2 == 1) else "-"
            ax.plot(
                h_loops, dice_fg, color=color, linewidth=1.7, linestyle=linestyle,
                label=f"$T_f={tf_val}$ ({peak_m:.3f}±{peak_s:.3f})"
            )
            ax.fill_between(h_loops, dice_fg - std_fg, dice_fg + std_fg, color=color, alpha=0.08)
            ax.errorbar(
                h_loops + stagger_offsets[idx_c],
                dice_fg, yerr=std_fg,
                fmt="o", color=color, markersize=3.0, capsize=2.0,
                elinewidth=0.9, alpha=0.80
            )

        ax.set_title(f"Stage 1 Loop {s1_loop}", fontsize=11)
        ax.set_xlabel("Stage 2B Head Loops ($E_{\mathrm{head}}$)")
        if c == 0:
            ax.set_ylabel("Validation Foreground Dice")
        ax.set_ylim(dice_ylim[0], dice_ylim[1])
        ax.legend(fontsize=7.2, loc="lower right", framealpha=0.88)

    for idx in range(n_s1, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=250)
    plt.close(fig)
    return save_path


def execute_stage3_plotting_pipeline(
    run_dir: Path,
    head_selection: str = "best",
    fixed_head_loop: Optional[int] = None,
    dice_ylim: Tuple[float, float] = (0.60, 1.00)
) -> None:
    """
    Loads Phase 3 summary metrics from `<run_dir>/phase3/` and generates all local and global
    validation plots, including both curve orientations and swapped-axis heatmaps.

    Args:
        run_dir (Path): Timestamped run root directory containing `phase3/`.
        head_selection (str): How to select across Stage 2B head loops.
        fixed_head_loop (Optional[int]): Specific Stage 2B head loop.
        dice_ylim (Tuple[float, float]): `(ymin, ymax)` range for Dice plots.
    """
    phase3_dir = run_dir / "phase3"
    if not phase3_dir.exists():
        raise FileNotFoundError(f"Phase 3 directory not found: {phase3_dir}")

    set_plot_style()
    df_summary = load_or_build_stage3_summary(phase3_dir)

    df_reduced = select_head_loop_slice(
        df_summary, head_selection=head_selection, fixed_head_loop=fixed_head_loop
    )

    global_plot_dir = phase3_dir / "plots"
    global_plot_dir.mkdir(parents=True, exist_ok=True)
    sel_tag = f"Head Selection: {head_selection}" + (f"={fixed_head_loop}" if head_selection == "fixed" else "")

    print(f"Generating Global Stage 3 Plots ({sel_tag})...")

    # Perspective 1A: Dice vs. Fit Steps (lines = Stage 1 runs, alternating dashed)
    p_dice_tf = plot_curves_by_fit_steps(
        df_reduced=df_reduced,
        output_dir=global_plot_dir,
        metric_mean_col="mean_full_vol_dice_fg",
        metric_std_col="std_full_vol_dice_fg",
        ylabel="Validation Foreground Dice",
        title=f"Stage 3 Validation Foreground Dice vs. Test-Time Fine-Tuning Steps ($\pm 1\sigma$)\n[{sel_tag}]",
        filename="global_stage3_dice_vs_fit_steps.png",
        ylim=dice_ylim,
        is_dice=True
    )
    print(f"  Saved Perspective 1 (Dice vs. Fit Steps)  -> {p_dice_tf.relative_to(run_dir)}")

    # Perspective 2A: Dice vs. Stage 1 Loops (lines = Fine-Tuning steps, alternating dashed)
    p_dice_to = plot_curves_by_stage1_loops(
        df_reduced=df_reduced,
        output_dir=global_plot_dir,
        metric_mean_col="mean_full_vol_dice_fg",
        metric_std_col="std_full_vol_dice_fg",
        ylabel="Validation Foreground Dice",
        title=f"Stage 3 Validation Foreground Dice vs. Stage 1 Meta-Learning Loops ($\pm 1\sigma$)\n[{sel_tag}]",
        filename="global_stage3_dice_vs_stage1_loops.png",
        ylim=dice_ylim,
        is_dice=True
    )
    print(f"  Saved Perspective 2 (Dice vs. S1 Loops)   -> {p_dice_to.relative_to(run_dir)}")

    # Perspective 1B: PSNR vs. Fit Steps (lines = Stage 1 runs, alternating dashed)
    p_psnr_tf = plot_curves_by_fit_steps(
        df_reduced=df_reduced,
        output_dir=global_plot_dir,
        metric_mean_col="mean_full_vol_psnr_db",
        metric_std_col="std_full_vol_psnr_db",
        ylabel="Validation Full-Volume PSNR (dB)",
        title="Stage 3 Validation PSNR vs. Test-Time Fine-Tuning Steps ($\pm 1\sigma$)",
        filename="global_stage3_psnr_vs_fit_steps.png",
        ylim=None,
        is_dice=False
    )
    print(f"  Saved Perspective 1 (PSNR vs. Fit Steps)  -> {p_psnr_tf.relative_to(run_dir)}")

    # Perspective 2B: PSNR vs. Stage 1 Loops (lines = Fine-Tuning steps, alternating dashed)
    p_psnr_to = plot_curves_by_stage1_loops(
        df_reduced=df_reduced,
        output_dir=global_plot_dir,
        metric_mean_col="mean_full_vol_psnr_db",
        metric_std_col="std_full_vol_psnr_db",
        ylabel="Validation Full-Volume PSNR (dB)",
        title="Stage 3 Validation PSNR vs. Stage 1 Meta-Learning Loops ($\pm 1\sigma$)",
        filename="global_stage3_psnr_vs_stage1_loops.png",
        ylim=None,
        is_dice=False
    )
    print(f"  Saved Perspective 2 (PSNR vs. S1 Loops)   -> {p_psnr_to.relative_to(run_dir)}")

    # Global Heatmaps (Stage 1 on x-axis, Stage 2A on y-axis, vmin=0.7, vmax=1.0)
    p_heat_dice = plot_global_stage3_heatmap(
        df_reduced=df_reduced,
        output_dir=global_plot_dir,
        metric_mean_col="mean_full_vol_dice_fg",
        metric_std_col="std_full_vol_dice_fg",
        title=f"Stage 3 Validation Foreground Dice Grid ($T_o$ vs. $T_f$) [{sel_tag}]",
        cbar_label="Validation Foreground Dice",
        filename="global_stage3_dice_heatmap.png",
        vmin=0.70,
        vmax=1.00,
        is_dice=True
    )
    print(f"  Saved Global Validation Dice Heatmap      -> {p_heat_dice.relative_to(run_dir)}")

    min_psnr = float(np.floor(df_reduced["mean_full_vol_psnr_db"].min()))
    max_psnr = float(np.ceil(df_reduced["mean_full_vol_psnr_db"].max()))
    p_heat_psnr = plot_global_stage3_heatmap(
        df_reduced=df_reduced,
        output_dir=global_plot_dir,
        metric_mean_col="mean_full_vol_psnr_db",
        metric_std_col="std_full_vol_psnr_db",
        title="Stage 3 Validation Full-Volume PSNR Grid ($T_o$ vs. $T_f$)",
        cbar_label="Validation PSNR (dB)",
        filename="global_stage3_psnr_heatmap.png",
        vmin=min_psnr,
        vmax=max_psnr,
        is_dice=False
    )
    print(f"  Saved Global Validation PSNR Heatmap      -> {p_heat_psnr.relative_to(run_dir)}")

    # Multi-panel Facet Grid across Stage 2B Head Loops (with alternating dashed lines)
    p_grid = plot_global_stage3_head_grid(
        df_summary=df_summary,
        output_dir=global_plot_dir,
        dice_ylim=dice_ylim
    )
    print(f"  Saved Global Head Convergence Facet Grid  -> {p_grid.relative_to(run_dir)}")
    print("\nAll Phase 3 plotting tasks completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot local and global Phase 3 validation metrics for MetaSeg."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the timestamped run directory (e.g., .../OASIS_METASEG/cf_10/2026-10-21_18-00-00)."
    )
    parser.add_argument(
        "--head_selection",
        type=str,
        default="best",
        choices=["best", "last", "fixed"],
        help="How to select across Stage 2B head loops for (T_o, T_f) plots (default: 'best')."
    )
    parser.add_argument(
        "--fixed_head_loop",
        type=int,
        default=None,
        help="Target Stage 2B head loop index when --head_selection='fixed'."
    )
    parser.add_argument(
        "--dice_ylim_min",
        type=float,
        default=0.60,
        help="Lower y-axis bound for Validation Dice line plots (default 0.60)."
    )
    parser.add_argument(
        "--dice_ylim_max",
        type=float,
        default=1.00,
        help="Upper y-axis bound for Validation Dice line plots (default 1.00)."
    )

    args = parser.parse_args()
    run_dir_path = Path(args.run_dir)

    if not run_dir_path.exists():
        raise FileNotFoundError(f"Specified run directory does not exist: {run_dir_path}")

    execute_stage3_plotting_pipeline(
        run_dir_path,
        head_selection=args.head_selection,
        fixed_head_loop=args.fixed_head_loop,
        dice_ylim=(args.dice_ylim_min, args.dice_ylim_max)
    )