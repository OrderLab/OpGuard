import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec
import numpy as np

# Embed TrueType fonts (avoid Type3 bitmap fonts in PDF submissions)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# Use larger default fonts for publication-ready figures
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    }
)

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
CSV_FILES = [
    "run_20251001_12292e6b.csv",   # baseline (correct) - GREEN
    "run_20251001_9e609e46.csv",   # buggy - RED
]

STEP_MIN = 3050
STEP_MAX = 3450

LOSS_PDF = "loss_with_diff_3050_3450.pdf"
GRADNORM_PDF = "gradnorm_with_diff_3050_3450.pdf"

# Steps to explicitly note diff values
NOTE_STEPS = [3080, 3081]

# Colors
BASELINE_COLOR = "#2ca02c"  # green
BUGGY_COLOR = "#d62728"     # red
DIFF_COLOR = "#d62728"      # red


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load_and_normalize(path: str) -> pd.DataFrame:
    """
    Load a single CSV and normalize column names:
        step        -> 'step'
        grad norm   -> 'grad_norm'
        loss        -> 'loss'
    """
    df = pd.read_csv(path)

    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if "step" in lower:
            col_map[col] = "step"
        elif "grad" in lower and "norm" in lower:
            col_map[col] = "grad_norm"
        elif "gradnorm" in lower and "grad_norm" not in col_map.values():
            col_map[col] = "grad_norm"
        elif "loss" in lower:
            col_map[col] = "loss"

    df = df.rename(columns=col_map)

    # Keep only relevant columns that exist
    needed = [c for c in ["step", "grad_norm", "loss"] if c in df.columns]
    df = df[needed]

    # Ensure numeric
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Filter by step range
    if "step" in df.columns:
        df = df[(df["step"] >= STEP_MIN) & (df["step"] <= STEP_MAX)]

    return df


def label_from_filename(path: str) -> str:
    """
    Map filenames to nice labels.
    """
    stem = Path(path).stem
    if stem == "run_20251001_12292e6b":
        return "baseline (correct)"
    if stem == "run_20251001_9e609e46":
        return "buggy"
    parts = stem.split("_")
    if len(parts) >= 3:
        return f"{parts[1]}-{parts[2]}"
    return stem


def build_combined_df(runs):
    """
    Combine per-run data into a single DataFrame with a 'run' column.
    `runs` is list of (label, df).
    """
    all_dfs = []
    for label, df in runs:
        tmp = df.copy()
        tmp["run"] = label
        all_dfs.append(tmp)
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def plot_metric_with_diff(
    runs,
    all_df: pd.DataFrame,
    metric_key: str,
    metric_label: str,
    title: str,
    out_pdf: str,
):
    """
    Create a figure with:
      - top: metric per run (buggy red, baseline green on top)
      - bottom (扁): difference from baseline (buggy - baseline) in red
      - arrow-labeled blocks for Δ at NOTE_STEPS.
    """
    if metric_key not in all_df.columns:
        print(f"Metric {metric_key} not found, skipping {out_pdf}")
        return

    # Define baseline: first run in the list
    baseline_label = runs[0][0]

    # Pivot to align runs by step: index = step, columns = run, values = metric
    pivot = (
        all_df.dropna(subset=["step", metric_key])
              .pivot(index="step", columns="run", values=metric_key)
              .sort_index()
    )

    pivot = pivot[(pivot.index >= STEP_MIN) & (pivot.index <= STEP_MAX)]

    fig = plt.figure(figsize=(6.5, 4.5))
    fig.patch.set_facecolor("white")
    gs = GridSpec(4, 1, height_ratios=[3, 0.1, 1, 0.1], hspace=0.25)

    ax_main = fig.add_subplot(gs[0])   # main metric
    ax_diff = fig.add_subplot(gs[2], sharex=ax_main)  # diff vs baseline

    ax_main.set_facecolor("#fafafa")
    ax_diff.set_facecolor("#fafafa")

    baseline_name = runs[0][0]
    buggy_name = runs[1][0]

    # ---- Main plot ----
    # Plot buggy first (red)
    if buggy_name in pivot.columns:
        ax_main.plot(
            pivot.index,
            pivot[buggy_name],
            linewidth=2.0,
            alpha=0.9,
            color=BUGGY_COLOR,
            label=buggy_name,
            zorder=1,
        )

    # Plot baseline second (green) so it stays on top
    if baseline_name in pivot.columns:
        ax_main.plot(
            pivot.index,
            pivot[baseline_name],
            linewidth=2.4,
            alpha=0.95,
            color=BASELINE_COLOR,
            label=baseline_name,
            zorder=2,
        )

    # Vertical guide lines at NOTE_STEPS
    for step in NOTE_STEPS:
        ax_main.axvline(step, linestyle=":", linewidth=1.0, color="grey", alpha=0.7)

    ax_main.set_ylabel(metric_label, fontsize=16)
    ax_main.set_title(title, fontsize=20, pad=10)
    ax_main.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)
    ax_main.minorticks_on()
    ax_main.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.3)
    ax_main.legend(title="Run", fontsize=13, title_fontsize=14)
    ax_main.set_xlim(STEP_MIN, STEP_MAX)
    y_min, y_max = ax_main.get_ylim()
    if metric_key == "grad_norm":
        # set y-axis using the 10/90th percentiles of grad_norm values
        metric_values = all_df.dropna(subset=[metric_key])[metric_key]
        if not metric_values.empty:
            y_min = np.percentile(metric_values, 5)
            y_max = np.percentile(metric_values, 95)
            if y_max > y_min:
                ax_main.set_ylim(y_min, y_max)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # ---- Diff subplot (扁) ----
    if baseline_label not in pivot.columns:
        print(f"Baseline run {baseline_label} not found in pivot; skipping diff plot.")
    else:
        baseline_series = pivot[baseline_name]

        if buggy_name in pivot.columns:
            diff_series = pivot[buggy_name] - baseline_series

            # Red diff line
            ax_diff.plot(
                diff_series.index,
                diff_series.values,
                linewidth=1.8,
                alpha=0.9,
                color=DIFF_COLOR,
                label=f"{buggy_name} - {baseline_name}",
            )

            # Horizontal zero reference
            ax_diff.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.7, color="black")

            # Vertical guide lines in diff axis as well
            for step in NOTE_STEPS:
                ax_diff.axvline(step, linestyle=":", linewidth=1.0, color="grey", alpha=0.7)

            # Add small blocks (labels) with arrows at NOTE_STEPS
            # Use slightly different offsets so they don't overlap
            offsets = {
                NOTE_STEPS[0]: (25, 25),
                NOTE_STEPS[1]: (25, -35),
            }

            for step in NOTE_STEPS:
                if step in diff_series.index:
                    val = float(diff_series.loc[step])
                    print(f"[{metric_key}] diff at step {step}: {val}")
                    # marker on the point
                    ax_diff.scatter(step, val, color=DIFF_COLOR, s=25, zorder=3)
                    # annotation box with arrow
                    dx, dy = offsets.get(step, (15, 15))
                    ax_diff.annotate(
                        f"step {step}\nΔ = {val:.5g}",
                        xy=(step, val),
                        xytext=(dx, dy),
                        textcoords="offset points",
                        fontsize=12,
                        ha="left",
                        va="center",
                        color="black",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="white",
                            edgecolor=DIFF_COLOR,
                            linewidth=0.8,
                            alpha=0.9,
                        ),
                        arrowprops=dict(
                            arrowstyle="->",
                            color=DIFF_COLOR,
                            linewidth=0.8,
                        ),
                    )

            ax_diff.set_ylabel(r"$\Delta$", fontsize=16, labelpad=8)
            ax_diff.set_xlabel("Step", fontsize=16)
            ax_diff.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)
            ax_diff.minorticks_on()
            ax_diff.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.3)

            # Make it visually flat/compressed but symmetric and non-clipping
            all_vals = list(diff_series.values)
            if all_vals:
                max_abs = max(abs(min(all_vals)), abs(max(all_vals))) * 0.5
                y_max = max_abs
                y_min = -max_abs
                if y_max > y_min:
                    padding = 0.05 * (y_max - y_min)
                    ax_diff.set_ylim(y_min - padding, y_max + padding)

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved {out_pdf}")


# ----------------------------------------------------------------------
# Load all runs
# ----------------------------------------------------------------------
runs = []
for f in CSV_FILES:
    df = load_and_normalize(f)
    if not df.empty:
        runs.append((label_from_filename(f), df))

if not runs:
    raise RuntimeError("No data loaded. Check file paths and column names / ranges.")

all_df = build_combined_df(runs)

# ----------------------------------------------------------------------
# Loss figure (curve + diff vs baseline)
# ----------------------------------------------------------------------
plot_metric_with_diff(
    runs,
    all_df,
    metric_key="loss",
    metric_label="Training Loss",
    title=f"Training Loss (steps {STEP_MIN}–{STEP_MAX})",
    out_pdf=LOSS_PDF,
)

# ----------------------------------------------------------------------
# Grad norm figure (curve + diff vs baseline)
# ----------------------------------------------------------------------
plot_metric_with_diff(
    runs,
    all_df,
    metric_key="grad_norm",
    metric_label="Gradient Norm",
    title=f"Gradient Norm (steps {STEP_MIN}–{STEP_MAX})",
    out_pdf=GRADNORM_PDF,
)
