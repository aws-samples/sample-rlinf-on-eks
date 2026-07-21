#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Plot training curves from TensorBoard event files.

Reads TensorBoard event files, deduplicates overlapping steps from checkpoint
resumptions (keeps the latest-written value per step), applies smoothing, and
renders a publication-quality PNG.

Usage:
    # Single run directory:
    python examples/scripts/plot_training_curve.py \
        --logdir /fsx/checkpoints/libero-pi0-ppo/libero_spatial_ppo_openpi/tensorboard \
        --metric success_once \
        --output examples/libero-pi0-ppo/assets/training_curve.png \
        --title "LIBERO + pi0 PPO — Training Progress"

    # Multiple event files (merged and deduplicated):
    python examples/scripts/plot_training_curve.py \
        --logdir /fsx/checkpoints/maniskill-openvla-ppo/maniskill_ppo_openvla/tensorboard \
        --metric success_once \
        --output examples/maniskill-openvla-ppo/assets/training_curve.png \
        --title "ManiSkill + OpenVLA PPO — Training Progress"

    # Custom smoothing window and threshold line:
    python examples/scripts/plot_training_curve.py \
        --logdir /path/to/events \
        --metric success_once \
        --output chart.png \
        --smooth-window 5 \
        --threshold 0.9 \
        --threshold-label "90% threshold"

Dependencies:
    pip install tensorboard matplotlib numpy
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("ERROR: tensorboard package required. Install with: pip install tensorboard", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Chart style
# ---------------------------------------------------------------------------
CHART_STYLE = {
    "figure.figsize": (10, 5),
    "figure.dpi": 150,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
}

# Per-example color palette
COLORS = {
    "libero": "#1E88E5",       # blue
    "openvla": "#43A047",      # green
    "openvlaoft": "#E53935",   # red
    "default": "#1E88E5",      # blue fallback
}


def detect_color(title: str) -> str:
    """Pick line color based on title keywords."""
    title_lower = title.lower()
    if "openvla-oft" in title_lower or "openvlaoft" in title_lower:
        return COLORS["openvlaoft"]
    elif "openvla" in title_lower:
        return COLORS["openvla"]
    elif "libero" in title_lower or "pi0" in title_lower:
        return COLORS["libero"]
    return COLORS["default"]


# ---------------------------------------------------------------------------
# Data loading and deduplication
# ---------------------------------------------------------------------------
def load_events(logdir: str, metric: str) -> list[tuple[float, int, float]]:
    """
    Load scalar events from all TensorBoard event files in logdir.

    Returns list of (wall_time, step, value) tuples.
    """
    logdir = Path(logdir)
    if not logdir.exists():
        print(f"ERROR: logdir does not exist: {logdir}", file=sys.stderr)
        sys.exit(1)

    all_events = []

    # EventAccumulator handles a single directory; walk for nested structures
    event_dirs = set()
    for event_file in logdir.rglob("events.out.tfevents.*"):
        event_dirs.add(str(event_file.parent))

    if not event_dirs:
        # Try the logdir itself
        event_dirs.add(str(logdir))

    for edir in sorted(event_dirs):
        try:
            ea = EventAccumulator(edir)
            ea.Reload()
            if metric in ea.Tags().get("scalars", []):
                for event in ea.Scalars(metric):
                    all_events.append((event.wall_time, event.step, event.value))
        except Exception as e:
            print(f"WARNING: Failed to read {edir}: {e}", file=sys.stderr)

    if not all_events:
        print(f"ERROR: No events found for metric '{metric}' in {logdir}", file=sys.stderr)
        print(f"  Available metrics: {ea.Tags().get('scalars', [])}", file=sys.stderr)
        sys.exit(1)

    return all_events


def deduplicate(events: list[tuple[float, int, float]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Deduplicate overlapping steps from checkpoint resumptions.

    When training is resumed from a checkpoint, TensorBoard logs may contain
    multiple values for the same step (from the original run and the resumed run).
    We keep the value with the latest wall_time for each step, which corresponds
    to the most recent (resumed) run's data.

    Returns (steps, values) arrays sorted by step.
    """
    # Group by step, keep latest wall_time
    step_data: dict[int, tuple[float, float]] = {}  # step -> (wall_time, value)
    for wall_time, step, value in events:
        if step not in step_data or wall_time > step_data[step][0]:
            step_data[step] = (wall_time, value)

    # Sort by step
    sorted_steps = sorted(step_data.keys())
    steps = np.array(sorted_steps)
    values = np.array([step_data[s][1] for s in sorted_steps])

    return steps, values


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Apply rolling average smoothing."""
    if window <= 1:
        return values.copy()
    kernel = np.ones(window) / window
    # Pad to handle edges
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_curve(
    steps: np.ndarray,
    values: np.ndarray,
    title: str,
    metric_label: str,
    output: str,
    smooth_window: int = 5,
    threshold: float | None = None,
    threshold_label: str | None = None,
    color: str | None = None,
):
    """Render the training curve to a PNG file."""
    plt.rcParams.update(CHART_STYLE)

    if color is None:
        color = detect_color(title)

    smoothed = smooth(values, smooth_window)

    fig, ax = plt.subplots()

    # Raw data (thin, transparent)
    ax.plot(steps, values, linewidth=0.5, alpha=0.3, color=color)

    # Smoothed (bold)
    ax.plot(
        steps,
        smoothed,
        linewidth=2.5,
        color=color,
        label=f"{metric_label} ({smooth_window}-step avg)",
    )

    # Threshold line
    if threshold is not None:
        label = threshold_label or f"{threshold*100:.0f}% threshold"
        ax.axhline(y=threshold, color="gray", linestyle="--", alpha=0.7, label=label)

    # Final value annotation
    final_val = smoothed[-1]
    ax.annotate(
        f"Final: {final_val*100:.1f}%",
        xy=(steps[-1], final_val),
        xytext=(-10, -25),
        textcoords="offset points",
        fontsize=10,
        color=color,
        alpha=0.8,
        ha="right",
    )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Success Rate")
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(steps[0], steps[-1])
    ax.legend(loc="lower right", fontsize=10)

    fig.tight_layout()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

    # Print summary stats
    peak_idx = np.argmax(smoothed)
    print(f"  Output: {output_path}")
    print(f"  Steps:  {steps[0]} → {steps[-1]} ({len(steps)} points)")
    print(f"  Peak:   {smoothed[peak_idx]*100:.1f}% at step {steps[peak_idx]}")
    print(f"  Final:  {final_val*100:.1f}% at step {steps[-1]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot training curves from TensorBoard events with deduplication.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--logdir",
        required=True,
        help="Path to TensorBoard event directory (supports nested dirs)",
    )
    parser.add_argument(
        "--metric",
        default="success_once",
        help="Scalar metric name to plot (default: success_once)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output PNG file path",
    )
    parser.add_argument(
        "--title",
        default="Training Progress",
        help="Chart title",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Rolling average window size (default: 5)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Draw a horizontal threshold line at this value (e.g., 0.9)",
    )
    parser.add_argument(
        "--threshold-label",
        default=None,
        help="Label for the threshold line",
    )
    parser.add_argument(
        "--color",
        default=None,
        help="Line color (hex or named). Auto-detected from title if not set.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Display label for the metric in the legend (default: metric name)",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Truncate data after this step (useful to exclude diverged continuation runs)",
    )

    args = parser.parse_args()

    print(f"Loading events from: {args.logdir}")
    events = load_events(args.logdir, args.metric)
    print(f"  Raw events: {len(events)}")

    steps, values = deduplicate(events)
    print(f"  After dedup: {len(steps)} unique steps")

    # Truncate if requested
    if args.max_step is not None:
        mask = steps <= args.max_step
        steps = steps[mask]
        values = values[mask]
        print(f"  After truncation (max_step={args.max_step}): {len(steps)} steps")

    plot_curve(
        steps=steps,
        values=values,
        title=args.title,
        metric_label=args.label or args.metric,
        output=args.output,
        smooth_window=args.smooth_window,
        threshold=args.threshold,
        threshold_label=args.threshold_label,
        color=args.color,
    )
    print("Done.")


if __name__ == "__main__":
    main()
