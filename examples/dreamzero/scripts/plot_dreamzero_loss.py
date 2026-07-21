#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Plot DreamZero SFT training loss curves from TensorBoard event files.

Produces a two-panel chart:
  - Top: Total Loss, Dynamics Loss, Action Loss (with checkpoint resume annotation)
  - Bottom: Learning Rate schedule

Reads TensorBoard event files, deduplicates overlapping steps from checkpoint
resumptions (keeps the latest-written value per step), and renders a
publication-quality PNG.

Usage:
    python examples/dreamzero/scripts/plot_dreamzero_loss.py \
        --logdir /fsx/checkpoints/dreamzero-sft/tensorboard \
        --output examples/dreamzero/assets/training_loss.png

    # With checkpoint resume annotation:
    python examples/dreamzero/scripts/plot_dreamzero_loss.py \
        --logdir /fsx/checkpoints/dreamzero-sft/tensorboard \
        --output examples/dreamzero/assets/training_loss.png \
        --resume-step 100

    # Custom title and subtitle:
    python examples/dreamzero/scripts/plot_dreamzero_loss.py \
        --logdir /fsx/checkpoints/dreamzero-sft/tensorboard \
        --output examples/dreamzero/assets/training_loss.png \
        --resume-step 100 \
        --title "DreamZero SFT Training Loss" \
        --subtitle "2x p5en.48xlarge (16x H200), FSDP2 full_shard, EFA RDMA"

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
    print(
        "ERROR: tensorboard package required. Install with: pip install tensorboard",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Chart style
# ---------------------------------------------------------------------------
CHART_STYLE = {
    "figure.dpi": 150,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
}

# Metric display configuration
METRICS = {
    "train/loss": {
        "label": "Total Loss",
        "color": "#1565C0",  # blue
        "marker": "o",
        "zorder": 3,
    },
    "train/dynamics_loss": {
        "label": "Dynamics Loss",
        "color": "#2E7D32",  # green
        "marker": "s",
        "zorder": 2,
    },
    "train/action_loss": {
        "label": "Action Loss",
        "color": "#C62828",  # red
        "marker": "^",
        "zorder": 2,
    },
}

LR_METRIC = "train/learning_rate"


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

    return all_events


def deduplicate(events: list[tuple[float, int, float]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Deduplicate overlapping steps from checkpoint resumptions.

    When training is resumed from a checkpoint, TensorBoard logs may contain
    multiple values for the same step. We keep the value with the latest
    wall_time for each step.

    Returns (steps, values) arrays sorted by step.
    """
    step_data: dict[int, tuple[float, float]] = {}
    for wall_time, step, value in events:
        if step not in step_data or wall_time > step_data[step][0]:
            step_data[step] = (wall_time, value)

    sorted_steps = sorted(step_data.keys())
    steps = np.array(sorted_steps)
    values = np.array([step_data[s][1] for s in sorted_steps])

    return steps, values


def discover_metrics(logdir: str) -> list[str]:
    """List available scalar metrics in the logdir."""
    logdir = Path(logdir)
    all_metrics = set()

    event_dirs = set()
    for event_file in logdir.rglob("events.out.tfevents.*"):
        event_dirs.add(str(event_file.parent))

    if not event_dirs:
        event_dirs.add(str(logdir))

    for edir in sorted(event_dirs):
        try:
            ea = EventAccumulator(edir)
            ea.Reload()
            all_metrics.update(ea.Tags().get("scalars", []))
        except Exception:
            pass

    return sorted(all_metrics)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_dreamzero_loss(
    logdir: str,
    output: str,
    title: str,
    subtitle: str,
    resume_step: int | None = None,
    max_step: int | None = None,
):
    """Render the two-panel DreamZero training loss chart."""
    plt.rcParams.update(CHART_STYLE)

    # Load loss metrics
    loss_data = {}
    for metric_key, config in METRICS.items():
        events = load_events(logdir, metric_key)
        if events:
            steps, values = deduplicate(events)
            if max_step is not None:
                mask = steps <= max_step
                steps, values = steps[mask], values[mask]
            loss_data[metric_key] = (steps, values, config)
            print(f"  {config['label']}: {len(steps)} points, final={values[-1]:.4f}")
        else:
            print(f"  WARNING: No events found for {metric_key}", file=sys.stderr)

    if not loss_data:
        print("ERROR: No loss metrics found. Available metrics:", file=sys.stderr)
        for m in discover_metrics(logdir):
            print(f"  - {m}", file=sys.stderr)
        sys.exit(1)

    # Load learning rate
    lr_events = load_events(logdir, LR_METRIC)
    lr_data = None
    if lr_events:
        lr_steps, lr_values = deduplicate(lr_events)
        if max_step is not None:
            mask = lr_steps <= max_step
            lr_steps, lr_values = lr_steps[mask], lr_values[mask]
        lr_data = (lr_steps, lr_values)
        print(f"  Learning Rate: {len(lr_steps)} points, final={lr_values[-1]:.2e}")
    else:
        print("  WARNING: No learning rate data found, skipping LR panel", file=sys.stderr)

    # Create figure
    if lr_data is not None:
        fig, (ax_loss, ax_lr) = plt.subplots(
            2, 1, figsize=(10, 7), height_ratios=[2, 1], sharex=True
        )
    else:
        fig, ax_loss = plt.subplots(1, 1, figsize=(10, 5))
        ax_lr = None

    # --- Top panel: Loss curves ---
    for metric_key, (steps, values, config) in loss_data.items():
        ax_loss.plot(
            steps,
            values,
            linewidth=2,
            color=config["color"],
            marker=config["marker"],
            markersize=5,
            markevery=max(1, len(steps) // 20),
            label=config["label"],
            zorder=config["zorder"],
        )

    # Resume annotation
    if resume_step is not None:
        ax_loss.axvline(
            x=resume_step,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            zorder=1,
        )
        # Shaded regions
        xlim_left = ax_loss.get_xlim()[0] if ax_loss.get_xlim()[0] > 0 else 0
        all_steps = np.concatenate([s for s, _, _ in loss_data.values()])
        x_max = all_steps.max()
        ax_loss.axvspan(xlim_left, resume_step, alpha=0.04, color="blue", zorder=0)
        ax_loss.axvspan(resume_step, x_max, alpha=0.04, color="green", zorder=0)
        # Label
        y_top = ax_loss.get_ylim()[1] if ax_loss.get_ylim()[1] > 0 else 0.6
        ax_loss.text(
            resume_step - 2,
            y_top * 0.85,
            f"Resume from\ncheckpoint-{resume_step}",
            ha="right",
            va="top",
            fontsize=9,
            color="gray",
            style="italic",
        )

    ax_loss.set_ylabel("Loss")
    ax_loss.set_ylim(bottom=0)
    ax_loss.legend(loc="upper right", fontsize=10)

    # Title (main + subtitle)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    ax_loss.set_title(subtitle, fontsize=10, color="gray", pad=4)

    # --- Bottom panel: Learning rate ---
    if ax_lr is not None and lr_data is not None:
        lr_steps, lr_values = lr_data
        ax_lr.plot(lr_steps, lr_values, linewidth=2.5, color="black")
        ax_lr.set_ylabel(r"Learning Rate ($\times 10^{-6}$)")
        ax_lr.set_xlabel("Training Step")

        # Scale y-axis to 1e-6 units for readability
        lr_max = lr_values.max()
        if lr_max < 1e-3:
            ax_lr.ticklabel_format(style="sci", axis="y", scilimits=(-6, -6))

        # Resume annotation on LR panel too
        if resume_step is not None:
            ax_lr.axvline(
                x=resume_step, color="gray", linestyle="--", linewidth=1.2, alpha=0.7
            )
    elif ax_lr is None:
        ax_loss.set_xlabel("Training Step")

    fig.tight_layout()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  Output: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot DreamZero SFT training loss from TensorBoard events.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--logdir",
        required=True,
        help="Path to TensorBoard event directory (supports nested dirs)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG file path (required unless --list-metrics)",
    )
    parser.add_argument(
        "--title",
        default="DreamZero SFT Training Loss",
        help="Main chart title (default: 'DreamZero SFT Training Loss')",
    )
    parser.add_argument(
        "--subtitle",
        default="2x p5en.48xlarge (16x H200), FSDP2 full_shard, EFA RDMA",
        help="Subtitle with infrastructure details",
    )
    parser.add_argument(
        "--resume-step",
        type=int,
        default=None,
        help="Training step where checkpoint resume occurred (draws annotation)",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Truncate data after this step",
    )
    parser.add_argument(
        "--loss-metric",
        default="train/loss",
        help="Total loss metric name (default: train/loss)",
    )
    parser.add_argument(
        "--dynamics-metric",
        default="train/dynamics_loss",
        help="Dynamics loss metric name (default: train/dynamics_loss)",
    )
    parser.add_argument(
        "--action-metric",
        default="train/action_loss",
        help="Action loss metric name (default: train/action_loss)",
    )
    parser.add_argument(
        "--lr-metric",
        default="train/learning_rate",
        help="Learning rate metric name (default: train/learning_rate)",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List available metrics in the logdir and exit",
    )

    args = parser.parse_args()

    if args.list_metrics:
        print(f"Available metrics in {args.logdir}:")
        for m in discover_metrics(args.logdir):
            print(f"  - {m}")
        return

    if args.output is None:
        parser.error("--output is required unless --list-metrics is used")

    # Allow overriding metric names
    global METRICS, LR_METRIC
    METRICS = {
        args.loss_metric: METRICS.get("train/loss", METRICS["train/loss"]),
        args.dynamics_metric: METRICS.get(
            "train/dynamics_loss", METRICS["train/dynamics_loss"]
        ),
        args.action_metric: METRICS.get(
            "train/action_loss", METRICS["train/action_loss"]
        ),
    }
    LR_METRIC = args.lr_metric

    print(f"Loading events from: {args.logdir}")
    plot_dreamzero_loss(
        logdir=args.logdir,
        output=args.output,
        title=args.title,
        subtitle=args.subtitle,
        resume_step=args.resume_step,
        max_step=args.max_step,
    )
    print("Done.")


if __name__ == "__main__":
    main()
