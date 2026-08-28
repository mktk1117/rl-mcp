"""Charts an agent can actually read.

Two families:

* :func:`plot_metric_series` -- iteration-level curves (rewards, losses,
  terrain level), one panel per metric.
* :func:`plot_trace` -- per-step signals for a single environment, which is
  where jerkiness, saturation and slip become visible.

Both return PNG bytes so the caller can hand them straight to an LLM or write
them to the session's artifact directory.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

_LINE = "#1f77b4"
_ACCENT = "#d62728"
_GRID = {"linestyle": "--", "alpha": 0.4}


def _finish(fig: Any, dpi: int = 110) -> bytes:
  buf = io.BytesIO()
  fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
  plt.close(fig)
  return buf.getvalue()


def _empty(message: str) -> bytes:
  fig, ax = plt.subplots(figsize=(6, 2.2))
  try:
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
    ax.axis("off")
    return _finish(fig)
  finally:
    # Idempotent after _finish; on an exception path it stops the Agg backend
    # accumulating one orphaned figure per failed plot for the life of the run.
    plt.close(fig)


def _grid_shape(n: int, max_cols: int = 2) -> tuple[int, int]:
  cols = min(max_cols, max(1, n))
  rows = (n + cols - 1) // cols
  return rows, cols


def plot_metric_series(
    series: dict[str, Sequence[tuple[float, float]]],
    title: str | None = None,
    smooth_window: int = 1,
    markers: Sequence[tuple[float, str]] | None = None,
) -> bytes:
  """Plot ``{metric_name: [(iteration, value), ...]}`` as a panel grid.

  ``markers`` draws labelled vertical lines -- curriculum stage changes, mostly.
  Seeing a reward dip land exactly on a stage boundary is the difference between
  "the policy broke" and "the task just got harder".
  """
  named = [(k, list(v)) for k, v in series.items()]
  populated = [(k, v) for k, v in named if v]
  if not populated:
    missing = ", ".join(k for k, _ in named) or "(none requested)"
    return _empty(f"No data recorded for: {missing}")

  rows, cols = _grid_shape(len(populated))
  fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 3.1 * rows), squeeze=False)
  try:
    flat = axes.flatten()

    for ax, (name, points) in zip(flat, populated, strict=False):
      xs = np.array([p[0] for p in points], dtype=np.float64)
      ys = np.array([p[1] for p in points], dtype=np.float64)
      ax.plot(xs, ys, color=_LINE, linewidth=1.0,
              alpha=0.85 if smooth_window > 1 else 1.0)
      if smooth_window > 1 and ys.size >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(ys, kernel, mode="valid")
        ax.plot(xs[smooth_window - 1 :], smoothed, color=_ACCENT, linewidth=1.8)
      for position, label in markers or ():
        if xs.size and not (xs.min() <= position <= xs.max()):
          continue
        ax.axvline(position, color="#666666", linestyle=":", linewidth=1.2)
        ax.annotate(
            label,
            xy=(position, 0.97),
            xycoords=("data", "axes fraction"),
            rotation=90,
            va="top",
            ha="right",
            fontsize=7,
            color="#444444",
        )
      ax.set_title(name, fontsize=10, fontweight="bold")
      ax.set_xlabel("iteration", fontsize=8)
      ax.tick_params(labelsize=8)
      ax.grid(True, **_GRID)
      if ys.size:
        ax.annotate(
            f"last {ys[-1]:.4g}",
            xy=(0.99, 0.03),
            xycoords="axes fraction",
            ha="right",
            fontsize=8,
            color="#444444",
        )

    for ax in flat[len(populated) :]:
      ax.axis("off")
    if title:
      fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _finish(fig)
  finally:
    plt.close(fig)  # Idempotent after _finish; frees the figure on exceptions.


def plot_trace(
    data: dict[str, np.ndarray],
    labels: dict[str, list[str]] | None = None,
    channels: Sequence[str] | None = None,
    components: Sequence[str] | None = None,
    title: str | None = None,
    max_components: int = 8,
) -> bytes:
  """Plot per-step trace channels, one panel per channel.

  Args:
    data: channel -> (num_steps, width) arrays plus an optional ``time`` vector.
    labels: channel -> component names, used for the legend and for
      ``components`` filtering.
    channels: which channels to draw; defaults to the joint/base essentials.
    components: substring filter on component names, e.g. ``["knee", "ankle"]``.
    max_components: cap on traces per panel, so a 29-DoF humanoid stays legible.
  """
  labels = labels or {}
  if not data:
    return _empty("No trace recorded yet -- run record_trace first.")

  default_order = [
      "joint_pos",
      "joint_vel",
      "joint_torque",
      "action",
      "command",
      "base_lin_vel",
      "base_ang_vel",
      "foot_contact",
  ]
  available = [k for k in data if k != "time"]
  channels = [c for c in (channels or default_order) if c in available] or available
  time = data.get("time")
  if time is None:
    first = data[channels[0]]
    time = np.arange(first.shape[0], dtype=np.float32)

  rows, cols = _grid_shape(len(channels), max_cols=2)
  fig, axes = plt.subplots(rows, cols, figsize=(7.0 * cols, 2.7 * rows), squeeze=False)
  try:
    flat = axes.flatten()

    for ax, channel in zip(flat, channels, strict=False):
      values = np.atleast_2d(data[channel])
      if values.shape[0] != time.shape[0] and values.shape[1] == time.shape[0]:
        values = values.T
      names = labels.get(channel) or [f"{channel}[{i}]" for i in range(values.shape[1])]
      indices = list(range(values.shape[1]))
      if components:
        needles = [c.lower() for c in components]
        filtered = [
            i
            for i in indices
            if i < len(names) and any(n in names[i].lower() for n in needles)
        ]
        if filtered:
          indices = filtered
      truncated = len(indices) > max_components
      indices = indices[:max_components]

      for i in indices:
        name = names[i] if i < len(names) else f"[{i}]"
        ax.plot(time[: values.shape[0]], values[:, i], linewidth=1.0, label=name)
      ax.set_title(
          channel + (f"  (showing {len(indices)} of {values.shape[1]})" if truncated else ""),
          fontsize=10,
          fontweight="bold",
      )
      ax.set_xlabel("time [s]", fontsize=8)
      ax.tick_params(labelsize=8)
      ax.grid(True, **_GRID)
      if len(indices) <= 10:
        ax.legend(fontsize=6, ncol=2, loc="upper right", framealpha=0.7)

    for ax in flat[len(channels) :]:
      ax.axis("off")
    if title:
      fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _finish(fig)
  finally:
    plt.close(fig)  # Idempotent after _finish; frees the figure on exceptions.


def plot_terrain_status(status: dict[str, Any], title: str | None = None) -> bytes:
  """Bar chart of environments per terrain and their mean difficulty level."""
  per_terrain = status.get("per_terrain") or []
  if not per_terrain:
    return _empty("No terrain grid in this environment.")

  names = [e["terrain"] for e in per_terrain]
  counts = [e.get("num_envs", 0) for e in per_terrain]
  means = [e.get("level_mean", 0.0) for e in per_terrain]
  ceiling = status.get("level_ceiling")

  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(7.0, len(names) * 1.1), 6.0))
  try:
    ax1.bar(names, counts, color=_LINE)
    ax1.set_ylabel("environments", fontsize=9)
    ax1.set_title("Environments per terrain", fontsize=10, fontweight="bold")
    ax1.tick_params(axis="x", labelrotation=30, labelsize=8)
    ax1.grid(True, axis="y", **_GRID)

    ax2.bar(names, means, color=_ACCENT)
    if ceiling:
      ax2.axhline(ceiling - 1, color="#444444", linestyle="--", linewidth=1.0,
                  label=f"ceiling (level {ceiling - 1})")
      ax2.legend(fontsize=8)
    ax2.set_ylabel("mean difficulty level", fontsize=9)
    ax2.set_title("Terrain level reached", fontsize=10, fontweight="bold")
    ax2.tick_params(axis="x", labelrotation=30, labelsize=8)
    ax2.grid(True, axis="y", **_GRID)

    if title:
      fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _finish(fig)
  finally:
    plt.close(fig)  # Idempotent after _finish; frees the figure on exceptions.


class TelemetryPlotter:
  """Thin adapter kept for callers holding a :class:`TelemetryBuffer`."""

  def __init__(self, buffer: Any):
    self.buffer = buffer

  def render_plots_png(
      self,
      metric_names: Sequence[str],
      last_n: int | None = 200,
      title: str | None = None,
      smooth_window: int = 1,
  ) -> bytes:
    names = list(metric_names) or self.buffer.list_metrics()[:4]
    series = {name: self.buffer.get_series(name, last_n=last_n) for name in names}
    return plot_metric_series(series, title=title, smooth_window=smooth_window)
