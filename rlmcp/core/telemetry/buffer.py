"""Thread-safe Circular Metric Ring Buffer."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from typing import Any


class TelemetryBuffer:
  """Ring buffer for streaming scalar metrics during RL training.

  Storage is kept both ways round: per-metric series for plotting, and
  per-iteration rows for trend analysis. Rows are maintained incrementally on
  :meth:`push`, so :meth:`as_rows` -- called from inside the training loop on
  every ``get_metrics`` -- costs the rows it returns, never a rebuild and sort
  of every recorded point.
  """

  def __init__(
      self,
      maxlen: int = 5000,
      on_drop: Callable[[str, Any], None] | None = None,
  ):
    """Args:
      maxlen: points kept per metric, and iterations kept as rows.
      on_drop: called as ``on_drop(key, value)`` the first time a key's value
        cannot be coerced to float -- so a misnamed or non-scalar metric is
        reported once instead of vanishing silently. Called outside the lock.
    """
    self.maxlen = maxlen
    self._lock = threading.Lock()
    self._series: dict[str, deque[tuple[int, float]]] = {}
    # Per-iteration rows, insertion-ordered by iteration (out-of-order pushes,
    # e.g. after a checkpoint rollback, re-sort the order list -- rare).
    self._rows: dict[int, dict[str, float]] = {}
    self._row_order: list[int] = []
    self._on_drop = on_drop
    self._warned_keys: set = set()

  def push(self, iteration: int, metrics: dict[str, float]) -> None:
    """Pushes a dictionary of scalar metrics at a given iteration.

    Re-pushing the same iteration updates it in place, last value wins -- the
    service loop legitimately runs several times at one iteration (commands
    answered while paused), and appending would double the x-values a plot
    draws for that iteration.
    """
    dropped: list[tuple[str, Any]] = []
    with self._lock:
      row = self._rows.get(iteration)
      if row is None:
        row = {"iteration": iteration}
        self._rows[iteration] = row
        if self._row_order and iteration < self._row_order[-1]:
          # Out-of-order iteration (e.g. rollback); keep the order sorted.
          self._row_order.append(iteration)
          self._row_order.sort()
        else:
          self._row_order.append(iteration)
        while len(self._row_order) > self.maxlen:
          oldest = self._row_order.pop(0)
          self._rows.pop(oldest, None)
      for key, val in metrics.items():
        try:
          f_val = float(val)
        except (ValueError, TypeError):
          if key not in self._warned_keys:
            self._warned_keys.add(key)
            dropped.append((key, val))
          continue
        dq = self._series.get(key)
        if dq is None:
          dq = deque(maxlen=self.maxlen)
          self._series[key] = dq
        if dq and dq[-1][0] == iteration:
          dq[-1] = (iteration, f_val)  # Same-iteration re-push: last wins.
        else:
          dq.append((iteration, f_val))
        row[key] = f_val
    if self._on_drop is not None:
      for key, val in dropped:
        try:
          self._on_drop(key, val)
        except Exception:
          pass  # A broken sink must not take the push down.

  def get_series(
      self, metric_name: str, last_n: int | None = None
  ) -> list[tuple[int, float]]:
    """Retrieves (iteration, value) pairs for a given metric.

    ``last_n=None`` returns everything; ``last_n=0`` returns nothing.
    """
    with self._lock:
      if metric_name not in self._series:
        return []
      if last_n is not None and last_n <= 0:
        return []
      data = list(self._series[metric_name])
      return data[-last_n:] if last_n else data

  def get_latest_metrics(self) -> dict[str, float]:
    """Returns the most recent value for all tracked metrics."""
    with self._lock:
      latest = {}
      for key, dq in self._series.items():
        if dq:
          latest[key] = dq[-1][1]
      return latest

  def list_metrics(self) -> list[str]:
    """Returns list of all metric names tracked."""
    with self._lock:
      return list(self._series.keys())

  def as_rows(self, last_n: int | None = None) -> list[dict[str, float]]:
    """Per-iteration rows, oldest first.

    Metrics are stored per name because they arrive at different rates; trend
    analysis wants them the other way round. The rows are maintained
    incrementally by :meth:`push`, so this returns copies of the requested
    window -- O(rows returned), independent of how many points are stored.
    ``last_n=None`` returns everything; ``last_n=0`` returns nothing.
    """
    with self._lock:
      if last_n is not None and last_n <= 0:
        return []
      keys = self._row_order[-last_n:] if last_n else list(self._row_order)
      return [dict(self._rows[k]) for k in keys]
