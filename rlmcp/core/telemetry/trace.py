"""Per-step state traces for one environment.

Iteration-level metrics (``Episode_Reward/*``, PPO losses) tell you *that* a
policy is bad. To see *why* -- a knee buzzing at 40 Hz, a foot skating, a torque
saturating -- you need the raw per-step signals. :class:`TraceRecorder` keeps a
fixed-size ring of them for a single env so the cost stays bounded and the GPU
sync happens only while a trace is armed.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


class TraceRecorder:
  """Ring buffer of per-step arrays for one environment.

  Channels are declared implicitly by the first :meth:`record` call. Every later
  call must supply the same channels with the same widths.
  """

  def __init__(self, capacity: int = 4000, dt: float = 0.02):
    self.capacity = int(capacity)
    self.dt = float(dt)
    self._lock = threading.Lock()
    self._channels: dict[str, np.ndarray] = {}
    self._labels: dict[str, list[str]] = {}
    self._count = 0  # Total records ever written.
    self._env_id: int | None = None
    self._armed = False
    self._stop_after: int | None = None
    self._started_at: float | None = None
    self.meta: dict[str, Any] = {}

  # Arming.

  @property
  def armed(self) -> bool:
    return self._armed

  @property
  def env_id(self) -> int | None:
    return self._env_id

  @property
  def num_records(self) -> int:
    return min(self._count, self.capacity)

  def arm(
      self,
      env_id: int,
      num_steps: int | None = None,
      labels: dict[str, list[str]] | None = None,
      meta: dict[str, Any] | None = None,
  ) -> None:
    """Start recording ``env_id``, clearing anything held from a prior trace."""
    with self._lock:
      self._channels.clear()
      self._count = 0
      self._env_id = int(env_id)
      self._armed = True
      self._stop_after = int(num_steps) if num_steps else None
      self._started_at = time.time()
      self._labels = dict(labels or {})
      self.meta = dict(meta or {})

  def disarm(self) -> None:
    with self._lock:
      self._armed = False

  # Recording.

  def record(self, sample: dict[str, np.ndarray]) -> bool:
    """Append one step. Returns True while the trace is still collecting."""
    with self._lock:
      if not self._armed:
        return False
      idx = self._count % self.capacity
      for name, value in sample.items():
        arr = np.atleast_1d(np.asarray(value, dtype=np.float32)).ravel()
        buf = self._channels.get(name)
        if buf is None:
          buf = np.zeros((self.capacity, arr.shape[0]), dtype=np.float32)
          self._channels[name] = buf
        if buf.shape[1] != arr.shape[0]:
          continue  # Width changed (e.g. sensor reconfigured); skip channel.
        buf[idx] = arr
      self._count += 1
      if self._stop_after is not None and self._count >= self._stop_after:
        self._armed = False
        return False
      return True

  # Reading.

  def channels(self) -> list[str]:
    with self._lock:
      return sorted(self._channels)

  def labels(self, channel: str) -> list[str]:
    """Human-readable component names, e.g. joint names for ``joint_pos``."""
    with self._lock:
      if channel in self._labels:
        return list(self._labels[channel])
      buf = self._channels.get(channel)
      width = buf.shape[1] if buf is not None else 0
      return [f"{channel}[{i}]" for i in range(width)]

  def snapshot(self) -> dict[str, np.ndarray]:
    """Return every channel in chronological order."""
    with self._lock:
      n = min(self._count, self.capacity)
      if n == 0:
        return {}
      out: dict[str, np.ndarray] = {}
      for name, buf in self._channels.items():
        if self._count <= self.capacity:
          out[name] = buf[:n].copy()
        else:
          start = self._count % self.capacity
          out[name] = np.concatenate([buf[start:], buf[:start]], axis=0)
      out["time"] = np.arange(n, dtype=np.float32) * self.dt
      return out

  def save_npz(self, path: Path | str) -> Path:
    """Persist the trace (plus labels and metadata) as a .npz archive.

    Labels and metadata are stored as JSON in 0-d unicode arrays, so the
    archive contains no pickled objects: metadata values keep their types
    (``{"iteration": 7}`` round-trips as an int, not ``"7"``) and
    :func:`load_npz` can read the file with ``allow_pickle=False``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = self.snapshot()
    payload: dict[str, Any] = dict(data)
    for name in list(data):
      if name == "time":
        continue
      payload[f"__labels__{name}"] = np.array(json.dumps(self.labels(name)))
    payload["__meta__"] = np.array(json.dumps(self.meta, default=str, sort_keys=True))
    np.savez_compressed(path, **payload)
    return path

  def info(self) -> dict[str, Any]:
    with self._lock:
      return {
          "armed": self._armed,
          "env_id": self._env_id,
          "num_records": min(self._count, self.capacity),
          "capacity": self.capacity,
          "dt": self.dt,
          "duration_s": round(min(self._count, self.capacity) * self.dt, 3),
          "channels": sorted(self._channels),
          "started_at": self._started_at,
          "meta": dict(self.meta),
      }


def _decode_labels(value: np.ndarray) -> list[str]:
  """Labels from either format: 0-d JSON string, or a legacy 1-d array."""
  if value.ndim == 0:
    try:
      return [str(x) for x in json.loads(str(value))]
    except (ValueError, TypeError):
      return []
  return [str(x) for x in value]


def _decode_meta(value: np.ndarray) -> dict[str, Any]:
  """Metadata from either format: 0-d JSON string, or legacy ``k=v`` items.

  The legacy format stringified every value, so it yields ``{"iteration":
  "7"}`` where the JSON format yields ``{"iteration": 7}``.
  """
  if value.ndim == 0:
    try:
      loaded = json.loads(str(value))
      return dict(loaded) if isinstance(loaded, dict) else {}
    except (ValueError, TypeError):
      return {}
  meta: dict[str, Any] = {}
  for item in value:
    text = str(item)
    if "=" in text:
      k, v = text.split("=", 1)
      meta[k] = v
  return meta


def load_npz(path: Path | str, allow_legacy: bool = False) -> dict[str, Any]:
  """Load a trace written by :meth:`TraceRecorder.save_npz`.

  Args:
    path: the ``.npz`` archive.
    allow_legacy: traces written before labels/meta moved to JSON hold pickled
      object arrays, and unpickling can execute code from the file -- so by
      default they are refused with an error naming the problem. Pass ``True``
      only for a legacy file whose origin you trust.
  """
  raw = np.load(path, allow_pickle=allow_legacy)
  data: dict[str, np.ndarray] = {}
  labels: dict[str, list[str]] = {}
  meta: dict[str, Any] = {}
  try:
    for key in raw.files:
      try:
        value = raw[key]
      except ValueError as exc:
        raise ValueError(
            f"'{path}' contains pickled arrays -- a trace written by an older "
            "rlmcp. Unpickling can execute code embedded in the file; pass "
            "allow_legacy=True only if you trust where this trace came from."
        ) from exc
      if key.startswith("__labels__"):
        labels[key[len("__labels__"):]] = _decode_labels(value)
      elif key == "__meta__":
        meta = _decode_meta(value)
      else:
        data[key] = value
  finally:
    raw.close()
  return {"data": data, "labels": labels, "meta": meta}
