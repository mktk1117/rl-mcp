"""Clips of the run taken on a schedule, so nobody has to remember to ask.

A run is watched through numbers, and numbers endorse exploits -- the metric
climbs while the robot skates on its ankles. Looking at the policy is the fix
that works, and it does not happen because looking costs a command at the
moment you are busy reading the last one. So the run films itself: a clip at
iteration 0 -- the untrained baseline every later clip is read against -- and
then at gaps that double, each one filed on the run record.

    0   50   100   200   400   800   1600   3200   5200   7200 ...

Doubling is dense while behaviour is changing fastest and thins out on its own,
which also bounds what the clips cost: their number grows with the *logarithm*
of the run length. The gap stops doubling at ``max_every`` so a long run still
shows what it is doing now, and :data:`DEFAULT_BUDGET_MB` stops the schedule
outright if the files ever add up -- a log directory filled overnight is a
worse outcome than a trajectory that stops and says why.

Configured with one value everywhere (``video_every=``, ``--video-every``,
``rlmcp video --every``): ``"double"``, ``"double:100:5000"`` to move the first
gap and the cap, a plain ``200`` for a flat interval, or ``0`` for no clips.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

DEFAULT_FIRST = 50
"""The first clip after the untrained baseline; every gap after it doubles."""

DEFAULT_MAX_EVERY = 2000
"""The longest the schedule will ever wait.

Without it the clip after 51 200 would be at 102 400 -- so the answer to "what
does it look like now" would be a clip from days ago.
"""

DEFAULT_BUDGET_MB = 200.0
"""Disk one run's clips may take before the schedule stops. ``0`` for no limit."""


def _flat(every: int) -> Optional["Cadence"]:
  """A flat interval, or None for zero. A negative one is a typo, not "off":
  reading it as "no clips" would be the silent reinterpretation this parser
  exists to prevent."""
  if every < 0:
    raise ValueError(f"A clip cadence cannot be negative ({every}); use 0 for none.")
  return Cadence(every, every) if every else None


class Cadence:
  """When clips happen: gaps of ``first``, doubling, capped at ``max_every``.

  A flat interval is the same object with the cap equal to the first gap, which
  is why there is one class here rather than a family of schedules.
  """

  def __init__(self, first: int = DEFAULT_FIRST,
               max_every: Optional[int] = DEFAULT_MAX_EVERY):
    if int(first) <= 0:
      raise ValueError(f"A clip cadence needs a positive gap, not {first}.")
    self.first = int(first)
    self.max_every = None if not max_every else max(int(max_every), self.first)

  @staticmethod
  def parse(spec: Any) -> Optional["Cadence"]:
    """Read a cadence from ``"double[:first[:cap]]"``, a number, or None.

    ``None`` in means the default; ``None`` out means "no clips" (``0``,
    ``"off"``), which is an answer rather than an error. Anything else that
    does not parse raises, because a mistyped schedule quietly becoming a
    different one is worse than a failed launch.
    """
    if spec is None:
      return Cadence()
    if isinstance(spec, Cadence):
      return spec
    if isinstance(spec, bool):  # bool is an int; nobody means True here.
      raise ValueError(f"A cadence is a number or a string, not {spec!r}.")
    if isinstance(spec, (int, float)):
      return _flat(int(spec))

    text = str(spec).strip().lower()
    if text in ("", "0", "off", "none", "never"):
      return None
    parts = text.split(":")
    try:
      if parts[0] in ("double", "auto", "x2"):
        numbers = [int(p) for p in parts[1:] if p]
        return Cadence(*(numbers or [DEFAULT_FIRST]))
      flat = int(parts[0])
    except ValueError:
      raise ValueError(
          f"Could not read '{spec}' as a cadence. Write 'double', "
          "'double:<first>:<cap>', a flat interval like '200', or '0'."
      ) from None
    return _flat(flat)

  def every_at(self, iteration: int) -> int:
    """The gap this iteration sits in -- the number a status line shows."""
    previous, step = 0, self.first
    while step <= max(0, int(iteration)):
      following = step * 2
      if self.max_every and following - step > self.max_every:
        return self.max_every  # Past the cap the sequence is evenly spaced.
      previous, step = step, following
    return step - previous

  def next_after(self, iteration: int) -> int:
    """The iteration the next clip is due at, strictly after ``iteration``."""
    position = max(0, int(iteration))
    step = self.first
    while step <= position:
      following = step * 2
      if self.max_every and following - step > self.max_every:
        return step + self.max_every * (1 + (position - step) // self.max_every)
      step = following
    return step

  @property
  def spec(self) -> str:
    return f"double:{self.first}:{self.max_every or 0}"

  def prose(self) -> str:
    if self.max_every == self.first:
      return f"every {self.first:,} iterations"
    return (f"0, {self.first:,}, {self.first * 2:,}, {self.first * 4:,} … each gap "
            f"twice the last" + (f", capped at {self.max_every:,}" if self.max_every else ""))


class ProgressVideoSchedule:
  """When the next clip is due, and what happened to the last one.

  Owns no simulator and starts no jobs -- the controller does both. This is the
  bookkeeping, including what the clips have cost, which is the part a silent
  scheduler would hide.
  """

  def __init__(self, every: Any = None, seconds: float = 4.0, env_id: int = 0,
               budget_mb: float = DEFAULT_BUDGET_MB):
    self.cadence: Optional[Cadence] = Cadence.parse(every)
    self.seconds = float(seconds)
    self.env_id = int(env_id)
    self.budget_mb = max(0.0, float(budget_mb))

    self.next_due = 0  # Iteration 0: the untrained baseline.
    self.count = 0
    self.skipped = 0
    self.bytes = 0
    self.last_iteration: Optional[int] = None
    self.last_path = ""
    self.last_error = ""
    self.stopped_because = ""
    self.seen_iteration = 0

  def configure(self, every: Any = None, seconds: Optional[float] = None,
                env_id: Optional[int] = None, budget_mb: Optional[float] = None,
                iteration: int = 0) -> None:
    """Change the schedule mid-run. A new cadence takes effect here, not after
    the current gap expires: somebody asking for every 50 means starting now."""
    if every is not None:
      self.cadence = Cadence.parse(every)
      self.stopped_because = ""
      self.next_due = int(iteration)
    if seconds is not None:
      self.seconds = float(seconds)
    if env_id is not None:
      self.env_id = int(env_id)
    if budget_mb is not None:
      self.budget_mb = max(0.0, float(budget_mb))
      if self.stopped_because and self.bytes < self.budget_mb * 1e6:
        self.stopped_because = ""  # Raising the budget starts them again.

  @property
  def active(self) -> bool:
    return self.cadence is not None and not self.stopped_because

  @property
  def megabytes(self) -> float:
    return round(self.bytes / 1e6, 2)

  def due(self, iteration: int) -> bool:
    self.seen_iteration = max(self.seen_iteration, int(iteration))
    return self.active and int(iteration) >= self.next_due

  def fired(self, iteration: int) -> None:
    """Mark a clip as started. Counted from the iteration that fired, not the
    one that was due, so a run whose iterations jump does not fire again to
    catch up on clips nobody would watch."""
    self.next_due = (self.cadence or Cadence()).next_after(iteration)

  def completed(self, iteration: int, path: str, size_bytes: int = 0) -> str:
    """Record a clip that landed. Returns a note if it spent the disk budget."""
    self.count += 1
    self.bytes += max(0, int(size_bytes))
    self.last_iteration = int(iteration)
    self.last_path = str(path)
    self.last_error = ""
    # Compared in bytes, not in the rounded megabytes a status line shows.
    if self.budget_mb and self.bytes >= self.budget_mb * 1e6:
      self.stopped_because = (
          f"progress clips reached their {self.budget_mb:g} MB budget at "
          f"iteration {int(iteration)} ({self.count} clips)")
      return (f"Progress clips are off: {self.stopped_because}. Raise it with "
              "`rlmcp video --budget-mb N`.")
    return ""

  def failed(self, iteration: int, error: str) -> None:
    """Record a clip that did not happen. Not fatal to the schedule: the gaps
    double, so a backend that cannot render costs a handful of events over a
    run rather than one per iteration."""
    self.skipped += 1
    self.last_iteration = int(iteration)
    self.last_error = str(error)

  def describe(self) -> Dict[str, Any]:
    """The status-payload view: what is scheduled and how it is going."""
    return {
        "enabled": self.active,
        "every": self.cadence.every_at(self.seen_iteration) if self.cadence else None,
        "cadence": self.cadence.spec if self.cadence else "",
        "cadence_prose": self.cadence.prose() if self.cadence else "no clips",
        "seconds": self.seconds,
        "env_id": self.env_id,
        "next_iteration": self.next_due if self.active else None,
        "clips": self.count,
        "skipped": self.skipped,
        "megabytes": self.megabytes,
        "budget_mb": self.budget_mb or None,
        "last_iteration": self.last_iteration,
        "last_path": self.last_path,
        "last_error": self.last_error,
        "stopped_because": self.stopped_because,
        "at": time.time(),
    }
