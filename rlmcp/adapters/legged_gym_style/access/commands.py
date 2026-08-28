"""Command ranges -- written where the sampler actually reads them.

This is the trap in this family, and it is a quiet one. ``command_cfg`` carries
``lin_vel_x_range`` and its siblings, so it looks like the place to write. It is
not: ``__init__`` reads those once into a pair of tensors (``commands_limits``)
and the resampler draws from *those* for the rest of the run. A write to
``command_cfg`` alone therefore succeeds, reads back correctly, and changes
nothing about the commands the robot is given.

Command ranges are the main curriculum lever -- "walk faster now" is a write to
one of these -- so a silent failure here is a curriculum that appears to climb
while the task stays where it started. That is worth more care than the rest of
this package put together.

So each range is a synthetic that writes the tensors *and* keeps ``command_cfg``
in step, and reads back from the tensors, which are the thing that decides what
the robot is asked to do. When an environment has no such tensors, the cfg is
the live surface and is written directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rlmcp.adapters.access.base import AccessProvider, Synthetic, Term
from rlmcp.core.parameters.spec import ParameterCategory

RANGE_SUFFIX = "_range"


class CommandAccess(AccessProvider):
  """``command.<name>`` as a ``[min, max]`` pair, per commanded channel."""

  domain = "command"
  category = ParameterCategory.CURRICULUM

  def __init__(self, env: Any, spec: Any):
    super().__init__(env)
    self.spec = spec

  @property
  def cfg(self) -> dict:
    cfg = self.spec.resolve(self.env, "command_cfg", None)
    return cfg if isinstance(cfg, dict) else {}

  @property
  def limits(self) -> tuple[Any, Any] | None:
    """The ``(lower, upper)`` tensors the sampler reads, when there are any."""
    limits = self.spec.resolve(self.env, "command_limits", None)
    if isinstance(limits, (tuple, list)) and len(limits) == 2:
      return limits[0], limits[1]
    return None

  def channel_names(self) -> list[str]:
    """One name per commanded channel, in the order the tensors are packed.

    The names come from the ``*_range`` keys of ``command_cfg`` in insertion
    order, which is the order they were zipped into the tensors. That is an
    assumption about the environment, so it is checked rather than trusted: if
    the counts disagree, the channels are numbered instead of being given names
    that might belong to the wrong axis.
    """
    named = [k[: -len(RANGE_SUFFIX)] for k in self.cfg if k.endswith(RANGE_SUFFIX)]
    limits = self.limits
    if limits is None:
      return named
    width = len(limits[0])
    if len(named) != width:
      return [f"channel_{i}" for i in range(width)]
    return named

  def available(self) -> bool:
    return bool(self.channel_names())

  def terms(self) -> list[Term]:
    return []

  def synthetic(self) -> list[Synthetic]:
    return [
        Synthetic(
            key=f"{self.domain}.{name}",
            getter=self._getter(index, name),
            setter=self._setter(index, name),
            default=self._read(index, name),
            description=(
                f"Range the '{name}' command is sampled from, as [min, max]"
            ),
            data_type="range",
        )
        for index, name in enumerate(self.channel_names())
    ]

  # Reads and writes go to the tensors when there are tensors.

  def _read(self, index: int, name: str) -> list[float]:
    limits = self.limits
    if limits is None:
      return [float(v) for v in self.cfg.get(f"{name}{RANGE_SUFFIX}", (0.0, 0.0))]
    lower, upper = limits
    return [float(lower[index]), float(upper[index])]

  def _getter(self, index: int, name: str):
    def get() -> list[float]:
      return self._read(index, name)
    return get

  def _setter(self, index: int, name: str):
    def put(value: Any) -> bool:
      pair = self._coerce(value, name)
      limits = self.limits
      if limits is not None:
        lower, upper = limits
        lower[index] = pair[0]
        upper[index] = pair[1]
      # Kept in step so the cfg still describes the run, and so an environment
      # that rebuilds its tensors on reset rebuilds them from the new numbers.
      key = f"{name}{RANGE_SUFFIX}"
      if key in self.cfg:
        self.cfg[key] = type(self.cfg[key])(pair)
      elif limits is None:
        raise KeyError(
            f"No command channel '{name}'. Available: {self.channel_names()}."
        )
      return True
    return put

  @staticmethod
  def _coerce(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
      raise ValueError(
          f"Command range '{name}' takes a [min, max] pair; got {value!r}."
      )
    try:
      low, high = float(value[0]), float(value[1])
    except (TypeError, ValueError):
      raise ValueError(
          f"Command range '{name}' takes two numbers; got {value!r}."
      ) from None
    if low > high:
      raise ValueError(
          f"Command range '{name}' has min above max: {low} > {high}."
      )
    return low, high

  def describe(self, term: Term, parts: Sequence[str], value: Any) -> str:
    return f"Command parameter '{'.'.join(parts)}'"
