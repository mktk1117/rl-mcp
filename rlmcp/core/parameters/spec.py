"""Declarative Parameter Specification and Schema."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class ParameterCategory(str, Enum):
  """Categories of parameters exposed to the agent."""
  REWARD = "reward"
  RL_HYPERPARAMETER = "rl"
  CURRICULUM = "curriculum"
  DOMAIN_RANDOMIZATION = "domain_randomization"
  PHYSICS = "physics"
  ACTION = "action"
  TERMINATION = "termination"
  OTHER = "other"


class Liveness(str, Enum):
  """When a write to a parameter actually reaches the running simulation."""

  LIVE = "live"
  """Re-read continuously; a write takes effect on the next step or batch."""

  AT_RESET = "at_reset"
  """Re-read on episode reset; a write takes effect from each env's next reset."""

  AT_STARTUP = "at_startup"
  """Only read when the environment is constructed; a write during this run
  changes nothing. Requires a config change and a restart."""

  INERT = "inert"
  """Cached by the owning term at construction and never re-read; a write can
  never take effect. Requires a config change and a restart."""


def _as_number(key: str, value: Any) -> float:
  """``value`` as a float, or a ValueError naming what a number looks like."""
  if isinstance(value, bool) or not isinstance(value, (int, float, str)):
    raise ValueError(f"Parameter '{key}' expects a number; got {value!r}.")
  try:
    return float(value)
  except ValueError:
    raise ValueError(f"Parameter '{key}' expects a number; got {value!r}.") from None


@dataclass
class ParameterSpec:
  """Schema definition for a mutable parameter.

  Attributes:
    key: Unique dotted identifier (e.g., 'reward.tracking_lin_vel.weight').
    data_type: Data type string ('float', 'int', 'bool', 'str', 'list', 'range').
    current_value: Current active value.
    default_value: Initial baseline default value.
    min_value: Optional minimum allowable bound for safety.
    max_value: Optional maximum allowable bound for safety.
    description: Human and LLM readable explanation of parameter effect.
    category: Category for grouping and querying.
    liveness: When a write actually reaches the simulation (see :class:`Liveness`).
  """
  key: str
  data_type: str
  current_value: Any
  default_value: Any
  min_value: Optional[float] = None
  max_value: Optional[float] = None
  description: str = ""
  category: ParameterCategory = ParameterCategory.OTHER
  liveness: Liveness = Liveness.LIVE

  def validate(self, value: Any) -> bool:
    """Validates the proposed value against shape, type and bounds.

    Raises ValueError before any write happens, so a rejected value leaves the
    live config untouched. Shape errors here are the cheap version of a crash
    that would otherwise surface minutes later inside the training loop.
    """
    if self.data_type == "range":
      if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            f"Parameter '{self.key}' expects a two-element [low, high] range; "
            f"got {value!r}."
        )
      low, high = (_as_number(self.key, v) for v in value)
      if low > high:
        raise ValueError(
            f"Parameter '{self.key}' range low {low} exceeds high {high}; "
            "expected [low, high] with low <= high."
        )
      numbers = [low, high]
    elif self.data_type == "bool":
      if not isinstance(value, bool) and not (
          isinstance(value, int) and value in (0, 1)
      ):
        raise ValueError(
            f"Parameter '{self.key}' expects true or false; got {value!r}."
        )
      return True
    elif self.data_type == "int":
      number = _as_number(self.key, value)
      if number != int(number):
        raise ValueError(
            f"Parameter '{self.key}' expects a whole number; {value!r} would "
            f"be truncated to {int(number)}."
        )
      numbers = [number]
    elif self.data_type == "float":
      numbers = [_as_number(self.key, value)]
    else:
      numbers = [value]

    for number in numbers:
      if self.min_value is not None and number < self.min_value:
        raise ValueError(
            f"Parameter '{self.key}' value {number} is below minimum allowed {self.min_value}."
        )
      if self.max_value is not None and number > self.max_value:
        raise ValueError(
            f"Parameter '{self.key}' value {number} is above maximum allowed {self.max_value}."
        )
    return True

  def to_dict(self) -> dict[str, Any]:
    return {
        "key": self.key,
        "type": self.data_type,
        "current": self.current_value,
        "default": self.default_value,
        "min": self.min_value,
        "max": self.max_value,
        "description": self.description,
        "category": self.category.value,
        "liveness": Liveness(self.liveness).value,
    }
