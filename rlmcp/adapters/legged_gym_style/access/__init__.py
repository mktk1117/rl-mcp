"""The parameter providers for legged-gym-shaped environments.

Same machinery as every other family -- :mod:`rlmcp.adapters.access` does the
walking, reading and writing -- and the same key vocabulary, so
``reward.tracking_lin_vel.weight`` means what it means on mjlab. What differs is
only where the values are found: dicts on the environment instance rather than
manager term configs.

The providers need to know which attribute holds which dict, so they take a
:class:`~rlmcp.adapters.legged_gym_style.spec.FlatEnvSpec` alongside the
environment. That is the only difference in how they are constructed, and it is
absorbed here so nothing downstream has to care.
"""

from __future__ import annotations

from functools import partial
from typing import Any, List, Optional

from rlmcp.adapters.access.registry import ParameterAccess as _ParameterAccess
from rlmcp.adapters.legged_gym_style.access.commands import CommandAccess
from rlmcp.adapters.legged_gym_style.access.env_cfg import (
    ActionAccess,
    EnvAccess,
    TerminationAccess,
)
from rlmcp.adapters.legged_gym_style.access.rewards import RewardAccess
from rlmcp.adapters.legged_gym_style.spec import FlatEnvSpec

PROVIDER_TYPES: List[type] = [
    RewardAccess,
    TerminationAccess,
    CommandAccess,
    ActionAccess,
    EnvAccess,
]

__all__ = [
    "ActionAccess",
    "CommandAccess",
    "EnvAccess",
    "ParameterAccess",
    "PROVIDER_TYPES",
    "RewardAccess",
    "TerminationAccess",
]


class ParameterAccess(_ParameterAccess):
  """The tunable surface of one legged-gym-shaped environment."""

  PROVIDER_TYPES = PROVIDER_TYPES

  def __init__(
      self,
      env: Any,
      spec: Optional[FlatEnvSpec] = None,
      provider_types: Optional[List[type]] = None,
  ):
    self.spec = spec or FlatEnvSpec()
    bound = [
        partial(provider, spec=self.spec)
        for provider in (provider_types or self.PROVIDER_TYPES)
    ]
    super().__init__(env, bound)
