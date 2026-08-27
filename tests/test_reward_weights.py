"""A reward weight is unbounded, and a dormant one can be raised to any weight.

The adapter used to derive a bound from the term's own value, which capped a
term shipped at 0.0 well below the scale its siblings ran at. These pin that no
new guess replaces it.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import pytest

from rlmcp.adapters.manager_based.access import ParameterAccess
from rlmcp.core.parameters.registry import ParameterRegistry


@dataclasses.dataclass
class _RewardCfg:
  weight: float
  func: Optional[Any] = None


class _RewardManager:
  """The shape `AccessProvider._term_cfgs` reads: `active_terms` + `get_term_cfg`."""

  def __init__(self, weights):
    self._cfgs = {k: _RewardCfg(weight=w) for k, w in weights.items()}
    self.active_terms = list(self._cfgs)

  def get_term_cfg(self, name):
    return self._cfgs[name]


class _Env:
  def __init__(self, weights):
    self.reward_manager = _RewardManager(weights)


# A task where one term ships dormant and its siblings run large.
TASK = {"catch": 300.0, "drop_penalty": -200.0, "hand_tracking": 0.0}


def _specs(weights=TASK):
  return {s.key: s for s in ParameterAccess(_Env(weights)).discover()}


def _registry(weights=TASK):
  """Discovery wired to the registry that validates a write, as the lab does."""
  access = ParameterAccess(_Env(weights))
  registry = ParameterRegistry()
  for spec in access.discover():
    registry.register(
        spec,
        setter=lambda value, k=spec.key: access.set(k, value) is not None,
        getter=lambda k=spec.key: access.get(k),
    )
  return access, registry


def test_a_reward_weight_carries_no_bounds():
  spec = _specs()["reward.hand_tracking.weight"]
  assert spec.min_value is None and spec.max_value is None


def test_no_reward_parameter_invents_a_bound():
  """Guards against a new magic number arriving on any reward leaf."""
  bounded = {
      key: (s.min_value, s.max_value)
      for key, s in _specs().items()
      if s.min_value is not None or s.max_value is not None
  }
  assert bounded == {}, f"reward leaves grew bounds out of nowhere: {bounded}"


@pytest.mark.parametrize("weight", [40.0, 600.0, -600.0, 1e6])
def test_a_dormant_knob_can_be_raised_to_any_weight(weight):
  """The regression: `max(10.0, abs(0.0) * 20.0)` refused everything past 10."""
  access, registry = _registry()

  assert registry.set_value("reward.hand_tracking.weight", weight)
  assert access.get("reward.hand_tracking.weight") == weight


def test_a_reward_can_still_be_turned_into_a_penalty():
  access, registry = _registry()

  registry.set_value("reward.catch.weight", -300.0)
  assert access.get("reward.catch.weight") == -300.0


def test_a_weight_still_has_to_be_a_number():
  """Dropping the bound drops no type checking."""
  _, registry = _registry()

  with pytest.raises(ValueError):
    registry.set_value("reward.catch.weight", "heavy")

