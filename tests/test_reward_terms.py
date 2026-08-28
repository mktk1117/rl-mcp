"""Adding a reward term mid-run: compiling it, installing it, keeping it.

Three layers, pinned separately because they fail differently:

* :mod:`rlmcp.core.reward_source` turns text into a function, and every way
  that can go wrong should produce a message naming the problem rather than a
  traceback out of ``exec``;
* :mod:`rlmcp.adapters.manager_based.reward_terms` performs the manager
  surgery, whose whole risk is doing *part* of it -- so the tests here check
  the buffers as much as the term list, and check that a refused term left
  nothing behind;
* the controller ties them together and has to make the addition outlive the
  process, which is what the session file and the event log are for.

The fake manager mirrors mjlab's real shape (``_term_names``/``_term_cfgs``
lists, ``_episode_sums`` dict, ``_step_reward`` matrix), because that shape is
the actual contract being relied on.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

import pytest
import torch

from rlmcp.adapters.base import NotSupported, SimAdapter
from rlmcp.adapters.manager_based.reward_terms import (
    RewardInstallError,
    install_reward_term,
)
from rlmcp.core.reward_source import RewardSourceError, compile_reward_source

NUM_ENVS = 4

UPRIGHT = """
def upright(env, scale: float = 1.0):
  return torch.ones(env.num_envs) * scale
"""


@dataclasses.dataclass
class _RewardTermCfg:
  func: Any = None
  weight: float = 0.0
  params: Dict[str, Any] = dataclasses.field(default_factory=dict)


class _RewardManager:
  """mjlab's RewardManager, reduced to the parts the surgery touches."""

  def __init__(self, terms: Dict[str, _RewardTermCfg], num_envs: int = NUM_ENVS):
    self.num_envs = num_envs
    self.device = "cpu"
    self.cfg = dict(terms)
    self._term_names: List[str] = list(terms)
    self._term_cfgs: List[_RewardTermCfg] = list(terms.values())
    self._class_term_cfgs: List[_RewardTermCfg] = []
    self._episode_sums = {
        name: torch.zeros(num_envs) for name in self._term_names
    }
    self._reward_buf = torch.zeros(num_envs)
    self._step_reward = torch.zeros((num_envs, len(self._term_names)))
    self.resolved: List[str] = []

  @property
  def active_terms(self) -> List[str]:
    return self._term_names

  def get_term_cfg(self, name: str) -> _RewardTermCfg:
    return self._term_cfgs[self._term_names.index(name)]

  def _resolve_common_term_cfg(self, name: str, cfg: Any) -> None:
    self.resolved.append(name)

  def compute(self, dt: float = 0.02) -> torch.Tensor:
    """The real loop, including the two buffers a partial install breaks."""
    self._reward_buf[:] = 0.0
    for index, (name, cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
      value = cfg.func(self._env, **cfg.params) * cfg.weight * dt
      self._reward_buf += value
      self._episode_sums[name] += value
      self._step_reward[:, index] = value / dt
    return self._reward_buf


class _Env:
  def __init__(self, terms: Optional[Dict[str, _RewardTermCfg]] = None):
    self.num_envs = NUM_ENVS
    self.device = "cpu"
    if terms is None:
      terms = {
          "alive": _RewardTermCfg(
              func=lambda env, **_: torch.ones(env.num_envs), weight=1.0)
      }
    self.reward_manager = _RewardManager(terms)
    self.reward_manager._env = self


def _compile(source: str = UPRIGHT, name: str = "upright"):
  return compile_reward_source(source, name=name, namespace={"torch": torch})


# Compiling what the agent wrote.


def test_compiles_a_function_and_digests_its_source():
  compiled = _compile()
  assert compiled.name == "upright"
  assert compiled.func_name == "upright"
  assert compiled.func(_Env()).shape == (NUM_ENVS,)
  assert compiled.digest == _compile().digest
  assert compiled.digest != _compile(UPRIGHT + "\n# changed\n").digest


def test_function_may_be_named_differently_from_the_term():
  compiled = compile_reward_source(
      "def torso_height_l2(env):\n  return torch.zeros(env.num_envs)\n",
      name="posture", namespace={"torch": torch})
  assert compiled.name == "posture"
  assert compiled.func_name == "torso_height_l2"


@pytest.mark.parametrize(
    "source, name, expected",
    [
        ("def f(env): return 1", "not an identifier", "identifier"),
        ("", "upright", "No source"),
        ("def upright(env)\n  return 1\n", "upright", "does not parse"),
        ("UPRIGHT = 3\n", "upright", "defines no function"),
        ("def upright():\n  return 1\n", "upright", "no positional argument"),
    ],
)
def test_bad_source_is_refused_with_a_message_naming_the_problem(
    source, name, expected):
  with pytest.raises(RewardSourceError, match=expected):
    compile_reward_source(source, name=name, namespace={"torch": torch})


def test_two_functions_and_no_match_is_ambiguous_rather_than_a_guess():
  source = "def a(env):\n  return 1\n\n\ndef b(env):\n  return 2\n"
  with pytest.raises(RewardSourceError, match="which one scores"):
    compile_reward_source(source, name="upright", namespace={"torch": torch})


def test_a_raising_module_body_says_so_is_not_the_reward_computing():
  with pytest.raises(RewardSourceError, match="module body"):
    compile_reward_source(
        "import nonexistent_module_xyz\n\n\ndef upright(env):\n  return 1\n",
        name="upright", namespace={"torch": torch})


# Installing it into a manager that is already running.


def test_installing_appends_the_term_and_widens_every_buffer():
  env = _Env()
  compiled = _compile()
  result = install_reward_term(
      env, name="upright", func=compiled.func, weight=2.0,
      params={"scale": 3.0})

  manager = env.reward_manager
  assert manager.active_terms == ["alive", "upright"]
  assert result["index"] == 1
  # The three places a half-install shows up.
  assert "upright" in manager._episode_sums
  assert manager._step_reward.shape == (NUM_ENVS, 2)
  assert "upright" in manager.cfg
  assert manager.resolved == ["upright"]

  # And it actually scores: 1*1 from alive, 3*2 from upright.
  rewards = manager.compute(dt=1.0)
  assert torch.allclose(rewards, torch.full((NUM_ENVS,), 7.0))
  assert torch.allclose(manager._episode_sums["upright"],
                        torch.full((NUM_ENVS,), 6.0))


def test_the_new_weight_is_live_and_tunable_like_any_other():
  env = _Env()
  install_reward_term(env, name="upright", func=_compile().func, weight=1.0)
  env.reward_manager.get_term_cfg("upright").weight = 5.0
  assert torch.allclose(env.reward_manager.compute(dt=1.0),
                        torch.full((NUM_ENVS,), 6.0))


def test_a_duplicate_name_is_refused_and_points_at_set_parameter():
  env = _Env()
  with pytest.raises(RewardInstallError, match="set_parameter"):
    install_reward_term(env, name="alive", func=_compile().func, weight=1.0)


@pytest.mark.parametrize(
    "source, expected",
    [
        ("def bad(env):\n  return torch.ones(3)\n", "one score per environment"),
        ("def bad(env):\n  return 1.0\n", "has to return a torch tensor"),
        ("def bad(env):\n  raise ValueError('nope')\n", "raised on its trial call"),
        ("def bad(env):\n  return torch.full((4,), float('nan'))\n", "non-finite"),
    ],
)
def test_a_term_that_fails_its_trial_leaves_the_manager_untouched(
    source, expected):
  env = _Env()
  manager = env.reward_manager
  before = (list(manager.active_terms), manager._step_reward.shape,
            set(manager._episode_sums))

  with pytest.raises(RewardInstallError, match=expected):
    install_reward_term(env, name="bad", func=_compile(source, "bad").func,
                        weight=1.0)

  assert (list(manager.active_terms), manager._step_reward.shape,
          set(manager._episode_sums)) == before
  assert "bad" not in manager.cfg
  # The run keeps going, which is the whole point of trialling first.
  manager.compute(dt=1.0)


def test_params_reach_the_function_on_every_call():
  env = _Env()
  install_reward_term(env, name="upright", func=_compile().func, weight=1.0,
                      params={"scale": 4.0})
  assert torch.allclose(env.reward_manager.compute(dt=1.0),
                        torch.full((NUM_ENVS,), 5.0))


def test_a_wrong_param_name_is_refused_as_a_call_error():
  env = _Env()
  with pytest.raises(RewardInstallError, match="could not be called"):
    install_reward_term(env, name="upright", func=_compile().func, weight=1.0,
                        params={"scal": 4.0})


def test_an_environment_without_a_reward_manager_says_so():
  class _Bare:
    num_envs = NUM_ENVS

  with pytest.raises(RewardInstallError, match="no reward_manager"):
    install_reward_term(_Bare(), name="upright", func=_compile().func,
                        weight=1.0)


def test_a_backend_that_cannot_grow_its_reward_function_raises_not_supported():
  class _Fixed(SimAdapter):
    def discover_parameters(self):
      return []

    def get_parameter(self, key):
      raise KeyError(key)

    def set_parameter(self, key, value):
      return True

  with pytest.raises(NotSupported):
    _Fixed().add_reward_term("upright", func=None, weight=1.0)
