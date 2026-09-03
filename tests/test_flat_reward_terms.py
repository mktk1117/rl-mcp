"""A term the task never had, appended to a Go2Env-shaped environment.

The fake models the mechanism rather than the fix: ``FakeFlatEnv.total_reward``
is Go2Env's step loop -- walk ``reward_functions``, multiply each result by
``reward_scales[name]`` -- and ``episode_sums`` is filled in ``__init__`` the
way Go2Env fills it. So a term that is really installed changes the next
step's reward by exactly ``value * weight * dt``, and one that only looks
installed does not.
"""

from __future__ import annotations

import pytest
import torch
from conftest import FakeFlatEnv

from rlmcp.adapters.genesis.sim_adapter import GenesisSimAdapter
from rlmcp.adapters.legged_gym_style.access import ParameterAccess
from rlmcp.adapters.legged_gym_style.reward_terms import install_reward_term
from rlmcp.adapters.legged_gym_style.spec import FlatEnvSpec
from rlmcp.adapters.reward_terms import RewardInstallError


def upright(env, scale: float = 1.0):
  return torch.ones(env.num_envs) * scale


def snapshot(env: FakeFlatEnv) -> tuple:
  return (dict(env.reward_scales), dict(env.reward_functions),
          dict(env.episode_sums))


@pytest.fixture
def env() -> FakeFlatEnv:
  return FakeFlatEnv()


@pytest.fixture
def spec() -> FlatEnvSpec:
  return FlatEnvSpec()


# The term scores.


def test_the_term_scores_from_the_next_step(env, spec):
  before = env.total_reward()
  install_reward_term(env, spec, name="upright", func=upright, weight=2.0,
                      params={"scale": 3.0})
  assert env.total_reward() == pytest.approx(before + 3.0 * 2.0 * env.dt)


def test_the_weight_is_stored_the_way_the_environment_stores_its_own(env, spec):
  install_reward_term(env, spec, name="upright", func=upright, weight=2.0)
  # Go2Env pre-multiplies every scale by dt at init; a term added later has to
  # match, or the loop would pay it dt times too much.
  assert env.reward_scales["upright"] == pytest.approx(2.0 * env.dt)


def test_an_env_that_did_not_premultiply_stores_the_weight_as_is(env):
  spec = FlatEnvSpec(scales_premultiplied_by_dt=False)
  install_reward_term(env, spec, name="upright", func=upright, weight=2.0)
  assert env.reward_scales["upright"] == pytest.approx(2.0)


def test_the_function_is_bound_so_the_loop_can_call_it_bare(env, spec):
  install_reward_term(env, spec, name="upright", func=upright, weight=1.0,
                      params={"scale": 5.0})
  # Go2Env calls reward_functions[name]() with no arguments.
  value = env.reward_functions["upright"]()
  assert torch.equal(value, torch.full((env.num_envs,), 5.0))


def test_the_episode_sum_is_allocated_like_the_others(env, spec):
  like = env.episode_sums["tracking_lin_vel"]
  install_reward_term(env, spec, name="upright", func=upright, weight=1.0)
  new = env.episode_sums["upright"]
  assert new.shape == like.shape
  assert new.dtype == like.dtype
  assert new.device == like.device
  assert not new.any(), "a term that has not scored yet has summed nothing"


def test_the_report_names_the_term_its_slot_and_what_it_scored(env, spec):
  out = install_reward_term(env, spec, name="upright", func=upright,
                            weight=0.5, params={"scale": 2.0})
  assert out["name"] == "upright"
  assert out["index"] == len(env.reward_functions) - 1
  assert out["weight"] == 0.5
  assert out["trial_value"] == {"min": 2.0, "max": 2.0, "mean": 2.0}
  assert out["class_based"] is False


# Afterwards it is a parameter like any other.


def test_the_weight_is_tunable_afterwards_in_config_units(env, spec):
  install_reward_term(env, spec, name="upright", func=upright, weight=2.0)
  access = ParameterAccess(env, spec)
  access.discover()
  assert access.get("reward.upright.weight") == pytest.approx(2.0)
  access.set("reward.upright.weight", 4.0)
  assert env.reward_scales["upright"] == pytest.approx(4.0 * env.dt)


# Refusals leave the environment exactly as it was.


def test_a_name_already_bound_is_refused(env, spec):
  before = snapshot(env)
  with pytest.raises(RewardInstallError, match="already exists"):
    install_reward_term(env, spec, name="action_rate", func=upright,
                        weight=1.0)
  assert snapshot(env) == before


def test_a_term_that_raises_on_trial_is_not_installed(env, spec):
  def broken(env):
    raise ZeroDivisionError("no")

  before = snapshot(env)
  with pytest.raises(RewardInstallError, match="raised on its trial call"):
    install_reward_term(env, spec, name="broken", func=broken, weight=1.0)
  assert snapshot(env) == before


def test_a_term_of_the_wrong_shape_is_not_installed(env, spec):
  def per_joint(env):
    return torch.ones(env.num_envs, 12)

  before = snapshot(env)
  with pytest.raises(RewardInstallError, match="one score per environment"):
    install_reward_term(env, spec, name="per_joint", func=per_joint,
                        weight=1.0)
  assert snapshot(env) == before


def test_a_term_that_cannot_take_its_params_is_not_installed(env, spec):
  before = snapshot(env)
  with pytest.raises(RewardInstallError, match="could not be called"):
    install_reward_term(env, spec, name="upright", func=upright, weight=1.0,
                        params={"no_such": 1})
  assert snapshot(env) == before


def test_an_environment_without_the_dicts_says_where_it_looked(spec):
  class Bare:
    def __init__(self):
      self.num_envs = 4
      self.reward_scales = {"a": 1.0}
      self.env_cfg = {}

  with pytest.raises(RewardInstallError, match="reward_functions"):
    install_reward_term(Bare(), spec, name="upright", func=upright,
                        weight=1.0)


def test_a_term_reads_buffers_the_way_step_does(env, spec):
  """rsl_rl steps inside inference mode, so the buffers a term reads are
  inference tensors; the trial call has to run where the real call will."""
  seen = {}

  def probe(env):
    seen["inference"] = torch.is_inference_mode_enabled()
    return torch.ones(env.num_envs)

  install_reward_term(env, spec, name="probe", func=probe, weight=1.0)
  assert seen["inference"] is True


# Through the Genesis adapter, where the controller reaches it.


def test_the_genesis_adapter_installs_and_then_discovers_the_weight(genesis_env):
  sim = GenesisSimAdapter(genesis_env)
  keys_before = {p.key for p in sim.discover_parameters()}
  out = sim.add_reward_term(name="upright", func=upright, weight=0.5)
  assert out["trial_value"]["mean"] == 1.0
  keys_after = {p.key for p in sim.discover_parameters()}
  assert keys_after - keys_before == {"reward.upright.weight"}
  assert sim.get_parameter("reward.upright.weight") == pytest.approx(0.5)


def test_discovery_after_an_add_keeps_the_defaults_it_recorded_at_wrap(env, spec):
  """The registry re-asks its providers so the new weight shows up, but the
  weights it already served keep the default they had: that is what a reset
  goes back to, and a set in between must not move it."""
  access = ParameterAccess(env, spec)
  access.discover()
  access.set("reward.tracking_lin_vel.weight", 7.0)
  install_reward_term(env, spec, name="upright", func=upright, weight=2.0)
  defaults = {p.key: p.default_value for p in access.discover()}
  assert defaults["reward.tracking_lin_vel.weight"] == pytest.approx(1.0)
  assert defaults["reward.upright.weight"] == pytest.approx(2.0)
