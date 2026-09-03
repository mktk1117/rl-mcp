"""The legged-gym-shaped parameter surface, against a fake env of that shape.

No Genesis, no GPU, no display. The fake mirrors Genesis's own
``examples/locomotion/go2_env.py`` in the three places that decide whether this
adapter is correct or merely plausible:

* ``reward_scales`` is pre-multiplied by ``dt``, in place, on the same dict
  object ``reward_cfg["reward_scales"]`` is bound to;
* ``commands_limits`` is built once from ``command_cfg`` and is what the
  sampler reads afterwards;
* ``reward_functions`` binds one function per key of ``reward_scales``, at
  construction, and never again.

Each of those is a way for a write to look like it worked and do nothing, which
is the failure this file exists to prevent.
"""

from __future__ import annotations

import pytest

from rlmcp.adapters.legged_gym_style.access import ParameterAccess
from rlmcp.adapters.legged_gym_style.spec import FlatEnvSpec, NotAFlatEnv, detect
from rlmcp.core.parameters.spec import Liveness, ParameterCategory


@pytest.fixture
def env(flat_env):
  return flat_env


@pytest.fixture
def access(env) -> ParameterAccess:
  return ParameterAccess(env)


def keys(access: ParameterAccess) -> set:
  return {spec.key for spec in access.discover()}


# Detection.


def test_a_conventional_env_needs_no_spec(env):
  assert detect(env) == FlatEnvSpec()


def test_an_env_of_the_wrong_shape_is_refused_by_name():
  class NotOne:
    def __init__(self):
      self.observations = {}

  with pytest.raises(NotAFlatEnv) as excinfo:
    detect(NotOne())
  message = str(excinfo.value)
  assert "reward_scales" in message and "env_cfg" in message
  assert "FlatEnvSpec" in message, "the error should say how to fix it"
  assert "observations" in message, "and what the env does have"


def test_an_explicit_spec_names_attributes_the_convention_does_not(env):
  env.weights = env.reward_scales
  del env.reward_scales
  spec = FlatEnvSpec(reward_scales="weights")
  assert detect(env, spec) is spec
  assert "reward.action_rate.weight" in keys(ParameterAccess(env, spec))


# Discovery.


def test_parameters_are_discovered_from_the_dicts_the_env_keeps(access):
  found = keys(access)
  assert {"reward.tracking_lin_vel.weight",
          "reward.lin_vel_z.weight",
          "reward.action_rate.weight"} <= found
  assert "reward.params.tracking_sigma" in found
  assert "termination.pitch_greater_than" in found
  assert "command.lin_vel_x" in found
  assert "action.scale" in found


def test_the_scales_dict_is_not_also_walked_as_config(access):
  """It is served as weights; walking it too would double every term up under
  a second key, in units nobody writes."""
  assert not [k for k in keys(access) if k.startswith("reward.params.")
              and "reward_scales" in k]


def test_an_unclaimed_env_cfg_entry_is_still_tunable(access):
  """A fork's extra knob should be reachable the day it is added."""
  assert "env.resampling_time_s" in keys(access)


def test_categories_let_an_agent_ask_for_one_family(access):
  by_key = {s.key: s for s in access.discover()}
  assert by_key["reward.action_rate.weight"].category is ParameterCategory.REWARD
  assert by_key["command.lin_vel_x"].category is ParameterCategory.CURRICULUM
  assert by_key["action.scale"].category is ParameterCategory.ACTION
  assert (by_key["termination.pitch_greater_than"].category
          is ParameterCategory.TERMINATION)


# The dt round trip.


def test_a_reward_weight_reads_back_in_the_units_the_config_used(access, env):
  """The stored number is -0.005 * dt. What an agent sees is -0.005."""
  assert env.reward_scales["action_rate"] == pytest.approx(-0.005 * env.dt)
  assert access.get("reward.action_rate.weight") == pytest.approx(-0.005)


def test_setting_a_weight_stores_the_dt_scaled_value(access, env):
  access.set("reward.action_rate.weight", -0.01)
  assert env.reward_scales["action_rate"] == pytest.approx(-0.01 * env.dt)
  assert access.get("reward.action_rate.weight") == pytest.approx(-0.01)


def test_a_new_weight_changes_the_next_reward_evaluation(access, env):
  before = env.total_reward()
  access.set("reward.tracking_lin_vel.weight", 2.0)
  assert env.total_reward() == pytest.approx(before + 1.0 * env.dt)


def test_an_env_that_did_not_premultiply_needs_no_conversion(env):
  env.reward_scales["action_rate"] = -0.005
  access = ParameterAccess(env, FlatEnvSpec(scales_premultiplied_by_dt=False))
  assert access.get("reward.action_rate.weight") == pytest.approx(-0.005)
  access.set("reward.action_rate.weight", -0.02)
  assert env.reward_scales["action_rate"] == pytest.approx(-0.02)


def test_a_weight_alone_cannot_add_a_term_and_the_refusal_says_what_can(access):
  with pytest.raises(KeyError) as excinfo:
    access.set("reward.feet_air_time.weight", 1.0)
  assert "bound once" in str(excinfo.value), (
      "the message should say why the term is missing, not only which exist")
  assert "add-reward" in str(excinfo.value), (
      "a weight is not a term; the message should name the verb that adds one")


# Command ranges: the write has to reach the sampler, not just the config.


def test_a_command_range_reads_from_what_the_sampler_reads(access, env):
  assert access.get("command.lin_vel_x") == [0.5, 1.5]
  env.commands_limits[1][0] = 2.5
  assert access.get("command.lin_vel_x") == [0.5, 2.5], (
      "reads must come from commands_limits, not from command_cfg"
  )


def test_setting_a_command_range_changes_what_gets_sampled(access, env):
  access.set("command.lin_vel_x", [1.0, 2.0])
  assert env.sample_command(0) == (1.0, 2.0), (
      "the write must land on commands_limits; command_cfg alone is inert"
  )


def test_setting_a_command_range_keeps_the_config_honest(access, env):
  access.set("command.ang_vel", [-2.0, 2.0])
  assert list(env.command_cfg["ang_vel_range"]) == [-2.0, 2.0]


def test_a_backwards_command_range_is_refused(access, env):
  with pytest.raises(ValueError, match="min above max"):
    access.set("command.lin_vel_x", [2.0, 1.0])
  assert env.sample_command(0) == (0.5, 1.5), "a refused write changes nothing"


def test_a_command_range_takes_a_pair_not_a_scalar(access):
  with pytest.raises(ValueError, match=r"\[min, max\]"):
    access.set("command.lin_vel_x", 1.0)


def test_channels_are_numbered_when_the_names_cannot_be_trusted(env):
  """Names come from command_cfg's *_range keys in order. If that disagrees
  with the tensor width, numbering beats labelling the wrong axis."""
  del env.command_cfg["ang_vel_range"]
  found = keys(ParameterAccess(env))
  assert "command.channel_0" in found
  assert "command.lin_vel_x" not in found


# What is refused, and why.


def test_a_value_baked_in_at_construction_is_refused_with_a_reason(access, env):
  spec = {s.key: s for s in access.discover()}["env.kp"]
  assert spec.liveness is Liveness.AT_STARTUP
  with pytest.raises(ValueError, match="at_startup"):
    access.set("env.kp", 40.0)
  assert env.env_cfg["kp"] == 20.0


def test_the_episode_length_is_refused_because_it_is_already_baked(access):
  with pytest.raises(ValueError, match="at_startup"):
    access.set("env.episode_length_s", 30.0)


def test_a_live_env_cfg_entry_is_writable(access, env):
  access.set("action.scale", 0.5)
  assert env.env_cfg["action_scale"] == 0.5


def test_a_termination_limit_is_writable(access, env):
  access.set("termination.pitch_greater_than", 25.0)
  assert env.env_cfg["termination_if_pitch_greater_than"] == 25.0


# The vocabulary is the one the other backends use.


def test_reward_weight_keys_match_the_manager_based_spelling(access):
  """An agent that has driven mjlab should not learn a second vocabulary."""
  assert access.get("reward.tracking_lin_vel.weight") == pytest.approx(1.0)
