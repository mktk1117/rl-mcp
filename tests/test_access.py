"""The mjlab adapter against fake managers shaped like mjlab's.

These run without mjlab or a GPU. The access tests pin down the property that
motivates the whole design: parameters are found by walking the environment, so
a term or field nobody wrote code for still shows up. The state-contract tests
at the bottom pin the other half of the adapter surface: the trace channel
vocabulary, labels derived truthfully from the live env, and the error
convention declared in :mod:`rlmcp.adapters.base`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from rlmcp.adapters import base
from rlmcp.adapters.base import NotSupported, RunnerAdapter, SimAdapter
from rlmcp.adapters.manager_based.access import ParameterAccess
from rlmcp.adapters.manager_based.access import paths
from rlmcp.adapters import rsl_rl_runner as runner_adapter
from rlmcp.adapters.rsl_rl_runner import RslRlRunnerAdapter
from rlmcp.adapters.manager_based import metrics as state_metrics
from rlmcp.adapters.mjlab.state import rendering
from rlmcp.adapters.manager_based.sampling import StateSampler
from rlmcp.core import diagnostics as diag
from rlmcp.core.parameters.spec import ParameterCategory


# Fakes shaped like mjlab's managers.


@dataclass
class SceneEntityCfg:
  """Stands in for the real selector: a dataclass that is not a tunable."""

  joint_names: tuple = (".*",)
  body_names: tuple = ()


@dataclass
class TermCfg:
  func: Any = None
  weight: float = 0.0
  params: Dict[str, Any] = field(default_factory=dict)


def _track_linear_velocity(env, std, command_name):
  """A function-based term: mjlab passes ``**cfg.params`` on every call."""
  del env, std, command_name
  return 0.0


class VariablePosture:
  """Mirrors mjlab's variable_posture: caches the std params at construction.

  The real term resolves the regex-keyed std dicts to tensors in ``__init__``
  and its ``__call__`` literally ``del``s the std arguments -- the config copy
  is never read again. ``walking_threshold`` is used from the call args, so it
  stays live.
  """

  def __init__(self, params: Dict[str, Any]):
    self.std_standing = dict(params["std_standing"])
    self.std_walking = dict(params["std_walking"])

  def __call__(self, env, std_standing, std_walking, walking_threshold=0.5):
    del std_standing, std_walking  # Unused, like the real term.
    return 0.0

  def reset(self, env_ids=None):
    pass


class ListManager:
  """A manager with a flat list of terms (rewards, terminations, curriculum)."""

  def __init__(self, terms: Dict[str, TermCfg]):
    self._terms = terms

  @property
  def active_terms(self) -> List[str]:
    return list(self._terms)

  def get_term_cfg(self, name: str) -> TermCfg:
    return self._terms[name]


class EventManager(ListManager):
  """Events are grouped by mode, and the same lookup serves every mode."""

  def __init__(self, by_mode: Dict[str, Dict[str, TermCfg]]):
    self._by_mode = by_mode
    flat: Dict[str, TermCfg] = {}
    for terms in by_mode.values():
      flat.update(terms)
    super().__init__(flat)

  @property
  def active_terms(self) -> Dict[str, List[str]]:
    return {mode: list(terms) for mode, terms in self._by_mode.items()}


@dataclass
class Ranges:
  lin_vel_x: tuple = (-1.0, 1.0)
  ang_vel_z: tuple = (-0.5, 0.5)


@dataclass
class CommandCfg:
  ranges: Ranges = field(default_factory=Ranges)


class CommandTerm:
  def __init__(self):
    self.cfg = CommandCfg()


class CommandManager:
  def __init__(self):
    self._terms = {"twist": CommandTerm()}

  @property
  def active_terms(self) -> List[str]:
    return list(self._terms)

  def get_term(self, name: str) -> CommandTerm:
    return self._terms[name]


class ActionTerm:
  def __init__(self, scale: float = 0.5):
    self._scale = scale


class ActionManager:
  def __init__(self):
    self._terms = {"joint_pos": ActionTerm()}

  @property
  def active_terms(self) -> List[str]:
    return list(self._terms)

  def get_term(self, name: str) -> ActionTerm:
    return self._terms[name]


class FakeEnv:
  """Just enough environment for the access layer to walk."""

  def __init__(self):
    posture_params = {
        "std_standing": {".*elbow.*": 0.15},
        "std_walking": {".*elbow.*": 0.25},
        "walking_threshold": 0.5,
    }
    self.reward_manager = ListManager(
        {
            "track_linear_velocity": TermCfg(
                func=_track_linear_velocity,
                weight=2.0,
                params={"std": 0.5, "command_name": "twist"},
            ),
            "foot_slip": TermCfg(
                weight=-0.1,
                params={"command_threshold": 0.05, "asset_cfg": SceneEntityCfg()},
            ),
            "variable_posture": TermCfg(
                func=VariablePosture(posture_params),
                weight=1.0,
                params=dict(posture_params),
            ),
        }
    )
    self.termination_manager = ListManager(
        {"fell_over": TermCfg(params={"limit_angle": 1.22})}
    )
    self.event_manager = EventManager(
        {
            "interval": {
                "push_robot": TermCfg(
                    params={"velocity_range": {"x": (-0.5, 0.5), "roll": (-0.52, 0.52)}}
                )
            },
            "startup": {
                "foot_friction": TermCfg(
                    params={"ranges": (0.3, 1.2), "shared_random": True}
                )
            },
            "reset": {
                "reset_base": TermCfg(
                    params={"pose_range": {"x": (-0.5, 0.5), "yaw": (-3.14, 3.14)}}
                ),
                "reset_robot_joints": TermCfg(
                    params={
                        "position_range": (0.9, 1.1),
                        "use_default_offset": True,
                        "attempts": 4,
                    }
                ),
            },
        }
    )
    self.command_manager = CommandManager()
    self.action_manager = ActionManager()
    self.curriculum_manager = ListManager(
        {
            "command_vel": TermCfg(
                params={
                    "command_name": "twist",
                    "velocity_stages": [
                        {"step": 0, "lin_vel_x": (-1.0, 1.0)},
                        {"step": 5000, "lin_vel_x": (-1.5, 2.0)},
                    ],
                }
            )
        }
    )


@pytest.fixture
def access() -> ParameterAccess:
  return ParameterAccess(FakeEnv())


def keys(access: ParameterAccess) -> List[str]:
  return [s.key for s in access.discover()]


# Discovery.


def test_discovers_every_family(access):
  found = keys(access)
  assert "reward.track_linear_velocity.weight" in found
  assert "reward.foot_slip.params.command_threshold" in found
  assert "termination.fell_over.params.limit_angle" in found
  assert "event.interval.push_robot.params.velocity_range.x" in found
  assert "command.twist.ranges.lin_vel_x" in found
  assert "action.joint_pos.scale_gain" in found


def test_does_not_offer_things_that_are_not_knobs(access):
  found = keys(access)
  # Entity selectors, callables and plain strings are not tunable.
  assert not [k for k in found if "asset_cfg" in k]
  assert not [k for k in found if k.endswith(".func")]
  assert not [k for k in found if "command_name" in k]


def test_a_field_nobody_wrote_code_for_still_appears():
  """The point of reflection: new config fields need no adapter change."""
  env = FakeEnv()
  before = set(keys(ParameterAccess(env)))

  env.reward_manager.get_term_cfg("foot_slip").params["invented_later"] = 0.25
  env.reward_manager._terms["brand_new_term"] = TermCfg(weight=-1.0)

  after = set(keys(ParameterAccess(env)))

  assert after - before == {
      "reward.foot_slip.params.invented_later",
      "reward.brand_new_term.weight",
  }


def test_types_and_categories_are_carried(access):
  specs = {s.key: s for s in access.discover()}
  assert specs["reward.foot_slip.weight"].category == ParameterCategory.REWARD
  assert specs["reward.foot_slip.weight"].data_type == "float"
  assert specs["event.startup.foot_friction.params.ranges"].data_type == "range"
  assert (
      specs["event.startup.foot_friction.params.shared_random"].data_type == "bool"
  )
  assert (
      specs["event.interval.push_robot.params.velocity_range.x"].category
      == ParameterCategory.DOMAIN_RANDOMIZATION
  )


def test_reward_weight_gets_a_sign_aware_description_and_no_bounds(access):
  specs = {s.key: s for s in access.discover()}
  penalty = specs["reward.foot_slip.weight"]
  reward = specs["reward.track_linear_velocity.weight"]
  assert "penalty" in penalty.description
  assert "reward" in reward.description
  assert penalty.min_value is None and penalty.max_value is None


# Reads and writes.


def test_round_trip_a_scalar(access):
  access.set("reward.foot_slip.weight", -0.35)
  assert access.get("reward.foot_slip.weight") == -0.35


def test_round_trip_a_range_keeps_the_container_type(access):
  # A reset-mode event: writes to startup events are refused (tested below).
  env = access.env
  access.set("event.reset.reset_robot_joints.params.position_range", [0.1, 2.0])
  live = env.event_manager.get_term_cfg("reset_robot_joints").params["position_range"]
  assert live == (0.1, 2.0)
  assert isinstance(live, tuple)  # mjlab unpacks these; a list would be a landmine.
  assert access.get("event.reset.reset_robot_joints.params.position_range") == [0.1, 2.0]


def test_round_trip_a_nested_dict_range(access):
  key = "event.interval.push_robot.params.velocity_range.x"
  access.set(key, [-2.0, 2.0])
  assert access.get(key) == [-2.0, 2.0]


def test_int_and_bool_keep_their_type(access):
  access.set("event.reset.reset_robot_joints.params.use_default_offset", False)
  live = access.env.event_manager.get_term_cfg("reset_robot_joints").params
  assert live["use_default_offset"] is False

  access.set("event.reset.reset_robot_joints.params.attempts", 6.0)
  assert live["attempts"] == 6
  assert type(live["attempts"]) is int  # An int leaf must stay an int.


def test_a_dict_keyed_by_integers_round_trips():
  """mjlab's base_com randomization keys its ranges by integer axis index.

  Registered here under reset mode so the write is allowed: what this test
  pins is the int-key addressing, and the real (startup) base_com's write
  refusal is covered by the liveness tests below.
  """
  env = FakeEnv()
  env.event_manager._by_mode["reset"]["base_com"] = TermCfg(
      params={"ranges": {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)}}
  )
  env.event_manager._terms["base_com"] = env.event_manager._by_mode["reset"]["base_com"]
  access = ParameterAccess(env)

  key = "event.reset.base_com.params.ranges.2"
  assert key in keys(access)
  assert access.get(key) == [-0.03, 0.03]

  access.set(key, [-0.1, 0.1])
  live = env.event_manager.get_term_cfg("base_com").params["ranges"]
  assert live[2] == (-0.1, 0.1)
  assert 2 in live and "2" not in live  # The int key was updated, not a new str one.


def test_unknown_domain_lists_what_exists(access):
  with pytest.raises(KeyError) as excinfo:
    access.get("rewrd.foot_slip.weight")
  assert "reward" in str(excinfo.value)


def test_unknown_term_lists_what_exists(access):
  with pytest.raises(KeyError) as excinfo:
    access.get("reward.no_such_term.weight")
  assert "foot_slip" in str(excinfo.value)


def test_unknown_field_points_at_the_bad_segment(access):
  with pytest.raises(KeyError) as excinfo:
    access.get("reward.foot_slip.params.nope")
  assert "nope" in str(excinfo.value)


def test_a_dict_key_containing_dots_round_trips():
  """mjlab keys the G1 pose std dict by joint-name regex, e.g. `.*knee.*`.

  Without escaping, discovery advertises a key that get/set cannot resolve --
  which is exactly the failure a hand-written adapter hides by never exposing
  the parameter at all.
  """
  env = FakeEnv()
  env.reward_manager._terms["pose"] = TermCfg(
      weight=1.0, params={"std_walking": {r".*knee.*": 0.35, r".*ankle_roll.*": 0.1}}
  )
  access = ParameterAccess(env)

  key = next(k for k in keys(access) if "knee" in k)
  assert access.get(key) == 0.35

  access.set(key, 0.42)
  assert env.reward_manager.get_term_cfg("pose").params["std_walking"][r".*knee.*"] == 0.42
  # The neighbouring regex key must not have been touched.
  assert env.reward_manager.get_term_cfg("pose").params["std_walking"][r".*ankle_roll.*"] == 0.1


def test_structural_fields_are_not_offered_as_knobs():
  """Flags that change semantics are traps, not tuning surface."""
  env = FakeEnv()
  env.termination_manager._terms["time_out"] = TermCfg(params={})
  env.termination_manager.get_term_cfg("time_out").time_out = True
  env.event_manager.get_term_cfg("push_robot").is_global_time = False

  found = keys(ParameterAccess(env))

  assert not [k for k in found if k.endswith(".time_out")]
  assert not [k for k in found if "is_global_time" in k]


def test_event_keys_are_mode_qualified(access):
  # Longest-prefix term matching is what makes "<mode>.<name>" work.
  assert access.get("event.interval.push_robot.params.velocity_range.roll") == [
      -0.52,
      0.52,
  ]


# Provider-specific behaviour.


def test_setting_a_command_range_claims_the_curriculum_stage_table(access):
  notes = access.set("command.twist.ranges.lin_vel_x", [-0.6, 1.4])

  assert notes["curriculum_overridden"] == ["command_vel"]
  stages = access.env.curriculum_manager.get_term_cfg("command_vel").params[
      "velocity_stages"
  ]
  # Every stage, so the value survives whatever step count the run reaches.
  assert all(stage["lin_vel_x"] == (-0.6, 1.4) for stage in stages)


def test_a_command_range_no_curriculum_owns_reports_nothing(access):
  notes = access.set("command.twist.ranges.ang_vel_z", [-1.0, 1.0])
  assert notes == {}


def test_action_gain_scales_the_configured_value(access):
  term = access.env.action_manager.get_term("joint_pos")
  assert access.get("action.joint_pos.scale_gain") == 1.0

  access.set("action.joint_pos.scale_gain", 0.5)
  assert term._scale == 0.25  # Half of the configured 0.5.
  assert access.get("action.joint_pos.scale_gain") == 0.5


def test_action_gain_carries_no_bounds(access):
  """Only the task knows the scale a gain works at, so the adapter invents none."""
  gain = {s.key: s for s in access.discover()}["action.joint_pos.scale_gain"]
  assert gain.min_value is None and gain.max_value is None


def test_action_gain_is_relative_to_the_baseline_not_the_last_value(access):
  term = access.env.action_manager.get_term("joint_pos")
  access.set("action.joint_pos.scale_gain", 0.5)
  access.set("action.joint_pos.scale_gain", 0.5)
  assert term._scale == 0.25  # Not 0.125.

  access.set("action.joint_pos.scale_gain", 1.0)
  assert term._scale == 0.5  # Exactly the original.


def test_missing_managers_are_simply_absent():
  class Bare:
    reward_manager = ListManager({"only": TermCfg(weight=1.0)})

  access = ParameterAccess(Bare())
  assert access.domains() == ["reward"]
  assert keys(access) == ["reward.only.weight"]


# Shape and type validation: a bad value is refused before anything is written.


def test_scalar_over_a_range_is_refused_before_any_write(access):
  """The F12 landmine: a scalar written over [lo, hi] kills uniform_() later."""
  key = "command.twist.ranges.lin_vel_x"
  with pytest.raises(ValueError) as excinfo:
    access.set(key, 0.5)
  assert "[low, high]" in str(excinfo.value)
  # The live config is untouched -- and so is the curriculum stage table that
  # after_set would have rewritten had the write been allowed to land first.
  assert access.get(key) == [-1.0, 1.0]
  stages = access.env.curriculum_manager.get_term_cfg("command_vel").params[
      "velocity_stages"
  ]
  assert stages[0]["lin_vel_x"] == (-1.0, 1.0)


def test_wrong_length_range_is_refused(access):
  key = "event.interval.push_robot.params.velocity_range.x"
  with pytest.raises(ValueError):
    access.set(key, [-1.0, 0.0, 1.0])
  assert access.get(key) == [-0.5, 0.5]


def test_inverted_range_is_refused(access):
  key = "event.interval.push_robot.params.velocity_range.x"
  with pytest.raises(ValueError) as excinfo:
    access.set(key, [2.0, -2.0])
  assert "low" in str(excinfo.value) and "high" in str(excinfo.value)
  assert access.get(key) == [-0.5, 0.5]


def test_non_numeric_range_entries_are_refused(access):
  key = "command.twist.ranges.ang_vel_z"
  with pytest.raises(ValueError):
    access.set(key, ["low", "high"])
  assert access.get(key) == [-0.5, 0.5]


def test_scalar_refuses_sequences_and_bad_strings(access):
  key = "reward.foot_slip.params.command_threshold"
  for bad in ([0.1, 0.2], "fast"):
    with pytest.raises(ValueError) as excinfo:
      access.set(key, bad)
    assert "number" in str(excinfo.value)
  assert access.get(key) == 0.05


def test_fractional_write_to_an_int_leaf_is_refused_not_floored(access):
  """Silently flooring 6.5 to 6 was a trap; the refusal must say why."""
  key = "event.reset.reset_robot_joints.params.attempts"
  with pytest.raises(ValueError) as excinfo:
    access.set(key, 6.5)
  assert "whole number" in str(excinfo.value)
  assert access.get(key) == 4


def test_bool_leaf_refuses_arbitrary_truthy_objects(access):
  key = "event.reset.reset_robot_joints.params.use_default_offset"
  for bad in ("yes", [1], 2):
    with pytest.raises(ValueError):
      access.set(key, bad)
  assert access.get(key) is True


# Liveness: a write that cannot take effect is refused, not reported applied.


def test_liveness_is_carried_on_discovered_specs(access):
  specs = {s.key: s for s in access.discover()}
  assert specs["reward.track_linear_velocity.weight"].liveness == "live"
  assert specs["reward.track_linear_velocity.params.std"].liveness == "live"
  assert (
      specs["event.interval.push_robot.params.velocity_range.x"].liveness
      == "live"
  )
  assert (
      specs["event.reset.reset_robot_joints.params.position_range"].liveness
      == "at_reset"
  )
  assert specs["event.startup.foot_friction.params.ranges"].liveness == "at_startup"
  inert_key = next(k for k in specs if "elbow" in k)
  assert specs[inert_key].liveness == "inert"


def test_startup_event_write_is_refused_with_liveness_in_the_message(access):
  """G1's foot_friction ranges: discovered, but mjlab read them exactly once."""
  key = "event.startup.foot_friction.params.ranges"
  with pytest.raises(ValueError) as excinfo:
    access.set(key, [0.1, 2.0])
  assert "at_startup" in str(excinfo.value)
  live = access.env.event_manager.get_term_cfg("foot_friction").params["ranges"]
  assert live == (0.3, 1.2)


def test_class_cached_reward_param_is_inert_and_refused(access):
  """variable_posture's __call__ dels the std params and uses tensors baked at
  __init__ -- the regex-keyed std dict is a permanent no-op, and saying
  "applied" would send an agent tuning a dead knob for the rest of the run."""
  key = next(
      k for k in keys(access) if "variable_posture" in k and "std_walking" in k
  )
  with pytest.raises(ValueError) as excinfo:
    access.set(key, 0.9)
  assert "inert" in str(excinfo.value)
  cfg = access.env.reward_manager.get_term_cfg("variable_posture")
  assert cfg.params["std_walking"] == {".*elbow.*": 0.25}


def test_class_term_pass_through_param_stays_live(access):
  """walking_threshold is read from the call args, not the cache: a
  wrongly-inert live knob would be nearly as bad as the reverse."""
  key = "reward.variable_posture.params.walking_threshold"
  specs = {s.key: s for s in access.discover()}
  assert specs[key].liveness == "live"

  access.set(key, 0.7)
  cfg = access.env.reward_manager.get_term_cfg("variable_posture")
  assert cfg.params["walking_threshold"] == 0.7


def test_class_term_weight_stays_live(access):
  # The manager multiplies by cfg.weight every compute, whatever func is.
  access.set("reward.variable_posture.weight", 0.5)
  assert access.env.reward_manager.get_term_cfg("variable_posture").weight == 0.5


def test_function_term_param_is_live_and_writable(access):
  access.set("reward.track_linear_velocity.params.std", 0.4)
  assert access.get("reward.track_linear_velocity.params.std") == 0.4


def test_reset_event_write_lands_and_reports_at_reset(access):
  notes = access.set("event.reset.reset_base.params.pose_range.x", [-1.0, 1.0])
  assert notes["liveness"] == "at_reset"
  assert access.get("event.reset.reset_base.params.pose_range.x") == [-1.0, 1.0]


class RebuildablePosture:
  """A class-based term that CAN rebuild its cache: it exposes update_params."""

  def __init__(self, std: Dict[str, float]):
    self.std = dict(std)
    self.rebuilt_with: List[Dict[str, Any]] = []

  def __call__(self, env, std):
    del std
    return 0.0

  def update_params(self, **params):
    self.rebuilt_with.append(params)
    if "std" in params:
      self.std = dict(params["std"])


def test_class_term_with_update_params_stays_live_and_is_rebuilt():
  env = FakeEnv()
  term = RebuildablePosture({".*hip.*": 0.2})
  env.reward_manager._terms["hooked"] = TermCfg(
      func=term, weight=1.0, params={"std": {".*hip.*": 0.2}}
  )
  access = ParameterAccess(env)

  key = next(k for k in keys(access) if "hooked" in k and "hip" in k)
  specs = {s.key: s for s in access.discover()}
  assert specs[key].liveness == "live"

  access.set(key, 0.6)
  # The hook received the whole updated top-level param -- the shape it cached
  # from -- not just the one regex-keyed entry that changed.
  assert term.rebuilt_with == [{"std": {".*hip.*": 0.6}}]
  assert term.std == {".*hip.*": 0.6}


def test_update_params_hook_failure_is_not_swallowed():
  env = FakeEnv()

  class BrokenHook(RebuildablePosture):
    def update_params(self, **params):
      raise RuntimeError("rebuild failed")

  env.reward_manager._terms["hooked"] = TermCfg(
      func=BrokenHook({".*hip.*": 0.2}), weight=1.0, params={"std": {".*hip.*": 0.2}}
  )
  access = ParameterAccess(env)
  key = next(k for k in keys(access) if "hooked" in k and "hip" in k)

  with pytest.raises(RuntimeError):
    access.set(key, 0.6)


# Command terms: what the constructor read is what it cached.


@dataclass
class KernelCommandCfg:
  ranges: Ranges = field(default_factory=Ranges)
  kernel_lambda: float = 0.8
  kernel_size: int = 1


class KernelCommandTerm:
  """A command term shaped like mjlab's MotionCommand.

  Its constructor folds ``kernel_lambda`` and ``kernel_size`` into a sampling
  kernel and validates one range; ``ranges.lin_vel_x`` is read on every
  resample. Editing the cfg fields the constructor read changes nothing --
  the kernel is already built -- so they must be reported inert.
  """

  def __init__(self, cfg=None):
    self.cfg = cfg or KernelCommandCfg()
    if self.cfg.ranges.ang_vel_z[0] > self.cfg.ranges.ang_vel_z[1]:
      raise ValueError("ang_vel_z range is inverted")
    self.kernel = [self.cfg.kernel_lambda**i for i in range(self.cfg.kernel_size)]

  def resample(self):
    return self.cfg.ranges.lin_vel_x


class RebuildableKernelCommandTerm(KernelCommandTerm):
  """The same term, able to rebuild its kernel when told a field changed."""

  def __init__(self, cfg=None):
    super().__init__(cfg)
    self.rebuilt_with: List[Dict[str, Any]] = []

  def update_params(self, **fields):
    self.rebuilt_with.append(fields)
    self.kernel = [self.cfg.kernel_lambda**i for i in range(self.cfg.kernel_size)]


def _access_with_command(term) -> ParameterAccess:
  env = FakeEnv()
  env.command_manager._terms["motion"] = term
  return ParameterAccess(env)


def test_constructor_cfg_reads_are_found_along_the_mro():
  from rlmcp.adapters.access.base import constructor_cfg_reads

  reads = constructor_cfg_reads(RebuildableKernelCommandTerm)
  assert reads == frozenset({"kernel_lambda", "kernel_size", "ranges.ang_vel_z"})
  # The fake velocity term reads nothing from its cfg at construction.
  assert constructor_cfg_reads(CommandTerm) == frozenset()


def test_command_field_read_by_constructor_is_inert_and_refused():
  term = KernelCommandTerm()
  access = _access_with_command(term)
  specs = {s.key: s for s in access.discover()}
  assert specs["command.motion.kernel_lambda"].liveness == "inert"
  assert specs["command.motion.kernel_size"].liveness == "inert"

  with pytest.raises(ValueError) as excinfo:
    access.set("command.motion.kernel_lambda", 0.5)
  assert "inert" in str(excinfo.value)
  assert term.cfg.kernel_lambda == 0.8
  assert term.kernel == [1.0]


def test_command_field_read_at_runtime_stays_live():
  """A validation read of ``ranges.ang_vel_z`` must not spread to its siblings:
  the term reads ``ranges.lin_vel_x`` on every resample, so that one is live."""
  term = KernelCommandTerm()
  access = _access_with_command(term)
  specs = {s.key: s for s in access.discover()}
  assert specs["command.motion.ranges.lin_vel_x"].liveness == "live"

  access.set("command.motion.ranges.lin_vel_x", [-0.3, 0.3])
  assert term.resample() == (-0.3, 0.3)


def test_command_term_with_update_params_stays_live_and_is_rebuilt():
  term = RebuildableKernelCommandTerm()
  access = _access_with_command(term)
  specs = {s.key: s for s in access.discover()}
  assert specs["command.motion.kernel_lambda"].liveness == "live"
  assert specs["command.motion.kernel_size"].liveness == "live"

  access.set("command.motion.kernel_size", 3)
  access.set("command.motion.kernel_lambda", 0.5)
  assert term.kernel == [1.0, 0.5, 0.25]

  # A nested write hands the hook the whole top-level field it cached from.
  access.set("command.motion.ranges.lin_vel_x", [-2.0, 2.0])
  assert term.rebuilt_with == [
      {"kernel_size": 3},
      {"kernel_lambda": 0.5},
      {"ranges": term.cfg.ranges},
  ]


# The walker itself.


def test_walker_stops_at_max_depth():
  deep: Dict[str, Any] = {"v": 1.0}
  for _ in range(paths.MAX_DEPTH + 2):
    deep = {"nest": deep}
  found = list(paths.walk_leaves(deep))
  assert found == []


def test_walker_skips_private_names():
  assert list(paths.walk_leaves({"_hidden": 1.0, "shown": 2.0})) == [(["shown"], 2.0)]


@pytest.mark.parametrize(
    "current,value,expected",
    [
        (1.0, 2, 2.0),
        (3, 4.0, 4),
        (True, 0, False),
        ((0.0, 1.0), [2, 3], (2.0, 3.0)),
    ],
)
def test_coerce_like_preserves_the_declared_shape(current, value, expected):
  result = paths.coerce_like(current, value)
  assert result == expected
  assert type(result) is type(expected)


# The state contract: channel vocabulary, truthful labels, error convention.
# Fakes below are shaped like mjlab's scene / command / action managers, with
# real tensors, so the sampler exercises the same code paths it does on GPU.


class _RobotData:
  def __init__(self, num_envs: int, n_joints: int):
    self.joint_pos = torch.zeros(num_envs, n_joints)
    self.joint_vel = torch.zeros(num_envs, n_joints)
    self.root_link_lin_vel_b = torch.zeros(num_envs, 3)
    self.root_link_ang_vel_b = torch.zeros(num_envs, 3)


class _Robot:
  def __init__(self, num_envs: int, joints):
    self.joint_names = list(joints)
    self.data = _RobotData(num_envs, len(joints))


class _Scene:
  """Indexable like mjlab's scene; sensors resolve by name, every other
  entity name resolves to the robot."""

  def __init__(self, robot):
    self._robot = robot
    self.sensors: Dict[str, Any] = {}

  def __getitem__(self, name):
    return self.sensors.get(name, self._robot)


class _TwistVelocityCommand:
  """Shaped like mjlab's UniformVelocityCommand: a (num_envs, 3) twist."""

  def __init__(self, num_envs: int):
    self.vel_command_b = torch.tensor([[0.5, 0.0, 0.1]]).repeat(num_envs, 1)

  @property
  def command(self):
    return self.vel_command_b


class _MotionTargetCommand:
  """Shaped like mjlab's MotionCommand: a wide target vector, not a twist."""

  def __init__(self, num_envs: int, width: int = 8):
    self._command = torch.zeros(num_envs, width)

  @property
  def command(self):
    return self._command


class _NoneCommand:
  """A term that yields no tensor, like a bare/unstarted command."""

  command = None


class _StateCommandManager:
  def __init__(self, terms: Dict[str, Any]):
    self._terms = dict(terms)

  @property
  def active_terms(self) -> List[str]:
    return list(self._terms)

  def get_command(self, name):
    return self._terms[name].command

  def get_term(self, name):
    return self._terms[name]


class _StateActionTerm:
  def __init__(self, dim: int, target_names=None):
    self.action_dim = dim
    if target_names is not None:
      self._target_names = list(target_names)


class _StateActionManager:
  def __init__(self, terms: Dict[str, _StateActionTerm], num_envs: int):
    self._terms = dict(terms)
    total = sum(t.action_dim for t in self._terms.values())
    self.action = torch.zeros(num_envs, total)

  @property
  def active_terms(self) -> List[str]:
    return list(self._terms)

  def get_term(self, name):
    return self._terms[name]


class _StateEnv:
  def __init__(self, command_terms, action_terms=None, num_envs: int = 2,
               joints=("left_knee", "right_knee")):
    self.num_envs = num_envs
    self.scene = _Scene(_Robot(num_envs, joints))
    self.command_manager = _StateCommandManager(command_terms)
    if action_terms:
      self.action_manager = _StateActionManager(action_terms, num_envs)


def _velocity_env(**kwargs) -> _StateEnv:
  return _StateEnv({"twist": _TwistVelocityCommand(2)}, **kwargs)


# Channel vocabulary: one set of names, shared by producers and diagnostics.


def test_every_documented_channel_feeds_its_diagnostics_section():
  """The base.py channel table is the real map, not aspiration."""
  rng = np.random.default_rng(0)
  n = 64
  trace = {
      base.CHANNEL_JOINT_POS: rng.normal(size=(n, 2)),
      base.CHANNEL_JOINT_VEL: rng.normal(size=(n, 2)),
      base.CHANNEL_JOINT_TORQUE: rng.normal(size=(n, 2)),
      base.CHANNEL_ACTION: rng.normal(size=(n, 2)),
      base.CHANNEL_COMMAND: np.tile([1.0, 0.0, 0.0], (n, 1)),
      base.CHANNEL_BASE_LIN_VEL: rng.normal(size=(n, 3)) * 0.1,
      base.CHANNEL_BASE_ANG_VEL: rng.normal(size=(n, 3)) * 0.1,
      base.CHANNEL_BASE_POS: rng.normal(size=(n, 3)) * 0.01,
      base.CHANNEL_PROJECTED_GRAVITY: np.tile([0.0, 0.0, -1.0], (n, 1)),
      base.CHANNEL_FOOT_CONTACT: (rng.random(size=(n, 2)) > 0.4).astype(np.float32),
      base.CHANNEL_REWARD: rng.random(size=(n, 1)),
  }
  report = diag.analyze_trace(trace, dt=0.02)
  assert report["smoothness"]["chatter_measured"] is True  # joint_vel
  assert "joint_jerk_rms" in report["smoothness"]  # joint_pos
  assert "action_rate_rms" in report["smoothness"]  # action
  assert "lin_vel_error_rms" in report["tracking"]  # command + base_lin_vel
  assert "ang_vel_error_rms" in report["tracking"]  # command + base_ang_vel
  assert "torque_rms" in report["effort"]  # joint_torque
  assert "tilt_deg_mean" in report["posture"]  # projected_gravity
  assert "base_height_mean" in report["posture"]  # base_pos
  assert "contact_fraction" in report["gait"]  # foot_contact
  assert "mean_step_reward" in report["reward"]  # reward


def test_samplers_emit_only_declared_channels(fake_sim):
  sampler = StateSampler(_velocity_env(), "robot")
  assert set(sampler.sample(0)) <= set(base.TRACE_CHANNELS)
  assert set(fake_sim.sample_state()) <= set(base.TRACE_CHANNELS)


def test_fake_sim_labels_cover_every_emitted_channel(fake_sim):
  """The conftest fakes are the minimal reference implementation."""
  sample = fake_sim.sample_state()
  labels = fake_sim.trace_labels()
  assert set(labels) == set(sample)
  for channel, values in sample.items():
    assert len(labels[channel]) == values.shape[0], channel


# Labels: derived from the live env, honest when the meaning is unknown.


def test_velocity_command_labels_match_its_true_width():
  """mjlab's velocity command is 3-wide; no fabricated cmd_heading."""
  sampler = StateSampler(_velocity_env(), "robot")
  labels = sampler.labels()
  assert labels[base.CHANNEL_COMMAND] == ["cmd_vx", "cmd_vy", "cmd_wz"]
  sample = sampler.sample(0)
  assert sample[base.CHANNEL_COMMAND].shape == (3,)


def test_non_velocity_command_is_not_recorded_as_a_velocity():
  """A motion target must not land under CHANNEL_COMMAND, where the tracking
  section would subtract it from the base velocity as if it were a twist."""
  env = _StateEnv({"motion": _MotionTargetCommand(2, width=8)})
  sampler = StateSampler(env, "robot")

  sample = sampler.sample(0)
  assert base.CHANNEL_COMMAND not in sample
  assert sample["command_motion"].shape == (8,)

  labels = sampler.labels()
  assert labels["command_motion"] == [f"cmd_{i}" for i in range(8)]


def test_a_bare_first_command_term_does_not_hide_the_second():
  env = _StateEnv({"idle": _NoneCommand(), "twist": _TwistVelocityCommand(2)})
  sampler = StateSampler(env, "robot")
  sample = sampler.sample(0)
  assert base.CHANNEL_COMMAND in sample
  assert sampler.labels()[base.CHANNEL_COMMAND] == ["cmd_vx", "cmd_vy", "cmd_wz"]


def test_action_labels_come_from_the_action_terms_not_the_joint_list():
  """A 3-wide action on a 2-joint robot: labels follow the term, honestly."""
  named = _StateEnv(
      {"twist": _TwistVelocityCommand(2)},
      action_terms={"joint_pos": _StateActionTerm(3, ["a", "b", "tail"])},
  )
  assert StateSampler(named, "robot").labels()[base.CHANNEL_ACTION] == [
      "act:a", "act:b", "act:tail"
  ]

  nameless = _StateEnv(
      {"twist": _TwistVelocityCommand(2)},
      action_terms={"ik": _StateActionTerm(3)},
  )
  assert StateSampler(nameless, "robot").labels()[base.CHANNEL_ACTION] == [
      "ik_0", "ik_1", "ik_2"
  ]


class _ContactData:
  def __init__(self, num_envs: int, width: int):
    self.found = torch.zeros(num_envs, width)


class _ContactSensor:
  """Shaped like mjlab's ContactSensor: per-column flags, named primaries."""

  def __init__(self, num_envs: int, width: int, primary_names=None):
    self.data = _ContactData(num_envs, width)
    if primary_names is not None:
      self.primary_names = list(primary_names)


def test_foot_contact_labels_come_from_the_sensor_not_a_biped_assumption():
  """mjlab's contact sensor names its columns; a quadruped's four feet must
  not be guessed at from the channel width."""
  env = _velocity_env()
  env.scene.sensors["feet_ground_contact"] = _ContactSensor(
      2, 4, primary_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"])

  labels = StateSampler(env, "robot").labels()

  assert labels[base.CHANNEL_FOOT_CONTACT] == [
      "FL_foot", "FR_foot", "RL_foot", "RR_foot"
  ]


def test_foot_contact_labels_are_numbered_when_the_sensor_has_no_names():
  """Width 2 alone is not evidence of a left and a right foot."""
  env = _velocity_env()
  env.scene.sensors["feet_ground_contact"] = _ContactSensor(2, 2)

  labels = StateSampler(env, "robot").labels()

  assert labels[base.CHANNEL_FOOT_CONTACT] == ["foot_0", "foot_1"]


# Metrics: the velocity-error metric verifies its command before existing.


def test_velocity_error_metric_requires_a_velocity_command():
  out = state_metrics.summary_metrics(_velocity_env(), "robot")
  assert "rlmcp/lin_vel_error_mean" in out
  assert "rlmcp/commanded_speed_mean" in out


def test_velocity_error_metric_omitted_for_a_motion_command():
  env = _StateEnv({"motion": _MotionTargetCommand(2, width=8)})
  out = state_metrics.summary_metrics(env, "robot")
  assert "rlmcp/lin_vel_error_mean" not in out
  assert "rlmcp/commanded_speed_mean" not in out


# Rendering: the private-attribute dependency is guarded, not assumed.


def test_renderer_guard_names_the_mjlab_assumption():
  class NoRendererSlot:
    pass

  env = NoRendererSlot()
  assert rendering.renderer_ready(env) is False
  with pytest.raises(NotSupported) as excinfo:
    rendering.ensure_renderer(env)
  assert "_offline_renderer" in str(excinfo.value)


# The declared contract: graceful defaults and one error convention.


class _MinimalSim(SimAdapter):
  """Implements exactly the three mandatory methods, nothing else."""

  def discover_parameters(self):
    return []

  def get_parameter(self, key):
    raise KeyError(key)

  def set_parameter(self, key, value):
    raise KeyError(key)


class _MinimalRunner(RunnerAdapter):
  def discover_hyperparameters(self):
    return []

  def set_hyperparameter(self, key, value):
    raise KeyError(key)


def test_optional_sim_surface_has_graceful_defaults():
  sim = _MinimalSim()
  assert sim.trace_labels() == {}
  assert sim.renderer_ready() is False
  assert sim.summary_metrics() == {}
  assert sim.get_env_state() == {}
  with pytest.raises(NotSupported):
    sim.sample_state()
  with pytest.raises(NotSupported):
    sim.render()


def test_request_stop_is_advisory():
  # Unimplemented: NotSupported, which the controller swallows.
  with pytest.raises(NotSupported):
    _MinimalRunner().request_stop()
  # mjlab: a documented no-op -- rsl_rl has no stop hook to arm, and stopping
  # is delivered by the loop polling should_stop().
  adapter = RslRlRunnerAdapter(SimpleNamespace(alg=None))
  assert adapter.request_stop() is False


def test_rl_bounds_are_only_the_definitional_ones():
  """Pins the whole table: re-adding a taste bound like `learning_rate max` fails here."""
  assert {
      name: (meta.get("min"), meta.get("max"))
      for name, meta in runner_adapter._ALG_PARAMS.items()
  } == {
      "learning_rate": (0.0, None),
      "entropy_coef": (None, None),
      "clip_param": (0.0, None),
      "desired_kl": (0.0, None),
      "gamma": (0.0, 1.0),
      "lam": (0.0, 1.0),
      "max_grad_norm": (0.0, None),
      "value_loss_coef": (None, None),
  }


def test_unknown_hyperparameter_raises_naming_what_exists():
  adapter = RslRlRunnerAdapter(
      SimpleNamespace(alg=SimpleNamespace(learning_rate=1e-3, entropy_coef=0.01))
  )
  with pytest.raises(KeyError) as excinfo:
    adapter.set_hyperparameter("rl.warmup_steps", 100)
  assert "warmup_steps" in str(excinfo.value)
  assert "learning_rate" in str(excinfo.value)


# --- Checkpoint rollback across the inference-mode boundary -----------------
#
# rsl_rl collects rollouts inside torch.inference_mode(), and its empirical
# normalizer reassigns a buffer there (`self._std = torch.sqrt(self._var)`), so
# that buffer is born an inference tensor. Rollback runs outside inference mode
# -- the only point it can run without racing the simulator -- where torch
# refuses to write to one. These fakes reproduce that exact shape; the fixture
# is not a mock of the fix, it is a mock of rsl_rl.


class _FakeNormalizer(torch.nn.Module):
  """Mirrors rsl_rl's EmpiricalNormalization buffer handling."""

  def __init__(self, width: int = 4):
    super().__init__()
    self.register_buffer("_mean", torch.zeros(1, width))
    self.register_buffer("_var", torch.ones(1, width))
    self.register_buffer("_std", torch.ones(1, width))

  def update(self, batch: torch.Tensor) -> None:
    self._mean += 0.1 * (batch.mean(0, keepdim=True) - self._mean)
    self._var += 0.1 * (batch.var(0, keepdim=True) - self._var)
    # Reassignment, not an in-place write: this is the line that makes `_std`
    # an inference tensor when it runs inside inference_mode().
    self._std = torch.sqrt(self._var)


class _FakePolicy(torch.nn.Module):
  def __init__(self, width: int = 4):
    super().__init__()
    self.obs_normalizer = _FakeNormalizer(width)
    self.linear = torch.nn.Linear(width, 2)


class _FakeRunner:
  """A runner shaped like rsl_rl's: modules hang off `alg`, load() is strict."""

  def __init__(self, width: int = 4):
    self.alg = SimpleNamespace(_raw_actor=_FakePolicy(width))
    self.current_learning_iteration = 0

  def save(self, path, infos=None):
    torch.save({"model": self.alg._raw_actor.state_dict(), "infos": infos}, path)

  def load(self, path):
    saved = torch.load(path, weights_only=False)
    self.alg._raw_actor.load_state_dict(saved["model"], strict=True)
    return saved["infos"]

  def collect_a_rollout(self) -> None:
    with torch.inference_mode():
      self.alg._raw_actor.obs_normalizer.update(torch.randn(8, 4))


def test_rollback_survives_buffers_reassigned_in_inference_mode(tmp_path):
  """The bug: every rollback failed once a rollout had touched the normalizer."""
  runner = _FakeRunner()
  checkpoint = tmp_path / "before-experiment.pt"
  runner.save(str(checkpoint), {"tag": "before-experiment"})
  saved_std = runner.alg._raw_actor.obs_normalizer._std.clone()

  runner.collect_a_rollout()

  # Precondition: the fake really does reproduce the trap, so this test cannot
  # pass for the wrong reason if rsl_rl's normalizer changes shape.
  normalizer = runner.alg._raw_actor.obs_normalizer
  assert normalizer._std.is_inference()
  with pytest.raises(RuntimeError, match="inference tensor"):
    runner.load(str(checkpoint))

  infos = RslRlRunnerAdapter(runner).load_checkpoint(str(checkpoint))

  assert infos == {"tag": "before-experiment"}
  restored = runner.alg._raw_actor.obs_normalizer._std
  assert not restored.is_inference()
  assert torch.equal(restored, saved_std)


def test_thaw_leaves_ordinary_buffers_alone(tmp_path):
  """It is a no-op on a run that has not entered inference mode yet."""
  runner = _FakeRunner()
  adapter = RslRlRunnerAdapter(runner)
  before = runner.alg._raw_actor.obs_normalizer._std

  assert adapter._thaw_inference_buffers() == 0
  assert runner.alg._raw_actor.obs_normalizer._std is before


def test_thaw_does_not_replace_parameters(tmp_path):
  """Parameters must keep their identity -- the optimizer holds references."""
  runner = _FakeRunner()
  runner.collect_a_rollout()
  weight = runner.alg._raw_actor.linear.weight

  RslRlRunnerAdapter(runner)._thaw_inference_buffers()

  assert runner.alg._raw_actor.linear.weight is weight


def test_thaw_finds_modules_without_evaluating_properties():
  """Module discovery must not trigger a property that runs training code."""

  class _Exploding:
    @property
    def boom(self):
      raise AssertionError("a property was evaluated while looking for tensors")

  runner = _FakeRunner()
  runner.alg = SimpleNamespace(_raw_actor=runner.alg._raw_actor, extra=_Exploding())
  runner.collect_a_rollout()

  assert RslRlRunnerAdapter(runner)._thaw_inference_buffers() == 1


def test_load_checkpoint_still_reports_a_missing_file(tmp_path):
  """Thawing runs before the load, but must not mask the file check."""
  with pytest.raises(FileNotFoundError):
    RslRlRunnerAdapter(_FakeRunner()).load_checkpoint(str(tmp_path / "nope.pt"))

# Restarting episodes.


class _ResettableEnv:
  """The reset surface of a manager-based env, and nothing else.

  ``_reset_idx(env_ids)`` is what such an environment calls for whichever of
  its environments terminated this tick; the adapter reuses it so a requested
  reset runs the same event and randomisation terms a natural one does.
  """

  def __init__(self, num_envs: int = 4, private: bool = True):
    self.num_envs = num_envs
    self.device = "cpu"
    self.episode_length_buf = torch.arange(num_envs)
    self.reset_calls: List[Any] = []
    self.full_resets = 0
    if private:
      self._reset_idx = self._record
    else:
      self.reset_idx = self._record

  def _record(self, env_ids):
    self.reset_calls.append([int(i) for i in env_ids])

  def reset(self):
    self.full_resets += 1


def _sim_for(env):
  """The adapter over one environment, without its constructor.

  ``reset_envs`` reads nothing but ``self.env``; building the whole adapter
  would drag in parameter discovery and state sampling, which have their own
  tests above and their own demands on the fake environment.
  """
  from rlmcp.adapters.mjlab.sim_adapter import MjlabSimAdapter

  sim = object.__new__(MjlabSimAdapter)
  sim.env = env
  return sim


def test_reset_envs_restarts_the_environments_it_was_given():
  env = _ResettableEnv()
  result = _sim_for(env).reset_envs([1, 3])

  assert env.reset_calls == [[1, 3]]
  assert result["num_reset"] == 2
  # The restarted episodes get a clock at zero; the others keep theirs.
  assert env.episode_length_buf.tolist() == [0, 0, 2, 0]


def test_reset_envs_with_nothing_named_restarts_all_of_them():
  env = _ResettableEnv()
  result = _sim_for(env).reset_envs()

  assert env.reset_calls == [[0, 1, 2, 3]]
  assert result["num_reset"] == 4


def test_reset_envs_prefers_a_public_spelling_when_the_backend_has_one():
  env = _ResettableEnv(private=False)
  assert _sim_for(env).reset_envs([0])["method"] == "reset_idx"


def test_reset_envs_falls_back_to_a_full_reset_only_when_all_were_asked_for():
  """Restarting every environment because the caller asked for two would be
  worse than refusing."""

  class OnlyFullReset(_ResettableEnv):
    def __init__(self):
      super().__init__()
      del self._reset_idx

  env = OnlyFullReset()
  assert _sim_for(env).reset_envs()["method"] == "reset"
  assert env.full_resets == 1

  with pytest.raises(NotSupported):
    _sim_for(OnlyFullReset()).reset_envs([1])


def test_reset_envs_refuses_an_environment_that_does_not_exist():
  env = _ResettableEnv(num_envs=4)
  with pytest.raises(ValueError) as caught:
    _sim_for(env).reset_envs([9])

  assert "9" in str(caught.value) and "0..3" in str(caught.value)
  assert env.reset_calls == []
