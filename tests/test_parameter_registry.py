import pytest

from rlmcp.core.parameters.registry import ParameterRegistry
from rlmcp.core.parameters.spec import Liveness, ParameterCategory, ParameterSpec


def test_parameter_registration_and_validation():
  reg = ParameterRegistry()
  spec = ParameterSpec(
      key="reward.test_penalty.weight",
      data_type="float",
      current_value=-0.01,
      default_value=-0.01,
      min_value=-1.0,
      max_value=0.0,
      description="Test penalty",
      category=ParameterCategory.REWARD
  )
  val_store = {"val": -0.01}
  reg.register(
      spec=spec,
      setter=lambda v: val_store.update(val=v) is None or True,
      getter=lambda: val_store["val"]
  )

  assert reg.get_value("reward.test_penalty.weight") == -0.01

  # Set valid
  assert reg.set_value("reward.test_penalty.weight", -0.05) is True
  assert reg.get_value("reward.test_penalty.weight") == -0.05

  # Set out of bounds
  with pytest.raises(ValueError):
    reg.set_value("reward.test_penalty.weight", 0.5)

def test_parameter_diff():
  reg = ParameterRegistry()
  reg.register(ParameterSpec(key="a", data_type="int", current_value=10, default_value=10))
  reg.register(ParameterSpec(key="b", data_type="float", current_value=0.5, default_value=0.5))

  baseline = reg.get_snapshot()
  reg.set_value("a", 20)

  diff = reg.compute_diff(baseline)
  assert "a" in diff
  assert diff["a"]["old"] == 10
  assert diff["a"]["new"] == 20
  assert "b" not in diff


def test_writes_that_cannot_take_effect_are_refused_before_the_setter():
  reg = ParameterRegistry()
  calls = []
  reg.register(
      ParameterSpec(
          key="event.startup.foot_friction.params.ranges",
          data_type="range",
          current_value=[0.3, 1.2],
          default_value=[0.3, 1.2],
          liveness=Liveness.AT_STARTUP,
      ),
      setter=lambda v: calls.append(v) or True,
  )
  reg.register(
      ParameterSpec(
          key="reward.pose.params.std",
          data_type="float",
          current_value=0.35,
          default_value=0.35,
          liveness=Liveness.INERT,
      ),
      setter=lambda v: calls.append(v) or True,
  )

  with pytest.raises(ValueError) as excinfo:
    reg.set_value("event.startup.foot_friction.params.ranges", [0.1, 2.0])
  assert "at_startup" in str(excinfo.value)

  with pytest.raises(ValueError) as excinfo:
    reg.set_value("reward.pose.params.std", 0.5)
  assert "inert" in str(excinfo.value)

  assert calls == []  # The setter never ran; nothing reached the simulator.
  assert reg.get_value("event.startup.foot_friction.params.ranges") == [0.3, 1.2]
  assert reg.get_value("reward.pose.params.std") == 0.35


def test_at_reset_writes_succeed_and_the_spec_says_so():
  reg = ParameterRegistry()
  key = "event.reset.reset_base.params.pose_range.x"
  reg.register(
      ParameterSpec(
          key=key,
          data_type="range",
          current_value=[-0.5, 0.5],
          default_value=[-0.5, 0.5],
          liveness=Liveness.AT_RESET,
      )
  )

  assert reg.set_value(key, [-1.0, 1.0]) is True
  assert reg.get_value(key) == [-1.0, 1.0]
  # Callers can annotate "takes effect on next reset" from the spec.
  assert reg.get_spec(key).liveness == Liveness.AT_RESET


def test_liveness_is_exported_in_the_schema():
  reg = ParameterRegistry()
  reg.register(
      ParameterSpec(key="a.weight", data_type="float", current_value=1.0,
                    default_value=1.0)
  )
  reg.register(
      ParameterSpec(key="b.ranges", data_type="range", current_value=[0.3, 1.2],
                    default_value=[0.3, 1.2], liveness=Liveness.AT_STARTUP)
  )

  schema = reg.export_schema_json()
  assert schema["a.weight"]["liveness"] == "live"
  assert schema["b.ranges"]["liveness"] == "at_startup"


def test_range_shape_is_validated_before_the_setter_runs():
  reg = ParameterRegistry()
  calls = []
  key = "command.twist.ranges.lin_vel_x"
  reg.register(
      ParameterSpec(key=key, data_type="range", current_value=[-1.0, 1.0],
                    default_value=[-1.0, 1.0]),
      setter=lambda v: calls.append(v) or True,
  )

  for bad in (0.5, [1.0, 2.0, 3.0], [2.0, -2.0], ["low", "high"]):
    with pytest.raises(ValueError):
      reg.set_value(key, bad)
  assert calls == []  # Rejected values never reached the simulator.

  assert reg.set_value(key, [-2.0, 2.0]) is True
  assert calls == [[-2.0, 2.0]]


def test_scalar_and_bool_types_are_validated():
  reg = ParameterRegistry()
  reg.register(
      ParameterSpec(key="epochs", data_type="int", current_value=5, default_value=5)
  )
  reg.register(
      ParameterSpec(key="flag", data_type="bool", current_value=True,
                    default_value=True)
  )
  reg.register(
      ParameterSpec(key="std", data_type="float", current_value=0.5,
                    default_value=0.5)
  )

  with pytest.raises(ValueError) as excinfo:
    reg.set_value("epochs", 4.7)
  assert "whole number" in str(excinfo.value)
  with pytest.raises(ValueError):
    reg.set_value("flag", "yes")
  with pytest.raises(ValueError):
    reg.set_value("std", "fast")
  with pytest.raises(ValueError):
    reg.set_value("std", [0.1, 0.2])

  assert reg.set_value("epochs", 6) is True
  assert reg.set_value("flag", False) is True
  assert reg.set_value("std", 0.25) is True
