"""A term added mid-run has to outlive the process that ran it.

The failure this pins is not a crash: it is a run that trained beautifully on a
reward term nobody can reconstruct afterwards, because the function only ever
existed as an object inside a training process that has since exited. So the
controller writes the source into the session and records the digest, and
``rlmcp rewards export`` turns that back into the two things a task package
needs -- the implementation and the config lines.

The weight the export writes is the one in force at the end, not the one the
term was added with: a term added at 0.5 and tuned to 3.0 belongs in the config
at 3.0, since 3.0 is what the run being reproduced actually used.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

import pytest
import torch
from conftest import FakeSimAdapter

from rlmcp import rewards_export
from rlmcp.core.controller import RlMcp
from rlmcp.core.parameters.spec import ParameterCategory, ParameterSpec
from rlmcp.core.reward_source import RewardSourceError
from rlmcp.session import Session

UPRIGHT = """
def upright(env, scale: float = 1.0):
  \"\"\"Reward standing tall.\"\"\"
  return torch.ones(env.num_envs) * scale
"""


NUM_ENVS_EXPORTED = 12


class _Env:
  """The environment the adapter wraps -- what a reward term is handed."""

  def __init__(self, num_envs: int):
    self.num_envs = num_envs


class _RewardSimAdapter(FakeSimAdapter):
  """A fake sim that can grow its reward function, as the real ones can."""

  def __init__(self, num_envs: int = 12):
    super().__init__(num_envs=num_envs)
    self.env = _Env(num_envs)
    self.added: Dict[str, Any] = {}

  def add_reward_term(self, name, func, weight, params=None):
    if name in self.added:
      raise RuntimeError(f"Reward term '{name}' already exists.")
    value = func(self.env, **(params or {}))
    self.added[name] = {"func": func, "params": dict(params or {})}
    # Exactly what the real adapters do: the weight becomes a parameter.
    self._params[f"reward.{name}.weight"] = float(weight)
    return {"name": name, "index": len(self.added), "class_based": False,
            "trial_value": {"mean": float(value.mean())}}


@pytest.fixture
def lab(tmp_path):
  sim = _RewardSimAdapter()
  controller = RlMcp(sim_adapter=sim, session_dir=tmp_path / "session",
                     session_info={"task": "Fake-Walk-v0"})
  yield controller
  controller.session  # No teardown needed; the session is a directory.


def _add(lab, **kwargs):
  args = {"name": "upright", "source": UPRIGHT, "weight": 2.0,
          "rationale": "nothing rewards standing up"}
  args.update(kwargs)
  return lab.cmd_add_reward(**args)


def test_adding_a_term_reports_where_it_went_and_what_it_is_called(lab):
  result = _add(lab)
  assert result["name"] == "upright"
  assert result["function"] == "upright"
  assert result["key"] == "reward.upright.weight"
  assert result["weight"] == 2.0
  assert result["digest"]
  assert "upright" in lab.sim.added


def test_the_source_is_saved_into_the_session_with_a_usable_header(lab):
  result = _add(lab, params={"scale": 3.0})
  saved = lab.session.rewards / "upright.py"
  assert saved.exists()
  text = saved.read_text()
  # The header alone should say how to put this term into a config.
  assert "weight:    2.0" in text
  assert "'scale': 3.0" in text
  assert "nothing rewards standing up" in text
  # And the agent's own source survives verbatim.
  assert "def upright(env, scale: float = 1.0):" in text
  assert str(saved) == result["source_path"]


def test_the_event_log_records_the_addition_with_its_digest(lab):
  result = _add(lab)
  events = [e for e in lab.session.events() if e["kind"] == "add_reward_term"]
  assert len(events) == 1
  assert events[0]["name"] == "upright"
  assert events[0]["digest"] == result["digest"]
  assert events[0]["rationale"] == "nothing rewards standing up"


def test_the_new_weight_becomes_a_tunable_parameter(lab):
  _add(lab)
  listed = lab.cmd_list_parameters(contains="upright")["parameters"]
  assert "reward.upright.weight" in listed
  assert lab.cmd_get_parameter("reward.upright.weight")["value"] == 2.0

  changed = lab.cmd_set_parameter("reward.upright.weight", 5.0, "too weak")
  assert changed["applied"] is True
  assert lab.cmd_get_parameter("reward.upright.weight")["value"] == 5.0


def test_resetting_parameters_returns_the_term_to_the_weight_it_arrived_with(lab):
  _add(lab)
  lab.cmd_set_parameter("reward.upright.weight", 5.0, "experiment")
  lab.cmd_reset_parameters()
  assert lab.cmd_get_parameter("reward.upright.weight")["value"] == 2.0


def test_bad_source_never_reaches_the_simulator(lab):
  with pytest.raises(RewardSourceError):
    _add(lab, source="def upright(env)\n  return 1\n")
  assert lab.sim.added == {}
  assert not (lab.session.rewards / "upright.py").exists()
  assert [e for e in lab.session.events() if e["kind"] == "add_reward_term"] == []


def test_add_reward_is_a_verb_the_run_accepts(lab):
  assert "add_reward" in lab.cmd_help()["commands"]


# Getting it back out again.


def test_export_writes_an_implementation_and_its_config_lines(lab, tmp_path):
  _add(lab, params={"scale": 3.0})
  payload = rewards_export.export_added_rewards(
      lab.session, tmp_path / "out", task="Fake-Walk-v0")

  assert payload["count"] == 1
  implementation = (tmp_path / "out" / "added_rewards.py").read_text()
  config = (tmp_path / "out" / "added_rewards_cfg.py").read_text()

  # The implementation is importable source, with the rlmcp header gone.
  assert "def upright(env, scale: float = 1.0):" in implementation
  assert "digest:" not in implementation
  compile(implementation, "added_rewards.py", "exec")

  # torch is in scope when a term is compiled but not when the export is
  # imported, so the module has to bring it itself or NameError on first call.
  assert "import torch" in implementation
  namespace: Dict[str, Any] = {}
  exec(compile(implementation, "added_rewards.py", "exec"), namespace)
  assert namespace["upright"](_Env(NUM_ENVS_EXPORTED)).shape == (
      NUM_ENVS_EXPORTED,)

  # The config names the function and carries the params it ran with.
  assert "upright = RewTerm(func=added_rewards.upright, weight=2" in config
  assert "'scale': 3.0" in config
  compile(config, "added_rewards_cfg.py", "exec")


def test_export_uses_the_weight_the_run_ended_on_not_the_one_it_started_with(
    lab, tmp_path):
  _add(lab, weight=0.5)
  lab.cmd_set_parameter("reward.upright.weight", 3.0, "it was too weak")

  payload = rewards_export.export_added_rewards(lab.session, tmp_path / "out")
  assert payload["rewards"][0]["weight"] == 3.0
  assert payload["rewards"][0]["added_weight"] == 0.5

  config = (tmp_path / "out" / "added_rewards_cfg.py").read_text()
  assert "weight=3" in config
  # And the history is not silently dropped.
  assert "added at weight 0.5" in config


def test_export_of_a_run_that_added_nothing_writes_nothing(lab, tmp_path):
  payload = rewards_export.export_added_rewards(lab.session, tmp_path / "out")
  assert payload["count"] == 0
  assert not (tmp_path / "out").exists()
  assert "nothing to export" in rewards_export.describe(payload)


def test_export_reads_a_finished_run_from_its_directory_alone(lab, tmp_path):
  """The exporting process is not the training process, and usually later."""
  _add(lab)
  reopened = Session(lab.session.dir)

  rewards = rewards_export.collect_added_rewards(reopened)
  assert [r.name for r in rewards] == ["upright"]
  assert "def upright" in rewards[0].source
