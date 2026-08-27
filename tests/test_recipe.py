"""Turning a finished run into a directory that runs again."""

from __future__ import annotations

import json
import subprocess

import pytest

from rlmcp.core.curriculum import StageSchedule
from rlmcp.records import snapshot
from rlmcp.records.filestore import FileStore
from rlmcp.records.recipe import build, distil


def _edit(iteration: int, what: str, why: str = "") -> dict:
  return {"iteration": iteration, "kind": "set_parameter", "layer": "parameter",
          "what": what, "why": why, "at": 0.0}


def _stage(iteration: int, name: str) -> dict:
  return {"iteration": iteration, "kind": "curriculum_stage",
          "layer": "curriculum", "what": f"stage → {name}", "why": "", "at": 0.0}


# What `rlmcp.core.replay.read_ladder` hands back: the planned stages by name.
LADDER = {
    "0_held": {"name": "0_held", "parameters": {"assist": 1.0},
               "promote_when": [{"metric": "rlmcp/catch_rate", "op": ">=",
                                 "value": 8.0}],
               "min_iterations": 400},
    "1_table": {"name": "1_table", "parameters": {"assist": 0.0}},
}
ENTERED = ["0_held", "1_table"]


# Distillation.


def test_an_edit_becomes_the_value_its_rung_starts_with(): 
  """Not "wait 400 iterations, then panic": the point of distilling is that the
  replay starts the rung with the value the original run needed."""
  schedule = distil([_stage(0, "0_held"),
                     _edit(400, "reward.goal.weight: 18.0 → 24.0", "too weak")],
                    LADDER, ENTERED)

  held = schedule.stages[0]
  assert held.parameters == {"assist": 1.0, "reward.goal.weight": 24.0}
  assert "too weak" in held.notes and "it 400" in held.notes


def test_the_rungs_promotion_conditions_survive_distillation():
  """Those were written before the run and are real; nothing here invents one."""
  schedule = distil([_stage(0, "0_held")], LADDER, ENTERED)

  condition = schedule.stages[0].promote_when[0]
  assert (condition.metric, condition.op, condition.value) == (
      "rlmcp/catch_rate", ">=", 8.0)


def test_an_edit_lands_in_the_rung_that_was_active_when_it_happened():
  schedule = distil([_stage(0, "0_held"), _stage(1500, "1_table"),
                     _edit(1600, "rl.entropy_coef: 0.005 → 0.01", "collapsed")],
                    LADDER, ENTERED)

  assert "rl.entropy_coef" not in schedule.stages[0].parameters
  assert schedule.stages[1].parameters["rl.entropy_coef"] == 0.01


def test_a_refused_edit_is_not_in_the_ladder():
  """It never applied. Replaying it would be replaying a mistake."""
  schedule = distil([_edit(10, "reward.drop.weight: -150.0 → -300.0 (refused)",
                           "outside the registered bounds")])

  assert schedule.stages[0].parameters == {}


def test_only_the_rungs_the_run_actually_climbed_are_in_the_recipe():
  """A rung it never reached is a plan, not a result."""
  schedule = distil([_stage(0, "0_held")], LADDER, ["0_held"])

  assert [s.name for s in schedule.stages] == ["0_held"]


def test_a_run_with_no_curriculum_gets_one_rung_per_change(): 
  """And each rung is held for as long as the original ran before the next."""
  schedule = distil([_edit(400, "reward.goal.weight: 18.0 → 24.0"),
                     _edit(900, "rl.entropy_coef: 0.005 → 0.01")])

  assert [s.min_iterations for s in schedule.stages] == [500, 100]
  assert schedule.stages[0].parameters == {"reward.goal.weight": 24.0}


def test_a_run_nobody_touched_still_produces_a_loadable_ladder():
  schedule = distil([])

  assert len(schedule.stages) == 1
  assert StageSchedule.from_dict(schedule.to_dict()).stages[0].name == "0_as_launched"


# Building the directory.


@pytest.fixture
def bundle(tmp_path):
  """A records store, a task repo, and a run that launched from it."""
  repo = tmp_path / "tasks"
  (repo / "shand").mkdir(parents=True)
  (repo / "shand" / "task.py").write_text("EMA = 0.8\n")

  def git(*args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

  subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
  git("config", "user.email", "t@example.com")
  git("config", "user.name", "T")
  git("add", "-A")
  git("commit", "-qm", "the task")

  store = FileStore(tmp_path / "records", slots=1)
  parent = store.new_record("baseline", task="Shand", change=["from scratch"],
                            verdict="falsified", outcome="1.1 goals/min",
                            metrics=[["goals_per_min", "1.1"]])
  record = store.new_record("ema_filter", task="Shand", parent=parent.id,
                            hypothesis="the action interface is the problem",
                            change=["EMA filter alpha 0.8"],
                            verdict="validated", outcome="16.8 goals/min",
                            metrics=[["goals_per_min", "16.8"]],
                            config={"reward.orientation.weight": 2.0})
  store.update_record(record.id, lambda r: setattr(
      r, "code", snapshot.capture(repo / "shand", record_id=r.id)))
  return store, repo, tmp_path


def test_a_recipe_is_a_directory_you_can_launch(bundle):
  store, _, tmp_path = bundle

  answer = build(store, "002", tmp_path / "recipe")

  written = {p.name for p in (tmp_path / "recipe").iterdir()}
  assert written == {"package", "config.json", "curriculum.json", "launch.sh",
                     "phases.md", "expect.json", "README.md"}
  assert answer["missing"] == []          # this run recorded everything
  assert (tmp_path / "recipe" / "launch.sh").stat().st_mode & 0o111


def test_the_package_is_the_code_that_run_actually_used(bundle):
  """Restored from the tree, so it is what ran -- not the repo as it is now."""
  store, repo, tmp_path = bundle
  (repo / "shand" / "task.py").write_text("EMA = 0.0   # changed since\n")

  build(store, "002", tmp_path / "recipe")

  assert (tmp_path / "recipe" / "package" / "shand" / "task.py").read_text() == "EMA = 0.8\n"


def test_the_ladder_loads_straight_back_into_a_schedule(bundle):
  """`curriculum.json` is not a document about the run; StageSchedule reads it."""
  store, _, tmp_path = bundle

  build(store, "002", tmp_path / "recipe")

  loaded = StageSchedule.from_dict(
      json.loads((tmp_path / "recipe" / "curriculum.json").read_text()))
  assert loaded.stages


def test_the_expectations_are_numbers_to_check_never_a_hash(bundle):
  store, _, tmp_path = bundle

  build(store, "002", tmp_path / "recipe")

  expect = json.loads((tmp_path / "recipe" / "expect.json").read_text())
  assert expect["metrics"] == {"goals_per_min": "16.8"}
  assert "not bit-reproducible" in expect["note"]


def test_the_phases_flatten_the_chain_this_policy_came_through(bundle):
  store, _, tmp_path = bundle
  from rlmcp.records.record import Weights

  store.update_record("002", lambda r: setattr(r, "weights", Weights(run="001")))

  build(store, "002", tmp_path / "recipe")

  phases = (tmp_path / "recipe" / "phases.md").read_text()
  assert "2 training segments" in phases
  assert "001-baseline" in phases and "002-ema_filter" in phases


def test_a_run_with_no_snapshot_still_gets_a_recipe_that_says_what_is_missing(
    tmp_path):
  """Refusing to build because one input is absent is a worse answer than
  building and naming the gap."""
  store = FileStore(tmp_path / "records", slots=1)
  store.new_record("no_stamp", task="Shand")

  answer = build(store, "001", tmp_path / "recipe")

  assert not (tmp_path / "recipe" / "package").exists()
  assert any("code snapshot" in item for item in answer["missing"])
  assert "does not have" in (tmp_path / "recipe" / "README.md").read_text()


def test_an_unknown_record_is_refused_by_name(tmp_path):
  store = FileStore(tmp_path / "records", slots=1)
  with pytest.raises(ValueError, match="No record '404'"):
    build(store, "404", tmp_path / "recipe")
