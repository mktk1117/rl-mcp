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


# Branches, restarts, and which runs a recipe is actually made of.


def _tree(tmp_path):
  """The topology from the review of this feature:

      1 -> 2 -> 3 (restart from scratch) -> 4 -> 5 -> 6
                                                 5 -> 7   (the best one)
      8 (from scratch) -> 9                                (an unrelated line)

  Config parents run the whole way down; weights restart at 3 and at 8.
  """
  from rlmcp.records.record import Weights

  store = FileStore(tmp_path / "records", slots=1)
  made = {}
  def add(slug, parent=None, warm=None):
    record = store.new_record(slug, task="T", parent=parent,
                              change=[f"change in {slug}"])
    if warm:
      store.update_record(record.id, lambda r: setattr(r, "weights", Weights(run=warm)))
    made[slug] = store.get_record(record.id).id
    return made[slug]

  one = add("one")
  two = add("two", parent=one, warm=one)
  three = add("three", parent=two)              # restart: no weights
  four = add("four", parent=three, warm=three)
  five = add("five", parent=four, warm=four)
  add("six", parent=five, warm=five)            # a sibling branch off 5
  seven = add("seven", parent=five, warm=five)  # the branch that mattered
  eight = add("eight")                          # an unrelated from-scratch line
  add("nine", parent=eight, warm=eight)
  return store, made


def test_a_recipe_starts_where_the_policy_did_not_where_the_config_did(tmp_path):
  """The point of the whole rule: 3 restarted from scratch, so 1 and 2 are
  ancestors of 3's *config* and not of these weights. Replaying them would be
  replaying history this policy does not contain."""
  store, made = _tree(tmp_path)

  answer = build(store, made["seven"], tmp_path / "recipe")

  assert answer["phases"] == [made["three"], made["four"], made["five"],
                              made["seven"]]
  assert made["one"] not in answer["phases"]
  assert made["two"] not in answer["phases"]


def test_a_sibling_branch_and_an_unrelated_line_are_not_in_it(tmp_path):
  store, made = _tree(tmp_path)

  phases = build(store, made["seven"], tmp_path / "recipe")["phases"]

  assert made["six"] not in phases      # branched off 5, but not on the way to 7
  assert made["eight"] not in phases and made["nine"] not in phases


def test_the_config_fold_is_cut_at_the_restart_too(tmp_path):
  """`config.json` already holds everything the restart launched with, so the
  history worth reading starts there as well."""
  store, made = _tree(tmp_path)

  build(store, made["seven"], tmp_path / "recipe")
  phases_md = (tmp_path / "recipe" / "phases.md").read_text()

  assert "change in three" in phases_md and "change in seven" in phases_md
  assert "change in one" not in phases_md and "change in two" not in phases_md


def test_each_phase_says_how_to_launch_it(tmp_path):
  """A four-phase recipe that only tells you how to start phase 1 is not a
  recipe: every later phase warm-starts from the one before."""
  store, made = _tree(tmp_path)

  build(store, made["seven"], tmp_path / "recipe")
  phases_md = (tmp_path / "recipe" / "phases.md").read_text()

  assert "4 training segments" in phases_md
  assert "./launch.sh <new-record-id>" in phases_md
  assert f"checkpoint from phase 3 ({made['five']})" in phases_md


def test_a_from_scratch_run_is_a_one_phase_recipe(tmp_path):
  store, made = _tree(tmp_path)

  answer = build(store, made["three"], tmp_path / "recipe")

  assert answer["phases"] == [made["three"]]
  assert "trained from scratch" in (tmp_path / "recipe" / "phases.md").read_text()


def test_a_warm_start_that_points_at_nothing_ends_the_chain_honestly(tmp_path):
  """A record whose parent store is elsewhere: a short chain beats one with a
  hole in it."""
  from rlmcp.records.record import Weights

  store = FileStore(tmp_path / "records", slots=1)
  orphan = store.new_record("orphan", task="T")
  store.update_record(orphan.id, lambda r: setattr(r, "weights", Weights(run="404")))

  answer = build(store, orphan.id, tmp_path / "recipe")

  assert answer["phases"] == [orphan.id]
  assert "404" in (tmp_path / "recipe" / "phases.md").read_text()


def test_a_cycle_in_the_warm_start_chain_terminates(tmp_path):
  """Two runs that each claim to warm-start from the other must not hang."""
  from rlmcp.records.record import Weights

  store = FileStore(tmp_path / "records", slots=1)
  a = store.new_record("a", task="T")
  b = store.new_record("b", task="T")
  store.update_record(a.id, lambda r: setattr(r, "weights", Weights(run=b.id)))
  store.update_record(b.id, lambda r: setattr(r, "weights", Weights(run=a.id)))

  answer = build(store, a.id, tmp_path / "recipe")

  assert set(answer["phases"]) == {a.id, b.id}
