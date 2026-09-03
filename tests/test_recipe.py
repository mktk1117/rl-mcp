"""Turning a finished run into a directory that runs again."""

from __future__ import annotations

import dataclasses
import json
import subprocess

import pytest

from rlmcp.core.curriculum import StageSchedule
from rlmcp.records import snapshot
from rlmcp.records.filestore import FileStore
from rlmcp.records.recipe import build, distil, verify


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
                     _edit(400, "reward.goal.weight: 18.0 → 24.0", "too weak"),
                     _stage(1200, "1_table")],
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
                     "recipe.json",
                     "phases.md", "expect.json", "README.md"}
  assert (tmp_path / "recipe" / "launch.sh").stat().st_mode & 0o111
  # This record has a code snapshot and a config, but no session -- so the
  # environment and the weights are honestly absent rather than empty files.
  assert answer["package"]
  assert [m.split(" (")[0] for m in answer["missing"]] == [
      "the materialised environment", "the trained policy",
      "the task packages", "the seed"]


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
  add("seven", parent=five, warm=five)          # the branch that mattered
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


# The whole pair: the environment, the ladder and the policy in one directory.


@pytest.fixture
def run_with_a_session(bundle):
  """A record whose run left a real session: captured env, metrics, weights.

  This is the shape the feature exists for -- a finished run you want to hand
  to somebody as the pair of environment and policy.
  """
  store, _repo, tmp_path = bundle
  session = tmp_path / "logs" / "run002" / "rlmcp"
  session.mkdir(parents=True)
  (session / "session.json").write_text(json.dumps(
      {"kind": "rlmcp-training-session", "task": "Shand", "num_envs": 4096,
       "device": "cuda:0", "started_at": 0.0,
       "task_packages": ["shand.tasks", "shand.rlmcp_ext"], "seed": 7}))
  (session / "env_terms.json").write_text(json.dumps({
      "task": "Shand",
      "rewards": [{
          "name": "goal",
          "_cfg_type": {"module": "test_recipe", "name": "RewTermCfg"},
          "weight": 2.0,
          "params": {"__map__": {}},
          "func": {"module": "shand.mdp", "qualname": "goal_bonus",
                   "name": "goal_bonus", "kind": "function", "available": True,
                   "source": "def goal_bonus(env):\n  return env.goals\n"},
      }],
      "observations": {},
      "actions": [],
      "problems": [],
      "term_cfg_types": {"reward": {"module": "test_recipe",
                                    "name": "RewTermCfg"}},
  }))
  (session / "metrics.jsonl").write_text(
      json.dumps({"iteration": 4300, "t": 1.0, "goals_per_min": 16.4}) + "\n")
  # The trainer's checkpoints sit in the run directory, beside the session.
  (session.parent / "model_4300.pt").write_bytes(b"weights")
  store.update_record("002", lambda r: setattr(r, "session", str(session)))
  return store, tmp_path, session


@dataclasses.dataclass
class RewTermCfg:
  """Stands in for the backend's reward term config, for the env export."""

  func: object
  weight: float
  params: dict = dataclasses.field(default_factory=dict)


def test_the_recipe_carries_the_env_the_ladder_and_the_policy(run_with_a_session):
  store, tmp_path, _ = run_with_a_session

  answer = build(store, "002", tmp_path / "recipe")
  recipe = tmp_path / "recipe"

  assert answer["missing"] == []
  # The four things that make it a pair you can hand over.
  assert (recipe / "env" / "env_cfg.py").exists()
  assert (recipe / "env" / "mdp_terms.py").exists()
  assert (recipe / "curriculum.json").exists()
  assert (recipe / "policy" / "model_4300.pt").read_bytes() == b"weights"
  assert answer["env"]["counts"]["rewards"] == 1


def test_the_env_in_the_recipe_is_inlined_not_imported(run_with_a_session):
  """It has to work without the task package, which is the point of env/."""
  store, tmp_path, _ = run_with_a_session

  build(store, "002", tmp_path / "recipe")

  terms = (tmp_path / "recipe" / "env" / "mdp_terms.py").read_text()
  assert "def goal_bonus(env):" in terms
  assert "import shand" not in terms


def test_the_policy_can_be_left_out(run_with_a_session):
  """The weights are the large part; a recipe meant to be read skips them."""
  store, tmp_path, _ = run_with_a_session

  answer = build(store, "002", tmp_path / "recipe", policy=False)

  assert not (tmp_path / "recipe" / "policy").exists()
  assert answer["policy"] == ""
  assert answer["missing"] == []       # not missing: not asked for


def test_expectations_carry_the_policy_and_the_numbers_it_ended_on(
    run_with_a_session):
  store, tmp_path, _ = run_with_a_session

  build(store, "002", tmp_path / "recipe")

  expect = json.loads((tmp_path / "recipe" / "expect.json").read_text())
  assert expect["metrics"] == {"goals_per_min": "16.8"}   # what was claimed
  assert expect["final"] == {"goals_per_min": 16.4}       # what it ended on
  assert expect["final_iteration"] == 4300
  assert expect["policy"]["file"] == "model_4300.pt"


# Verifying a retrain against it.


def _candidate(tmp_path, name: str, value: float):
  """A session standing in for a run launched from the recipe."""
  session = tmp_path / name / "rlmcp"
  session.mkdir(parents=True)
  (session / "session.json").write_text(json.dumps({"task": "Shand"}))
  (session / "metrics.jsonl").write_text(
      json.dumps({"iteration": 4300, "goals_per_min": value}) + "\n")
  return session


def test_a_retrain_inside_the_band_counts_as_reproduced(run_with_a_session):
  store, tmp_path, _ = run_with_a_session
  build(store, "002", tmp_path / "recipe")

  report = verify(tmp_path / "recipe", _candidate(tmp_path, "again", 15.9))

  assert report["reproduced"] is True
  assert report["outside"] == 0
  assert "statistically equivalent" in report["verdict"]


def test_a_retrain_outside_the_band_is_reported_not_rounded_off(
    run_with_a_session):
  store, tmp_path, _ = run_with_a_session
  build(store, "002", tmp_path / "recipe")

  report = verify(tmp_path / "recipe", _candidate(tmp_path, "worse", 4.0))

  assert report["reproduced"] is False
  assert report["outside"] == 1
  check = report["checks"][0]
  assert check["metric"] == "goals_per_min"
  assert check["status"] == "outside"
  assert check["got"] == 4.0


def test_a_metric_written_as_text_is_still_compared(run_with_a_session):
  """`record close` takes metrics as text, so "16.8" is the normal shape."""
  store, tmp_path, _ = run_with_a_session
  build(store, "002", tmp_path / "recipe")

  report = verify(tmp_path / "recipe", _candidate(tmp_path, "again", 16.8))

  assert report["compared"] == 1
  # Reported as the number it was compared against, not as the text it was
  # written as: the check is arithmetic, and the report should show what the
  # arithmetic used.
  assert report["checks"][0]["expected"] == 16.8
  assert report["checks"][0]["status"] == "within"
  assert report["reproduced"] is True


def test_a_metric_the_retrain_never_published_is_missing_not_failed(
    run_with_a_session):
  """Usually a run that has not got far enough; calling it a regression is wrong."""
  store, tmp_path, _ = run_with_a_session
  build(store, "002", tmp_path / "recipe")
  session = tmp_path / "short" / "rlmcp"
  session.mkdir(parents=True)
  (session / "session.json").write_text(json.dumps({"task": "Shand"}))
  (session / "metrics.jsonl").write_text(
      json.dumps({"iteration": 10, "reward": 0.2}) + "\n")

  report = verify(tmp_path / "recipe", session)

  assert report["missing"] == 1
  assert report["compared"] == 0
  assert report["reproduced"] is False      # no evidence is not a pass
  assert "no claimed metrics" in report["verdict"]


def test_verify_needs_a_recipe_and_a_run_and_says_which_is_wrong(
    run_with_a_session, tmp_path):
  store, bundle_path, _ = run_with_a_session
  build(store, "002", bundle_path / "recipe")

  with pytest.raises(ValueError, match=r"expect\.json"):
    verify(bundle_path / "not-a-recipe", _candidate(bundle_path, "x", 1.0))
  with pytest.raises(ValueError, match="No metrics"):
    verify(bundle_path / "recipe", bundle_path / "nowhere")


# The three ways a recipe silently described a different run than the one it
# was built from, each reproduced on a real record first.


def _session_with_events(tmp_path, events, ladder, checkpoints=()):
  """A run whose event log names its rung entries the way the controller does."""
  run = tmp_path / "logs" / "run"
  session = run / "rlmcp"
  session.mkdir(parents=True)
  (session / "session.json").write_text(json.dumps({"task": "Shand"}))
  (session / "events.jsonl").write_text(
      "".join(json.dumps(e) + "\n" for e in events))
  (run / "params").mkdir()
  (run / "params" / "curriculum.json").write_text(json.dumps(list(ladder.values())))
  for name in checkpoints:
    path = session.parent / name if "/" not in name else session / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"weights")
  return session


THREE_RUNGS = {
    "0_turn": {"name": "0_turn", "parameters": {"reward.smooth.weight": -0.005}},
    "1_hold": {"name": "1_hold", "parameters": {"reward.fall.weight": -180.0}},
    "2_precise": {"name": "2_precise", "parameters": {}},
}


def _stage_event(iteration, frm, to):
  # Exactly the shape RlMcp._apply_stage writes: the transition as from/to.
  return {"t": 0.0, "kind": "curriculum_stage", "iteration": iteration,
          "from": frm, "to": to, "applied": {}, "notes": ""}


def _edit_event(iteration, key, old, new):
  return {"t": 0.0, "kind": "set_parameter", "iteration": iteration, "key": key,
          "old": old, "new": new, "applied": True, "rationale": "too taxing"}


def test_an_edit_folds_into_the_rung_the_event_log_says_was_active(bundle, tmp_path):
  """Run 010: three edits made on the smoothness rung came out folded into
  every rung, rung 0 included, because the stage lines could not be read."""
  store, _, _ = bundle
  session = _session_with_events(tmp_path, [
      _stage_event(0, "-", "0_turn"),
      _stage_event(1000, "0_turn", "1_hold"),
      _edit_event(1500, "reward.smooth.weight", -0.6, -0.01),
      _stage_event(2000, "1_hold", "2_precise"),
  ], THREE_RUNGS)
  store.update_record("002", lambda r: setattr(r, "session", str(session)))

  payload = build(store, "002", tmp_path / "recipe", policy=False)

  ladder = json.loads((tmp_path / "recipe" / "curriculum.json").read_text())
  by_name = {s["name"]: s for s in ladder["stages"]}
  assert by_name["1_hold"]["parameters"]["reward.smooth.weight"] == -0.01
  assert by_name["0_turn"]["parameters"] == {"reward.smooth.weight": -0.005}
  assert by_name["2_precise"]["parameters"] == {}
  assert not [m for m in payload["missing"] if "edit" in m]


def test_edits_are_named_not_smeared_when_the_log_has_no_rung_entries(
    bundle, tmp_path):
  """A ladder whose entries were never logged must not start every rung at the
  value of an edit made on one of them."""
  store, _, _ = bundle
  session = _session_with_events(tmp_path, [
      _edit_event(1500, "reward.smooth.weight", -0.6, -0.01),
  ], THREE_RUNGS)
  store.update_record("002", lambda r: setattr(r, "session", str(session)))

  payload = build(store, "002", tmp_path / "recipe", policy=False)

  ladder = json.loads((tmp_path / "recipe" / "curriculum.json").read_text())
  for stage in ladder["stages"]:
    assert stage["parameters"].get("reward.smooth.weight") != -0.01, stage["name"]
  assert any("not folded" in m and "reward.smooth.weight" in m
             for m in payload["missing"])


def test_the_policy_is_the_checkpoint_the_run_ended_on_not_a_named_one(
    bundle, tmp_path):
  """`<session>/checkpoints/` holds what an agent saved by name mid-run;
  the run's final weights are in the run directory above it."""
  store, _, _ = bundle
  session = _session_with_events(
      tmp_path, [], THREE_RUNGS,
      checkpoints=["checkpoints/pre-smoothness-fix.pt", "model_final_6906.pt"])
  store.update_record("002", lambda r: setattr(r, "session", str(session)))

  payload = build(store, "002", tmp_path / "recipe")

  assert payload["policy"].endswith("model_final_6906.pt")
  expect = json.loads((tmp_path / "recipe" / "expect.json").read_text())
  assert expect["policy"] == {"file": "model_final_6906.pt", "iteration": 6906,
                              "size_mb": expect["policy"]["size_mb"]}


def test_verify_matches_a_claimed_name_to_its_namespaced_telemetry_key(
    run_with_a_session):
  """A record says `goals_per_min`; the run publishes `rlmcp/goals_per_min`."""
  store, tmp_path, _ = run_with_a_session
  build(store, "002", tmp_path / "recipe")
  session = tmp_path / "again" / "rlmcp"
  session.mkdir(parents=True)
  (session / "session.json").write_text(json.dumps({"task": "Shand"}))
  (session / "metrics.jsonl").write_text(
      json.dumps({"iteration": 4300, "rlmcp/goals_per_min": 16.0}) + "\n")

  report = verify(tmp_path / "recipe", session)

  assert report["compared"] == 1
  assert report["checks"][0]["key"] == "rlmcp/goals_per_min"
  assert report["reproduced"] is True


# launch.sh is a launch, not a document: recipe.json carries the recipe,
# `rlmcp train --recipe` reads it, and the record for the rerun says what it is.


def test_the_manifest_carries_what_a_launcher_needs(run_with_a_session):
  """Run 010's launch.sh had no task package, never applied curriculum.json or
  config.json, and did not say how long to train. Everything a launcher needs
  is in recipe.json now, and launch.sh is one command over it."""
  store, tmp_path, _ = run_with_a_session

  payload = build(store, "002", tmp_path / "recipe")

  manifest = json.loads((tmp_path / "recipe" / "recipe.json").read_text())
  assert manifest["task"] == "Shand"
  assert manifest["task_packages"] == ["shand.tasks", "shand.rlmcp_ext"]
  assert manifest["seed"] == 7
  assert manifest["iterations"] == 4300              # where the original stopped
  assert manifest["num_envs"] == 4096
  assert manifest["package"] == "package"            # the restored task package
  assert manifest["policy"] == "policy/model_4300.pt"
  assert manifest["expect"] == {"goals_per_min": "16.8"}
  assert manifest["from_run"] == "002"
  script = (tmp_path / "recipe" / "launch.sh").read_text()
  assert 'rlmcp train --recipe "$HERE"' in script
  assert not [m for m in payload["missing"] if "seed" in m or "packages" in m]


def test_a_recipe_says_which_packages_it_does_not_know(bundle, tmp_path):
  """A run made before task packages were recorded still gets a launchable
  recipe, with the gap named in `missing` and launch.sh reading $TASK_PACKAGES."""
  store, _, _ = bundle
  session = _session_with_events(tmp_path, [], THREE_RUNGS)
  (session.parent / "params" / "env.yaml").write_text(json.dumps({"seed": 42}))
  store.update_record("002", lambda r: setattr(r, "session", str(session)))

  payload = build(store, "002", tmp_path / "recipe", policy=False)

  manifest = json.loads((tmp_path / "recipe" / "recipe.json").read_text())
  assert manifest["task_packages"] == [] and manifest["seed"] == 42
  assert 'read -r -a TASK_PACKAGES <<< "${TASK_PACKAGES:-}"' in (
      tmp_path / "recipe" / "launch.sh").read_text()
  assert any(m.startswith("the task packages") for m in payload["missing"])


def test_the_rerun_record_is_named_after_the_recipe_not_numbered(run_with_a_session):
  """`recipe-002`, then `recipe-002-2`: an id that says what the run is."""
  from rlmcp.records.recipe import load_manifest, open_reproduction_record
  store, tmp_path, _ = run_with_a_session
  build(store, "002", tmp_path / "recipe")
  manifest = load_manifest(tmp_path / "recipe")

  first = open_reproduction_record(store, manifest, tmp_path / "recipe")
  second = open_reproduction_record(store, manifest, tmp_path / "recipe")

  assert (first.id, second.id) == ("recipe-002", "recipe-002-2")
  assert first.parent == "002" and first.task == "Shand"
  assert "16.8" in first.prediction and first.proposed_by == "recipe"
  # `recipe-002-*` also matches the second's directory; the store must not
  # hand back the wrong record for a named id that prefixes another.
  assert store.get_record("recipe-002").id == "recipe-002"
  assert store.get_record("recipe-002-2").id == "recipe-002-2"
  assert (tmp_path / "records" / "runs" / "recipe-002-recipe_002" / "PLAN.md").exists()


def test_a_recipe_with_no_manifest_is_refused_by_name(tmp_path):
  from rlmcp.records.recipe import load_manifest
  with pytest.raises(ValueError, match=r"recipe\.json"):
    load_manifest(tmp_path)
