"""What each parameter did, along the tree.

The x-axis is the whole point and the easy thing to get wrong: every run starts
its own iteration counter at zero, so plotting raw iteration stacks the runs on
top of each other and hides the ancestry. These pin the offsetting, the three
sources a value can come from, and the two places the naive version lies.
"""

from __future__ import annotations

import json

from rlmcp.records.graph import build
from rlmcp.records.params import build_history, leaf_paths
from rlmcp.records.record import RunRecord


def _r(rid: str, seq: int, parent=None, verdict="provisional", **kw) -> RunRecord:
  return RunRecord(id=rid, slug=f"run_{rid}", seq=seq, parent=parent,
                   verdict=verdict, **kw)


def _session(tmp_path, name: str, iterations: int, events) -> str:
  """A session on disk with just enough in it to be opened and read."""
  d = tmp_path / name
  d.mkdir(parents=True, exist_ok=True)
  (d / "session.json").write_text(json.dumps({"pid": 1, "started_at": 0.0}))
  (d / "status.json").write_text(json.dumps({"iteration": iterations}))
  (d / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
  return str(d)


def test_a_launch_snapshot_alone_draws_a_flat_line_per_run():
  """A key the run never touched still has a value, and a flat line at the
  inherited value is information: it says nobody moved it."""
  records = [_r("001", 1, config={"reward.a.weight": 1.0, "mode": "adaptive"}),
             _r("002", 2, parent="001", config={"reward.a.weight": 1.0})]

  history = build_history(build(records))

  assert [r["id"] for r in history["runs"]] == ["001", "002"]
  assert history["runs"][0]["series"]["reward.a.weight"] == [
      {"x": 0, "v": 1.0, "source": "launch"}]
  # A string-valued setting is worth listing as changed, never coerced into a line.
  assert "mode" not in history["runs"][0]["series"]
  assert [e["key"] for e in history["index"]] == ["reward.a.weight"]
  assert history["index"][0]["changes"] == 0


def test_relaunching_with_a_different_value_counts_as_a_change():
  """A weight that moved because the next run was configured differently moved
  just as surely as one edited mid-run -- and that is where most of a tuning
  history actually lives."""
  records = [_r("001", 1, config={"reward.a.weight": 1.0}),
             _r("002", 2, parent="001", config={"reward.a.weight": 4.0})]

  entry = build_history(build(records))["index"][0]

  assert entry["changes"] == 1
  assert entry["relaunched_in"] == ["002"]
  assert (entry["first"], entry["last"]) == ("1", "4")


def test_each_run_is_offset_by_where_its_config_parent_finished(tmp_path):
  """Two children of one run start at the same x and diverge, which is what
  makes a fork read as a fork rather than as two overlaid runs."""
  parent = _r("001", 1, config={"reward.a.weight": 1.0},
              session=_session(tmp_path, "one", 600, []))
  left = _r("002", 2, parent="001", config={"reward.a.weight": 2.0},
            session=_session(tmp_path, "two", 300, []))
  right = _r("003", 3, parent="001", config={"reward.a.weight": 3.0},
             session=_session(tmp_path, "three", 100, []))

  runs = {r["id"]: r for r in build_history(build([parent, left, right]))["runs"]}

  assert (runs["001"]["offset"], runs["001"]["end"]) == (0, 600)
  assert runs["002"]["offset"] == runs["003"]["offset"] == 600
  assert runs["002"]["end"] == 900 and runs["003"]["end"] == 700


def test_curriculum_and_manual_edits_are_both_steps_and_say_which(tmp_path):
  """The three sources are the launch snapshot, the ladder moving a knob on its
  own, and a live edit with the rationale that was typed. A viewer that cannot
  tell them apart cannot answer "who decided this?"."""
  events = [
      {"kind": "curriculum_stage", "iteration": 400, "to": "1_rough",
       "reason": "goal rate held", "applied": {"parameters": {"reward.a.weight": 2.0}}},
      {"kind": "set_parameter", "iteration": 620, "key": "reward.a.weight",
       "new": 3.0, "applied": True, "rationale": "it plateaued"},
  ]
  record = _r("001", 1, config={"reward.a.weight": 1.0},
              session=_session(tmp_path, "one", 800, events))

  points = build_history(build([record]))["runs"][0]["series"]["reward.a.weight"]

  assert [(p["x"], p["v"], p["source"]) for p in points] == [
      (0, 1.0, "launch"), (400, 2.0, "curriculum"), (620, 3.0, "manual"),
      (800, 3.0, "end"),  # carried to where the run actually stopped
  ]
  assert points[1]["stage"] == "1_rough"
  assert points[2]["why"] == "it plateaued"


def test_an_edit_at_iteration_zero_replaces_the_launch_value(tmp_path):
  """The curriculum's opening stage is applied before the first rollout, so it
  is what the run trained with. Keeping both draws a vertical spike at a value
  no policy ever saw; the config snapshot survives as config_value."""
  events = [{"kind": "curriculum_stage", "iteration": 0, "to": "0_flat",
             "reason": "opening stage",
             "applied": {"parameters": {"reward.a.weight": -30.0}}}]
  record = _r("001", 1, config={"reward.a.weight": -55.0},
              session=_session(tmp_path, "one", 200, events))

  points = build_history(build([record]))["runs"][0]["series"]["reward.a.weight"]

  assert points[0] == {"x": 0, "v": -30.0, "source": "curriculum",
                       "why": "stage 0_flat: opening stage", "stage": "0_flat",
                       "config_value": -55.0}


def test_a_stage_that_restates_a_value_is_not_drawn_as_a_change(tmp_path):
  """A ladder re-applying the value it already has is the ladder restating
  itself, and drawing it as a step is a lie about what happened."""
  events = [{"kind": "curriculum_stage", "iteration": 300, "to": "1_same",
             "reason": "promotion", "applied": {"parameters": {"reward.a.weight": 1.0}}}]
  record = _r("001", 1, config={"reward.a.weight": 1.0},
              session=_session(tmp_path, "one", 500, events))

  run = build_history(build([record]))["runs"][0]

  assert [p["source"] for p in run["series"]["reward.a.weight"]] == ["launch", "end"]
  assert run["changed"] == []


def test_a_session_that_is_gone_still_draws_its_launch_snapshot():
  """An ancestry whose logs were cleaned up still renders; it just loses the
  edits, which is a smaller loss than losing the page."""
  record = _r("001", 1, config={"reward.a.weight": 1.0},
              session="/nowhere/at/all")

  run = build_history(build([record]))["runs"][0]

  assert run["iterations"] == 0 and run["live"] is False
  assert run["series"]["reward.a.weight"] == [{"x": 0, "v": 1.0, "source": "launch"}]


def test_only_a_running_record_with_a_living_process_counts_as_live():
  """"Still training" is a fact about the process, not about the verdict: a run
  whose trainer exited an hour ago is not an active path just because nobody
  has closed the record yet."""
  record = _r("001", 1, verdict="running", config={"reward.a.weight": 1.0},
              session="/nowhere/at/all")

  run = build_history(build([record]))["runs"][0]

  assert run["open"] is True
  assert run["live"] is False


def test_leaf_paths_are_the_chains_a_viewer_lights_up():
  """The highlight is a path, not a node: "how did we get here" is a question
  about the chain of runs that produced it."""
  records = [_r("001", 1), _r("002", 2, parent="001"),
             _r("003", 3, parent="001"), _r("004", 4, parent="003")]

  paths = leaf_paths(build(records))

  assert paths == {"002": ["001", "002"], "004": ["001", "003", "004"]}


def test_the_index_is_ordered_by_how_much_a_parameter_moved():
  """A parameter that never moved is noise in a chooser built to answer "what
  did we tune?", so the busiest key sorts first and the still ones sink."""
  records = [_r("001", 1, config={"a": 1.0, "b": 1.0, "c": 1.0}),
             _r("002", 2, parent="001", config={"a": 2.0, "b": 1.0, "c": 2.0}),
             _r("003", 3, parent="002", config={"a": 3.0, "b": 1.0, "c": 2.0})]

  index = build_history(build(records))["index"]

  assert [e["key"] for e in index] == ["a", "c", "b"]
  assert [e["changes"] for e in index] == [2, 1, 0]
  assert (index[0]["min"], index[0]["max"]) == (1.0, 3.0)
