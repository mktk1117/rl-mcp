"""Reading back what was done to a run, in the run's own vocabulary."""

from __future__ import annotations

import pytest
from conftest import FakeSimAdapter

from rlmcp.core.controller import RlMcp
from rlmcp.records.interventions import KINDS, from_events, from_session
from rlmcp.session import Session


def test_only_decisions_count_as_interventions():
  """A rendered clip and a finished job are things that happened, not things
  anybody decided -- and the log is mostly those."""
  chosen = from_events([
      {"kind": "session_start", "t": 1},
      {"kind": "progress_clip", "iteration": 200, "t": 2},
      {"kind": "job_complete", "iteration": 201, "t": 3},
      {"kind": "telemetry_drop", "iteration": 202, "t": 4},
      {"kind": "set_parameter", "iteration": 900, "key": "rl.entropy_coef",
       "old": 0.005, "new": 0.01, "applied": True, "t": 5},
  ])

  assert [i["kind"] for i in chosen] == ["set_parameter"]


def test_each_intervention_carries_the_reason_given_at_the_time():
  """The reason is the whole value: 'raised at 900' is trivia, 'raised at 900
  because entropy had collapsed' is the answer to what happened."""
  chosen = from_events([
      {"kind": "set_parameter", "iteration": 900, "key": "rl.entropy_coef",
       "old": 0.005, "new": 0.01, "applied": True,
       "rationale": "entropy collapsed", "t": 1},
      {"kind": "curriculum_stage", "iteration": 1500, "stage": "1_rough",
       "reason": "flat is solved", "t": 2},
  ])

  assert chosen[0]["what"] == "rl.entropy_coef: 0.005 → 0.01"
  assert chosen[0]["why"] == "entropy collapsed"
  assert chosen[1]["what"] == "stage → 1_rough"
  assert chosen[1]["why"] == "flat is solved"


def test_a_refused_edit_is_shown_as_refused_rather_than_as_a_change():
  """It is in the log precisely because it did not happen."""
  chosen = from_events([
      {"kind": "set_parameter", "iteration": 10, "key": "reward.w",
       "old": 1.0, "new": 99.0, "applied": False, "t": 1}])

  assert "refused" in chosen[0]["what"]


def test_they_come_back_in_the_order_they_happened():
  chosen = from_events([
      {"kind": "note", "iteration": 1500, "text": "later", "t": 9},
      {"kind": "note", "iteration": 20, "text": "earlier", "t": 1},
      {"kind": "note", "iteration": 1500, "text": "last", "t": 11},
  ])

  assert [i["what"] for i in chosen] == ["earlier", "later", "last"]


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_every_listed_kind_phrases_itself_without_crashing(kind):
  """A kind in the table with no phrasing would print its own name at a user."""
  chosen = from_events([{"kind": kind, "iteration": 1, "text": "said",
                         "keys": ["a"], "t": 1}])
  assert chosen and chosen[0]["what"] and chosen[0]["layer"]


def test_a_real_run_reports_what_was_done_to_it(tmp_path):
  """End to end through the controller, so the kinds cannot drift from what
  rlmcp actually writes."""
  lab = RlMcp(sim_adapter=FakeSimAdapter(), session_dir=tmp_path / "session",
              video_every=0)
  try:
    lab.service(iteration=5)
    lab.run_command("set_parameter", key="reward.action_rate_l2.weight",
                    value=-0.25, rationale="ankles chattering")
    lab.run_command("note", text="looks better")
    chosen = from_session(lab.session.dir)
  finally:
    lab.close()

  assert [i["kind"] for i in chosen] == ["set_parameter", "note"]
  assert chosen[0]["why"] == "ankles chattering"
  assert chosen[0]["iteration"] == 5


def test_a_session_with_no_events_is_empty_not_an_error(tmp_path):
  session = Session(tmp_path / "empty").create({"task": "x"})
  assert from_session(session.dir) == []
